#!/usr/bin/env python3
"""Inspect saved single-circuit checkpoints for low-rank/shared-basis potential.

Read-only diagnostics, not retraining or compressed-model accuracy evaluation.
CUDA is the default. Metadata-only preflight does not import torch.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK = ROOT / 'final_exam/single_circuit_pin'
SUPPORTED = ('device_op_cascade_pin', 'width_decoder_pin', 'width_decoder_op_pin', 'zerosim_supervised_pin')


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def write_json(path, value):
    # All probe directories are newly reserved; snapshots never touch sources.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inspect_sources_pin(sources, models, seeds, num_samples):
    """Check every source before sbatch/output creation, without deserialization."""
    from final_exam.gate_probe_pin import identifier, select_ids
    if not models or len(set(models)) != len(models) or set(models) - set(SUPPORTED):
        raise ValueError(f'Choose unique supported models: {", ".join(SUPPORTED)}')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Choose unique checkpoint seeds in [0, 2**32)')
    if type(num_samples) is not int or num_samples < 0:
        raise ValueError('num_samples must be nonnegative; 0 means weights only')
    result, circuits = [], set()
    for entry in sources:
        if '=' not in entry:
            raise ValueError('Each --source must be CIRCUIT=RESULT_PATH')
        circuit, source = entry.split('=', 1)
        identifier(circuit)
        if not source.strip() or circuit in circuits:
            raise ValueError('Supply one nonempty source run per unique circuit')
        circuits.add(circuit)
        source = Path(source).expanduser().resolve()
        training = source / 'training_pin' if (source / 'training_pin/run_pin.json').is_file() else source
        record = read_json(training / 'run_pin.json')
        if record.get('circuit') != circuit or record.get('granularity') != 'pin':
            raise ValueError(f'Source circuit/granularity mismatch: {source}')
        if not set(models) <= set(record.get('models', [])) or not set(seeds) <= set(record.get('seeds', [])):
            raise ValueError(f'{circuit}: requested models/seeds are absent from this run')
        config = read_json(training / 'config_pin.json')
        if config.get('dataset', {}).get('train_circuits') != [circuit] or config['dataset'].get('ood_circuits'):
            raise ValueError('This probe compares single-circuit ID models; do not include held-out OOD models in a fitted basis')
        manifest = read_json(training / 'manifest_pin.json')
        checkpoints = []
        for model, seed in itertools.product(models, seeds):
            path = training / f'seed_{seed}_pin' / model / 'checkpoint_pin.pt'
            if not path.is_file():
                raise ValueError(f'Completed model checkpoint is missing: {path}')
            checkpoints.append(dict(model=model, seed=seed, path=str(path)))
        data = None
        if num_samples:
            select_ids(manifest.get('train'), num_samples, 17)
            data = training.parent / 'data_pin.json'
            if not data.is_file():
                pointer = read_json(training.parent / 'data_source_pin.json')
                if pointer.get('dataset_fingerprint') != record.get('dataset_fingerprint'):
                    raise ValueError('Data pointer fingerprint differs from source run')
                if not isinstance(pointer.get('data_path'), str) or not pointer['data_path']:
                    raise ValueError('Missing data_path in saved data pointer')
                data = Path(pointer['data_path']).expanduser()
                if not data.is_absolute():
                    data = training.parent / data
            data = data.resolve()
            if not data.is_file():
                raise ValueError(f'Calibration dataset is missing: {data}; set N_SAMPLES=0 for weights only')
            if not record.get('dataset_fingerprint'):
                raise ValueError('Source run lacks a dataset fingerprint')
        result.append(dict(circuit=circuit, training=str(training), source_run=str(training.parent),
            data=None if data is None else str(data), manifest=manifest,
            dataset_fingerprint=record.get('dataset_fingerprint'), checkpoints=checkpoints,
            source_code_commit=record.get('code_commit')))
    if not result:
        raise ValueError('Supply at least one --source CIRCUIT=RESULT_PATH')
    return result


def calibration_rows_pin(source, count, seed):
    from device_jepa.data import dataset_fingerprint, load_dataset, verify_manifest
    from final_exam.gate_probe_pin import select_ids
    if not count:
        return None, None, []
    print(f'Loading and validating {source["circuit"]} canonical JSON on CPU; keeping {count} train samples', flush=True)
    dataset = load_dataset(source['data'], sampling_config=source['manifest'].get('sampling'))
    verify_manifest(dataset, source['manifest'])
    if dataset_fingerprint(dataset) != source['dataset_fingerprint']:
        raise ValueError('Dataset changed since training')
    selected = select_ids(source['manifest']['train'], count, seed)
    indexed = {r['sample_id']: r for r in dataset['samples']}
    rows = [indexed[s] for s in selected]
    if any(r['circuit_id'] != source['circuit'] for r in rows):
        raise ValueError('Calibration sample belongs to a different circuit')
    return rows, dataset['feature_spec'], selected


def capture_activations_pin(model, metadata, specs, rows, feature_spec, device, batch_size, max_tokens, seed):
    import math
    import torch
    from final_exam.data_pin import PinResidentDataset
    from final_exam.spectral_analysis_pin import ActivationCapturePin
    if rows is None:
        return None
    prep = metadata['preprocessing']
    if prep.get('raw_feature_spec', prep['feature_spec']) != feature_spec:
        raise ValueError('Saved checkpoint feature order differs from its dataset')
    if metadata.get('requires_measured_op'):
        raise ValueError('This calibration is design-input-only; measured OP is not permitted')
    cache = PinResidentDataset(rows, prep, device, include_op=False, include_performance=False,
        max_tokens=metadata['model_config']['max_devices'], label='spectral calibration')
    capture = ActivationCapturePin(specs, max_tokens=max_tokens, seed=seed)
    batches = math.ceil(len(rows) / batch_size)
    with capture.install(), torch.inference_mode():
        for index, start in enumerate(range(0, len(rows), batch_size)):
            ids = list(range(start, min(start + batch_size, len(rows))))
            _, batch = next(cache.batches(len(ids), indices=ids, metadata=True))
            capture.set_batch(batch, index, batches)
            prediction = model(batch)
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError('Non-finite checkpoint prediction during calibration')
            print(f'  calibration {start + len(ids)}/{len(rows)} train samples', flush=True)
    return capture


def run_probe_pin(*, sources, models, seeds, ranks=(8, 16, 32, 64), num_samples=128,
                  max_tokens=512, sampling_seed=17, batch_size=16, scope='ffn', component='both',
                  device='cuda', output=None, log_dir=None, max_plots=12):
    import torch
    from device_jepa.resident import require_device
    from device_jepa.training import state_digest
    from final_exam.logging_utils import log_to
    from final_exam.run_pin import load_checkpoint_pin
    from final_exam.spectral_analysis_pin import (matrices_pin, parameter_counts_pin,
        analyze_matrix_pin, shared_basis_pin, subspace_overlap_pin)
    from final_exam.spectral_report_pin import write_report_pin, require_plotting_pin
    inspected = inspect_sources_pin(sources, models, seeds, num_samples)
    if not ranks or any(type(r) is not int or r < 1 for r in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError('Choose unique positive ranks')
    ranks = sorted(ranks)
    if any(type(v) is not int or v < 1 for v in (max_tokens, batch_size)) or max_plots < 0:
        raise ValueError('max_tokens/batch_size must be positive, max_plots nonnegative')
    if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32:
        raise ValueError('sampling_seed must be in [0, 2**32)')
    if scope not in ('ffn', 'all') or component not in ('both', 'performance', 'op_predictor'):
        raise ValueError('Invalid matrix scope/component')
    device = require_device(device)
    if max_plots:
        require_plotting_pin()  # dependency check before any output writes
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = f'spectral_probe_{stamp}_{uuid.uuid4().hex[:8]}_pin'
    output = Path(output or TASK / 'results_pin' / name).resolve()
    logs = Path(log_dir or TASK / 'logs_pin' / output.name).resolve()
    for destination in (output, logs):
        if destination.exists():
            raise ValueError(f'Use a new output/log directory: {destination}')
        for source in inspected:
            protected = Path(source['source_run'])
            if destination.is_relative_to(protected) or protected.is_relative_to(destination):
                raise ValueError('Probe outputs must be separate from every source run')
    if output.is_relative_to(logs) or logs.is_relative_to(output):
        raise ValueError('Output/log directories must be separate')
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', version=1, sources=sources, models=models, seeds=seeds,
        ranks=ranks, num_samples=num_samples, max_tokens_per_matrix=max_tokens,
        sampling_seed=sampling_seed, batch_size=batch_size, scope=scope, component=component,
        device=str(device), svd_dtype='float64', max_plots=max_plots,
        calibration_split='train_only', coordinates='saved_no_alignment',
        checkpoint_modified=False, retrained=False, compressed_model_evaluated=False,
        basis_fit='unit_Frobenius_per_circuit_joint_row_column_SVD',
        selected_sample_ids={}, checkpoints=[], results=str(output), logs=str(logs))
    tables = defaultdict(list)
    write_json(output / 'probe_pin.json', record)
    try:
        with log_to(logs / 'spectral_probe_pin.log'):
            # Selected raw rows only are cached across model/seed groups.
            calibration = {}
            for source in inspected:
                rows, features, ids = calibration_rows_pin(source, num_samples, sampling_seed)
                calibration[source['circuit']] = (rows, features)
                record['selected_sample_ids'][source['circuit']] = ids
            for model_name, seed in itertools.product(models, seeds):
                print(f'Spectral group: {model_name}, seed={seed}', flush=True)
                group = defaultdict(list)
                for source in inspected:
                    circuit = source['circuit']
                    checkpoint = next(c['path'] for c in source['checkpoints'] if c['model'] == model_name and c['seed'] == seed)
                    identity = dict(circuit=circuit, model=model_name, seed=seed)
                    print(f'Loading {circuit}: {checkpoint}', flush=True)
                    model, saved = load_checkpoint_pin(checkpoint, str(device))
                    if (saved.get('mode') != model_name or saved.get('seed') != seed or
                            saved.get('manifest') != source['manifest']):
                        raise ValueError('Checkpoint identity/split differs from run metadata')
                    saved.pop('state_dict', None)
                    before = state_digest(model)
                    specs = matrices_pin(model, scope, component)
                    rows, features = calibration[circuit]
                    captures = capture_activations_pin(model, saved, specs, rows, features,
                        device, batch_size, max_tokens, sampling_seed)
                    config = saved['model_config']
                    # Capacity limits do not change shared operator dimensions.
                    architecture = {k: v for k, v in config.items() if k != 'max_devices'}
                    architecture['width_decoder_pin'] = saved.get('config', {}).get('width_decoder_pin', {})
                    signature = json.dumps(architecture, sort_keys=True)
                    entry = dict(**identity, path=checkpoint, sha256=sha256(checkpoint),
                        total_parameters=sum(p.numel() for p in model.parameters()),
                        analyzed_parameters=sum(s.weight.numel() for s in specs),
                        calibration_samples=0 if rows is None else len(rows),
                        model_config=config, config=saved.get('config'),
                        source_code_commit=source['source_code_commit'],
                        normalization_refitted=False, preprocessing_sha256=hashlib.sha256(
                            json.dumps(saved['preprocessing'], sort_keys=True).encode()).hexdigest())
                    record['checkpoints'].append(entry)
                    tables['Parameters'].extend(dict(**identity, **v) for v in parameter_counts_pin(model))
                    for index, spec in enumerate(specs):
                        activation = None if captures is None else captures.get(spec.name)
                        summary, independent, item = analyze_matrix_pin(spec, ranks, activation)
                        tables['Matrices'].append(dict(**identity, **summary))
                        tables['IndependentLowRank'].extend(dict(**identity, matrix=spec.name, **v) for v in independent)
                        singular = item['singular']
                        energy = singular.square()
                        cumulative = (energy.cumsum(0) / energy.sum()).tolist() if float(energy.sum()) else [0.] * len(singular)
                        tables['SingularValues'].extend(dict(**identity, matrix=spec.name, index=i+1,
                            singular_value=float(s), cumulative_energy=e) for i, (s, e) in enumerate(zip(singular, cumulative)))
                        if captures is not None and activation is None:
                            tables['Notes'].append(dict(**identity, matrix=spec.name,
                                reason='Activation unavailable: functional projection/not executed; blank is not zero'))
                        item.update(identity=identity, architecture=signature)
                        group[(spec.name, tuple(spec.weight.shape), signature)].append(item)
                        print(f'  SVD {index+1}/{len(specs)} {spec.name}: rank99={summary["rank_99"]}/{min(spec.weight.shape)}', flush=True)
                    if state_digest(model) != before:
                        raise ValueError('Read-only probe changed model parameters/buffers')
                    del captures, specs, model, saved, activation
                    write_json(output / 'probe_pin.json', record)
                for (matrix, shape, signature), items in group.items():
                    common = dict(model=model_name, seed=seed, matrix=matrix)
                    if len(items) < 2:
                        tables['Notes'].append(dict(**common, circuit=items[0]['identity']['circuit'],
                            reason='No second circuit with matching matrix shape and architecture; shared basis not fitted'))
                        continue
                    print(f'  Shared basis: {matrix}, {len(items)} circuits', flush=True)
                    for left, right in itertools.combinations(items, 2):
                        for rank in ranks:
                            tables['SubspaceOverlap'].append(dict(**common,
                                circuit_a=left['identity']['circuit'], circuit_b=right['identity']['circuit'],
                                **subspace_overlap_pin(left, right, rank, device)))
                    groups, errors = shared_basis_pin(items, ranks, device)
                    circuits = ', '.join(item['identity']['circuit'] for item in items)
                    tables['SharedBudget'].extend(dict(**common, circuit_names=circuits, **v) for v in groups)
                    for error in errors:
                        index = error.pop('item_index')
                        tables['SharedErrors'].append(dict(**common, circuit=items[index]['identity']['circuit'],
                            fit_circuits=circuits, **error))
                del group
            write_report_pin(output, tables, record, max_plots=max_plots)
            record['status'] = 'complete'
            write_json(output / 'probe_pin.json', record)
            print(f'Excel: {output / "spectral_summary_pin.xlsx"}\nPlots: {output / "report_pin.html"}', flush=True)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        write_json(output / 'probe_pin.json', record)
        with (logs / 'spectral_probe_pin.log').open('a', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, dest='sources', metavar='CIRCUIT=RESULT_PATH')
    parser.add_argument('--models', nargs='+', choices=SUPPORTED, default=list(SUPPORTED[:3]))
    parser.add_argument('--seeds', nargs='+', type=int, default=[17])
    parser.add_argument('--ranks', nargs='+', type=int, default=[8, 16, 32, 64])
    parser.add_argument('--num-samples', type=int, default=128, help='Saved ID train samples; 0 disables calibration/data loading')
    parser.add_argument('--max-tokens', type=int, default=512, help='Maximum actual activation vectors retained per matrix')
    parser.add_argument('--sampling-seed', type=int, default=17)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--scope', choices=('ffn', 'all'), default='ffn')
    parser.add_argument('--component', choices=('both', 'performance', 'op_predictor'), default='both')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-plots', type=int, default=12, help='0 skips matplotlib; CSV/Excel always contain every matrix')
    parser.add_argument('--output')
    parser.add_argument('--log-dir')
    parser.add_argument('--inspect-source', action='store_true')
    args = vars(parser.parse_args(argv))
    inspect = args.pop('inspect_source')
    try:
        if inspect:
            result = inspect_sources_pin(args['sources'], args['models'], args['seeds'], args['num_samples'])
            print(json.dumps([dict(circuit=s['circuit'], training=s['training'], checkpoints=len(s['checkpoints'])) for s in result]))
            return
        run_probe_pin(**args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
