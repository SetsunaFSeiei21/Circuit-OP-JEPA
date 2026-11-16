#!/usr/bin/env python3
"""Analyze shared down bases across PE decoders INSIDE each saved checkpoint.

Sources, models and seeds are independent analyses. No cross-circuit fitting,
training, test-set evaluation or checkpoint writes are performed. GPU is the
default for inference/SVD; metadata preflight requires only the standard library.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from final_exam.low_rank_registry_pin import LOW_RANK_MODELS_PIN
from final_exam.spectral_probe_pin import read_json, write_json, sha256

TASK = ROOT / 'final_exam/single_circuit_pin'
SUPPORTED = ('width_decoder_pin', 'width_decoder_op_pin', *LOW_RANK_MODELS_PIN)


def inspect_sources_pin(sources, models, seeds, num_samples=128, num_val_samples=128,
                        component='performance'):
    """Validate all source metadata before reserving outputs or submitting jobs."""
    from final_exam.gate_probe_pin import identifier, select_ids
    if not models or len(set(models)) != len(models) or set(models) - set(SUPPORTED):
        raise ValueError(f'Choose distinct multi-decoder models: {SUPPORTED}')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Checkpoint seeds must be unique integers in [0, 2**32)')
    if any(type(n) is not int or n < 0 for n in (num_samples, num_val_samples)):
        raise ValueError('Sample counts must be nonnegative; 0 disables that split')
    if component not in ('performance', 'op_predictor', 'both'):
        raise ValueError('Invalid component')
    if component == 'op_predictor' and any(m != 'width_decoder_op_pin' for m in models):
        raise ValueError('Only width_decoder_op_pin has multiple OP-predictor decoders')
    inspected, circuits = [], set()
    for entry in sources:
        if '=' not in entry:
            raise ValueError('Each source must be CIRCUIT=RESULT_PATH')
        circuit, raw = entry.split('=', 1)
        identifier(circuit)
        if not raw.strip() or circuit in circuits:
            raise ValueError('Use one nonempty source path per unique circuit')
        circuits.add(circuit)
        path = Path(raw).expanduser().resolve()
        training = path / 'training_pin' if (path / 'training_pin/run_pin.json').is_file() else path
        run = read_json(training / 'run_pin.json')
        config = read_json(training / 'config_pin.json')
        manifest = read_json(training / 'manifest_pin.json')
        if run.get('circuit') != circuit or run.get('granularity') != 'pin':
            raise ValueError(f'Source circuit/granularity mismatch: {path}')
        if config.get('dataset', {}).get('train_circuits') != [circuit] or config['dataset'].get('ood_circuits'):
            raise ValueError('Each source must be a saved single-circuit run')
        if not set(models) <= set(run.get('models', [])) or not set(seeds) <= set(run.get('seeds', [])):
            raise ValueError(f'{circuit}: requested model/seed absent from this run')
        checkpoints = []
        for model, seed in itertools.product(models, seeds):
            checkpoint = training / f'seed_{seed}_pin' / model / 'checkpoint_pin.pt'
            if not checkpoint.is_file():
                raise ValueError(f'Model checkpoint is missing: {checkpoint}')
            effective = read_json(checkpoint.parent / 'config_pin.json')
            if effective.get('model', {}).get('cross_layers', 0) < 2:
                raise ValueError('Shared decoder analysis requires at least two PE decoder branches')
            checkpoints.append(dict(model=model, seed=seed, path=str(checkpoint)))
        data = None
        for split, count in (('train', num_samples), ('val', num_val_samples)):
            if count:
                select_ids(manifest.get(split), count, 17)
        if num_samples or num_val_samples:
            data = training.parent / 'data_pin.json'
            if not data.is_file():
                pointer = read_json(training.parent / 'data_source_pin.json')
                if pointer.get('dataset_fingerprint') != run.get('dataset_fingerprint'):
                    raise ValueError('Saved data pointer fingerprint mismatch')
                if not isinstance(pointer.get('data_path'), str) or not pointer['data_path']:
                    raise ValueError('Saved data pointer lacks data_path')
                data = Path(pointer['data_path']).expanduser()
                if not data.is_absolute():
                    data = training.parent / data
            data = data.resolve()
            if not data.is_file() or not run.get('dataset_fingerprint'):
                raise ValueError('Canonical data or its saved fingerprint is unavailable; use both sample counts=0 for weights only')
        inspected.append(dict(circuit=circuit, source_run=str(training.parent), training=str(training),
            data=None if data is None else str(data), manifest=manifest, checkpoints=checkpoints,
            dataset_fingerprint=run.get('dataset_fingerprint'), source_code_commit=run.get('code_commit')))
    if not inspected:
        raise ValueError('Provide at least one source')
    return inspected


def selected_rows_pin(source, num_samples, num_val_samples, seed):
    from device_jepa.data import load_dataset, verify_manifest, dataset_fingerprint
    from final_exam.gate_probe_pin import select_ids
    selected = {split: select_ids(source['manifest'][split], count, seed) if count else []
                for split, count in (('train', num_samples), ('val', num_val_samples))}
    if not num_samples and not num_val_samples:
        return dict(train=[], val=[]), None, selected
    print(f'Loading {source["circuit"]} canonical JSON and verifying identity; retaining selected train/val rows', flush=True)
    dataset = load_dataset(source['data'], sampling_config=source['manifest'].get('sampling'))
    verify_manifest(dataset, source['manifest'])
    if dataset_fingerprint(dataset) != source['dataset_fingerprint']:
        raise ValueError('Canonical dataset changed since training')
    wanted = set(selected['train'] + selected['val'])
    indexed = {r['sample_id']: r for r in dataset['samples'] if r['sample_id'] in wanted}
    rows = {split: [indexed[sid] for sid in ids] for split, ids in selected.items()}
    if any(row['circuit_id'] != source['circuit'] for part in rows.values() for row in part):
        raise ValueError('Selected data includes a different circuit')
    return rows, dataset['feature_spec'], selected


def design_inputs_pin(batch):
    from final_exam.models.op_transfer_pin import design_input_pin
    from final_exam.op_gate_data_pin import _OWNER_INPUTS
    inputs = design_input_pin(batch)
    inputs.update({key: batch[key] for key in _OWNER_INPUTS if key in batch})
    return inputs


def resident_pin(rows, saved, device, *, labels):
    import torch
    from final_exam.data_pin import PinResidentDataset
    prep = saved['preprocessing']
    return PinResidentDataset(rows, prep, device, include_op=False, include_performance=labels,
        performance_metric_mask=[n > 0 for n in prep['performance']['count']] if labels else None,
        performance_dtype=torch.float64, max_tokens=saved['model_config']['max_devices'],
        label='shared decoder validation' if labels else 'shared decoder train calibration')


def capture_inputs_pin(model, groups, cache, batch_size, max_tokens, seed):
    import torch
    from final_exam.shared_decoder_analysis_pin import SharedDecoderCapturePin
    if cache is None:
        return None
    batches = math.ceil(len(cache.rows) / batch_size)
    with SharedDecoderCapturePin(groups, max_tokens=max_tokens, seed=seed) as capture, torch.inference_mode():
        for index, (_, batch) in enumerate(cache.batches(batch_size)):
            inputs = design_inputs_pin(batch)
            capture.set_batch(inputs, index, batches)
            prediction = model(inputs)
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError('Original checkpoint produced nonfinite calibration predictions')
            print(f'  train activation batch {index + 1}/{batches}', flush=True)
        return capture.finish()


def evaluate_validation_pin(model, cache, saved, batch_size):
    """Read labels only for scoring; capture decoder outputs on the same val rows."""
    import torch
    from device_jepa.model import masked_mse
    from device_jepa.tensors import denormalize
    from device_jepa.training import _metrics
    predictions, targets, physical, masks = [], [], [], []
    features = defaultdict(list)
    handles, current = [], {}
    def hook_for(name, owner):
        def hook(_module, _args, output):
            values = output.detach()
            if owner:
                values = values[current['batch']['owner_mask']]
            features[name].append(values.reshape(-1, values.shape[-1]).clone())
        return hook
    for name, module in model.named_modules():
        if name.rsplit('.', 1)[0] in ('decoder.branches', 'predictor.head.branches') and name.split('.')[-1].isdigit():
            handles.append(module.register_forward_hook(hook_for(name, name.startswith('predictor.'))))
    try:
        with torch.inference_mode():
            for _, batch in cache.batches(batch_size):
                current['batch'] = batch
                prediction = model(design_inputs_pin(batch))
                predictions.append(prediction.double())
                targets.append(batch['performance'])
                physical.append(batch['physical_performance'])
                masks.append(batch['performance_mask'])
    finally:
        for handle in handles:
            handle.remove()
    pred, target, raw, mask = map(torch.cat, (predictions, targets, physical, masks))
    values = {key: torch.cat(items) for key, items in features.items()}
    report = dict(split='val', count=pred.shape[0], status='ok', normalized_mse=None, metrics={})
    if not bool(torch.isfinite(pred).all()):
        report['status'] = 'nonfinite_prediction'
        return report, values, pred
    decoded = denormalize(pred, saved['preprocessing']['performance'])
    if not bool(torch.isfinite(decoded[mask]).all()):
        report['status'] = 'nonfinite_decoded_prediction'
        return report, values, pred
    report['normalized_mse'] = float(masked_mse(pred, target, mask)) if bool(mask.any()) else None
    report['metrics'] = _metrics(decoded, raw, mask, saved['preprocessing'], pred, target)
    return report, values, pred


def add_validation_pin(tables, identity, report, baseline, features, original_features, prediction, original_prediction):
    import torch
    from final_exam.spectral_analysis_pin import relative_error_pin
    common = dict(**identity, split='val', sample_count=report['count'], status=report['status'],
                  normalized_mse=report['normalized_mse'],
                  delta_normalized_mse=None if report['normalized_mse'] is None else
                  report['normalized_mse'] - baseline['normalized_mse'])
    finite = bool(torch.isfinite(prediction).all())
    common['relative_prediction_change'] = relative_error_pin(prediction - original_prediction, original_prediction) if finite else None
    if not report['metrics']:
        tables['Validation'].append(dict(**common, metric=None))
    for metric, scores in report['metrics'].items():
        before = baseline['metrics'].get(metric, {})
        tables['Validation'].append(dict(**common, metric=metric, **scores,
            delta_r2=None if scores.get('r2') is None or before.get('r2') is None else scores['r2'] - before['r2']))
    for branch, value in features.items():
        reference = original_features[branch]
        error = relative_error_pin(value - reference, reference) if bool(torch.isfinite(value).all()) else None
        tables['DecoderFeatures'].append(dict(**identity, branch=branch, split='val',
            tokens=value.shape[0], relative_feature_error=error,
            status='ok' if error is not None else 'undefined_or_nonfinite'))


def run_shared_decoder_probe_pin(*, sources, models, seeds, ranks=(8, 16, 32, 64),
        num_samples=128, num_val_samples=128, max_tokens=512, sampling_seed=17, batch_size=16,
        scope='all', component='performance', validation_scope='all', device='cuda',
        output=None, log_dir=None, max_plots=12, vector_limit=64):
    inspected = inspect_sources_pin(sources, models, seeds, num_samples, num_val_samples, component)
    if not ranks or len(set(ranks)) != len(ranks) or any(type(r) is not int or r < 1 for r in ranks):
        raise ValueError('Ranks must be distinct positive integers')
    ranks = sorted(ranks)
    if any(type(v) is not int or v < 1 for v in (max_tokens, batch_size)) or type(max_plots) is not int or max_plots < 0:
        raise ValueError('Invalid batch/token/plot limits')
    if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32:
        raise ValueError('sampling_seed must be in [0, 2**32)')
    if scope not in ('all', 'attention', 'ffn') or validation_scope not in ('all', 'group', 'both'):
        raise ValueError('Invalid matrix/validation scope')
    if type(vector_limit) is not int or not 0 <= vector_limit <= 256:
        raise ValueError('vector_limit must be an integer in [0, 256]')
    import torch
    from device_jepa.resident import require_device
    from device_jepa.training import state_digest
    from final_exam.run_pin import load_checkpoint_pin
    from final_exam.logging_utils import log_to
    from final_exam.shared_decoder_analysis_pin import decoder_groups_pin, analyze_group_pin, apply_approximations_pin
    from final_exam.spectral_analysis_pin import parameter_counts_pin
    from final_exam.shared_decoder_report_pin import write_shared_decoder_report_pin
    device = require_device(device)
    if max_plots:
        from final_exam.spectral_report_pin import require_plotting_pin
        require_plotting_pin()
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = f'shared_decoder_probe_{stamp}_{uuid.uuid4().hex[:8]}_pin'
    output = Path(output or TASK / 'results_pin' / name).resolve()
    logs = Path(log_dir or TASK / 'logs_pin' / output.name).resolve()
    for destination in (output, logs):
        if destination.exists():
            raise ValueError(f'Use a new output/log directory: {destination}')
        for source in inspected:
            protected = Path(source['source_run'])
            if destination.is_relative_to(protected) or protected.is_relative_to(destination):
                raise ValueError('Diagnostic outputs must be separate from all source runs')
    if output.is_relative_to(logs) or logs.is_relative_to(output):
        raise ValueError('Output and log directories must be separate')
    output.mkdir(parents=True)
    logs.mkdir(parents=True)
    record = dict(version=2, status='running', sources=sources, models=models, seeds=seeds, ranks=ranks,
        num_samples=num_samples, num_val_samples=num_val_samples, max_tokens=max_tokens,
        sampling_seed=sampling_seed, batch_size=batch_size, component=component, scope=scope,
        validation_scope=validation_scope, device=str(device), max_plots=max_plots,
        vector_limit=vector_limit, plot_limit_unit='projection_groups',
        pairwise_basis='right_singular_subspaces_in_saved_coordinates',
        pairwise_rank='min_requested_and_both_source_dtype_numerical_ranks',
        random_overlap_reference='analytic_equal_rank_Haar_expectation_r_over_input_dim',
        principal_angle_order='ascending_degrees',
        fit_scope='within_one_checkpoint_across_corresponding_PE_decoder_branches',
        basis_fit='raw_vertical_stack_SVD_shared_right_factor',
        cross_circuit_fit=False, normalization_refitted=False, retrained=False,
        checkpoint_modified=False, test_evaluated=False, svd_dtype='float64',
        validation_method='temporary_dense_reconstruction_error_only_not_factorized_runtime',
        selected_sample_ids={}, checkpoints=[], results=str(output), logs=str(logs))
    aggregate = defaultdict(list)
    write_json(output / 'probe_pin.json', record)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with log_to(logs / 'shared_decoder_probe_pin.log'):
            for source in inspected:
                rows, feature_spec, ids = selected_rows_pin(source, num_samples, num_val_samples, sampling_seed)
                record['selected_sample_ids'][source['circuit']] = ids
                for item in source['checkpoints']:
                    identity = dict(circuit=source['circuit'], model=item['model'], seed=item['seed'])
                    print(f'Independent checkpoint analysis: {identity}', flush=True)
                    path = item['path']
                    digest = sha256(path)
                    model, saved = load_checkpoint_pin(path, str(device))
                    if saved.get('mode') != item['model'] or saved.get('seed') != item['seed'] or saved.get('manifest') != source['manifest']:
                        raise ValueError('Checkpoint identity/manifest differs from source run')
                    saved.pop('state_dict', None)
                    prep = saved['preprocessing']
                    if feature_spec is not None and prep.get('raw_feature_spec', prep['feature_spec']) != feature_spec:
                        raise ValueError('Checkpoint feature order differs from dataset')
                    if saved.get('requires_measured_op'):
                        raise ValueError('Only design-input inference is supported')
                    before = state_digest(model)
                    groups = decoder_groups_pin(model, component, scope)
                    local = defaultdict(list)
                    if component == 'both' and not any(g.side == 'op_predictor' for g in groups):
                        local['Notes'].append(dict(**identity,
                            reason='OP predictor has no multi-PE decoder; only performance decoders analyzed'))
                    local['Parameters'].extend(dict(**identity, **r) for r in parameter_counts_pin(model))
                    train_cache = resident_pin(rows['train'], saved, device, labels=False) if rows['train'] else None
                    captures = capture_inputs_pin(model, groups, train_cache, batch_size, max_tokens, sampling_seed)
                    del train_cache
                    analyses = {}
                    for index, group in enumerate(groups):
                        print(f'  shared down basis and pairwise similarity {index + 1}/{len(groups)}: {group.name}', flush=True)
                        analysis = analyze_group_pin(group, ranks, captures, vector_limit=vector_limit)
                        analyses[group.name] = analysis['approximations']
                        local['Groups'].append(dict(**identity, **analysis['summary']))
                        for table, key in (('Budgets', 'budget_rows'), ('Errors', 'error_rows'), ('Spectra', 'spectrum_rows'),
                                           ('SubspaceOverlap', 'overlap_rows'), ('PrincipalAngles', 'angle_rows'),
                                           ('CrossBasis', 'cross_basis_rows'), ('VectorCosines', 'vector_cosine_rows')):
                            local[table].extend(dict(**identity, group=group.name, component=group.side, **row)
                                                for row in analysis[key])
                        if any(s.kind == 'functional_out' for s in group.matrices):
                            local['Notes'].append(dict(**identity, group=group.name,
                                reason='Dense MHA output projection is functional: input activations unavailable; weight and end-to-end validation diagnostics remain available'))
                    del captures
                    if rows['val']:
                        cache = resident_pin(rows['val'], saved, device, labels=True)
                        base, base_features, base_prediction = evaluate_validation_pin(model, cache, saved, batch_size)
                        if base['status'] != 'ok' or base['normalized_mse'] is None:
                            raise ValueError('Original checkpoint has no finite scored validation predictions')
                        add_validation_pin(local, dict(**identity, group='original', strategy='original', requested_rank=None),
                            base, base, base_features, base_features, base_prediction, base_prediction)
                        cases = []
                        if validation_scope in ('all', 'both'):
                            cases.append(('all_selected_groups', groups))
                        if validation_scope in ('group', 'both'):
                            cases.extend((g.name, [g]) for g in groups)
                        for label, selected_groups in cases:
                            for rank, strategy in itertools.product(ranks, ('independent', 'shared')):
                                print(f'  validation {label}, rank={rank}, {strategy}', flush=True)
                                with apply_approximations_pin(selected_groups, analyses, rank, strategy=strategy):
                                    report, features, prediction = evaluate_validation_pin(model, cache, saved, batch_size)
                                add_validation_pin(local, dict(**identity, group=label, strategy=strategy, requested_rank=rank),
                                    report, base, features, base_features, prediction, base_prediction)
                                del features, prediction
                        del cache, base_features, base_prediction
                    if state_digest(model) != before or sha256(path) != digest:
                        raise ValueError('Read-only diagnostic changed checkpoint/model state')
                    destination = output / source['circuit'] / f'seed_{item["seed"]}_pin' / item['model']
                    destination.mkdir(parents=True)
                    entry = dict(**identity, path=path, sha256=digest, group_count=len(groups),
                        total_parameters=sum(p.numel() for p in model.parameters()),
                        calibration_samples=len(rows['train']), validation_samples=len(rows['val']),
                        source_code_commit=source['source_code_commit'], model_config=saved['model_config'],
                        config=saved.get('config'), dataset_fingerprint=source['dataset_fingerprint'],
                        preprocessing_sha256=hashlib.sha256(json.dumps(prep, sort_keys=True).encode()).hexdigest(),
                        output=str(destination.relative_to(output)))
                    record['checkpoints'].append(entry)
                    local_record = {**record, 'status': 'complete', 'checkpoints': [entry],
                        'sources': [f'{source["circuit"]}={source["source_run"]}'],
                        'models': [item['model']], 'seeds': [item['seed']],
                        'selected_sample_ids': {source['circuit']: ids}, 'results': str(destination)}
                    write_json(destination / 'probe_pin.json', local_record)
                    write_shared_decoder_report_pin(destination, local, local_record, max_plots=max_plots)
                    for name, entries in local.items():
                        aggregate[name].extend(entries)
                    write_json(output / 'probe_pin.json', record)
                    del model, saved, prep, groups, analyses, local
                del rows
            record['status'] = 'complete'
            write_shared_decoder_report_pin(output, aggregate, record, max_plots=max_plots)
            write_json(output / 'probe_pin.json', record)
            print(f'Completed. Excel: {output / "shared_decoder_summary_pin.xlsx"}', flush=True)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        write_json(output / 'probe_pin.json', record)
        with (logs / 'shared_decoder_probe_pin.log').open('a', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, dest='sources', metavar='CIRCUIT=RESULT_PATH')
    parser.add_argument('--models', nargs='+', choices=SUPPORTED, default=['width_decoder_pin'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[17])
    parser.add_argument('--ranks', nargs='+', type=int, default=[8, 16, 32, 64])
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--num-val-samples', type=int, default=128)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--sampling-seed', type=int, default=17)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--scope', choices=('all', 'attention', 'ffn'), default='all')
    parser.add_argument('--component', choices=('performance', 'op_predictor', 'both'), default='performance')
    parser.add_argument('--validation-scope', choices=('all', 'group', 'both'), default='all')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-plots', type=int, default=12)
    parser.add_argument('--vector-limit', type=int, default=64,
                        help='Leading singular vectors per heatmap axis, 0..256; 0 disables vector tables/plots only')
    parser.add_argument('--output')
    parser.add_argument('--log-dir')
    parser.add_argument('--inspect-source', action='store_true')
    args = vars(parser.parse_args(argv))
    inspect = args.pop('inspect_source')
    try:
        if inspect:
            checked = inspect_sources_pin(args['sources'], args['models'], args['seeds'],
                args['num_samples'], args['num_val_samples'], args['component'])
            print(json.dumps([dict(circuit=s['circuit'], checkpoints=len(s['checkpoints']),
                                   training=s['training']) for s in checked]))
        else:
            run_shared_decoder_probe_pin(**args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
