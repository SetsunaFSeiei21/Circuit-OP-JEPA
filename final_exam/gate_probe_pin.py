#!/usr/bin/env python3
"""Inspect actual checkpoint gate outputs on a fixed subset of a saved split.

No fitting, optimization, checkpoint conversion or measured-OP substitution is
performed. Hooks observe the gate used by the model's real inference forward.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK_ROOT = ROOT / 'final_exam/single_circuit_pin'
ROLES = ('d', 'g', 's', 'b')


def read_object(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Expected a simple circuit/model directory name: {value!r}')
    return value


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Use a positive integer')
    return number


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_source(source, circuit, models, seed, *, allowed_models=None):
    """Login-node preflight: small metadata only, with no torch dependency."""
    from final_exam.gate_version_registry_pin import VERSIONED_GATE_MODELS_PIN
    from final_exam.staged_gate_registry_pin import ALL_STAGED_GATE_MODELS_PIN
    allowed = set(VERSIONED_GATE_MODELS_PIN) | set(ALL_STAGED_GATE_MODELS_PIN) | {
        'device_op_cascade_pin', 'device_op_gate_pin', 'device_op_gate_oracle_pin',
        'device_op_gate_shuffle_pin', 'device_op_gate_only_base_pin',
        'device_op_gate_only_pin', 'device_op_gate_only_oracle_pin',
        'device_op_gate_only_shuffle_pin',
    }
    if allowed_models is not None:
        allowed = set(allowed_models)
    identifier(circuit)
    if not models or len(set(models)) != len(models) or set(models) - allowed:
        raise ValueError('Choose unique supported models for this probe')
    for model in models:
        identifier(model)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Checkpoint seed must be an integer in [0, 2**32)')
    path = Path(source).expanduser().resolve()
    training = path / 'training_pin' if (path / 'training_pin/run_pin.json').is_file() else path
    record = read_object(training / 'run_pin.json')
    if record.get('granularity') != 'pin' or record.get('circuit') != circuit:
        raise ValueError('Requested circuit does not match the saved pin run')
    if not set(models) <= set(record.get('models', [])) or seed not in record.get('seeds', []):
        raise ValueError('Requested models/checkpoint seed were not selected in this source run')
    checkpoints = {mode: training / f'seed_{seed}_pin' / mode / 'checkpoint_pin.pt'
                   for mode in models}
    for path in checkpoints.values():
        if not path.is_file():
            raise ValueError(f'Completed model checkpoint is missing: {path}')
    config = read_object(training / 'config_pin.json')
    if config.get('dataset', {}).get('train_circuits') != [circuit] or config['dataset'].get('ood_circuits'):
        raise ValueError('This probe expects a single-circuit final_exam pin run')
    manifest = read_object(training / 'manifest_pin.json')
    fingerprint = record.get('dataset_fingerprint')
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError('Source run is missing its dataset fingerprint')
    run_root = training.parent
    data = run_root / 'data_pin.json'
    if not data.is_file():
        pointer_path = run_root / 'data_source_pin.json'
        if not pointer_path.is_file():
            raise ValueError('Cannot find data_pin.json or data_source_pin.json beside training_pin')
        pointer = read_object(pointer_path)
        if pointer.get('dataset_fingerprint') != fingerprint:
            raise ValueError('Saved data reference fingerprint differs from the source run')
        if not isinstance(pointer.get('data_path'), str) or not pointer['data_path']:
            raise ValueError('Saved data reference lacks data_path')
        data = Path(pointer['data_path']).expanduser()
        if not data.is_absolute():
            data = run_root / data
    data = data.resolve()
    if not data.is_file():
        raise ValueError(f'Canonical data is unavailable: {data}')
    return dict(training=str(training), source_run=str(run_root), data=str(data),
                circuit=circuit, seed=seed, models=list(models),
                checkpoints={key: str(value) for key, value in checkpoints.items()},
                config=config, manifest=manifest, dataset_fingerprint=fingerprint,
                source_code_commit=record.get('code_commit'))


def select_ids(ids, count, seed):
    """Select without replacement, independently of model names or list order."""
    if type(count) is not int or count < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Use a positive sample count and sampling seed in [0, 2**32)')
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(s, str) for s in ids):
        raise ValueError('Saved split must contain unique sample IDs')
    return sorted(ids, key=lambda sid: (hashlib.sha256(f'{seed}:{sid}'.encode()).digest(), sid))[:count]


def capture_forward(model, batch):
    """Capture the exact MLP logits, gate inputs and applied bias in one forward.

    The gate is shared by layers. Its table is B,D,Q-role,K-role,H; the applied
    bias includes a zero graph-token row/column and physical-edge protection.
    """
    import torch
    gate = getattr(model, 'gate', None)
    active = gate is not None and getattr(model, 'gate_enabled', True)
    captured = {}

    def score_hook(module, inputs, output):
        if 'logits' in captured:
            raise ValueError('Expected one shared gate score table per model forward')
        captured['logits'] = output.detach()

    def gate_hook(module, inputs, output):
        if 'bias' in captured:
            raise ValueError('Expected one shared gate call per model forward')
        _, values, supported = inputs
        captured.update(values=values.detach(), supported=supported.detach(), bias=output.detach())

    model.eval()
    with ExitStack() as stack, torch.no_grad():
        if gate is not None:
            stack.callback(gate.score_head.register_forward_hook(score_hook).remove)
            stack.callback(gate.register_forward_hook(gate_hook).remove)
        prediction = model(batch)
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError('Model produced a non-finite prediction')
        if not active:
            if captured:
                raise ValueError('A disabled gate unexpectedly participated in inference')
            return None
        if set(captured) != {'logits', 'values', 'supported', 'bias'}:
            raise ValueError('Active gate did not execute the expected inference path')
        values, supported = captured['values'], captured['supported']
        owners, roles, eligible = gate._layout(batch, values, supported)
        logits = captured['logits']
        expected = (*values.shape[:2], 4, 4, gate.cfg.heads)
        if tuple(logits.shape) != expected or not bool(torch.isfinite(logits).all()):
            raise ValueError('Unexpected or non-finite gate logits')
        # Scores are computed on the inference device, just like the real gate.
        captured['gate_score'] = torch.sigmoid(logits)
        reproduced = gate._bias_from_logits(logits, owners, roles, eligible)
        torch.testing.assert_close(reproduced, captured['bias'], rtol=0, atol=0)
        if bool((captured['bias'][:, :, 0, :] != 0).any()) or bool((captured['bias'][:, :, :, 0] != 0).any()):
            raise ValueError('Graph-token protection was violated')
        captured['eligible'] = eligible
        captured['relative'] = (gate.relative_op(batch, values, supported)
                                if gate.version in ('v2', 'v3') else None)
        return {key: value.detach().cpu() if value is not None else None
                for key, value in captured.items()}


def gate_rows(model, batch, rows, captured, mode, seed):
    """Report physical MOS pairs only; protected pairs retain raw and applied scores."""
    gate = model.gate
    layout = {key: batch[key].detach().cpu() for key in ('owner_index', 'pin_terminal_index', 'device_mask')}
    terminal_roles = gate.terminal_roles.detach().cpu()
    for b, sample in enumerate(rows):
        for owner, dev in enumerate(sorted(sample['devices'], key=lambda item: item['device_id'])):
            if dev['device_type'].lower() not in ('nmos', 'pmos'):
                continue
            pins = {}
            for p in range(layout['owner_index'].shape[1]):
                if bool(layout['device_mask'][b, p]) and int(layout['owner_index'][b, p]) == owner:
                    terminal = int(layout['pin_terminal_index'][b, p])
                    role = int(terminal_roles[terminal])
                    if role >= 0:
                        pins[role] = p
            for q, qp in sorted(pins.items()):
                for k, kp in sorted(pins.items()):
                    applied = bool(captured['eligible'][b, qp, kp])
                    reason = ('self' if q == k else 'same_net' if dev['terminals'][ROLES[q]] == dev['terminals'][ROLES[k]]
                              else 'no_supported_op' if not bool(captured['supported'][b, owner].any()) else '')
                    if applied != (reason == ''):
                        raise ValueError('Physical pair labels disagree with the actual gate eligibility')
                    for head in range(gate.cfg.heads):
                        bias = float(captured['bias'][b, head, qp + 1, kp + 1])
                        yield dict(model=mode, seed=seed, sample_id=sample['sample_id'],
                            circuit=sample['circuit_id'], device_id=dev['device_id'], device_type=dev['device_type'],
                            query_role=ROLES[q].upper(), key_role=ROLES[k].upper(),
                            query_net=dev['terminals'][ROLES[q]], key_net=dev['terminals'][ROLES[k]],
                            head=head + 1, gate_version=gate.version, attention_scope=model.attention_scope,
                            applied=applied, protection_reason=reason,
                            gate_logit=float(captured['logits'][b, owner, q, k, head]),
                            gate_score=float(captured['gate_score'][b, owner, q, k, head]),
                            attention_bias=bias, effective_multiplier=math.exp(bias))


def run_probe(source_run, *, circuit, models, seed=17, num_samples=20, split='test',
              sampling_seed=17, batch_size=32, viewer_samples=20, device='cuda',
              output=None, log_dir=None):
    import torch
    from device_jepa.data import dataset_fingerprint, load_dataset, verify_manifest
    from device_jepa.resident import require_device
    from device_jepa.training import state_digest
    from final_exam.data_pin import PinResidentDataset
    from final_exam.gate_probe_report_pin import GateReport
    from final_exam.logging_utils import log_to
    from final_exam.models.op_gate_pin import DeviceOPGateModelPin
    from final_exam.run_pin import load_checkpoint_pin

    if split not in ('train', 'val', 'test'):
        raise ValueError('Choose train, val or test from the saved single-circuit split')
    if type(batch_size) is not int or batch_size < 1 or type(viewer_samples) is not int or viewer_samples < 1:
        raise ValueError('batch_size and viewer_samples must be positive integers')
    source = inspect_source(source_run, circuit, models, seed)
    selected = select_ids(source['manifest'].get(split), num_samples, sampling_seed)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'{circuit}_gate_probe_{stamp}_{uuid.uuid4().hex[:8]}_pin'
    output = Path(output or TASK_ROOT / 'results_pin' / run_id).resolve()
    logs = Path(log_dir or TASK_ROOT / 'logs_pin' / output.name).resolve()
    protected = Path(source['source_run'])
    for destination in (output, logs):
        if destination.exists():
            raise ValueError(f'Choose a new output/log directory: {destination}')
        if destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError('Probe outputs must be separate from the source run')
    if output.is_relative_to(logs) or logs.is_relative_to(output):
        raise ValueError('Probe output and log directories must be separate')
    target_device = require_device(device)
    torch.set_num_threads(1)
    loaded = []
    cohort = None
    for mode, path in source['checkpoints'].items():
        model, metadata = load_checkpoint_pin(path, 'cpu')
        if not isinstance(model, DeviceOPGateModelPin):
            raise ValueError(f'{mode} does not expose a device-OP pin gate/control')
        if metadata.get('mode') != mode or metadata.get('seed') != seed or metadata.get('manifest') != source['manifest']:
            raise ValueError('Checkpoint model, seed or split differs from the requested source')
        actual_ids = metadata.get('selected_sample_ids', {}).get(split)
        if not isinstance(actual_ids, list) or not set(actual_ids) <= set(source['manifest'][split]):
            raise ValueError('Checkpoint lacks a valid saved evaluation cohort')
        if cohort is None:
            cohort = actual_ids
        elif set(cohort) != set(actual_ids):
            raise ValueError('Selected models used different saved cohorts; inspect them separately')
        # Do not retain the redundant state_dict after constructing the model.
        metadata.pop('state_dict')
        loaded.append((mode, path, model, metadata))
    selected = select_ids(cohort, num_samples, sampling_seed)
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', source_run=source['source_run'], training=source['training'],
        data=source['data'], circuit=circuit, models=models, checkpoint_seed=seed,
        split=split, requested_samples=num_samples, available_samples=len(cohort),
        selected_sample_ids=selected, sampling_seed=sampling_seed, batch_size=batch_size,
        viewer_samples=min(viewer_samples, len(selected)), device=str(target_device),
        source_dataset_fingerprint=source['dataset_fingerprint'],
        source_code_commit=source['source_code_commit'], results=str(output), logs=str(logs),
        score_meaning='sigmoid MLP gate, not softmax attention or a calibrated connection probability',
        bias_meaning='actual additive gate bias; effective_multiplier=exp(bias)',
        pair_direction='query row, key column; heads are 1-based',
        shared_layers=True, refit_preprocessing=False, retrained=False, checkpoints={})

    def save_record():
        (output / 'probe_pin.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')

    save_record()
    try:
        with log_to(logs / 'probe_pin.log'):
            print(f'Probe {circuit}: {len(selected)}/{len(cohort)} saved {split} samples, seed={seed}', flush=True)
            print(f'Loading canonical JSON on CPU for identity checks: {source["data"]}', flush=True)
            dataset = load_dataset(source['data'], sampling_config=source['manifest'].get('sampling'))
            verify_manifest(dataset, source['manifest'])
            if dataset_fingerprint(dataset) != source['dataset_fingerprint']:
                raise ValueError('Canonical data fingerprint changed since training')
            by_id = {row['sample_id']: row for row in dataset['samples']}
            rows = [by_id[sid] for sid in selected]
            if any(row['circuit_id'] != circuit for row in rows):
                raise ValueError('Selected samples belong to a different circuit')
            feature_spec = dataset['feature_spec']
            del by_id, dataset
            (output / 'selection_pin.json').write_text(json.dumps(dict(split=split, sample_ids=selected), indent=2), encoding='utf-8')
            with GateReport(output, selected[:viewer_samples]) as report:
                for mode, path, model, metadata in loaded:
                    print(f'Inspecting {mode}: frozen checkpoint; no optimization', flush=True)
                    prep = metadata['preprocessing']
                    if prep.get('raw_feature_spec', prep['feature_spec']) != feature_spec:
                        raise ValueError('Checkpoint feature schema differs from the source data')
                    model.to(target_device).eval()
                    before = state_digest(model)
                    active = model.gate is not None and getattr(model, 'gate_enabled', True)
                    measured = bool(model.requires_measured_op)
                    model_rows = rows
                    if measured:
                        names = {name for name, count in zip(prep['feature_spec']['op_names'], prep['op']['count']) if count > 0}
                        model_rows = [{**row, 'devices': [{**dev, 'op': {k: v for k, v in dev['op'].items() if k in names}}
                                                       for dev in row['devices']]} for row in rows]
                    cache = PinResidentDataset(model_rows, prep, target_device,
                        include_op=measured, include_performance=False,
                        max_tokens=metadata['model_config']['max_devices'], label=f'gate probe {mode}')
                    scope = model.attention_scope if active else 'none'
                    cfg = model.encoder.cfg
                    first_layer = cfg.encoder_layers - cfg.cross_layers if scope == 'fusion_self_attention' else 0
                    info = dict(checkpoint=path, sha256=file_digest(path),
                        gate_active=active, gate_version=model.gate_version if active else 'none',
                        gate_source=model.gate_source if active else 'none',
                        requires_measured_op=measured, attention_scope=scope,
                        self_attention_layers_1based=list(range(first_layer + 1, cfg.encoder_layers + 1)) if active else [],
                        heads=cfg.heads, shared_gate_across_layers=True,
                        gate_settings=model.gate.settings if active else None,
                        gate_normalization=getattr(model.gate, 'normalization_contract', None) if active else None,
                        config=metadata['config'])
                    if active and model.gate_version == 'v3':
                        info.update(gate_architecture=model.gate.architecture_contract,
                                    gate_network='mixed_mlp',
                                    gate_n_experts=model.gate.settings['n_experts'],
                                    gate_mlp_dropout=model.gate.settings['mlp_dropout'],
                                    gate_network_parameters=sum(p.numel() for p in model.gate.parameters()))
                    record['checkpoints'][mode] = info
                    report.add_model(mode, info)
                    for start in range(0, len(rows), batch_size):
                        indices = list(range(start, min(start + batch_size, len(rows))))
                        _, batch = next(cache.batches(len(indices), indices=indices, metadata=True))
                        capture = capture_forward(model, batch)
                        if capture is not None:
                            part = [rows[i] for i in indices]
                            report.add_scores(gate_rows(model, batch, part, capture, mode, seed))
                            report.add_op_inputs(mode, part, model.gate.field_names, capture)
                        print(f'{mode}: {start + len(indices)}/{len(rows)} samples inspected', flush=True)
                    if state_digest(model) != before:
                        raise ValueError('Read-only inference changed checkpoint weights or buffers')
                    model.cpu()
                    del cache
                report.finish(record)
            record['status'] = 'complete'
            save_record()
            print(f'Gate viewer: {output / "gate_viewer_pin.html"}\nExcel: {output / "gate_scores_pin.xlsx"}\nFull CSV: {output / "gate_scores_pin.csv"}', flush=True)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save_record()
        with (logs / 'probe_pin.log').open('a', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, help='Saved run root or its training_pin directory')
    parser.add_argument('--circuit', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=17, help='Checkpoint training seed')
    parser.add_argument('--num-samples', type=positive, default=20)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--sampling-seed', type=int, default=17)
    parser.add_argument('--batch-size', type=positive, default=32)
    parser.add_argument('--viewer-samples', type=positive, default=20, help='HTML sample cap; CSV always contains all selected samples')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    parser.add_argument('--log-dir')
    parser.add_argument('--inspect-source', action='store_true', help='Metadata-only preflight; no torch, output files or inference')
    args = vars(parser.parse_args(argv))
    inspect_only = args.pop('inspect_source')
    try:
        if inspect_only:
            source = inspect_source(args['source_run'], args['circuit'], args['models'], args['seed'])
            select_ids(source['manifest'].get(args['split']), args['num_samples'], args['sampling_seed'])
            print(json.dumps({k: source[k] for k in ('training', 'data', 'circuit', 'seed', 'models', 'checkpoints')}))
            return
        return run_probe(**args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
