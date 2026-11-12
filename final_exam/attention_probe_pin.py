#!/usr/bin/env python3
"""Read-only PE self-attention inspection of natively ungated pin checkpoints.

Observe the actual downstream attention inputs without changing the forward.
Reconstruct scaled QK, original additive bias, logits and full-sequence softmax;
only then select each physical MOS's D/G/S/B submatrix. No gate is disabled.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final_exam.gate_probe_pin import (
    TASK_ROOT, ROLES, file_digest, inspect_source as _inspect_source, positive, select_ids,
)

UNGATED_MODELS_PIN = (
    'device_op_cascade_pin', 'zerosim_supervised_pin', 'device_op_gate_only_base_pin',
    'predicted_op_pin', 'student_predicted_op_pin', 'student_predicted_op_distill_pin',
    'op_oracle_pin', 'design_op_oracle_pin',
)


def inspect_source(source, circuit, models, seed):
    if set(models) - set(UNGATED_MODELS_PIN):
        raise ValueError('This A-mode probe requires a natively ungated model; '
                         'it does not disable gates on trained gated checkpoints. '
                         'Use --list-models for supported names.')
    return _inspect_source(source, circuit, models, seed, allowed_models=UNGATED_MODELS_PIN)


def downstream_encoder(model):
    """Explicitly exclude the OP predictor and teacher from layer selection."""
    from device_jepa.model import DeviceEncoder
    from final_exam.models.models_pin import PinReadoutModel
    from final_exam.models.op_gate_pin import DeviceOPGateModelPin
    from final_exam.models.staged_op_gate_pin import StagedOPGateModelPin
    from final_exam.models.op_transfer_pin import PredictedOPCascadePin, ReadoutModelPin

    if isinstance(model, DeviceOPGateModelPin):
        # In versioned studies device_op_cascade_pin is saved as a copy of
        # the shared supervised base, whose dormant gate is never called.
        # Accept the ORIGINAL saved ungated phase; never change set_phase().
        saved_ungated_base = (isinstance(model, StagedOPGateModelPin)
                              and model.phase == 'supervised' and not model.gate_enabled)
        if not saved_ungated_base and (model.gate is not None or model.gate_source != 'none'
                                       or model.attention_scope != 'none'):
            raise ValueError('Checkpoint contains a gate; A-mode never disables or bypasses it')
        encoder, path = model.encoder, 'encoder'
    else:
        path = ''
        if isinstance(model, PredictedOPCascadePin):
            model, path = model.downstream, 'downstream.'
        if not isinstance(model, (PinReadoutModel, ReadoutModelPin)):
            raise ValueError('Checkpoint has no supported ungated ZeroSim downstream encoder')
        encoder, path = model.encoder, path + 'encoder'
        if hasattr(encoder, 'core'):
            encoder, path = encoder.core, path + '.core'
    if not isinstance(encoder, DeviceEncoder):
        raise ValueError('PE inspection requires the ZeroSim-family fusion_blocks encoder')
    if len(encoder.fusion_blocks) != encoder.cfg.cross_layers:
        raise ValueError('Saved PE layer count differs from the encoder configuration')
    return encoder, path


def reconstruct_attention(module, args, kwargs):
    """Semantic replay of this checkpoint's MHA, without replacing its kernel.

    Supported architecture: batch-first, packed self-QKV, no extra KV tokens,
    eval mode. The complete key axis (including graph and external ports) is
    retained through softmax. Boolean True masks and -inf both exclude keys.
    """
    import torch
    from torch.nn import functional as F
    if module.training or not module.batch_first or not module._qkv_same_embed_dim:
        raise ValueError('Probe requires eval-mode batch-first packed self-attention')
    if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError('Extra key/value tokens are outside this pin attention contract')
    if len(args) != 3 or args[0] is not args[1] or args[0] is not args[2]:
        raise ValueError('Only the PE self-attention call is supported')
    if kwargs.get('is_causal', False):
        raise ValueError('Unexpected causal attention in a circuit encoder')
    x = args[0]
    batch_size, length, dim = x.shape
    heads, head_dim = module.num_heads, dim // module.num_heads
    query, key, value = F.linear(x, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
    reshape = lambda tensor: tensor.reshape(batch_size, length, heads, head_dim).transpose(1, 2)
    query, key, value = map(reshape, (query, key, value))
    qk = (query * (head_dim ** -.5)) @ key.transpose(-2, -1)
    bias = torch.zeros_like(qk)
    for name in ('attn_mask', 'key_padding_mask'):
        mask = kwargs.get(name)
        if mask is None:
            continue
        if name == 'key_padding_mask':
            if mask.shape != (batch_size, length):
                raise ValueError('Unexpected key padding mask shape')
            mask = mask[:, None, None, :]
        elif mask.ndim == 2 and mask.shape == (length, length):
            mask = mask[None, None]
        elif mask.ndim == 3 and mask.shape == (batch_size * heads, length, length):
            mask = mask.reshape(batch_size, heads, length, length)
        else:
            raise ValueError('Unexpected self-attention mask shape')
        if mask.dtype == torch.bool:
            bias = bias.masked_fill(mask, float('-inf'))
        elif mask.is_floating_point():
            bias = bias + mask
        else:
            raise ValueError('Attention mask must be boolean or floating point')
    logits = qk + bias
    weights = logits.softmax(-1)
    if not bool(torch.isfinite(qk).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError('Non-finite QK or softmax attention in the saved model')
    replay = (weights @ value).transpose(1, 2).reshape(batch_size, length, dim)
    replay = F.linear(replay, module.out_proj.weight, module.out_proj.bias)
    return dict(qk_scaled=qk, additive_bias=bias, attention_logit=logits,
                attention_weight=weights, replay_output=replay)


def capture_pe_forward(model, batch):
    """Hooks do not alter kwargs, return values, training state, or parameters."""
    import torch
    encoder, path = downstream_encoder(model)
    if model.training or any(block.self_attn.training for block in encoder.fusion_blocks):
        raise ValueError('Call eval() before read-only attention inspection')
    prefix_layers = len(encoder.structure_encoder.blocks)
    captured = {}

    def before(layer):
        def hook(module, args, kwargs):
            if layer in captured:
                raise ValueError('A PE attention layer ran more than once in one forward')
            captured[layer] = reconstruct_attention(module, args, kwargs)
        return hook

    def after(layer):
        def hook(module, args, kwargs, output):
            item = captured[layer]
            expected = item.pop('replay_output')
            actual = output[0]
            # The native fused kernel may round differently from explicit QK
            # + softmax. Validate the replay while preserving the native result.
            low_precision = actual.dtype in (torch.float16, torch.bfloat16)
            torch.testing.assert_close(actual, expected,
                rtol=3e-2 if low_precision else 3e-4,
                atol=3e-3 if low_precision else 3e-5)
            item['output_max_abs_error'] = float((actual - expected).abs().max())
            for key in ('qk_scaled', 'additive_bias', 'attention_logit', 'attention_weight'):
                item[key] = item[key].detach().cpu()
            item.update(pe_layer=layer + 1, encoder_layer=prefix_layers + layer + 1,
                        mask_scope='local' if (prefix_layers + layer) % 2 == 0 else 'global',
                        module_path=f'{path}.fusion_blocks.{layer}.self_attn')
        return hook

    with ExitStack() as stack, torch.inference_mode():
        dormant_gate = getattr(model, 'gate', None)
        if dormant_gate is not None:
            def forbid_gate(module, args):
                raise ValueError('An ungated checkpoint unexpectedly executed a gate')
            stack.callback(dormant_gate.register_forward_pre_hook(forbid_gate).remove)
        for layer, block in enumerate(encoder.fusion_blocks):
            stack.callback(block.self_attn.register_forward_pre_hook(before(layer), with_kwargs=True).remove)
            stack.callback(block.self_attn.register_forward_hook(after(layer), with_kwargs=True).remove)
        prediction = model(batch)
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError('Checkpoint generated non-finite performance predictions')
    if set(captured) != set(range(len(encoder.fusion_blocks))):
        raise ValueError('Not all downstream PE layers were observed')
    return [captured[i] for i in sorted(captured)]


def attention_rows(batch, samples, captured, *, mode, seed, terminal_names):
    """Map physical devices to real tensor indices, never assume four adjacent pins."""
    import torch
    layout = {key: batch[key].detach().cpu() for key in (
        'owner_index', 'pin_terminal_index', 'pin_net_index', 'device_mask')}
    for b, sample in enumerate(samples):
        valid = layout['device_mask'][b]
        owner_index = layout['owner_index'][b]
        for owner, dev in enumerate(sorted(sample['devices'], key=lambda d: d['device_id'])):
            if dev['device_type'].lower() not in ('nmos', 'pmos'):
                continue
            pins = {}
            for pin in torch.where(valid & (owner_index == owner))[0].tolist():
                terminal = int(layout['pin_terminal_index'][b, pin])
                if not 0 <= terminal < len(terminal_names):
                    raise ValueError('Owned pin lacks a valid terminal identity')
                role = terminal_names[terminal].lower()
                if role in ROLES:
                    if role in pins:
                        raise ValueError('Duplicate MOS terminal role')
                    pins[role] = pin
            if set(pins) != set(ROLES):
                raise ValueError(f'MOS {dev["device_id"]} lacks a complete D/G/S/B pin mapping')
            owner_mask = valid & (owner_index == owner)
            other_mask = valid & (owner_index >= 0) & (owner_index != owner)
            external_mask = valid & (owner_index < 0)
            for item in captured:
                matrices = {name: item[name][b] for name in (
                    'qk_scaled', 'additive_bias', 'attention_logit', 'attention_weight')}
                heads = matrices['attention_weight'].shape[0]
                for head in range(heads + 1):
                    values = {name: matrix[head] if head < heads else matrix.mean(0)
                              for name, matrix in matrices.items()}
                    for qr in ROLES:
                        qp = pins[qr]
                        full = values['attention_weight'][qp + 1]
                        base = dict(model=mode, seed=seed, sample_id=sample['sample_id'],
                            circuit=sample['circuit_id'], pe_layer=item['pe_layer'],
                            encoder_layer=item['encoder_layer'], mask_scope=item['mask_scope'],
                            device_id=dev['device_id'], device_type=dev['device_type'],
                            head=head + 1 if head < heads else 'mean', query_role=qr,
                            query_pin_index=qp, query_sequence_index=qp + 1, query_net=dev['terminals'][qr])
                        yield 'mass', dict(base, intra_device_mass=float(full[1:][owner_mask].sum()),
                            other_device_mass=float(full[1:][other_mask].sum()),
                            external_port_mass=float(full[1:][external_mask].sum()),
                            graph_token_mass=float(full[0]), padding_mass=float(full[1:][~valid].sum()),
                            total_mass=float(full.sum()))
                        for kr in ROLES:
                            kp = pins[kr]
                            same_net = dev['terminals'][qr] == dev['terminals'][kr]
                            if same_net != bool(layout['pin_net_index'][b, qp] == layout['pin_net_index'][b, kp]):
                                raise ValueError('Physical net identity differs from tensor layout')
                            numbers = {name: float(value[qp + 1, kp + 1]) for name, value in values.items()}
                            masked = math.isinf(numbers['attention_logit']) and numbers['attention_logit'] < 0
                            # JSON null / CSV blank denotes hard -inf; the
                            # separate masked flag disambiguates it from zero.
                            numbers = {name: value if math.isfinite(value) else None for name, value in numbers.items()}
                            yield 'score', dict(base, key_role=kr, key_pin_index=kp,
                                key_sequence_index=kp + 1, key_net=dev['terminals'][kr], same_net=same_net,
                                hard_masked=masked, **numbers)


def run_probe(source_run, *, circuit, models, seed=17, num_samples=20, split='test',
              sampling_seed=17, batch_size=8, viewer_samples=20, device='cuda', output=None, log_dir=None):
    import torch
    from device_jepa.data import dataset_fingerprint, load_dataset, verify_manifest
    from device_jepa.resident import require_device
    from device_jepa.training import state_digest
    from final_exam.attention_probe_report_pin import AttentionReport
    from final_exam.data_pin import PinResidentDataset
    from final_exam.logging_utils import log_to
    from final_exam.run_pin import load_checkpoint_pin

    if split not in ('train', 'val', 'test'):
        raise ValueError('Choose a saved train/val/test split')
    if any(type(x) is not int or x < 1 for x in (batch_size, viewer_samples)):
        raise ValueError('batch_size and viewer_samples must be positive integers')
    source = inspect_source(source_run, circuit, models, seed)
    select_ids(source['manifest'].get(split), num_samples, sampling_seed)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'{circuit}_pe_attention_probe_{stamp}_{uuid.uuid4().hex[:8]}_pin'
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
    loaded, cohort = [], None
    for mode, checkpoint in source['checkpoints'].items():
        model, metadata = load_checkpoint_pin(checkpoint, 'cpu')
        downstream_encoder(model)
        if (metadata.get('mode') != mode or metadata.get('seed') != seed
                or metadata.get('manifest') != source['manifest']):
            raise ValueError('Checkpoint model, seed or split differs from the requested source')
        ids = metadata.get('selected_sample_ids', {}).get(split)
        if not isinstance(ids, list) or not set(ids) <= set(source['manifest'][split]):
            raise ValueError('Checkpoint lacks a valid saved evaluation cohort')
        if cohort is not None and set(ids) != set(cohort):
            raise ValueError('Selected models used different cohorts; inspect separately')
        cohort = ids
        metadata.pop('state_dict')
        loaded.append((mode, checkpoint, model, metadata))
    selected = select_ids(cohort, num_samples, sampling_seed)
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', probe='ungated_PE_self_attention',
        source_run=source['source_run'], training=source['training'], data=source['data'],
        circuit=circuit, models=models, checkpoint_seed=seed, split=split,
        requested_samples=num_samples, available_samples=len(cohort), selected_sample_ids=selected,
        sampling_seed=sampling_seed, batch_size=batch_size, viewer_samples=min(viewer_samples, len(selected)),
        device=str(target_device), source_dataset_fingerprint=source['dataset_fingerprint'],
        source_code_commit=source['source_code_commit'], results=str(output), logs=str(logs),
        pair_direction='query row, key column; D/G/S/B; heads and layers 1-based; pin indices 0-based',
        score_meaning='QK/sqrt(head_dim) + original bias/mask; softmax over full sequence before MOS extraction',
        head_mean='arithmetic mean of per-head weights; never softmax of mean logits',
        gate_policy='natively ungated checkpoints only; no gate disabling',
        capture_method='replay from observed downstream MHA inputs; native output is preserved and checked',
        refit_preprocessing=False, retrained=False, checkpoints={})

    def save_record():
        (output / 'probe_pin.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')

    save_record()
    try:
        with log_to(logs / 'probe_pin.log'):
            print(f'PE attention: {circuit}, {len(selected)}/{len(cohort)} saved {split} samples', flush=True)
            print(f'Loading canonical JSON on CPU for identity checks: {source["data"]}', flush=True)
            dataset = load_dataset(source['data'], sampling_config=source['manifest'].get('sampling'))
            verify_manifest(dataset, source['manifest'])
            if dataset_fingerprint(dataset) != source['dataset_fingerprint']:
                raise ValueError('Canonical data fingerprint changed since training')
            by_id = {row['sample_id']: row for row in dataset['samples']}
            samples = [by_id[sid] for sid in selected]
            if any(row['circuit_id'] != circuit for row in samples):
                raise ValueError('Selected samples belong to a different circuit')
            feature_spec = dataset['feature_spec']
            del dataset, by_id
            (output / 'selection_pin.json').write_text(json.dumps(dict(split=split, sample_ids=selected), indent=2))
            with AttentionReport(output, selected[:viewer_samples]) as report:
                for mode, checkpoint, model, metadata in loaded:
                    prep = metadata['preprocessing']
                    if prep.get('raw_feature_spec', prep['feature_spec']) != feature_spec:
                        raise ValueError('Checkpoint feature schema differs from source data')
                    model.to(target_device).eval()
                    before = state_digest(model)
                    encoder, path = downstream_encoder(model)
                    measured = bool(metadata['requires_measured_op'])
                    rows = samples
                    if measured:
                        supported = {name for name, count in zip(prep['feature_spec']['op_names'], prep['op']['count']) if count > 0}
                        rows = [{**row, 'devices': [{**dev, 'op': {k: v for k, v in dev['op'].items() if k in supported}}
                                                   for dev in row['devices']]} for row in rows]
                    cache = PinResidentDataset(rows, prep, target_device, include_op=measured,
                        include_performance=False, max_tokens=metadata['model_config']['max_devices'],
                        label=f'PE attention {mode}')
                    prefix_layers = len(encoder.structure_encoder.blocks)
                    info = dict(checkpoint=checkpoint, sha256=file_digest(checkpoint), gate_active=False,
                        saved_phase=getattr(model, 'phase', None),
                        requires_measured_op=measured, encoder_path=path, heads=encoder.cfg.heads,
                        pe_layers=len(encoder.fusion_blocks), structure_layers=prefix_layers,
                        encoder_layers_1based=list(range(prefix_layers + 1, encoder.cfg.encoder_layers + 1)),
                        config=metadata['config'], native_output_max_abs_error=0.)
                    record['checkpoints'][mode] = info
                    report.add_model(mode, info)
                    for start in range(0, len(rows), batch_size):
                        indices = list(range(start, min(start + batch_size, len(rows))))
                        _, batch = next(cache.batches(len(indices), indices=indices, metadata=True))
                        captures = capture_pe_forward(model, batch)
                        info['native_output_max_abs_error'] = max(info['native_output_max_abs_error'],
                                                                 *(item['output_max_abs_error'] for item in captures))
                        report.add_rows(attention_rows(batch, [rows[i] for i in indices], captures,
                            mode=mode, seed=seed, terminal_names=prep['feature_spec']['terminal_names']))
                        print(f'{mode}: {start + len(indices)}/{len(rows)} samples, '
                              f'{len(captures)} PE layers x {encoder.cfg.heads} heads', flush=True)
                        del captures, batch
                    if state_digest(model) != before:
                        raise ValueError('Read-only inference changed checkpoint weights or buffers')
                    model.cpu()
                    del cache
                report.finish(record)
            record['status'] = 'complete'
            save_record()
            print(f'Viewer: {output / "attention_viewer_pin.html"}\nExcel: {output / "attention_scores_pin.xlsx"}', flush=True)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save_record()
        with (logs / 'probe_pin.log').open('a', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--source-run')
    parser.add_argument('--circuit')
    parser.add_argument('--models', nargs='+', default=['device_op_cascade_pin'])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num-samples', type=positive, default=20)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--sampling-seed', type=int, default=17)
    parser.add_argument('--batch-size', type=positive, default=8)
    parser.add_argument('--viewer-samples', type=positive, default=20)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    parser.add_argument('--log-dir')
    parser.add_argument('--inspect-source', action='store_true', help='Metadata-only check without Torch or inference')
    args = vars(parser.parse_args(argv))
    if args.pop('list_models'):
        print('\n'.join(UNGATED_MODELS_PIN))
        return
    if not args['source_run'] or not args['circuit']:
        parser.error('--source-run and --circuit are required')
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
