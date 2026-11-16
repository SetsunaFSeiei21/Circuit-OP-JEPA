"""Bounded, identity-preserving per-head score export after checkpoint selection."""
import csv
import json

import torch


def export_lasd_scores_pin(ctx, model, options, directory):
    limit = model.lasd_config['diagnostic_samples']
    if not limit or not model.lasd_spec['enabled']:
        return None
    device = torch.device(ctx.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    flags = [(module, module.training) for module in model.modules()]
    rows, sample_ids, sizes, statistics = [], [], [], []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for batch in ctx.batches('test', min(limit, options['batch_size']), seed=None, op=False):
                _, observed = model(batch, return_scores=True)
                # Detach ONLY at this diagnostic boundary, never in model.forward.
                scores = observed['scores'].detach().cpu()
                deltas = observed['differences'].detach().cpu()
                valid = observed['valid'].detach().cpu()
                for b, sample_id in enumerate(batch['_sample_ids']):
                    if len(sample_ids) == limit:
                        break
                    positions = torch.where(valid[b])[0].tolist()
                    names = ['GRAPH'] + [bytes.fromhex(key.rsplit(':', 1)[-1]).decode('utf-8')
                                         for key in batch['_group_keys'][b]]
                    if len(names) != len(positions) or positions[0] != 0:
                        raise ValueError('LASD diagnostic token identities do not match valid nodes')
                    sample_ids.append(sample_id)
                    sizes.append(len(names))
                    for layer, scope in enumerate(model.lasd_spec['layer_masks']):
                        selected = scores[layer, b, positions]
                        total = selected.sum(0)
                        # Report receive variability per head, not a claim of information flow.
                        statistics.append(dict(sample_id=sample_id, pe_layer=layer+1, mask=scope,
                            valid_tokens=len(names),
                            receive_std_by_head=selected[..., 0].std(0, unbiased=False).tolist(),
                            transmit_std_by_head=selected[..., 1].std(0, unbiased=False).tolist(),
                            max_conservation_error=float((total[:, 0]-total[:, 1]).abs().max())))
                        for position, token in zip(positions, names):
                            for head in range(scores.shape[3]):
                                r, t = scores[layer, b, position, head].tolist()
                                dr, dt = deltas[layer, b, position, head].tolist()
                                rows.append(dict(sample_id=sample_id, pe_layer=layer+1, mask=scope,
                                    token_index=position, token_id=token, is_graph=position == 0,
                                    head=head, receive=r, transmit=t, delta_receive=dr, delta_transmit=dt,
                                    delta_is_boundary_fill=layer == 0, valid_tokens=len(names)))
                if len(sample_ids) >= limit:
                    break
    finally:
        for module, training in flags:
            module.training = training
    if not rows:
        raise ValueError('LASD diagnostics require at least one test sample')
    with (directory / 'lasd_scores_pin.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(model=model.mode, split='test', checkpoint_selection=False,
        sample_ids=sample_ids, valid_token_counts=sizes, contract=model.lasd_spec,
        score_file='lasd_scores_pin.csv', statistics=statistics,
        notes=['R/T are raw attention weights, not physical information amounts.',
               'GRAPH/ports are included; padding and diagonal are excluded without renormalization.',
               'Local/global masks alternate, so delta includes changes in allowed connectivity.',
               'Same-index cross-layer heads need not have the same learned semantic meaning.',
               'First PE delta is boundary padding; score-only model does not consume deltas.',
               'Each sample preserves its native token order; no aggregation across graphs.'])
    (directory / 'lasd_diagnostics_pin.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return dict(samples=len(sample_ids), csv='lasd_scores_pin.csv', metadata='lasd_diagnostics_pin.json')
