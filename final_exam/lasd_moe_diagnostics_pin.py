"""Bounded validation observations and pre-clip gradient diagnostics for LASD-MoE.

This module never supplies a training loss or checkpoint-selection objective.
Score/route exports are detached here only. Training norm accumulators remain
on the training device and synchronize once per epoch when serializing history.
"""
import csv
import json
from pathlib import Path

import torch


class EncoderGradientNormsPin:
    """Observe each optimizer batch after backward and before gradient clipping.

    SE/PE blocks are listed separately; only PE blocks contribute the scores and
    memories fused by LASD-MoE. Norms include the whole block, including its FFN
    and (for PE) cross-attention, not just Q/K or a causal attribution to scores.
    """
    def __init__(self, model):
        self.model = model
        blocks = [('SE', i + 1, block) for i, block in
                  enumerate(model.encoder.structure_encoder.blocks)]
        blocks += [('PE', i + 1, block) for i, block in enumerate(model.encoder.fusion_blocks)]
        blocks.append(('FINAL', 1, model.final_graph_encoder))
        self.groups = [(kind, index, tuple(p for p in block.parameters() if p.requires_grad))
                       for kind, index, block in blocks]
        self._epoch = None
        self._totals = None
        self._batches = 0

    @torch.no_grad()
    def __call__(self, model, epoch):
        if model is not self.model:
            raise ValueError('Gradient observer belongs to a different model')
        if self._epoch != epoch:
            self._epoch, self._totals, self._batches = epoch, [], 0
        for i, (_, _, parameters) in enumerate(self.groups):
            reference = next(model.parameters())
            pieces = [torch.linalg.vector_norm(p.grad.detach().float())
                      for p in parameters if p.grad is not None]
            norm = (torch.linalg.vector_norm(torch.stack(pieces)) if pieces else
                    torch.zeros((), device=reference.device, dtype=torch.float32))
            present = int(bool(pieces))
            if self._batches == 0:
                self._totals.append([norm.clone(), norm.clone(), present])
            else:
                self._totals[i][0].add_(norm)
                self._totals[i][1] = torch.maximum(self._totals[i][1], norm)
                self._totals[i][2] += present
        self._batches += 1

    def epoch_record(self, epoch):
        if epoch != self._epoch or not self._batches:
            raise ValueError('No gradient observations for the requested epoch')
        values = torch.stack([torch.stack((total / self._batches, maximum))
                              for total, maximum, _ in self._totals]).detach().cpu().tolist()
        layers = []
        for (kind, index, params), (_, _, present), (mean, maximum) in zip(
                self.groups, self._totals, values):
            layers.append(dict(layer=f'{kind}{index}', stage=kind, layer_index=index,
                               parameters=sum(p.numel() for p in params),
                               batches_with_grad=present, mean_l2=mean, max_l2=maximum))
        return dict(epoch=epoch, batches=self._batches, timing='after_backward_before_clipping',
                    averaging='equal_optimizer_batches', norm='block_parameter_gradient_l2',
                    includes_embedding_parameters=False, layers=layers)


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f'No diagnostic rows for {path.name}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _identities(batch, b, positions):
    keys = batch['_group_keys'][b]
    names = ['GRAPH'] + [bytes.fromhex(key.rsplit(':', 1)[1]).decode('utf-8') for key in keys]
    if len(names) != len(positions) or not positions or positions[0] != 0:
        raise ValueError('LASD-MoE diagnostic identities do not match valid node ordering')
    prefixes = {key.rsplit(':', 1)[0] for key in keys}
    if len(prefixes) > 1:
        raise ValueError('LASD-MoE sample contains mixed circuit/topology identities')
    circuit, topology = json.loads(next(iter(prefixes))) if prefixes else (None, None)
    return names, circuit, topology


def export_lasd_moe_diagnostics_pin(ctx, model, options, directory):
    """Export first bounded ID-validation samples after loading the selected model.

    Statistics include valid GRAPH and pin tokens. Per-sample counts and both
    sample-weighted and node-weighted layer means expose graph-size effects.
    The final graph block is not part of the PE score sequence.
    """
    limit = model.lasd_moe_config['diagnostic_samples']
    if not limit:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(ctx.device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == 'cuda' else [])
    flags = [(module, module.training) for module in model.modules()]
    score_rows, weight_rows, expert_rows = [], [], []
    global_rows = []
    sample_ids, sizes, identities, statistics = [], [], [], []
    sample_layer_means, all_weights, all_probs, all_choices, all_mix = [], [], [], [], []
    spec = model.lasd_moe_spec
    headwise = spec.get('layer_weight_axes') == 'B,N,H,L'
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for batch in ctx.batches('val', min(limit, options['batch_size']), seed=None, op=False):
                _, observed = model(batch, return_scores=True)
                data = {key: observed[key].detach().cpu() for key in (
                    'scores', 'differences', 'valid', 'layer_weights', 'router_probs',
                    'selected_experts', 'selected_weights')}
                if 'global_context' in spec:
                    for key in ('global_statistics', 'global_differences'):
                        data[key] = observed[key].detach().cpu()
                        if data[key].shape != (data['valid'].shape[0], spec['base_layers'], spec['heads'], 3):
                            raise ValueError('LASD-MoE global diagnostic axes disagree with the saved contract')
                scores, differences, valid = data['scores'], data['differences'], data['valid']
                layer_shape = (spec['heads'],) if headwise else ()
                weight_shape = (*valid.shape, *layer_shape, len(spec['layer_masks']))
                if (scores.ndim != 5 or scores.shape[-1] != 2 or differences.shape != scores.shape
                        or scores.shape[:2] != valid.shape or valid.dtype != torch.bool
                        or scores.shape[2] != len(spec['layer_masks'])
                        or scores.shape[3] != spec['heads']
                        or data['layer_weights'].shape != weight_shape
                        or data['router_probs'].shape != (*valid.shape, spec['experts'])
                        or data['selected_experts'].shape != (*valid.shape, spec['top_k'])
                        or data['selected_weights'].shape != data['selected_experts'].shape):
                    raise ValueError('LASD-MoE diagnostic axes disagree with the saved contract')
                if len(batch['_sample_ids']) != valid.shape[0]:
                    raise ValueError('LASD-MoE sample IDs disagree with the diagnostic batch')
                for b, sample_id in enumerate(batch['_sample_ids']):
                    if len(sample_ids) >= limit:
                        break
                    positions = torch.where(valid[b])[0].tolist()
                    names, circuit, topology = _identities(batch, b, positions)
                    sample_ids.append(sample_id)
                    sizes.append(len(positions))
                    identities.append(dict(sample_id=sample_id, circuit=circuit, topology=topology))
                    weights = data['layer_weights'][b, positions]
                    probabilities = data['router_probs'][b, positions]
                    choices = data['selected_experts'][b, positions]
                    mixing = data['selected_weights'][b, positions]
                    if ((choices < 0).any() or (choices >= spec['experts']).any()
                            or (choices[:, 0] == choices[:, 1]).any()):
                        raise ValueError('Valid nodes must select two distinct valid experts')
                    sample_layer_means.append(weights.double().mean(0))
                    all_weights.append(weights.double())
                    all_probs.append(probabilities.double())
                    all_choices.append(choices)
                    all_mix.append(mixing.double())
                    for layer, mask in enumerate(spec['layer_masks']):
                        selected = scores[b, positions, layer]
                        total = selected.double().sum(0)
                        statistics.append(dict(sample_id=sample_id, pe_layer=layer+1, mask=mask,
                            valid_tokens=len(names),
                            receive_std_by_head=selected.std(0, unbiased=False)[:, 0].tolist(),
                            transmit_std_by_head=selected.std(0, unbiased=False)[:, 1].tolist(),
                            max_conservation_error=float((total[:, 0] - total[:, 1]).abs().max())))
                        if 'global_context' in spec:
                            for head in range(spec['heads']):
                                mean_r, std_r, std_t = data['global_statistics'][b, layer, head].tolist()
                                dmean_r, dstd_r, dstd_t = data['global_differences'][b, layer, head].tolist()
                                global_rows.append(dict(sample_id=sample_id, circuit=circuit, topology=topology,
                                    valid_tokens=len(names), pe_layer=layer+1, mask=mask, head=head,
                                    mean_receive=mean_r, std_receive=std_r, std_transmit=std_t,
                                    delta_mean_receive=dmean_r, delta_std_receive=dstd_r,
                                    delta_std_transmit=dstd_t, delta_is_boundary_fill=layer == 0,
                                    std_epsilon=spec['global_context']['epsilon']))
                    for row, (position, name) in enumerate(zip(positions, names)):
                        identity = dict(sample_id=sample_id, circuit=circuit, topology=topology,
                                        token_index=position, token_id=name, is_graph=position == 0,
                                        valid_tokens=len(names))
                        for layer, mask in enumerate(spec['layer_masks']):
                            if not headwise:
                                weight_rows.append(dict(**identity, pe_layer=layer+1,
                                                        layer_weight=float(weights[row, layer])))
                            for head in range(scores.shape[3]):
                                if headwise:
                                    weight_rows.append(dict(**identity, pe_layer=layer+1, head=head,
                                                            layer_weight=float(weights[row, head, layer])))
                                r, t = scores[b, position, layer, head].tolist()
                                dr, dt = differences[b, position, layer, head].tolist()
                                score_rows.append(dict(**identity, pe_layer=layer+1, mask=mask,
                                    head=head, receive=r, transmit=t, delta_receive=dr,
                                    delta_transmit=dt, delta_is_boundary_fill=layer == 0))
                        selected = choices[row].tolist()
                        selected_mix = mixing[row].tolist()
                        for expert in range(spec['experts']):
                            slot = selected.index(expert) if expert in selected else None
                            expert_rows.append(dict(**identity, expert=expert,
                                router_probability=float(probabilities[row, expert]),
                                selected=slot is not None,
                                selected_weight=selected_mix[slot] if slot is not None else 0.))
                if len(sample_ids) >= limit:
                    break
    finally:
        for module, training in flags:
            module.training = training
    if not sample_ids:
        raise ValueError('LASD-MoE diagnostics require at least one validation sample')
    all_weights, all_probs = torch.cat(all_weights), torch.cat(all_probs)
    all_choices, all_mix = torch.cat(all_choices), torch.cat(all_mix)
    node_count = sum(sizes)
    counts = torch.bincount(all_choices.flatten(), minlength=spec['experts'])
    expert_mass = torch.zeros(spec['experts'], dtype=torch.float64)
    expert_mass.scatter_add_(0, all_choices.flatten(), all_mix.flatten())
    files = dict(scores='lasd_moe_scores_pin.csv', weights='lasd_moe_weights_pin.csv',
                 experts='lasd_moe_experts_pin.csv', metadata='lasd_moe_diagnostics_pin.json')
    _write_csv(directory / files['scores'], score_rows)
    _write_csv(directory / files['weights'], weight_rows)
    _write_csv(directory / files['experts'], expert_rows)
    if 'global_context' in spec:
        files['global_scores'] = 'lasd_moe_global_scores_pin.csv'
        _write_csv(directory / files['global_scores'], global_rows)
    sample_mean = torch.stack(sample_layer_means).mean(0)
    head_summary = {}
    if headwise:
        head_summary = dict(per_head=dict(axes='H,L',
            node_weighted_mean=all_weights.mean(0).tolist(),
            sample_weighted_mean=sample_mean.tolist(),
            node_weighted_std=all_weights.std(0, unbiased=False).tolist()))
        # For comparison with the scalar variants, average groups within each node.
        all_weights = all_weights.mean(1)
        sample_mean = sample_mean.mean(0)
    report = dict(model=model.mode, split='val', checkpoint_selection=False,
        sample_ids=sample_ids, valid_token_counts=sizes, identities=identities,
        contract=spec, files=files, statistics=statistics,
        layer_weight_summary=dict(
            node_weighted_mean=all_weights.mean(0).tolist(),
            sample_weighted_mean=sample_mean.tolist(),
            node_weighted_std=all_weights.std(0, unbiased=False).tolist(),
            valid_nodes=node_count, includes_graph_token=True, **head_summary),
        expert_summary=dict(
            selection_counts=counts.tolist(), valid_nodes=node_count,
            selection_fraction_of_nodes=(counts.double()/node_count).tolist(),
            selection_fraction_of_routes=(counts.double()/(node_count*spec['top_k'])).tolist(),
            mean_selected_mixing_weight=(expert_mass/node_count).tolist(),
            mean_router_probability=all_probs.mean(0).tolist()),
        notes=[
            'ID validation diagnostics only; no checkpoint selection or training loss uses these exports.',
            'PE layers only; SE and final graph block do not enter the score sequence.',
            'Statistics include GRAPH and native pin tokens; padding is excluded.',
            'R/T measure attention weights, not physical information transfer.',
            'Diagonal is excluded after original masked softmax without renormalization.',
            'First PE delta is boundary zero; same-index heads across layers need not share semantics.',
            'Alternating local/global masks contribute to observed cross-layer changes.',
            'Pre-clip gradient norms in stage history measure full block gradients, not score-path attribution.'])
    if 'global_context' in spec:
        report['notes'].append(
            'global_scores CSV contains the actual MoE graph inputs: mean R, sqrt(population variance R + '
            'epsilon), sqrt(population variance T + epsilon), and adjacent-PE differences. '
            'Existing statistics std fields remain unstabilized descriptive values. No mean T input, '
            'Gaussian fit, running statistics or batch-level pooling is used.')
    (directory / files['metadata']).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return dict(samples=len(sample_ids), split='val', **files)
