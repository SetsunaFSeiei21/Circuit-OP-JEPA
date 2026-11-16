"""Read-only, exact weight SVD and bounded activation diagnostics.

Matrices use y = W @ x. Cross-circuit bases are diagnostic fits in the SAVED
coordinates, not neuron/head alignment, deployable compression, or OOD results.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn


@dataclass
class MatrixPin:
    name: str
    weight: torch.Tensor
    module: nn.Module
    module_path: str
    kind: str
    expert: int | None = None


def component_pin(name):
    side = 'op_predictor' if name.startswith('predictor.') else 'performance'
    if 'structure_encoder.' in name:
        part = 'SE'
    elif 'fusion_blocks.' in name:
        part = 'PE'
    elif 'encoder.' in name:
        part = 'encoder_other'
    elif '.head.' in name or name.startswith('decoder.'):
        part = 'readout'
    elif name.startswith('gate.'):
        part = 'gate'
    else:
        part = 'other'
    return side + '/' + part


def parameter_counts_pin(model):
    counts = defaultdict(lambda: [0, 0])
    # named_parameters removes tied aliases; buffers are not model parameters.
    for name, parameter in model.named_parameters():
        row = counts[component_pin(name)]
        row[0] += parameter.numel()
        row[1] += parameter.numel() if parameter.requires_grad else 0
    return [dict(component=k, parameters=v[0], trainable_parameters=v[1],
                 frozen_parameters=v[0] - v[1]) for k, v in sorted(counts.items())]


def matrices_pin(model, scope='ffn', component='both'):
    """Cover FFNs, linear projections, split Q/K/V and wide-readout experts.

Embedding tables, query tokens, normalization and biases remain in the parameter
ledger, but are not linear-map compression candidates. MHA out_proj calls are
functional in PyTorch: their activation column is explicitly unavailable.
    """
    if scope not in ('ffn', 'all') or component not in ('both', 'performance', 'op_predictor'):
        raise ValueError('Invalid matrix scope/component')
    result, seen = [], set()
    for path, module in model.named_modules():
        side = 'op_predictor' if path.startswith('predictor.') else 'performance'
        if component != 'both' and component != side:
            continue
        is_ffn = '.ffn.' in path or path.endswith(('.linear1', '.linear2'))
        if isinstance(module, nn.Linear) and (scope == 'all' or is_ffn):
            if id(module.weight) not in seen:
                result.append(MatrixPin(path + '.weight', module.weight, module, path, 'linear'))
                seen.add(id(module.weight))
        if scope == 'all' and isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                if id(module.in_proj_weight) in seen:
                    continue
                seen.add(id(module.in_proj_weight))
                for label, weight in zip(('q', 'k', 'v'), module.in_proj_weight.chunk(3)):
                    result.append(MatrixPin(path + '.in_proj_weight::' + label,
                        weight, module, path, 'mha_' + label))
            else:
                for label in ('q', 'k', 'v'):
                    weight = getattr(module, label + '_proj_weight')
                    if id(weight) not in seen:
                        result.append(MatrixPin(path + '.' + label + '_proj_weight',
                            weight, module, path, 'mha_' + label))
                        seen.add(id(weight))
        # DenseMoEMLPPin stores [expert,input,output], unlike nn.Linear.
        if scope == 'all' and module.__class__.__name__ == 'DenseMoEMLPPin':
            for number in (1, 2):
                weights = getattr(module, 'weight' + str(number))
                if id(weights) in seen:
                    continue
                seen.add(id(weights))
                for expert, weight in enumerate(weights):
                    result.append(MatrixPin(f'{path}.weight{number}::expert_{expert}',
                        weight.T, module, path, 'moe_' + str(number), expert))
    if not result:
        raise ValueError('No supported matrices matched this scope/component')
    return result


def svd_pin(matrix):
    if matrix.ndim != 2 or not matrix.numel() or not bool(torch.isfinite(matrix).all()):
        raise ValueError('SVD requires a nonempty, finite matrix')
    options = {'driver': 'gesvd'} if matrix.is_cuda else {}
    return torch.linalg.svd(matrix, full_matrices=False, **options)


def relative_error_pin(residual, reference):
    den = float(reference.square().sum())
    num = float(residual.square().sum())
    if den == 0:
        return 0. if num == 0 else None  # undefined relative error, never infinity
    return math.sqrt(num / den)


def spectrum_summary_pin(singular, input_epsilon, shape):
    energy = singular.square()
    total = float(energy.sum())
    if total == 0:
        return dict(weight_fro_norm=0., stable_rank=0., effective_rank_energy=0.,
                    numerical_rank=0, rank_95=0, rank_99=0, rank_999=0)
    probability = energy / total
    cumulative = probability.cumsum(0)
    return dict(weight_fro_norm=math.sqrt(total), stable_rank=float(energy.sum() / energy[0]),
        effective_rank_energy=float(torch.exp(-(probability[probability > 0] *
                                               probability[probability > 0].log()).sum())),
        numerical_rank=int((singular > input_epsilon * max(shape) * singular[0]).sum()),
        **{key: min(len(singular), int(torch.searchsorted(cumulative,
                       cumulative.new_tensor(value))) + 1)
           for key, value in [('rank_95', .95), ('rank_99', .99), ('rank_999', .999)]})


def analyze_matrix_pin(spec, ranks, activations=None):
    weight = spec.weight.detach().to(dtype=torch.float64)
    u, singular, vh = svd_pin(weight)
    summary = spectrum_summary_pin(singular, torch.finfo(spec.weight.dtype).eps, weight.shape)
    summary.update(matrix=spec.name, kind=spec.kind, component=component_pin(spec.module_path),
                   rows=weight.shape[0], columns=weight.shape[1], parameters=weight.numel(),
                   activation_tokens=0 if activations is None else activations.shape[1])
    x = None if activations is None else activations.to(device=weight.device, dtype=weight.dtype)
    y = None if x is None else weight @ x
    # Best rank-r output approximation on this SAME calibration subset. It is
    # achievable by W' = U_y[:, :r] U_y[:, :r].T W; not a held-out error estimate.
    output_singular = None if y is None else svd_pin(y)[1]
    rows = []
    for requested in ranks:
        rank = min(requested, len(singular))
        approx = (u[:, :rank] * singular[:rank]) @ vh[:rank]
        numerator = float(singular[rank:].square().sum())
        denominator = float(singular.square().sum())
        output_energy = None if y is None else float(output_singular.square().sum())
        rows.append(dict(requested_rank=requested, actual_rank=rank,
            dense_parameters=weight.numel(), low_rank_parameters=rank * sum(weight.shape),
            parameter_saving_fraction=1 - rank * sum(weight.shape) / weight.numel(),
            relative_weight_error=math.sqrt(numerator / denominator) if denominator else 0.,
            relative_layer_output_error=None if y is None else relative_error_pin((weight - approx) @ x, y),
            optimal_calibration_output_error=(None if y is None else
                math.sqrt(float(output_singular[rank:].square().sum()) / output_energy)
                if output_energy else 0.)))
    # Keep one comparison group on host; SVD/reconstruction run on requested GPU.
    keep = min(max(ranks), len(singular))
    item = dict(weight=weight.cpu(), u=u[:, :keep].cpu(), v=vh[:keep].T.cpu(),
                singular=singular.cpu(), x=None if x is None else x.cpu(), summary=summary)
    return summary, rows, item


def subspace_overlap_pin(left, right, rank, device):
    """Principal angles in saved coordinates. Rank-zero overlap is undefined."""
    ra = min(rank, left['summary']['numerical_rank'], left['u'].shape[1])
    rb = min(rank, right['summary']['numerical_rank'], right['u'].shape[1])
    result = dict(requested_rank=rank, rank_a=ra, rank_b=rb,
                  alignment='saved_coordinates_no_neuron_or_head_alignment')
    for side in ('u', 'v'):
        if min(ra, rb) == 0:
            values = [None] * 4
        else:
            a, b = left[side][:, :ra].to(device), right[side][:, :rb].to(device)
            cosines = svd_pin(a.T @ b)[1].clamp(0, 1)
            overlap = float(cosines.square().sum())
            values = [overlap / min(ra, rb), overlap / ra, overlap / rb,
                      float(torch.rad2deg(torch.acos(cosines)).mean())]
        for key, value in zip(('overlap', 'a_captured', 'b_captured', 'mean_angle_deg'), values):
            result[side + '_' + key] = value
    return result


def shared_basis_pin(items, ranks, device):
    """Joint row/column SVD initialization, not an optimal joint Tucker fit.

Normalize every nonzero circuit matrix to unit Frobenius norm for fitting the
common bases. Error is measured on each original matrix/its own activations.
No dense base/residual weights are retained in the theoretical parameter cost.
    """
    if len(items) < 2 or len({tuple(i['weight'].shape) for i in items}) != 1:
        raise ValueError('Shared bases require at least two identically shaped matrices')
    weights = [i['weight'].to(device) for i in items]
    scaled = [w / torch.linalg.vector_norm(w).clamp_min(torch.finfo(w.dtype).tiny) for w in weights]
    u = svd_pin(torch.cat(scaled, dim=1))[0]
    v = svd_pin(torch.cat(scaled, dim=0))[2].T
    m, n = weights[0].shape
    groups, errors = [], []
    for requested in ranks:
        rank = min(requested, m, n)
        a, b = u[:, :rank], v[:, :rank]
        cost = rank * (m + n) + len(items) * rank * rank
        groups.append(dict(requested_rank=requested, actual_rank=rank, circuits=len(items),
            dense_parameters=len(items) * m * n,
            independent_low_rank_parameters=len(items) * rank * (m + n),
            shared_parameters=cost, parameter_saving_fraction=1 - cost / (len(items) * m * n),
            alignment='saved_coordinates_no_neuron_or_head_alignment',
            basis_fit='unit_Frobenius_per_circuit_joint_row_column_SVD'))
        for index, (weight, item) in enumerate(zip(weights, items)):
            residual = weight - a @ (a.T @ weight @ b) @ b.T
            x = None if item['x'] is None else item['x'].to(device)
            errors.append(dict(item_index=index, requested_rank=requested, actual_rank=rank,
                relative_weight_error=relative_error_pin(residual, weight),
                relative_layer_output_error=None if x is None else relative_error_pin(residual @ x, weight @ x)))
    return groups, errors


class ActivationCapturePin:
    """Bounded real-input samples; no gradients, refits or forward modifications.

Every batch gets an equal token quota (at most max_tokens overall per matrix).
All numerical sampling stays on the inference device. Padding is excluded.
MHA projections are weight-only in this first probe version; no fabricated zeros.
    """
    def __init__(self, specs, max_tokens=512, seed=17):
        self.specs, self.max_tokens, self.seed = specs, max_tokens, seed
        self.values = defaultdict(list)
        self.counts = defaultdict(int)
        self.batch = None
        self.target = 0

    def set_batch(self, batch, index, total):
        self.batch = batch
        self.target = self.max_tokens * (index + 1) // total

    def _take(self, spec, values):
        quota = self.target - self.counts[spec.name]
        if quota <= 0:
            return
        values = values.detach()
        if values.shape[-1] != spec.weight.shape[1]:
            raise ValueError(f'Activation width mismatch: {spec.name}')
        if values.ndim >= 3 and values.shape[0] == self.batch['device_mask'].shape[0]:
            # Performance decoders operate on metric tokens, including their MoE.
            metric_axis = spec.module_path.startswith('decoder.')
            mask = self.batch['owner_mask' if spec.module_path.startswith('predictor.') else 'device_mask']
            if not metric_axis and values.shape[1] in (mask.shape[1], mask.shape[1] + 1):
                if values.shape[1] == mask.shape[1] + 1:
                    mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
                values = values[mask]
        values = values.reshape(-1, values.shape[-1])
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f'Non-finite captured activations: {spec.name}')
        count = min(quota, len(values))
        digest = int.from_bytes(hashlib.sha256(spec.name.encode()).digest()[:8], 'little')
        generator = torch.Generator(device=values.device).manual_seed((digest + self.seed + self.counts[spec.name]) % (2**63))
        indices = torch.randperm(len(values), generator=generator, device=values.device)[:count]
        self.values[spec.name].append(values[indices].T.contiguous())
        self.counts[spec.name] += count

    def install(self):
        stack = ExitStack()
        for spec in self.specs:
            if spec.kind == 'linear':
                def hook(module, args, spec=spec):
                    self._take(spec, args[0])
                stack.callback(spec.module.register_forward_pre_hook(hook).remove)
            elif spec.kind.startswith('moe_'):
                def hook(module, args, spec=spec):
                    values = args[0]
                    if spec.kind == 'moe_2':
                        values = module.activation(values @ module.weight1[spec.expert] + module.bias1[spec.expert])
                    self._take(spec, values)
                stack.callback(spec.module.register_forward_pre_hook(hook).remove)
        return stack

    def get(self, name):
        return torch.cat(self.values[name], dim=1) if self.counts[name] else None
