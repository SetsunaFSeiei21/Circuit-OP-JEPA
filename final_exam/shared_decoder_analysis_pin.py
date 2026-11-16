"""Within-checkpoint shared DOWN-basis diagnostics for per-PE decoders.

For weights W_i[out,in], fit W_i ~= B_i A using a vertical stack of the
unscaled original matrices. No circuit/checkpoint/seed is ever combined here.
This is post-hoc weight approximation, not evidence for trained shared models.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from itertools import combinations
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from final_exam.models.low_rank_decoder_pin import LowRankLinearPin, LowRankMultiheadAttentionPin
from final_exam.spectral_analysis_pin import svd_pin, spectrum_summary_pin, relative_error_pin


@dataclass
class DecoderMatrixPin:
    branch: int
    path: str
    weight: torch.Tensor
    module: nn.Module
    kind: str
    side: str


@dataclass
class DecoderGroupPin:
    name: str
    side: str
    layer: int
    role: str
    matrices: list[DecoderMatrixPin]


def _linear_weight(module):
    if isinstance(module, nn.Linear):
        return module.weight
    if isinstance(module, LowRankLinearPin):
        return module.up.weight.detach() @ module.down.weight.detach()
    raise ValueError(f'Unsupported decoder linear operator: {type(module).__name__}')


def decoder_groups_pin(model, component='performance', scope='all'):
    """Corresponding branch projections; self/cross attention never share groups."""
    if component not in ('performance', 'op_predictor', 'both') or scope not in ('all', 'attention', 'ffn'):
        raise ValueError('Invalid component or scope')
    heads = []
    for side, prefix, head in (
            ('performance', 'decoder', getattr(model, 'decoder', None)),
            ('op_predictor', 'predictor.head', getattr(getattr(model, 'predictor', None), 'head', None))):
        if component != 'both' and component != side:
            continue
        if head is None or not hasattr(head, 'branches'):
            if component != 'both':
                raise ValueError(f'{side} has no per-PE decoder branches')
            continue
        if len(head.branches) < 2:
            raise ValueError('Shared-basis analysis requires at least two PE decoder branches')
        depths = {len(branch.decoder.layers) for branch in head.branches}
        if len(depths) != 1:
            raise ValueError('Decoder branches have unequal depths')
        heads.append((side, prefix, head))
    result = []
    for side, prefix, head in heads:
        grouped = defaultdict(list)
        for branch_index, branch in enumerate(head.branches):
            for layer_index, layer in enumerate(branch.decoder.layers):
                base = f'{prefix}.branches.{branch_index}.decoder.layers.{layer_index}'
                if scope in ('all', 'attention'):
                    for attention_name in ('self_attn', 'multihead_attn'):
                        attention = getattr(layer, attention_name)
                        path = base + '.' + attention_name
                        if isinstance(attention, nn.MultiheadAttention):
                            if not attention.batch_first or not attention._qkv_same_embed_dim or attention.in_proj_weight is None:
                                raise ValueError('Only same-width packed decoder MHA is supported')
                            for letter, weight in zip(('q', 'k', 'v'), attention.in_proj_weight.chunk(3)):
                                grouped[(layer_index, attention_name + '.' + letter)].append(DecoderMatrixPin(
                                    branch_index + 1, path + '.in_proj_weight::' + letter,
                                    weight, attention, 'mha_' + letter, side))
                            grouped[(layer_index, attention_name + '.out')].append(DecoderMatrixPin(
                                branch_index + 1, path + '.out_proj.weight', attention.out_proj.weight,
                                attention.out_proj, 'functional_out', side))
                        elif isinstance(attention, LowRankMultiheadAttentionPin):
                            for letter, attr in (('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('out', 'out_proj')):
                                projection = getattr(attention, attr)
                                grouped[(layer_index, attention_name + '.' + letter)].append(DecoderMatrixPin(
                                    branch_index + 1, path + '.' + attr + '.effective_weight',
                                    _linear_weight(projection), projection,
                                    'factorized' if isinstance(projection, LowRankLinearPin) else 'linear', side))
                        else:
                            raise ValueError('Unsupported decoder attention implementation')
                if scope in ('all', 'ffn'):
                    for attr in ('linear1', 'linear2'):
                        module = getattr(layer, attr)
                        grouped[(layer_index, attr)].append(DecoderMatrixPin(
                            branch_index + 1, base + '.' + attr + '.weight', _linear_weight(module),
                            module, 'factorized' if isinstance(module, LowRankLinearPin) else 'linear', side))
        for (layer, role), matrices in sorted(grouped.items()):
            if len({tuple(spec.weight.shape) for spec in matrices}) != 1:
                raise ValueError('Corresponding decoder projections must have equal dimensions')
            result.append(DecoderGroupPin(f'{side}.layer_{layer+1}.{role}', side, layer+1, role, matrices))
    if not result:
        raise ValueError('No multi-PE decoder groups matched the selection')
    return result


def pairwise_down_bases_pin(specs, weights, decompositions, ranks, activations, vector_limit):
    """Compare input subspaces in saved coordinates, independently per pair.

    Use an equal, non-null rank on both sides. Otherwise arbitrary SVD null
    vectors can manufacture agreement. Vector heatmaps are auxiliary: signs
    are removed, but rotations within repeated singular values remain possible.
    All linear algebra stays on the weights' device; only report values leave it.
    """
    summaries = [spectrum_summary_pin(s, torch.finfo(spec.weight.dtype).eps, weight.shape)
                 for spec, weight, (_, s, _) in zip(specs, weights, decompositions)]
    overlap_rows, angle_rows, cross_rows, vector_rows = [], [], [], []
    for ia, ib in combinations(range(len(specs)), 2):
        sa, sb = summaries[ia], summaries[ib]
        _, sing_a, va = decompositions[ia]
        _, sing_b, vb = decompositions[ib]
        dimension = weights[ia].shape[1]
        identity = dict(branch_a=specs[ia].branch, branch_b=specs[ib].branch,
                        matrix_a=specs[ia].path, matrix_b=specs[ib].path,
                        input_dim=dimension, numerical_rank_a=sa['numerical_rank'],
                        numerical_rank_b=sb['numerical_rank'])
        # Save only a bounded heatmap once per pair, not r*r duplicated rows
        # for every requested rank. Each CSV/Excel row is one cosine row.
        na = min(max(ranks), vector_limit, sa['numerical_rank'])
        nb = min(max(ranks), vector_limit, sb['numerical_rank'])
        if na and nb:
            absolute = (va[:na] @ vb[:nb].T).abs().clamp(0, 1).cpu().tolist()
            for index, values in enumerate(absolute):
                vector_rows.append(dict(**identity, vector_rank_a=na, vector_rank_b=nb,
                    vector_index_a=index+1, absolute_cosines=values,
                    interpretation='columns are branch_b right singular vectors 1..vector_rank_b; auxiliary only'))
        for requested in ranks:
            rank = min(requested, sa['numerical_rank'], sb['numerical_rank'])
            common = dict(**identity, requested_rank=requested, actual_rank=rank,
                          status='ok' if rank else 'no_nonzero_common_rank')
            if rank:
                cosines = svd_pin(va[:rank] @ vb[:rank].T)[1].clamp(0, 1)
                angles = torch.rad2deg(torch.acos(cosines))
                overlap = max(0., min(1., float(cosines.square().mean())))
                random_expected = rank / dimension
                # Full input spaces must overlap by 1. This is tautological,
                # not evidence for meaningful low-dimensional sharing.
                adjusted = ((overlap-random_expected)/(1-random_expected)
                            if rank < dimension else None)
                values = dict(overlap=overlap, random_expected_overlap=random_expected,
                    chance_adjusted_overlap=adjusted, full_input_space=rank == dimension,
                    min_angle_deg=float(angles.min()), mean_angle_deg=float(angles.mean()),
                    max_angle_deg=float(angles.max()))
                for index, (cosine, angle) in enumerate(zip(cosines.cpu().tolist(), angles.cpu().tolist())):
                    angle_rows.append(dict(**common, angle_index=index+1, cosine=cosine,
                                           angle_deg=angle))
            else:
                values = dict(overlap=None, random_expected_overlap=None,
                    chance_adjusted_overlap=None, full_input_space=False,
                    min_angle_deg=None, mean_angle_deg=None, max_angle_deg=None)
            for suffix, idx, singular in (('a', ia, sing_a), ('b', ib, sing_b)):
                gap = (float((singular[rank-1]-singular[rank])/singular[0])
                       if rank and rank < len(singular) else None)
                cutoff = torch.finfo(specs[idx].weight.dtype).eps * max(weights[idx].shape)
                values['cutoff_relative_gap_'+suffix] = gap
                values['cutoff_ambiguous_'+suffix] = gap is not None and gap <= cutoff
                values['retained_energy_'+suffix] = (float(singular[:rank].square().sum()/singular.square().sum())
                                                     if rank else None)
            overlap_rows.append(dict(**common, **values))
            # Direction matters. Reconstruct target W with the donor's right
            # basis, and compare against target's OWN best rank-r basis.
            for donor, target in ((ia, ib), (ib, ia)):
                wd = weights[target]
                donor_v = decompositions[donor][2][:rank]
                own_v = decompositions[target][2][:rank]
                row = dict(requested_rank=requested, actual_rank=rank,
                    donor_branch=specs[donor].branch, target_branch=specs[target].branch,
                    donor_matrix=specs[donor].path, target_matrix=specs[target].path,
                    status=common['status'], activation_tokens=0,
                    cross_weight_error=None, own_weight_error=None,
                    cross_retained_energy=None, own_retained_energy=None,
                    retained_vs_own=None, excess_relative_weight_sse=None,
                    cross_output_error=None, own_output_error=None)
                x = None if activations is None else activations.get(specs[target].path)
                if x is not None:
                    if x.ndim != 2 or x.shape[0] != dimension or not bool(torch.isfinite(x).all()):
                        raise ValueError(f'Invalid calibration activations: {specs[target].path}')
                    x = x.to(device=wd.device, dtype=wd.dtype)
                    row['activation_tokens'] = x.shape[1]
                if rank:
                    cross = (wd @ donor_v.T) @ donor_v
                    own = (wd @ own_v.T) @ own_v
                    total = float(wd.square().sum())
                    cross_energy = min(1., max(0., float(cross.square().sum())/total))
                    own_energy = min(1., max(0., float(own.square().sum())/total))
                    cross_error = relative_error_pin(wd-cross, wd)
                    own_error = relative_error_pin(wd-own, wd)
                    row.update(cross_weight_error=cross_error, own_weight_error=own_error,
                        cross_retained_energy=cross_energy, own_retained_energy=own_energy,
                        retained_vs_own=min(1., cross_energy/own_energy),
                        excess_relative_weight_sse=max(0., cross_error**2-own_error**2),
                        cross_output_error=None if x is None else relative_error_pin((wd-cross) @ x, wd @ x),
                        own_output_error=None if x is None else relative_error_pin((wd-own) @ x, wd @ x))
                row['activation_note'] = ('target branch train inputs; bias excluded' if x is not None
                                          else 'unavailable; blank is not zero')
                cross_rows.append(row)
    return dict(overlap_rows=overlap_rows, angle_rows=angle_rows,
                cross_basis_rows=cross_rows, vector_cosine_rows=vector_rows)


def analyze_group_pin(group, ranks, activations=None, *, vector_limit=64):
    """Raw stacked SVD, independent-SVD comparator and per-branch input errors.

    Rank for the common A may exceed one matrix's output width: the union of
    row spaces can have a larger rank. Independent ranks are capped separately.
    All budgets count weights only and use the actual ranks of each strategy.
    """
    if not ranks or len(set(ranks)) != len(ranks) or any(type(r) is not int or r < 1 for r in ranks):
        raise ValueError('Ranks must be distinct positive integers')
    if type(vector_limit) is not int or not 0 <= vector_limit <= 256:
        raise ValueError('vector_limit must be an integer in [0, 256]')
    specs = group.matrices
    if len(specs) < 2 or len({tuple(s.weight.shape) for s in specs}) != 1:
        raise ValueError('At least two equally shaped branch weights are required')
    weights = [spec.weight.detach().to(dtype=torch.float64) for spec in specs]
    stacked = torch.cat(weights, dim=0)
    _, singular, vh = svd_pin(stacked)
    independent_svd = [svd_pin(w) for w in weights]
    out_dim, in_dim = weights[0].shape
    branches = len(weights)
    summary = spectrum_summary_pin(singular, torch.finfo(specs[0].weight.dtype).eps, stacked.shape)
    summary.update(group=group.name, side=group.side, layer=group.layer, role=group.role,
                   branches=branches, output_dim=out_dim, input_dim=in_dim,
                   dense_weight_parameters=branches*out_dim*in_dim,
                   fit='raw_vertical_stack_SVD', shared_factor='down_input_basis',
                   source_parameterization='effective_weights' if any(s.kind == 'factorized' for s in specs) else 'dense_weights')
    total_energy = float(singular.square().sum())
    spectrum_rows = [dict(index=i+1, singular_value=float(value), cumulative_energy=(
        float(singular[:i+1].square().sum()) / total_energy if total_energy else 0.))
        for i, value in enumerate(singular)]
    budget_rows, error_rows, approximations = [], [], {}
    for requested in ranks:
        shared_rank = min(requested, len(singular))
        independent_rank = min(requested, out_dim, in_dim)
        a = vh[:shared_rank]
        shared = [(w @ a.T) @ a for w in weights]
        independent = [(u[:, :independent_rank] * values[:independent_rank]) @ v[:independent_rank]
                       for u, values, v in independent_svd]
        dense_cost = branches*out_dim*in_dim
        independent_cost = branches*independent_rank*(in_dim+out_dim)
        shared_cost = shared_rank*in_dim+branches*out_dim*shared_rank
        shared_weight_sse = sum(float((w-p).square().sum()) for w, p in zip(weights, shared))
        independent_weight_sse = sum(float((w-p).square().sum()) for w, p in zip(weights, independent))
        budget_rows.append(dict(requested_rank=requested, shared_rank=shared_rank,
            independent_rank=independent_rank, dense_parameters=dense_cost,
            independent_parameters=independent_cost, shared_parameters=shared_cost,
            independent_saving_fraction=1-independent_cost/dense_cost,
            shared_saving_fraction=1-shared_cost/dense_cost,
            aggregate_shared_weight_error=math.sqrt(shared_weight_sse/total_energy) if total_energy else 0.,
            aggregate_independent_weight_error=math.sqrt(independent_weight_sse/total_energy) if total_energy else 0.))
        for spec, weight, sapprox, iapprox in zip(specs, weights, shared, independent):
            x = None if activations is None else activations.get(spec.path)
            if x is not None:
                if x.ndim != 2 or x.shape[0] != in_dim or not bool(torch.isfinite(x).all()):
                    raise ValueError(f'Invalid calibration activations: {spec.path}')
                x = x.to(device=weight.device, dtype=weight.dtype)
            error_rows.append(dict(branch=spec.branch, matrix=spec.path,
                requested_rank=requested, shared_rank=shared_rank, independent_rank=independent_rank,
                activation_tokens=0 if x is None else x.shape[1],
                shared_weight_error=relative_error_pin(weight-sapprox, weight),
                independent_weight_error=relative_error_pin(weight-iapprox, weight),
                shared_output_error=None if x is None else relative_error_pin((weight-sapprox) @ x, weight @ x),
                independent_output_error=None if x is None else relative_error_pin((weight-iapprox) @ x, weight @ x),
                activation_note='unavailable; blank is not zero' if x is None else 'train calibration inputs; bias excluded'))
        approximations[requested] = dict(shared=[w.cpu() for w in shared], independent=[w.cpu() for w in independent])
    pairs = pairwise_down_bases_pin(specs, weights, independent_svd, ranks, activations, vector_limit)
    return dict(summary=summary, budget_rows=budget_rows, error_rows=error_rows,
                spectrum_rows=spectrum_rows, approximations=approximations, **pairs)


class SharedDecoderCapturePin:
    """Bounded train-only inputs to Q/K/V/FFN; functional dense O stays blank.

    Captures actual arguments before original forward execution. No projection
    or attention is recomputed. Dense MHA out_proj is called functionally, so
    its input cannot be captured by a module hook; low-rank O is hookable.
    """
    def __init__(self, groups, max_tokens=512, seed=17):
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        self.specs = [spec for group in groups for spec in group.matrices]
        self.max_tokens, self.seed = max_tokens, seed
        self.values, self.counts = defaultdict(list), defaultdict(int)
        self.batch, self.target = None, 0
        self.stack = None

    def set_batch(self, batch, index, total):
        if total < 1 or not 0 <= index < total:
            raise ValueError('Invalid calibration batch index')
        self.batch = batch
        self.target = self.max_tokens*(index+1)//total

    def _take(self, spec, values, valid=None):
        quota = self.target-self.counts[spec.path]
        if quota <= 0:
            return
        values = values.detach()
        if values.ndim != 3 or values.shape[-1] != spec.weight.shape[1]:
            raise ValueError(f'Expected batch-first decoder activations: {spec.path}')
        if valid is not None:
            if tuple(valid.shape) != tuple(values.shape[:2]):
                raise ValueError(f'Calibration padding shape mismatch: {spec.path}')
            values = values[valid]
        values = values.reshape(-1, values.shape[-1])
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f'Nonfinite calibration activations: {spec.path}')
        count = min(quota, len(values))
        if not count:
            return
        digest = int.from_bytes(hashlib.sha256(spec.path.encode()).digest()[:8], 'little')
        generator = torch.Generator(device=values.device).manual_seed((digest+self.seed+self.counts[spec.path]) % 2**63)
        indexes = torch.randperm(len(values), generator=generator, device=values.device)[:count]
        self.values[spec.path].append(values[indexes].T.contiguous())
        self.counts[spec.path] += count

    def _query_mask(self, spec, values):
        if spec.side != 'op_predictor':
            return None  # Performance queries are real metric tokens.
        mask = self.batch.get('owner_mask')
        if mask is None or tuple(mask.shape) != tuple(values.shape[:2]):
            raise ValueError('Missing physical-device query mask for OP decoder capture')
        return mask

    def __enter__(self):
        self.stack = ExitStack()
        try:
            for spec in self.specs:
                if spec.kind.startswith('mha_'):
                    position = {'mha_q':0, 'mha_k':1, 'mha_v':2}[spec.kind]
                    def hook(module, args, kwargs, spec=spec, position=position):
                        values = args[position] if len(args) > position else kwargs[('query','key','value')[position]]
                        valid = self._query_mask(spec, values) if position == 0 else None
                        if position:
                            padding = kwargs.get('key_padding_mask')
                            if padding is not None:
                                valid = ~padding if padding.dtype == torch.bool else ~torch.isneginf(padding)
                        self._take(spec, values, valid)
                    self.stack.callback(spec.module.register_forward_pre_hook(hook, with_kwargs=True).remove)
                elif spec.kind != 'functional_out':
                    def hook(module, args, spec=spec):
                        values = args[0]
                        # Low-rank K/V projection inputs in cross attention can
                        # include GRAPH + padded owner/pin memory tokens.
                        if '.multihead_attn.' in spec.path and any('.'+name+'.' in spec.path for name in ('k_proj','v_proj')):
                            mask = self.batch['owner_mask' if spec.side == 'op_predictor' else 'device_mask']
                            valid = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
                        else:
                            valid = self._query_mask(spec, values)
                        self._take(spec, values, valid)
                    self.stack.callback(spec.module.register_forward_pre_hook(hook).remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def finish(self):
        return {spec.path: torch.cat(self.values[spec.path], dim=1) if self.counts[spec.path] else None
                for spec in self.specs}


@contextmanager
def apply_approximations_pin(groups, analyses, requested_rank, strategy='shared'):
    """Temporarily simulate approximate dense weights, then restore exactly.

    This measures numerical output effects, NOT compressed runtime/memory.
    Dense operators are copied in-place; existing factorized operators receive
    a temporary functional dense forward without altering their saved factors.
    """
    if strategy not in ('shared', 'independent'):
        raise ValueError('Unknown approximation strategy')
    planned = []
    for group in groups:
        analysis = analyses[group.name] if isinstance(analyses, dict) else analyses[groups.index(group)]
        approximations = analysis.get('approximations', analysis)[requested_rank][strategy]
        if len(approximations) != len(group.matrices):
            raise ValueError('Approximation branch count mismatch')
        for spec, approximate in zip(group.matrices, approximations):
            if tuple(approximate.shape) != tuple(spec.weight.shape) or not bool(torch.isfinite(approximate).all()):
                raise ValueError('Invalid approximation weight')
            planned.append((spec, approximate.to(device=spec.weight.device, dtype=spec.weight.dtype)))
    with torch.no_grad(), ExitStack() as stack:
        for spec, approximate in planned:
            if spec.kind == 'factorized':
                module = spec.module
                had_forward = 'forward' in module.__dict__
                old_forward = module.__dict__.get('forward')
                def restore(module=module, had=had_forward, old=old_forward):
                    if had:
                        module.forward = old
                    else:
                        del module.forward
                stack.callback(restore)
                module.forward = lambda values, weight=approximate, module=module: F.linear(values, weight, module.up.bias)
            else:
                original = spec.weight.detach().clone()
                stack.callback(lambda weight=spec.weight, original=original: weight.copy_(original))
                spec.weight.copy_(approximate)
        yield
