"""Differentiable, pre-dropout R/T statistics; no change to attention messages.

The repository uses dense masked MultiheadAttention, not sparse edge attention.
Recompute only Q/K scores in query chunks; never retain attention matrices for
diagnostics. Optional activation checkpointing bounds saved score intermediates.
All computations stay on the input device, including backward recomputation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def score_from_attention_pin(weights, valid):
    """[B,H,N,N] -> [B,N,H,2] in [receive(row), transmit(column)] order.

Input is already normalized using the original attention mask. Exclusion of
self/padding is a statistic, not a second softmax. Mask before summing, rather
than multiplying by zero, so padded NaN entries cannot contaminate real nodes.
"""
    if weights.ndim != 4 or weights.shape[-1] != weights.shape[-2]:
        raise ValueError('Scores require batched square multi-head attention')
    b, _, n, _ = weights.shape
    if valid.shape != (b, n) or valid.dtype != torch.bool or valid.device != weights.device:
        raise ValueError('Valid node mask must match batch, node order and device')
    keep = valid[:, None, :, None] & valid[:, None, None, :]
    keep = keep & ~torch.eye(n, device=weights.device, dtype=torch.bool)[None, None]
    off_diagonal = weights.masked_fill(~keep, 0.)
    return torch.stack((off_diagonal.sum(-1), off_diagonal.sum(-2)), -1).permute(0, 2, 1, 3)


def layer_score_features_pin(scores, *, dynamics, node_ids=None):
    """Keep all heads and PE layers; first difference is boundary-filled zero.

The native encoder never samples/reorders tokens. External callers that do
reorder must provide per-layer node_ids; mismatch is rejected, not subtracted.
"""
    if not scores or any(s.shape != scores[0].shape for s in scores):
        raise ValueError('Layer scores must preserve batch/head/node axes')
    if scores[0].ndim != 4 or scores[0].shape[-1] != 2:
        raise ValueError('Layer score shape must be [B,N,H,2]')
    if node_ids is not None:
        if (len(node_ids) != len(scores) or any(ids.shape != scores[0].shape[:2] for ids in node_ids)
                or any(not torch.equal(ids, node_ids[0]) for ids in node_ids)):
            raise ValueError('Cross-layer node identities are not aligned')
    result, differences = [], []
    for i, current in enumerate(scores):
        delta = torch.zeros_like(current) if i == 0 else current - scores[i-1]
        differences.append(delta)
        raw = current.flatten(-2)
        result.append(torch.cat((raw, delta.flatten(-2)), -1) if dynamics else raw)
    return tuple(result), tuple(differences)


def _chunk_scores(q, k, mask, valid_q, valid_k, positions):
    # FP32 score softmax for low-precision inputs. FP64 remains FP64 for checks.
    work_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    logits = (q.to(work_dtype) * (q.shape[-1] ** -.5)) @ k.to(work_dtype).transpose(-1, -2)
    logits = logits.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool else logits + mask
    logits = logits.masked_fill(~valid_k[:, None, None, :], float('-inf'))
    # Padded/all-masked query rows get a finite dummy softmax then exactly zero.
    has_keys = ~torch.isneginf(logits).all(-1, keepdim=True)
    weights = torch.softmax(torch.where(has_keys, logits, 0.), -1)
    diagonal = positions[:, None] == torch.arange(k.shape[-2], device=k.device)[None]
    keep = has_keys & valid_q[:, None, :, None] & valid_k[:, None, None, :] & ~diagonal[None, None]
    weights = weights.masked_fill(~keep, 0.)
    return weights.sum(-1), weights.sum(-2)


class AttentionScoreExtractorPin(nn.Module):
    """Stateless observer of the exact inputs/weights/masks used by one MHA."""
    def __init__(self, chunk_size=128, checkpoint_scores=True):
        super().__init__()
        if type(chunk_size) is not int or chunk_size < 1 or type(checkpoint_scores) is not bool:
            raise ValueError('Invalid attention score extraction options')
        self.chunk_size = chunk_size
        self.checkpoint_scores = checkpoint_scores

    def forward(self, attention, x, mask, valid):
        if (not isinstance(attention, nn.MultiheadAttention) or not attention.batch_first
                or not attention._qkv_same_embed_dim or attention.bias_k is not None
                or attention.bias_v is not None or attention.add_zero_attn):
            raise ValueError('LASD requires the native square self-attention layout')
        if x.ndim != 3 or valid.shape != x.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('LASD inputs require aligned [B,N,D] and boolean valid nodes')
        b, n, dim = x.shape
        h, depth = attention.num_heads, dim // attention.num_heads
        if dim != attention.embed_dim or valid.device != x.device:
            raise ValueError('LASD input dimension/device disagrees with native attention')
        # Project only Q and K, with the same packed weights as the message path.
        bias = None if attention.in_proj_bias is None else attention.in_proj_bias[:2*dim]
        q, k = F.linear(x, attention.in_proj_weight[:2*dim], bias).chunk(2, -1)
        q = q.reshape(b, n, h, depth).transpose(1, 2)
        k = k.reshape(b, n, h, depth).transpose(1, 2)
        if mask is None:
            mask = x.new_zeros(1, 1, 1, n).expand(b, h, n, n)
        elif mask.shape == (n, n):
            mask = mask[None, None].expand(b, h, -1, -1)
        elif mask.shape == (b*h, n, n):
            mask = mask.reshape(b, h, n, n)
        elif mask.shape != (b, h, n, n):
            raise ValueError('LASD mask shape differs from the native MHA mask')
        if mask.device != x.device or (mask.dtype != torch.bool and not mask.is_floating_point()):
            raise ValueError('LASD attention mask must be boolean or floating on the input device')
        # nn.MultiheadAttention boolean masks use True=blocked. Internal chunks
        # use True=allowed, mirroring the float mask without renormalizing edges.
        if mask.dtype == torch.bool:
            mask = ~mask
        receives, transmit = [], None
        for start in range(0, n, self.chunk_size):
            end = min(start+self.chunk_size, n)
            arguments = (q[:, :, start:end], k, mask[:, :, start:end],
                         valid[:, start:end], valid, torch.arange(start, end, device=x.device))
            if self.checkpoint_scores and torch.is_grad_enabled() and (q.requires_grad or mask.requires_grad):
                receive, part = checkpoint(_chunk_scores, *arguments, use_reentrant=False, preserve_rng_state=False)
            else:
                receive, part = _chunk_scores(*arguments)
            receives.append(receive)
            transmit = part if transmit is None else transmit + part
        return torch.stack((torch.cat(receives, -1), transmit), -1).permute(0, 2, 1, 3).to(x.dtype)


class ScoreMemoryAdapterPin(nn.Module):
    """One independent adapter per PE; changes decoder memory only."""
    def __init__(self, feature_dim, score_dim, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim+score_dim, hidden_dim), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, feature_dim))

    def forward(self, memory, scores, padding):
        if memory.shape[:2] != scores.shape[:2] or padding.shape != memory.shape[:2]:
            raise ValueError('Score adapter requires aligned memory, scores and padding')
        joined = torch.cat((memory, scores), -1).masked_fill(padding.unsqueeze(-1), 0.)
        return self.network(joined).masked_fill(padding.unsqueeze(-1), 0.)
