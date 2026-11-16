"""From-scratch low-rank decoder projections, preserving dense token attention.

Only linear weight matrices are factorized. No activation is inserted between
the factors, no full-size weight is retained, and attention masks keep the
``nn.MultiheadAttention`` convention (True means blocked).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _rank_pin(in_features, out_features, rank):
    if any(type(v) is not int or v < 1 for v in (in_features, out_features, rank)):
        raise ValueError('Linear dimensions and rank must be positive integers')
    if rank * (in_features + out_features) >= in_features * out_features:
        raise ValueError('Rank must strictly reduce the number of weight parameters')
    return rank


class LowRankLinearPin(nn.Module):
    """Linear(in, out) represented by two independently trained small factors."""
    def __init__(self, in_features, out_features, rank, bias=True, *, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.rank = _rank_pin(in_features, out_features, rank)
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.down.reset_parameters()
        self.up.reset_parameters()
        # Match the marginal variance of an ordinary Linear weight product.
        # Both factors remain nonzero so both receive gradients at step one.
        with torch.no_grad():
            self.up.weight.mul_(math.sqrt(3.0))
            if self.up.bias is not None:
                nn.init.uniform_(self.up.bias, -1 / math.sqrt(self.in_features),
                                 1 / math.sqrt(self.in_features))

    def forward(self, values):
        return self.up(self.down(values))


class LowRankMultiheadAttentionPin(nn.Module):
    """Batch-first MHA with independent low-rank Q/K/V/output projections.

The normal training path uses SDPA directly on the input device and never
materializes a dense projection weight. Attention probabilities are computed
explicitly only when requested for diagnostics.
"""
    def __init__(self, embed_dim, num_heads, rank, dropout=0., bias=True,
                 batch_first=True, *, device=None, dtype=None):
        super().__init__()
        _rank_pin(embed_dim, embed_dim, rank)
        if type(num_heads) is not int or num_heads < 1 or embed_dim % num_heads:
            raise ValueError('embed_dim must be divisible by a positive num_heads')
        if not batch_first:
            raise ValueError('Low-rank decoder attention requires batch_first=True')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.embed_dim, self.num_heads, self.rank = embed_dim, num_heads, rank
        self.head_dim = embed_dim // num_heads
        self.dropout, self.batch_first = float(dropout), True
        self.kdim = self.vdim = embed_dim
        self._qkv_same_embed_dim = True
        factory = dict(device=device, dtype=dtype)
        self.q_proj = LowRankLinearPin(embed_dim, embed_dim, rank, bias, **factory)
        self.k_proj = LowRankLinearPin(embed_dim, embed_dim, rank, bias, **factory)
        self.v_proj = LowRankLinearPin(embed_dim, embed_dim, rank, bias, **factory)
        self.out_proj = LowRankLinearPin(embed_dim, embed_dim, rank, bias, **factory)
        # Packed dense MHA QKV uses Xavier variance 1/(2*d), whereas an
        # ordinary Linear uses 1/(3*d). Match those marginal product variances;
        # this is fresh initialization, not equal dense weights or a dense SVD.
        with torch.no_grad():
            for projection in (self.q_proj, self.k_proj, self.v_proj):
                projection.up.weight.mul_(math.sqrt(1.5))
        # Match standard MHA's zero projection biases; factors are fresh.
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if projection.up.bias is not None:
                nn.init.zeros_(projection.up.bias)

    @staticmethod
    def _additive_mask(mask, dtype, device):
        if mask.device != device:
            raise ValueError('Attention masks and inputs must be on the same device')
        if mask.dtype == torch.bool:
            return torch.zeros_like(mask, dtype=dtype).masked_fill(mask, float('-inf'))
        if not mask.is_floating_point():
            raise TypeError('Attention masks must be boolean or floating point')
        return mask.to(dtype=dtype)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        unbatched = query.ndim == 2
        if unbatched:
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        if any(t.ndim != 3 or t.shape[-1] != self.embed_dim for t in (query, key, value)):
            raise ValueError('Expected query/key/value with shape (batch, tokens, embed_dim)')
        batch, target, _ = query.shape
        source = key.shape[1]
        if key.shape[0] != batch or value.shape[:2] != (batch, source):
            raise ValueError('Query/key/value batch sizes and key/value lengths must match')
        def project(layer, values):
            return layer(values).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = project(self.q_proj, query), project(self.k_proj, key), project(self.v_proj, value)
        mask = None
        if attn_mask is not None:
            if attn_mask.ndim == 2 and tuple(attn_mask.shape) == (target, source):
                mask = attn_mask[None, None]
            elif attn_mask.ndim == 3 and tuple(attn_mask.shape) == (batch * self.num_heads, target, source):
                mask = attn_mask.reshape(batch, self.num_heads, target, source)
            else:
                raise ValueError('attn_mask must have shape (target, source) or (batch * heads, target, source)')
            mask = self._additive_mask(mask, q.dtype, q.device)
        if key_padding_mask is not None:
            if tuple(key_padding_mask.shape) != (batch, source):
                raise ValueError('key_padding_mask must have shape (batch, source)')
            padding = self._additive_mask(key_padding_mask[:, None, None, :], q.dtype, q.device)
            mask = padding if mask is None else mask + padding
        if is_causal:
            causal = torch.ones(target, source, dtype=torch.bool, device=q.device).triu(1)
            causal = self._additive_mask(causal[None, None], q.dtype, q.device)
            mask = causal if mask is None else mask + causal
        dropout = self.dropout if self.training else 0.
        weights = None
        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            # A fully masked row has no attended values, matching SDPA's zero.
            # Avoid an undefined softmax backward, without hiding unrelated NaNs.
            blocked = torch.isneginf(scores).all(dim=-1, keepdim=True)
            weights = torch.softmax(scores.masked_fill(blocked, 0.), dim=-1).masked_fill(blocked, 0.)
            weights = F.dropout(weights, p=dropout, training=self.training)
            attended = weights @ v
            if average_attn_weights:
                weights = weights.mean(dim=1)
        else:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                      dropout_p=dropout)
        output = self.out_proj(attended.transpose(1, 2).reshape(batch, target, self.embed_dim))
        if unbatched:
            output = output.squeeze(0)
            if weights is not None:
                weights = weights.squeeze(0)
        return output, weights


def factorize_decoder_pin(decoder, *, attention_rank=None, ffn_rank=None):
    """Replace selected projections in-place with fresh low-rank modules.

    ``None`` leaves a projection family untouched. This is an architectural
    replacement before training, not an SVD approximation of saved weights.
    All ranks are validated before any mutation. Returns ``decoder``.
    """
    if not isinstance(decoder, nn.TransformerDecoder):
        raise TypeError('Expected nn.TransformerDecoder')
    if not decoder.layers:
        raise ValueError('Decoder must contain at least one layer')
    for layer in decoder.layers:
        if attention_rank is not None:
            for attention in (layer.self_attn, layer.multihead_attn):
                if not isinstance(attention, nn.MultiheadAttention) or not attention.batch_first:
                    raise ValueError('Expected unmodified batch-first MultiheadAttention')
                if not attention._qkv_same_embed_dim or attention.bias_k is not None or attention.add_zero_attn:
                    raise ValueError('Unsupported nonstandard decoder attention')
                _rank_pin(attention.embed_dim, attention.embed_dim, attention_rank)
        if ffn_rank is not None:
            for linear in (layer.linear1, layer.linear2):
                if not isinstance(linear, nn.Linear):
                    raise ValueError('Expected unmodified decoder FFN linear projections')
                _rank_pin(linear.in_features, linear.out_features, ffn_rank)
    for layer in decoder.layers:
        if attention_rank is not None:
            for name in ('self_attn', 'multihead_attn'):
                old = getattr(layer, name)
                replacement = LowRankMultiheadAttentionPin(old.embed_dim, old.num_heads,
                    attention_rank, dropout=old.dropout, bias=old.in_proj_bias is not None,
                    device=old.out_proj.weight.device, dtype=old.out_proj.weight.dtype)
                replacement.train(old.training)
                setattr(layer, name, replacement)
        if ffn_rank is not None:
            for name in ('linear1', 'linear2'):
                old = getattr(layer, name)
                replacement = LowRankLinearPin(old.in_features, old.out_features, ffn_rank,
                    bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
                replacement.train(old.training)
                setattr(layer, name, replacement)
    return decoder
