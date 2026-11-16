"""Numerical and architectural checks for factorized performance readouts."""
import copy
import io

import pytest
import torch
from torch import nn

from final_exam.models.low_rank_decoder_pin import (
    LowRankLinearPin, LowRankMultiheadAttentionPin, factorize_decoder_pin,
)


def dense_attention(low):
    dense = nn.MultiheadAttention(low.embed_dim, low.num_heads, dropout=low.dropout,
                                  bias=low.q_proj.up.bias is not None, batch_first=True,
                                  dtype=low.q_proj.down.weight.dtype,
                                  device=low.q_proj.down.weight.device)
    with torch.no_grad():
        dense.in_proj_weight.copy_(torch.cat([
            proj.up.weight @ proj.down.weight for proj in (low.q_proj, low.k_proj, low.v_proj)]))
        if dense.in_proj_bias is not None:
            dense.in_proj_bias.copy_(torch.cat([
                proj.up.bias for proj in (low.q_proj, low.k_proj, low.v_proj)]))
        dense.out_proj.weight.copy_(low.out_proj.up.weight @ low.out_proj.down.weight)
        if dense.out_proj.bias is not None:
            dense.out_proj.bias.copy_(low.out_proj.up.bias)
    dense.train(low.training)
    return dense


def test_linear_matches_materialized_weight_and_gradients():
    torch.manual_seed(101)
    module = LowRankLinearPin(16, 32, 4, dtype=torch.float64)
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.linear(x, module.up.weight @ module.down.weight, module.up.bias)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().mean().backward()
    assert module.down.bias is None
    assert sum(p.numel() for p in module.parameters()) == 4 * (16 + 32) + 32
    for p in module.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize('cross', [False, True])
@pytest.mark.parametrize('float_mask', [False, True])
@pytest.mark.parametrize('need_weights', [False, True])
def test_attention_matches_dense_materialized_factors(cross, float_mask, need_weights):
    torch.manual_seed(18)
    low = LowRankMultiheadAttentionPin(16, 4, 3, dropout=0., dtype=torch.float64)
    q = torch.randn(2, 5, 16, dtype=torch.float64)
    memory = torch.randn(2, 7 if cross else 5, 16, dtype=torch.float64) if cross else q
    mask = torch.zeros(5, memory.shape[1], dtype=torch.bool)
    mask[:, 1] = True
    padding = torch.zeros(2, memory.shape[1], dtype=torch.bool)
    padding[1, -1] = True
    if float_mask:
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, float('-inf'))
        padding = torch.zeros_like(padding, dtype=q.dtype).masked_fill(padding, float('-inf'))
        mask[:, 2] = -.37
    kw = dict(attn_mask=mask, key_padding_mask=padding,
              need_weights=need_weights, average_attn_weights=False)
    actual, weights = low(q, memory, memory, **kw)
    expected, dense_weights = dense_attention(low)(q, memory, memory, **kw)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    if need_weights:
        torch.testing.assert_close(weights, dense_weights, rtol=1e-10, atol=1e-10)
        assert (weights[..., 1] == 0).all()
        assert (weights[1, ..., -1] == 0).all()
    else:
        assert weights is None


def test_causal_per_head_mask_and_unbatched_attention():
    torch.manual_seed(42)
    low = LowRankMultiheadAttentionPin(16, 4, 3, dtype=torch.float64)
    x = torch.randn(5, 16, dtype=torch.float64)
    mask = torch.zeros(4, 5, 5, dtype=torch.float64)
    mask[0, :, 0] = -.7
    causal = torch.ones(5, 5, dtype=torch.bool).triu(1)
    expected_mask = mask.masked_fill(causal, float('-inf'))
    actual, w = low(x, x, x, attn_mask=mask, is_causal=True,
                    average_attn_weights=False)
    expected, ew = dense_attention(low)(x, x, x, attn_mask=expected_mask,
                                        average_attn_weights=False)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(w, ew, rtol=1e-10, atol=1e-10)


def test_bias_free_attention_keeps_bias_free_contract():
    low = LowRankMultiheadAttentionPin(16, 4, 3, bias=False)
    assert all('bias' not in name for name, _ in low.named_parameters())
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(low(x, x, x, need_weights=False)[0],
                               dense_attention(low)(x, x, x, need_weights=False)[0])


def test_sdpa_matches_explicit_attention_with_trainable_bias_and_gradients():
    torch.manual_seed(31)
    low = LowRankMultiheadAttentionPin(16, 4, 3, dtype=torch.float64)
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(8, 6, 6, dtype=torch.float64, requires_grad=True)
    sdpa, _ = low(x, x, x, attn_mask=bias, need_weights=False)
    explicit, _ = low(x, x, x, attn_mask=bias, need_weights=True)
    torch.testing.assert_close(sdpa, explicit, rtol=1e-10, atol=1e-10)
    sdpa.square().mean().backward()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()
    for p in low.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize('attention_rank,ffn_rank', [(3, None), (None, 4), (3, 4)])
def test_decoder_replacement_preserves_layout_and_checkpoint(attention_rank, ffn_rank):
    torch.manual_seed(11)
    dense = nn.TransformerDecoder(nn.TransformerDecoderLayer(
        16, 4, dim_feedforward=48, dropout=0., batch_first=True), 2)
    dense_parameters = sum(p.numel() for p in dense.parameters())
    old = copy.deepcopy(dense.state_dict())
    model = factorize_decoder_pin(dense, attention_rank=attention_rank, ffn_rank=ffn_rank)
    for i, layer in enumerate(model.layers):
        for name in ('norm1', 'norm2', 'norm3'):
            torch.testing.assert_close(getattr(layer, name).weight, old[f'layers.{i}.{name}.weight'])
        assert isinstance(layer.self_attn, LowRankMultiheadAttentionPin) == (attention_rank is not None)
        assert isinstance(layer.multihead_attn, LowRankMultiheadAttentionPin) == (attention_rank is not None)
        assert isinstance(layer.linear1, LowRankLinearPin) == (ffn_rank is not None)
        assert isinstance(layer.linear2, LowRankLinearPin) == (ffn_rank is not None)
    saved = 0
    if attention_rank is not None:
        saved += 2 * 2 * 4 * (16 * 16 - attention_rank * 32)
    if ffn_rank is not None:
        saved += 2 * 2 * (16 * 48 - ffn_rank * (16 + 48))
    assert dense_parameters - sum(p.numel() for p in model.parameters()) == saved
    q, memory = torch.randn(2, 3, 16), torch.randn(2, 7, 16)
    padding = torch.tensor([[False] * 7, [False] * 5 + [True] * 2])
    model.train()
    y = model(q, memory, memory_key_padding_mask=padding)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = factorize_decoder_pin(nn.TransformerDecoder(nn.TransformerDecoderLayer(
        16, 4, dim_feedforward=48, dropout=0., batch_first=True), 2),
        attention_rank=attention_rank, ffn_rank=ffn_rank)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval(); restored.eval()
    torch.testing.assert_close(model(q, memory, memory_key_padding_mask=padding),
                               restored(q, memory, memory_key_padding_mask=padding))


@pytest.mark.parametrize('rank', [True, 0, -1, 1.5, 8, 16])
def test_invalid_attention_rank_rejected(rank):
    with pytest.raises(ValueError):
        LowRankMultiheadAttentionPin(16, 4, rank)


def test_invalid_rank_does_not_partially_mutate_decoder():
    decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
        16, 4, dim_feedforward=48, batch_first=True), 1)
    with pytest.raises(ValueError):
        factorize_decoder_pin(decoder, attention_rank=3, ffn_rank=12)
    assert isinstance(decoder.layers[0].self_attn, nn.MultiheadAttention)
    assert isinstance(decoder.layers[0].linear1, nn.Linear)


def test_eval_disables_dropout_and_fully_masked_rows_are_finite():
    attention = LowRankMultiheadAttentionPin(16, 4, 3, dropout=.8).eval()
    x = torch.randn(2, 5, 16)
    mask = torch.zeros(5, 5, dtype=torch.bool)
    mask[0] = True
    a, _ = attention(x, x, x, attn_mask=mask, need_weights=False)
    b, _ = attention(x, x, x, attn_mask=mask, need_weights=False)
    c, _ = attention(x, x, x, attn_mask=mask, need_weights=True)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, c)
    assert torch.isfinite(a).all()
    c.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in attention.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_cuda_sdpa_forward_backward(dtype):
    low = LowRankMultiheadAttentionPin(128, 8, 32, device='cuda', dtype=dtype)
    q = torch.randn(2, 11, 128, device='cuda', dtype=dtype, requires_grad=True)
    memory = torch.randn(2, 19, 128, device='cuda', dtype=dtype, requires_grad=True)
    padding = torch.zeros(2, 19, dtype=torch.bool, device='cuda')
    padding[:, -3:] = True
    actual = low(q, memory, memory, key_padding_mask=padding, need_weights=False)[0]
    expected = dense_attention(low)(q, memory, memory,
                                   key_padding_mask=padding, need_weights=False)[0]
    tolerance = 5e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    actual.float().square().mean().backward()
    for p in low.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
