"""LASD math, native-mask fidelity, unchanged encoder messages and replay."""
import copy

import pytest
import torch

from device_jepa.model import EncoderBlock, ModelConfig
from device_jepa.training import state_digest
from final_exam.lasd_registry_pin import LASD_MODELS_PIN, lasd_settings_pin
from final_exam.models.attention_score_pin import (
    AttentionScoreExtractorPin, score_from_attention_pin, layer_score_features_pin,
)
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.width_decoder_pin import WidthDecoderModelPin
from final_exam.tests.test_width_decoder_models_pin import inputs
from final_exam.width_registry_pin import validate_width_contract_pin


def test_direction_diagonal_batch_isolation_and_conservation():
    weights = torch.tensor([[[[.5, .3, .2], [.1, .8, .1], [.2, .4, .4]]],
                            [[[1., 0., float('nan')], [0., 1., float('nan')],
                              [float('nan'), float('nan'), float('nan')]]]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    scores = score_from_attention_pin(weights, valid)
    torch.testing.assert_close(scores[0, :, 0, 0], torch.tensor([.5, .2, .6]))
    torch.testing.assert_close(scores[0, :, 0, 1], torch.tensor([.3, .7, .3]))
    assert torch.count_nonzero(scores[1]) == 0
    torch.testing.assert_close(scores[..., 0].sum(1), scores[..., 1].sum(1))


@pytest.mark.parametrize('mask_kind', ['none', 'bool', 'float'])
@pytest.mark.parametrize('chunks', [1, 2, 32])
@pytest.mark.parametrize('recompute', [False, True])
def test_matches_native_mha_scores_and_gradients(mask_kind, chunks, recompute):
    torch.manual_seed(31)
    torch.set_num_threads(1)
    mha = torch.nn.MultiheadAttention(8, 2, dropout=0., batch_first=True).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True]*5, [True]*3+[False]*2])
    allowed = torch.rand(2, 2, 5, 5) > .3
    allowed[..., 0] = True
    allowed &= valid[:, None, None, :]
    mask = (~allowed).reshape(4, 5, 5)
    if mask_kind == 'float':
        mask = torch.randn(4, 5, 5, dtype=torch.float64).masked_fill(mask, float('-inf')).requires_grad_()
    elif mask_kind == 'none':
        mask = None
    _, weights = mha(x, x, x, attn_mask=mask,
        key_padding_mask=(torch.zeros_like(valid, dtype=x.dtype).masked_fill(~valid, float('-inf'))
                          if mask_kind == 'float' else ~valid),
        need_weights=True, average_attn_weights=False)
    expected = score_from_attention_pin(weights, valid)
    actual = AttentionScoreExtractorPin(chunks, recompute)(mha, x, mask, valid)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    coefficients = torch.randn_like(expected)
    params = (x, mha.in_proj_weight, mha.in_proj_bias) + ((mask,) if mask_kind == 'float' else ())
    a = torch.autograd.grad((actual * coefficients).sum(), params, retain_graph=True)
    b = torch.autograd.grad((expected * coefficients).sum(), params)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-11)
    assert a[1][:16].abs().sum() > 0  # Q/K
    assert torch.count_nonzero(a[1][16:]) == 0  # scores do not use V


def test_all_masked_rows_are_finite_and_receive_degeneracy_is_preserved():
    torch.manual_seed(4)
    mha = torch.nn.MultiheadAttention(8, 2, batch_first=True)
    x = torch.randn(1, 4, 8, requires_grad=True)
    valid = torch.ones(1, 4, dtype=torch.bool)
    extractor = AttentionScoreExtractorPin(2)
    mask = torch.eye(4, dtype=torch.bool)  # disallow self; all other nodes allowed
    scores = extractor(mha, x, mask, valid)
    torch.testing.assert_close(scores[..., 0], torch.ones(1, 4, 2))
    assert scores[..., 1].std() > 0
    mask[:] = True
    zero = extractor(mha, x, mask, valid)
    assert torch.count_nonzero(zero) == 0
    zero.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_head_features_differences_and_node_identity_alignment():
    first = torch.arange(24.).reshape(2, 3, 2, 2).requires_grad_()
    second = (first.detach()+4).requires_grad_()
    features, delta = layer_score_features_pin([first, second], dynamics=True)
    assert features[0].shape == (2, 3, 8)
    assert torch.count_nonzero(delta[0]) == 0
    torch.testing.assert_close(delta[1], torch.full_like(first, 4))
    torch.testing.assert_close(features[1][..., :4], second.flatten(-2))
    torch.testing.assert_close(features[1][..., 4:], delta[1].flatten(-2))
    delta[1].sum().backward()
    torch.testing.assert_close(first.grad, -torch.ones_like(first))
    torch.testing.assert_close(second.grad, torch.ones_like(second))
    only, _ = layer_score_features_pin([first, second], dynamics=False)
    assert only[0].shape[-1] == 4
    ids = torch.arange(3).expand(2, -1)
    layer_score_features_pin([first, second], dynamics=True, node_ids=[ids, ids])
    with pytest.raises(ValueError, match='identities'):
        layer_score_features_pin([first, second], dynamics=True, node_ids=[ids, ids.flip(-1)])
    # A joint permutation preserves statistics under the same permutation.
    p = torch.tensor([2, 0, 1])
    weights = torch.randn(2, 2, 3, 3).softmax(-1)
    valid = torch.ones(2, 3, dtype=torch.bool)
    torch.testing.assert_close(score_from_attention_pin(weights[:, :, p][:, :, :, p], valid),
                               score_from_attention_pin(weights, valid)[:, p])


def test_observer_preserves_message_values_dropout_rng_and_gradients():
    cfg = ModelConfig(dim=8, heads=2, encoder_layers=4, cross_layers=2, ffn_dim=16, dropout=.25)
    block = EncoderBlock(cfg, cross=True).train()
    x = torch.randn(2, 4, 8, requires_grad=True)
    params = torch.randn(2, 5, 8, requires_grad=True)
    valid = torch.tensor([[True]*4, [True]*3+[False]])
    mask = torch.zeros(4, 4, 4).masked_fill(~valid[:, None, :].repeat_interleave(2, 0), float('-inf'))
    extractor = AttentionScoreExtractorPin(2)
    scores = []
    def observe(attn, value, bias, live):
        scores.append(extractor(attn, value, bias, live))
    torch.manual_seed(42)
    baseline = block(x, mask, params, None, valid)
    rng = torch.get_rng_state()
    grad = torch.autograd.grad(baseline.square().sum(), x, retain_graph=True)[0]
    torch.manual_seed(42)
    tapped = block(x, mask, params, None, valid, attention_observer=observe)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(baseline, tapped, atol=0, rtol=0)
    torch.testing.assert_close(grad, torch.autograd.grad(tapped.square().sum(), x, retain_graph=True)[0], atol=0, rtol=0)
    # The score-only loss has its own live path back through Q/K.
    score_grad = torch.autograd.grad(scores[0].square().sum(), block.self_attn.in_proj_weight)[0]
    assert torch.isfinite(score_grad).all() and score_grad[:16].abs().sum() > 0


@pytest.mark.parametrize('mode', LASD_MODELS_PIN)
def test_model_trains_replays_and_only_changes_decoder_memory(inputs, mode):
    cfg, prep, support, batch = inputs
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    initial_predictor = state_digest(predictor)
    torch.manual_seed(17)
    base = WidthDecoderModelPin('width_decoder_pin', cfg, prep, predictor).eval()
    torch.manual_seed(17)
    model = WidthDecoderModelPin(mode, cfg, prep, predictor,
                                 lasd_config=dict(hidden_dim=12, score_chunk_size=3)).eval()
    assert state_digest(base.encoder) == state_digest(model.encoder)
    for key, value in base.decoder.state_dict().items():
        assert torch.equal(value, model.decoder.state_dict()[key])
    seen = []
    handles = [block.register_forward_hook(lambda module, args, output: seen.append(output))
               for block in model.encoder.fusion_blocks]
    output, diagnostic = model(batch, return_scores=True)
    assert len(seen) == cfg['cross_layers']
    for handle in handles:
        handle.remove()
    before, pad = base.memory(batch)
    after, pad2 = model.memory(batch)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(pad, pad2)
    assert torch.count_nonzero(diagnostic['differences'][0]) == 0
    assert diagnostic['scores'].requires_grad
    model.train()
    live_scores = []
    def retain_scores(module, arguments, value):
        value.retain_grad()
        live_scores.append(value)
    handle = model.score_extractor.register_forward_hook(retain_scores)
    model(batch).square().sum().backward()
    handle.remove()
    assert len(live_scores) == cfg['cross_layers']
    assert all(s.grad is not None and torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0
               for s in live_scores)
    for adapter in model.decoder.score_adapters:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert all(p.grad is None for p in predictor.parameters())
    assert state_digest(predictor) == initial_predictor
    assert len({branch.query.data_ptr() for branch in model.decoder.branches}) == cfg['cross_layers']
    model.eval()
    poisoned = copy.deepcopy(batch)
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        if key in poisoned:
            poisoned[key].fill_(float('nan'))
    torch.testing.assert_close(model(batch), model(poisoned), atol=0, rtol=0)
    recipe = describe_model_pin(model)
    restored = rebuild_model_pin(recipe, cfg, prep).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(batch), restored(batch), atol=0, rtol=0)
    validate_width_contract_pin(mode, model.architecture_contract)
    assert sum(p.numel() for p in model.decoder.score_adapters.parameters()) == model.lasd_spec['adapter_parameters']
    bad = copy.deepcopy(recipe)
    del bad['lasd_config']
    with pytest.raises(ValueError, match='saved score configuration'):
        rebuild_model_pin(bad, cfg, prep)


def test_disabled_switch_is_exact_original_width_model(inputs):
    cfg, prep, support, batch = inputs
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    torch.manual_seed(43)
    base = WidthDecoderModelPin('width_decoder_pin', cfg, prep, predictor).eval()
    torch.manual_seed(43)
    disabled = WidthDecoderModelPin('width_decoder_lasd_pin', cfg, prep, predictor,
                                    lasd_config={'enabled': False}).eval()
    assert state_digest(base) == state_digest(disabled)
    torch.testing.assert_close(base(batch), disabled(batch), atol=0, rtol=0)
    validate_width_contract_pin(disabled.mode, disabled.architecture_contract)


@pytest.mark.parametrize('cfg', [{'enabled': 1}, {'dropout': float('nan')}, {'hidden_dim': 0},
    {'score_chunk_size': 0}, {'diagnostic_samples': -1}, {'head_average': True}])
def test_settings_reject_invalid(cfg):
    with pytest.raises(ValueError):
        lasd_settings_pin(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_scores_cuda_backward():
    module = torch.nn.MultiheadAttention(16, 4, batch_first=True).cuda()
    x = torch.randn(2, 7, 16, device='cuda', requires_grad=True)
    value = AttentionScoreExtractorPin(3)(module, x, None, torch.ones(2, 7, dtype=torch.bool, device='cuda'))
    value.square().sum().backward()
    assert value.is_cuda and x.grad.is_cuda and torch.isfinite(x.grad).all()
