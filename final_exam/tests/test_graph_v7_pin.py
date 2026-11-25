"""Multi-order attention semantics, exact v3 control and v3 gradient boundaries."""
import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from device_jepa.model import ModelConfig
from final_exam.lasd_moe_registry_pin import (
    GRAPH_LAYER_POWER_MODEL_PIN as V7, lasd_moe_settings_pin, validate_lasd_moe_contract_pin,
)
from final_exam.models.attention_power_score_pin import (
    AttentionPowerScoreExtractorPin, attention_power_scores_pin,
)
from final_exam.models.attention_score_pin import AttentionScoreExtractorPin, score_from_attention_pin
from final_exam.models.graph_layer_score_pin import MultiOrderGraphLayerScoreRouterPin
from final_exam.models.lasd_moe_pin import LASDMoEModelPin, score_dynamics_pin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.tests.test_width_decoder_models_pin import inputs

V3 = 'lasd_graph_layer_headwise_detach_pin_v3'


@pytest.mark.parametrize('chunk', [1, 2, 8])
def test_powers_preserve_walk_self_loops_and_do_not_remask_or_normalize(chunk):
    a = torch.tensor([[[[.5, .5, 0.], [0., .25, .75], [.2, 0., .8]]]], dtype=torch.float64)
    valid = torch.ones(1, 3, dtype=torch.bool)
    scores = attention_power_scores_pin(a, valid, 3, chunk)
    torch.testing.assert_close(scores[0, :, 0, 0], torch.tensor(
        [[.5, .2], [.75, .5], [.2, .75]], dtype=a.dtype))
    torch.testing.assert_close(scores[0, :, 1, 0], torch.tensor(
        [[.75, .41], [.9375, .475], [.36, 1.1625]], dtype=a.dtype))
    for k in range(3):
        power = torch.linalg.matrix_power(a, k+1)
        torch.testing.assert_close(scores[:, :, k], score_from_attention_pin(power, valid))
        torch.testing.assert_close(scores[:, :, k, :, 0], (1-power.diagonal(dim1=-2, dim2=-1)).transpose(1, 2))
    assert (a @ a)[0, 0, 0, 2] > 0 and a[0, 0, 0, 2] == 0
    no_self = a.clone()
    no_self.diagonal(dim1=-2, dim2=-1).zero_()
    assert not torch.allclose(scores[:, :, 1], score_from_attention_pin(no_self @ no_self, valid))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_padding_head_batch_isolation_permutation_and_degenerate_rows(dtype):
    torch.manual_seed(12)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False], [False]*4])
    a = torch.randn(3, 2, 4, 4, dtype=dtype).masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
    a = a.masked_fill(~(valid[:, None, :, None] & valid[:, None, None]), float('nan'))
    result = attention_power_scores_pin(a, valid, 4, 2)
    assert torch.isfinite(result).all() and torch.count_nonzero(result[~valid]) == 0
    clean = a.masked_fill(~(valid[:, None, :, None] & valid[:, None, None]), 0.)
    for k in range(4):
        torch.testing.assert_close(result[:, :, k], score_from_attention_pin(torch.linalg.matrix_power(clean, k+1), valid))
    torch.testing.assert_close(result[..., 0].sum(1), result[..., 1].sum(1))
    for b in range(3):
        for h in range(2):
            single = attention_power_scores_pin(a[b:b+1, h:h+1], valid[b:b+1], 4, 2)
            torch.testing.assert_close(single[0, :, :, 0], result[b, :, :, h])
    order = torch.tensor([2, 0, 3, 1])
    permuted = attention_power_scores_pin(a[:, :, order][:, :, :, order], valid[:, order], 4, 2)
    torch.testing.assert_close(permuted, result[:, order])
    identity = torch.eye(4, dtype=dtype)[None, None]
    assert torch.count_nonzero(attention_power_scores_pin(identity, valid[:1], 4)) == 0


@pytest.mark.parametrize('mask_type', ['boolean', 'additive', 'none'])
def test_observer_matches_native_pre_dropout_attention_and_distance_one(mask_type):
    torch.manual_seed(12)
    attention = nn.MultiheadAttention(8, 2, batch_first=True, dropout=.7).double().eval()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True]*5, [True, True, True, False, False]])
    blocked = torch.eye(5, dtype=torch.bool).roll(1, -1)
    mask = blocked if mask_type == 'boolean' else None
    if mask_type == 'additive':
        mask = torch.randn(4, 5, 5, dtype=x.dtype).masked_fill(blocked, -torch.inf)
    with torch.no_grad():
        # Match key-padding mask type to the native attention mask.
        key_mask = (torch.zeros_like(valid, dtype=x.dtype).masked_fill(~valid, -torch.inf)
                    if mask_type == 'additive' else ~valid)
        _, a = attention(x, x, x, attn_mask=mask, key_padding_mask=key_mask,
                         need_weights=True, average_attn_weights=False)
        a = a.masked_fill(~valid[:, None, :, None], 0.)
    expected = attention_power_scores_pin(a, valid, 3, 2)
    # Observations must not apply MHA's .7 attention dropout even in train mode.
    attention.train()
    actual = AttentionPowerScoreExtractorPin(3, 2)(attention, x, mask, valid)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    one = AttentionPowerScoreExtractorPin(1, 2)(attention, x, mask, valid)
    legacy = AttentionScoreExtractorPin(2)(attention, x, mask, valid)
    torch.testing.assert_close(one[:, :, 0], legacy, atol=0, rtol=0)


def test_independent_order_projections_sum_before_shared_encoders_and_mask_biases():
    cfg = ModelConfig(dim=8, heads=2, encoder_layers=5, cross_layers=3, ffn_dim=16, dropout=0.)
    module = MultiOrderGraphLayerScoreRouterPin(60, cfg, distance=3, hidden_dim=12).eval()
    valid = torch.tensor([[True, True, False], [True, True, True]])
    features = torch.randn(2, 3, 3, 60)
    features[~valid] = float('nan')
    safe = features.masked_fill(~valid[:, :, None, None], 0.)
    expected = sum(p(safe[:, :, k]) for k, p in enumerate(module.order_projections))
    expected = expected.masked_fill(~valid.unsqueeze(-1), 0.)
    mask = torch.zeros(2, 2, 3, 3).masked_fill(~valid[:, None, None], -torch.inf).flatten(0, 1)
    seen = []
    handle = module.graph_blocks[0].register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    try:
        logits, _ = module(features, valid, mask)
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
    assert len(module.graph_blocks) == 3 and len(module.order_projections) == 3
    assert len({id(p[0].weight) for p in module.order_projections}) == 3
    logits.square().sum().backward()
    assert all(p[0].weight.grad.abs().sum() > 0 for p in module.order_projections)


def test_distance_one_exact_v3_initialization_forward_backward_and_dropout(inputs):
    cfg, prep, support, batch = inputs
    models = []
    for mode in (V3, V7):
        torch.manual_seed(9)
        predictor = DeviceOPPredictorPin(cfg, prep, support)
        models.append(LASDMoEModelPin(mode, {**cfg, 'dropout': .2}, prep, predictor,
            {'aggregation_distance': 1, 'dropout': .15, 'score_chunk_size': 3}))
    a, b = models
    assert list(a.state_dict()) == list(b.state_dict())
    for name, p in a.state_dict().items():
        torch.testing.assert_close(p, b.state_dict()[name], atol=0, rtol=0)
    for training in (False, True):
        results, rngs = [], []
        for model in models:
            model.train(training)
            model.zero_grad()
            torch.manual_seed(17)
            result = model(batch)
            result.square().sum().backward()
            results.append(result)
            rngs.append(torch.get_rng_state())
        torch.testing.assert_close(*results, atol=0, rtol=0)
        assert torch.equal(*rngs)
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.grad is None:
                assert q.grad is None, name
            else:
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_v7_score_and_all_fusion_taps_detached_but_residual_and_main_chain_live(inputs):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3}
    torch.manual_seed(43)
    model = LASDMoEModelPin(V7, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
        {'aggregation_distance': 3, 'hidden_dim': 12, 'score_chunk_size': 3}).train()
    memories, raw_states, routing_inputs, decoder_inputs = [], [], [], []
    original = model.encoder.forward_fusion_memories
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        memories.extend(result[0])
        return result
    handles = [block.register_forward_hook(lambda _, args, value: raw_states.append(value))
               for block in model.encoder.fusion_blocks]
    handles += [model.layer_fusion.register_forward_pre_hook(lambda _, args: routing_inputs.append(args[0])),
                model.decoder.register_forward_pre_hook(lambda _, args: decoder_inputs.append(args[0]))]
    try:
        with patch.object(model.encoder, 'forward_fusion_memories', side_effect=capture):
            output, info = model(batch, return_scores=True)
    finally:
        for handle in handles: handle.remove()
    assert not routing_inputs[0].requires_grad and not info['power_scores'].requires_grad
    assert model.final_graph_encoder is None and model.decoder_input_norm is None
    weights, actual = info['layer_weights'], decoder_inputs[0]
    values = torch.stack(memories, 2).unflatten(-1, (cfg['heads'], -1)).transpose(2, 3)
    fused = (values * weights.unsqueeze(-1)).sum(3).flatten(-2)
    torch.testing.assert_close(actual, memories[-1] + fused, atol=0, rtol=0)
    cotangent = torch.randn_like(actual)
    grad = torch.autograd.grad((actual*cotangent).sum(), memories, allow_unused=True, retain_graph=True)
    assert grad[0] is grad[1] is None
    torch.testing.assert_close(grad[2], cotangent, atol=0, rtol=0)
    pe_qk = [block.self_attn.in_proj_weight for block in model.encoder.fusion_blocks]
    score_only_grad = torch.autograd.grad(weights[..., 0].square().sum(), pe_qk,
                                         allow_unused=True, retain_graph=True)
    assert all(g is None for g in score_only_grad)
    for state in raw_states: state.retain_grad()
    output.square().sum().backward()
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in raw_states)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in pe_qk)
    assert all(p[0].weight.grad.abs().sum() > 0 for p in model.layer_fusion.order_projections)
    assert all(p.grad is None for p in model.predictor.parameters())
    # Every order computes deltas ACROSS PE, not across order, with boundary zeros.
    for k in range(3):
        node, delta = score_dynamics_pin(info['power_scores'][:, :, k])
        context, stats, dglobal = model.global_scores(info['power_scores'][:, :, k], info['valid'])
        expected = torch.cat((node, context[:, None].expand(-1, node.size(1), -1)), -1)
        expected = expected.masked_fill(~info['valid'].unsqueeze(-1), 0.)
        torch.testing.assert_close(expected, info['score_features'][:, :, k])
        torch.testing.assert_close(delta, info['power_differences'][:, :, k])
        torch.testing.assert_close(stats, info['power_global_statistics'][:, k])
        torch.testing.assert_close(dglobal, info['power_global_differences'][:, k])
        assert torch.count_nonzero(delta[:, :, 0]) == 0


@pytest.mark.parametrize('layers,distance', [(1, 1), (1, 3), (3, 2), (4, 3)])
def test_variable_depth_and_strict_saved_contract(inputs, layers, distance):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': layers+2, 'cross_layers': layers}
    model = LASDMoEModelPin(V7, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
        {'aggregation_distance': distance}).eval()
    with torch.no_grad():
        out, info = model(batch, return_scores=True)
    assert out.shape == batch['performance'].shape and torch.isfinite(out).all()
    assert info['power_scores'].shape[2:] == (distance, layers, cfg['heads'], 2)
    assert info['layer_weights'].shape[2:] == (cfg['heads'], layers)
    torch.testing.assert_close(info['layer_weights'][info['valid']].sum(-1),
                               torch.ones_like(info['layer_weights'][info['valid']][..., 0]))
    spec = model.architecture_contract
    validate_lasd_moe_contract_pin(V7, spec)
    validate_lasd_moe_contract_pin(V7, spec, cfg, model.lasd_moe_config)
    for key, value in [('aggregation_distance', distance+1), ('score_detach', False),
                       ('hidden_state_detach', False), ('final_encoder_layers', 1)]:
        bad = copy.deepcopy(spec); bad[key] = value
        with pytest.raises(ValueError): validate_lasd_moe_contract_pin(V7, bad)


@pytest.mark.parametrize('distance', [0, -1, True, 1.5, '2', None])
def test_invalid_distance_rejected(distance):
    with pytest.raises(ValueError, match='aggregation_distance'):
        lasd_moe_settings_pin({'aggregation_distance': distance})
