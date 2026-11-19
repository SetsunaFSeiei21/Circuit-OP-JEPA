"""Global score arithmetic, graph isolation, contracts and score-only gradients."""
import copy

import pytest
import torch

from final_exam.lasd_moe_registry_pin import (
    GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN, lasd_moe_settings_pin,
    lasd_moe_spec_pin, validate_lasd_moe_contract_pin,
)
from final_exam.models.attention_score_pin import score_from_attention_pin
from final_exam.models.global_score_pin import GlobalScoreStatisticsPin
from final_exam.models.lasd_moe_pin import LASDMoEModelPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.tests.test_width_decoder_models_pin import inputs


def test_population_statistics_order_differences_padding_and_batch_isolation():
    scores = torch.arange(2*4*3*2*2, dtype=torch.float64).reshape(2, 4, 3, 2, 2) / 9
    valid = torch.tensor([[True]*4, [True, True, False, False]])
    scores[~valid] = float('nan')
    module = GlobalScoreStatisticsPin(epsilon=.01)
    features, stats, delta = module(scores, valid)
    expected = torch.empty(2, 3, 2, 3, dtype=torch.float64)
    for b in range(2):
        for layer in range(3):
            for head in range(2):
                values = scores[b, valid[b], layer, head]
                mean = values.mean(0)
                std = (values.var(0, correction=0) + .01).sqrt()
                expected[b, layer, head] = torch.stack((mean[0], std[0], std[1]))
    torch.testing.assert_close(stats, expected)
    assert features.shape == (2, 36)
    torch.testing.assert_close(features.reshape(2, 3, 2, 6)[..., :3], stats)
    torch.testing.assert_close(features.reshape(2, 3, 2, 6)[..., 3:], delta)
    assert torch.count_nonzero(delta[:, 0]) == 0
    torch.testing.assert_close(delta[:, 1:], stats[:, 1:] - stats[:, :-1])
    for b in range(2):
        separate = module(scores[b:b+1], valid[b:b+1])
        for value, combined in zip(separate, (features, stats, delta)):
            torch.testing.assert_close(value, combined[b:b+1])
    order = torch.tensor([2, 0, 3, 1])
    for original, reordered in zip((features, stats, delta), module(scores[:, order], valid[:, order])):
        torch.testing.assert_close(original, reordered)


def test_constant_single_node_empty_graph_and_padding_gradients():
    valid = torch.tensor([[True]*3, [True, False, False], [False]*3])
    scores = torch.full((3, 3, 2, 2, 2), .6, dtype=torch.float64)
    scores[~valid] = float('nan')
    scores.requires_grad_()
    features, stats, delta = GlobalScoreStatisticsPin(1e-8)(scores, valid)
    torch.testing.assert_close(stats[:2, ..., 0], torch.full_like(stats[:2, ..., 0], .6))
    torch.testing.assert_close(stats[:2, ..., 1:], torch.full_like(stats[:2, ..., 1:], 1e-4))
    assert torch.count_nonzero(stats[2]) == torch.count_nonzero(delta) == 0
    features.square().sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad[~valid]) == 0


def test_no_mean_t_feature_but_both_standard_deviations_survive():
    # Row-normalized attention gives equal graph means R and T, not equal spreads.
    weights = torch.tensor([[[[.5, .4, .1], [.2, .6, .2], [.7, .2, .1]]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    scores = score_from_attention_pin(weights, valid)
    torch.testing.assert_close(scores[..., 0].mean(1), scores[..., 1].mean(1))
    features, stats, _ = GlobalScoreStatisticsPin()(scores.unsqueeze(2), valid)
    assert features.shape == (1, 6)  # Three statistics + three boundary differences.
    assert not torch.isclose(stats[..., 1], stats[..., 2]).all()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_reductions_accumulate_in_float32(dtype):
    scores = torch.ones(1, 512, 2, 2, 2, dtype=dtype)
    scores[:, 0, ..., 1] = 400
    valid = torch.ones(1, 512, dtype=torch.bool)
    actual = GlobalScoreStatisticsPin()(scores, valid)
    reference = GlobalScoreStatisticsPin()(scores.float(), valid)
    assert actual[0].dtype == torch.float32
    for a, b in zip(actual, reference):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_global_gradient_is_within_graph_and_matches_finite_differences():
    torch.manual_seed(13)
    scores = torch.rand(2, 3, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True]*3, [True, True, False]])
    module = GlobalScoreStatisticsPin()
    assert torch.autograd.gradcheck(lambda x: module(x, valid)[0], (scores,))
    context, _, _ = module(scores, valid)
    context[0].square().sum().backward()
    assert scores.grad[0].abs().sum() > 0
    assert torch.count_nonzero(scores.grad[1]) == 0


@pytest.mark.parametrize('epsilon', [0, -1, float('nan'), float('inf'), True])
def test_invalid_epsilon_rejected_in_config_and_module(epsilon):
    with pytest.raises(ValueError):
        GlobalScoreStatisticsPin(epsilon)
    with pytest.raises(ValueError):
        lasd_moe_settings_pin({'global_std_epsilon': epsilon})


def test_global_contract_has_240_inputs_24_outputs_and_custom_epsilon_replays():
    cfg = dict(dim=128, heads=8, encoder_layers=6, cross_layers=3, decoder_layers=2)
    for mode in GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN:
        spec = lasd_moe_spec_pin(mode, {'global_std_epsilon': 3e-7}, cfg)
        assert spec['input_dim'] == 240 and spec['expert_output_dim'] == 24
        assert spec['global_context']['input_dim'] == 144
        assert spec['global_context']['statistics'] == ['mean_R', 'std_R', 'std_T']
        assert not spec['hidden_state_detach']
        validate_lasd_moe_contract_pin(mode, spec)  # Summary path without separate config.
        validate_lasd_moe_contract_pin(mode, spec, cfg, {'global_std_epsilon': 3e-7})
        bad = copy.deepcopy(spec)
        bad['global_context']['statistics'].append('mean_T')
        with pytest.raises(ValueError):
            validate_lasd_moe_contract_pin(mode, bad)
        with pytest.raises(ValueError):
            validate_lasd_moe_contract_pin(mode, spec, cfg, {'global_std_epsilon': 1e-8})
    assert 'global_std_epsilon' not in lasd_moe_settings_pin()  # Old checkpoint configs unchanged.
    old = lasd_moe_spec_pin('lasd_moe_headwise_pin', None, cfg)
    assert old['input_dim'] == 96 and 'global_context' not in old


@pytest.mark.parametrize('mode', GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN)
def test_real_model_global_only_score_gradient_and_forward_features(inputs, mode):
    cfg, prep, support, batch = inputs
    torch.manual_seed(31)
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
                           {'hidden_dim': 12, 'score_chunk_size': 3}).eval()
    local_width = 4*cfg['cross_layers']*cfg['heads']
    # Isolate the global score route by zeroing all local score input columns.
    with torch.no_grad():
        model.layer_fusion.router.weight[:, :local_width].zero_()
        for expert in model.layer_fusion.experts:
            expert[0].weight[:, :local_width].zero_()
    output, observed = model(batch, return_scores=True)
    features, valid = observed['score_features'], observed['valid']
    assert features.shape[-1] == 10*cfg['cross_layers']*cfg['heads']
    torch.testing.assert_close(features[..., :local_width], observed['node_score_features'])
    expected = observed['global_score_features'][:, None].expand_as(features[..., local_width:])
    torch.testing.assert_close(features[..., local_width:][valid], expected[valid])
    assert torch.count_nonzero(features[~valid]) == 0
    loss = observed['layer_weights'].square().sum()  # No hidden-state path in this loss.
    gradients = torch.autograd.grad(loss, (observed['scores'],
        model.encoder.fusion_blocks[0].self_attn.in_proj_weight), retain_graph=True, allow_unused=True)
    if model.lasd_moe_spec['score_detach']:
        assert all(g is None for g in gradients)
    else:
        assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    loss.backward(retain_graph=True)
    assert model.layer_fusion.router.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layer_fusion.experts.parameters())
    model.zero_grad(set_to_none=True)
    output.square().sum().backward()  # Hidden-state training remains live in both variants.
    assert model.encoder.fusion_blocks[0].self_attn.in_proj_weight.grad.abs().sum() > 0
