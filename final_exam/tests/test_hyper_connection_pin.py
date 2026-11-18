"""Residual stream equations, native pin PE equivalence and real LASD taps."""
import copy

import pytest
import torch

from final_exam.lasd_moe_registry_pin import HYPER_LASD_MOE_MODEL_PIN, validate_lasd_moe_contract_pin
from final_exam.models.hyper_connection_pin import HyperConnectionPin
from final_exam.models.lasd_moe_pin import LASDMoEModelPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.tests.test_width_decoder_models_pin import inputs


def test_stream_read_carry_and_write_with_token_dependent_coefficients():
    torch.manual_seed(11)
    mixer = HyperConnectionPin(dim=5, streams=3, layer_index=4).double()
    x = torch.randn(2, 4, 3, 5, dtype=torch.double, requires_grad=True)
    update = torch.randn(2, 4, 5, dtype=torch.double, requires_grad=True)
    read, carry, beta = mixer(x)
    torch.testing.assert_close(read, x[:, :, 1])
    torch.testing.assert_close(carry, x)
    torch.testing.assert_close(beta, torch.ones_like(beta))
    torch.testing.assert_close(mixer.write(carry, beta, update), x + update.unsqueeze(-2))

    # Non-identity and asymmetric coefficients catch mixing the wrong stream axis.
    with torch.no_grad():
        for parameter in mixer.parameters():
            parameter.normal_()
    read, carry, beta = mixer(x)
    normalized = mixer.norm(x)
    alpha = mixer.static_alpha + mixer.alpha_scale * (normalized @ mixer.alpha_projection).tanh()
    expected = torch.einsum('bnsq,bnsd->bnqd', alpha, x)
    expected_beta = mixer.static_beta + mixer.beta_scale * (normalized @ mixer.beta_projection).tanh()
    torch.testing.assert_close(read, expected[..., 0, :])
    torch.testing.assert_close(carry, expected[..., 1:, :])
    torch.testing.assert_close(beta, expected_beta)
    result = mixer.write(carry, beta, update)
    torch.testing.assert_close(result, expected[..., 1:, :] + torch.einsum('bns,bnd->bnsd', expected_beta, update))
    (read.square().mean() + result.square().mean()).backward()
    for parameter in (x, update, *mixer.parameters()):
        assert parameter.grad is not None and parameter.grad.isfinite().all() and parameter.grad.abs().sum() > 0


def test_one_stream_initialization_matches_native_detached_model_even_with_dropout(inputs):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dropout': .2}
    torch.manual_seed(17)
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    options = dict(hyper_connection_streams=1, dropout=.2)
    torch.manual_seed(29)
    base = LASDMoEModelPin('lasd_moe_headwise_detach_pin', cfg, prep, predictor, options)
    torch.manual_seed(29)
    hyper = LASDMoEModelPin(HYPER_LASD_MOE_MODEL_PIN, cfg, prep, predictor, options)
    for training in [False, True]:
        base.train(training)
        hyper.train(training)
        torch.manual_seed(43)
        expected, baseline_info = base(batch, return_scores=True)
        rng = torch.get_rng_state()
        torch.manual_seed(43)
        actual, info = hyper(batch, return_scores=True)
        assert torch.equal(rng, torch.get_rng_state())
        torch.testing.assert_close(actual, expected)
        for key in ('scores', 'differences', 'layer_weights'):
            torch.testing.assert_close(info[key], baseline_info[key])


def test_four_streams_continue_between_pes_score_actual_queries_and_use_headwise_readouts(inputs):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dim': 128, 'heads': 8, 'encoder_layers': 6, 'cross_layers': 3}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    model = LASDMoEModelPin(HYPER_LASD_MOE_MODEL_PIN, cfg, prep, predictor).eval()
    spec = model.architecture_contract
    assert spec['hyper_connection']['streams'] == 4
    validate_lasd_moe_contract_pin(HYPER_LASD_MOE_MODEL_PIN, spec)
    changed = copy.deepcopy(spec)
    changed['hyper_connection']['streams'] = 2
    with pytest.raises(ValueError, match='contract'):
        validate_lasd_moe_contract_pin(HYPER_LASD_MOE_MODEL_PIN, changed, cfg, model.lasd_moe_config)

    mixers = [module for module in model.modules() if isinstance(module, HyperConnectionPin)]
    assert len(mixers) == 9
    assert sum(p.numel() for mixer in mixers for p in mixer.parameters()) == 9450
    assert set(mixers) == {m for pe in model.encoder.fusion_blocks for m in
                           (pe.self_connection, pe.cross_connection, pe.ffn_connection)}
    # Force stream asymmetry; collapsing/re-expanding at a tap would now change PE2.
    with torch.no_grad():
        model.encoder.fusion_blocks[0].ffn_connection.static_beta[0] = -1.

    inputs_seen, states, taps, observed, messages = [], [], [], [], []
    for pe in model.encoder.fusion_blocks:
        pe.register_forward_pre_hook(lambda module, args: inputs_seen.append(args[0]))
        pe.register_forward_hook(lambda module, args, state: states.append(state))
        pe.self_attn.register_forward_pre_hook(
            lambda module, args, kwargs: messages.append((module, args[0], kwargs['attn_mask'])), with_kwargs=True)
    model.score_extractor.register_forward_pre_hook(lambda module, args: observed.append(args))
    model.encoder.output_norm.register_forward_pre_hook(lambda module, args: taps.append(args[0]))
    output, diag = model(batch, return_scores=True)
    assert len(inputs_seen) == len(states) == len(taps) == len(observed) == len(messages) == 3
    assert inputs_seen[0].ndim == 3
    assert inputs_seen[1] is states[0] and inputs_seen[2] is states[1]
    assert not torch.allclose(states[0][..., 0, :], states[0][..., 1, :])
    assert diag['layer_weights'].shape == (*diag['valid'].shape, 8, 3)
    for state, tap, observation, message in zip(states, taps, observed, messages):
        assert state.shape == (*diag['valid'].shape, 4, 128)
        assert torch.count_nonzero(state[~diag['valid']]) == 0
        torch.testing.assert_close(tap, state.sum(-2))
        assert all(left is right for left, right in zip(observation[:3], message))

    diag['score_features'].retain_grad()
    output.square().mean().backward()
    assert diag['score_features'].grad is None
    for mixer in mixers:
        grad = mixer.static_beta.grad
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0
