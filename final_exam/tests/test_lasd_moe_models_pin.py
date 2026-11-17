"""PE-layer alignment, Top-2 routing, replay and score-only stop-gradient boundaries."""
import copy

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.lasd_moe_registry_pin import (
    LASD_MOE_MODELS_PIN, lasd_moe_settings_pin, validate_lasd_moe_contract_pin,
)
from final_exam.models.attention_score_pin import AttentionScoreExtractorPin
from final_exam.models.lasd_moe_pin import (
    LASDMoEModelPin, Top2LayerMoEPin, score_dynamics_pin,
    describe_lasd_moe_model_pin, rebuild_lasd_moe_model_pin,
)
from final_exam.models.op_gate_pin import DeviceOPGateModelPin, DeviceOPPredictorPin
from final_exam.op_gate_data_pin import device_input_pin
from final_exam.tests.test_width_decoder_models_pin import inputs


def test_score_feature_order_deltas_and_reorder_rejection():
    scores = torch.arange(2*3*4*2*2.).reshape(2, 3, 4, 2, 2).requires_grad_()
    features, delta = score_dynamics_pin(scores)
    assert features.shape == (2, 3, 32)
    torch.testing.assert_close(delta[:, :, 0], torch.zeros_like(delta[:, :, 0]))
    torch.testing.assert_close(delta[:, :, 1:], scores[:, :, 1:] - scores[:, :, :-1])
    torch.testing.assert_close(features.reshape(2, 3, 4, 2, 4)[..., :2], scores)
    torch.testing.assert_close(features.reshape(2, 3, 4, 2, 4)[..., 2:], delta)
    delta[:, :, 1].sum().backward()
    torch.testing.assert_close(scores.grad[:, :, 0], -torch.ones_like(scores[:, :, 0]))
    ids = torch.arange(3).expand(2, -1)
    score_dynamics_pin(scores, node_ids=[ids]*4)
    with pytest.raises(ValueError, match='identities'):
        score_dynamics_pin(scores, node_ids=[ids, ids.flip(-1), ids, ids])


def test_sparse_moe_matches_top2_reference_with_padding_and_live_router():
    torch.manual_seed(83)
    moe = Top2LayerMoEPin(16, 3, {'hidden_dim': 9})
    features = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.tensor([[True]*5, [True, True, False, False, False]])
    features.data[~valid] = float('nan')
    actual, info = moe(features, valid)
    safe = features.masked_fill(~valid.unsqueeze(-1), 0.)
    logits, selected = moe.router(safe).topk(2, dim=-1)
    weights = logits.softmax(-1)
    expected = sum(expert(safe) * ((selected == i).to(safe.dtype)*weights).sum(-1, keepdim=True)
                   for i, expert in enumerate(moe.experts)).masked_fill(~valid.unsqueeze(-1), 0.)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(info['selected_weights'][valid].sum(-1), torch.ones(int(valid.sum())))
    assert (info['selected_experts'][~valid] == -1).all()
    assert torch.count_nonzero(actual[~valid]) == 0
    actual.square().sum().backward()
    assert torch.isfinite(features.grad).all() and torch.count_nonzero(features.grad[~valid]) == 0
    assert moe.router.weight.grad.abs().sum() > 0
    for i in range(2):
        out, _ = moe(features[i:i+1], valid[i:i+1])
        torch.testing.assert_close(out, actual[i:i+1])


def test_sparse_dispatch_preserves_autocast_gradient_dtype():
    module = Top2LayerMoEPin(16, 3, {'hidden_dim': 8})
    x = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.ones(2, 5, dtype=torch.bool)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output, _ = module(x, valid)
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_pe_encoder_taps_preserve_messages_rng_order_and_masks(inputs):
    cfg, prep, support, batch = inputs
    predictor = DeviceOPPredictorPin({**cfg, 'dropout': .2}, prep, support).train()
    encoder = predictor.encoder
    source = device_input_pin(batch)
    torch.manual_seed(7)
    reference, pad = encoder(source)
    expected_rng = torch.get_rng_state()
    reference_grad = torch.autograd.grad(reference.square().sum(), encoder.parameter_names.weight)[0]
    observed = []
    extractor = AttentionScoreExtractorPin(3)
    def observe(index, attention, hidden, mask, valid):
        observed.append((index, extractor(attention, hidden, mask, valid)))
    torch.manual_seed(7)
    memories, padding, masks = encoder.forward_fusion_memories(source, attention_observer=observe, return_self_masks=True)
    assert torch.equal(expected_rng, torch.get_rng_state())
    torch.testing.assert_close(memories[-1], reference, atol=0, rtol=0)
    actual_grad = torch.autograd.grad(memories[-1].square().sum(), encoder.parameter_names.weight)[0]
    torch.testing.assert_close(actual_grad, reference_grad, atol=0, rtol=0)
    assert len(memories) == len(observed) == cfg['cross_layers']
    assert [x[0] for x in observed] == list(range(cfg['cross_layers']))
    assert torch.equal(padding, pad) and len(masks) == 2
    assert all(torch.count_nonzero(memory[padding]) == 0 for memory in memories)


@pytest.mark.parametrize('mode', LASD_MOE_MODELS_PIN)
def test_model_gradients_replay_no_measured_op_and_one_final_decoder(inputs, mode):
    cfg, prep, support, batch = inputs
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    before = state_digest(predictor)
    model = LASDMoEModelPin(mode, cfg, prep, predictor,
                           {'hidden_dim': 12, 'score_chunk_size': 3}).train()
    retained_memories = []
    real_forward = model.encoder.forward_fusion_memories
    def retain(*args, **kwargs):
        result = real_forward(*args, **kwargs)
        for memory in result[0]:
            memory.retain_grad()
            retained_memories.append(memory)
        return result
    model.encoder.forward_fusion_memories = retain
    output, diag = model(batch, return_scores=True)
    diag['score_features'].retain_grad()
    assert output.shape == batch['performance'].shape
    assert diag['scores'].shape[2:] == (cfg['cross_layers'], cfg['heads'], 2)
    assert torch.count_nonzero(diag['differences'][:, :, 0]) == 0
    torch.testing.assert_close(diag['layer_weights'][diag['valid']].sum(-1),
                               torch.ones(int(diag['valid'].sum())))
    assert torch.count_nonzero(diag['layer_weights'][~diag['valid']]) == 0
    output.square().sum().backward()
    if mode == 'lasd_moe_detach_pin':
        assert diag['score_features'].grad is None
    else:
        assert diag['score_features'].grad.abs().sum() > 0
    assert all(memory.grad is not None and memory.grad.abs().sum() > 0 for memory in retained_memories)
    assert model.layer_fusion.router.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layer_fusion.experts.parameters())
    for block in list(model.encoder.structure_encoder.blocks) + list(model.encoder.fusion_blocks):
        assert block.self_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.final_graph_encoder.self_attn.in_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in predictor.parameters()) and state_digest(predictor) == before
    model.encoder.forward_fusion_memories = real_forward
    model.eval()
    poison = copy.deepcopy(batch)
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        if key in poison:
            poison[key].fill_(float('nan'))
    torch.testing.assert_close(model(batch), model(poison), atol=0, rtol=0)
    recipe = describe_lasd_moe_model_pin(model)
    rebuilt = rebuild_lasd_moe_model_pin(recipe, cfg, prep).eval()
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(batch), rebuilt(batch), atol=0, rtol=0)
    validate_lasd_moe_contract_pin(mode, model.architecture_contract, cfg, model.lasd_moe_config)
    bad = copy.deepcopy(recipe)
    bad['architecture']['score_detach'] = not bad['architecture']['score_detach']
    with pytest.raises(ValueError, match='architecture'):
        rebuild_lasd_moe_model_pin(bad, cfg, prep)


def test_detach_pair_same_parameters_forward_rng_and_native_initialization(inputs):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dropout': .2}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    torch.manual_seed(11)
    baseline = DeviceOPGateModelPin('device_op_cascade_pin', cfg, prep, predictor)
    torch.manual_seed(11)
    model = LASDMoEModelPin('lasd_moe_pin', cfg, prep, predictor, {'dropout': .15})
    torch.manual_seed(11)
    detached = LASDMoEModelPin('lasd_moe_detach_pin', cfg, prep, predictor, {'dropout': .15})
    assert state_digest(model) == state_digest(detached)
    assert state_digest(model.encoder) == state_digest(baseline.encoder)
    assert state_digest(model.decoder) == state_digest(baseline.decoder)
    for training in (False, True):
        model.train(training)
        detached.train(training)
        torch.manual_seed(42)
        value = model(batch)
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        other = detached(batch)
        torch.testing.assert_close(value, other, atol=0, rtol=0)
        assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('config', [{'n_experts': 3}, {'top_k': 1}, {'hidden_dim': 0},
    {'checkpoint_scores': 1}, {'dropout': float('nan')}, {'head_average': True}])
def test_reject_invalid_settings(config):
    with pytest.raises(ValueError):
        lasd_moe_settings_pin(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_top2_cuda_deterministic_backward():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = Top2LayerMoEPin(16, 4, {'hidden_dim': 12}).cuda()
        x = torch.randn(2, 9, 16, device='cuda', requires_grad=True)
        out, _ = model(x, torch.ones(2, 9, dtype=torch.bool, device='cuda'))
        out.square().sum().backward()
        assert torch.isfinite(x.grad).all()
    finally:
        torch.use_deterministic_algorithms(previous)
