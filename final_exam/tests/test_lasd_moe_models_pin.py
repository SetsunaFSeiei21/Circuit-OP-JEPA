"""PE-layer alignment, Top-2 routing, replay and score-only stop-gradient boundaries."""
import copy

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.lasd_moe_registry_pin import (
    HEADWISE_LASD_MOE_MODELS_PIN, HYPER_LASD_MOE_MODEL_PIN, HYPER_LASD_MOE_MODELS_PIN, LASD_MOE_MODELS_PIN,
    RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN, GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN,
    GRAPH_LAYER_LASD_MODELS_PIN, GRAPH_LAYER_LASD_DETACH_MODEL_PIN,
    lasd_moe_settings_pin, validate_lasd_moe_contract_pin,
)
from final_exam.models.attention_score_pin import AttentionScoreExtractorPin
from final_exam.models.lasd_moe_pin import (
    LASDMoEModelPin, Top2LayerMoEPin, _fuse_pe_memories_pin, score_dynamics_pin,
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


@pytest.mark.parametrize('output_heads', [None, 1, 8])
def test_sparse_moe_matches_top2_reference_with_padding_and_live_router(output_heads):
    torch.manual_seed(83)
    moe = Top2LayerMoEPin(16, 3, {'hidden_dim': 9}, output_heads=output_heads)
    features = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.tensor([[True]*5, [True, True, False, False, False]])
    features.data[~valid] = float('nan')
    actual, info = moe(features, valid)
    safe = features.masked_fill(~valid.unsqueeze(-1), 0.)
    logits, selected = moe.router(safe).topk(2, dim=-1)
    weights = logits.softmax(-1)
    expected = sum(expert(safe) * ((selected == i).to(safe.dtype)*weights).sum(-1, keepdim=True)
                   for i, expert in enumerate(moe.experts)).masked_fill(~valid.unsqueeze(-1), 0.)
    if output_heads is not None:
        expected = expected.unflatten(-1, (output_heads, 3))
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


def test_heads_mix_their_own_projected_channels_across_pe_layers():
    # Each layer has four channels; the two heads use different PE mixtures.
    memories = [torch.tensor(values, dtype=torch.float32).expand(1, 3, 4)
                for values in ([1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400])]
    mixture = torch.tensor([[.2, .3, .5], [.7, .2, .1]])
    logits = torch.stack((mixture, mixture.flip(0), mixture)).log().unsqueeze(0)
    padding = torch.tensor([[False, False, True]])
    fused, weights = _fuse_pe_memories_pin(memories, logits, padding)
    expected = torch.tensor([[[53.2, 106.4, 38.1, 50.8],
                              [12.7, 25.4, 159.6, 212.8], [0., 0., 0., 0.]]])
    torch.testing.assert_close(fused, expected)
    torch.testing.assert_close(weights[0, 0], mixture)
    assert torch.count_nonzero(weights[padding]) == 0
    # A shared distribution on every head recovers the original full-vector fusion.
    scalar = logits[:, :, 0, :]
    original, _ = _fuse_pe_memories_pin(memories, scalar, padding)
    headwise, _ = _fuse_pe_memories_pin(memories, scalar.unsqueeze(2).expand_as(logits), padding)
    torch.testing.assert_close(headwise, original)


def test_default_eight_heads_each_return_three_pe_weights(inputs):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dim': 128, 'heads': 8, 'encoder_layers': 6, 'cross_layers': 3}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    shared = LASDMoEModelPin('lasd_moe_pin', cfg, prep, predictor)
    model = LASDMoEModelPin('lasd_moe_headwise_pin', cfg, prep, predictor).eval()
    assert (sum(p.numel() for p in model.parameters()) -
            sum(p.numel() for p in shared.parameters())) == 10836
    with torch.no_grad():
        memory, padding, info = model.memory(batch, return_scores=True)
    assert memory.shape == (*padding.shape, 128)
    assert info['selected_experts'].shape == (*padding.shape, 2)
    assert info['score_features'].shape == (*padding.shape, 96)
    assert info['layer_weights'].shape == (*padding.shape, 8, 3)
    sums = info['layer_weights'][~padding].sum(-1)
    torch.testing.assert_close(sums, torch.ones_like(sums))


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
    count = int(diag['valid'].sum())
    headwise = model.architecture_contract.get('layer_weight_axes') == 'B,N,H,L'
    expected_axes = (count, cfg['heads']) if headwise else (count,)
    node_weights = diag['layer_weights'][diag['valid']]
    assert node_weights.shape == (*expected_axes, cfg['cross_layers'])
    torch.testing.assert_close(node_weights.sum(-1), torch.ones(expected_axes))
    assert torch.count_nonzero(diag['layer_weights'][~diag['valid']]) == 0
    output.square().sum().backward()
    contract = model.architecture_contract
    if contract['score_detach']:
        assert diag['score_features'].grad is None
    else:
        assert diag['score_features'].grad.abs().sum() > 0
    if 'fusion_memory_detach' in contract:
        for memory, stop in zip(retained_memories, contract['fusion_memory_detach']):
            if stop:
                assert memory.grad is None
            else:
                assert memory.grad.abs().sum() > 0
    elif contract['hidden_state_detach'] and contract.get('residual_connection') == 'last_pe_memory_add':
        assert all(memory.grad is None for memory in retained_memories[:-1])
        assert retained_memories[-1].grad.abs().sum() > 0
    elif contract['hidden_state_detach']:
        assert all(memory.grad is None for memory in retained_memories)
    else:
        assert all(memory.grad is not None and memory.grad.abs().sum() > 0 for memory in retained_memories)
    if mode in GRAPH_LAYER_LASD_MODELS_PIN:
        assert model.layer_fusion.input_projection[0].weight.grad.abs().sum() > 0
        assert model.layer_fusion.head_projection[-1].weight.grad.abs().sum() > 0
        for block in [*model.layer_fusion.graph_blocks, model.layer_fusion.layer_encoder]:
            assert block.self_attn.in_proj_weight.grad.abs().sum() > 0
    else:
        assert model.layer_fusion.router.weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layer_fusion.experts.parameters())
    for block in list(model.encoder.structure_encoder.blocks) + list(model.encoder.fusion_blocks):
        if (contract['hidden_state_detach'] and contract['score_detach']
                and 'fusion_memory_detach' not in contract
                and contract.get('residual_connection') == 'none'):
            assert block.self_attn.in_proj_weight.grad is None
        else:
            assert block.self_attn.in_proj_weight.grad.abs().sum() > 0
    if model.final_graph_encoder is not None:
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


@pytest.mark.parametrize('modes', [('lasd_moe_pin', 'lasd_moe_detach_pin'), HEADWISE_LASD_MOE_MODELS_PIN,
                                  RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN, GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN,
                                  GRAPH_LAYER_LASD_MODELS_PIN])
def test_detach_pair_same_parameters_forward_rng_and_native_initialization(inputs, modes):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dropout': .2}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    torch.manual_seed(11)
    baseline = DeviceOPGateModelPin('device_op_cascade_pin', cfg, prep, predictor)
    torch.manual_seed(11)
    model = LASDMoEModelPin(modes[0], cfg, prep, predictor, {'dropout': .15})
    torch.manual_seed(11)
    detached = LASDMoEModelPin(modes[1], cfg, prep, predictor, {'dropout': .15})
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
    {'checkpoint_scores': 1}, {'dropout': float('nan')}, {'head_average': True},
    {'hyper_connection_streams': 0}, {'hyper_connection_streams': True}, {'hyper_connection_streams': 2.5}])
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
