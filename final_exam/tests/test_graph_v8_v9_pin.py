"""v8: v7 powers + v5 HC. v9: add v6's feature LayerNorm after the residual sum."""
import copy
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.lasd_moe_registry_pin import (
    GRAPH_LAYER_POWER_HC_MODEL_PIN as V8, lasd_moe_spec_pin, validate_lasd_moe_contract_pin,
    GRAPH_LAYER_POWER_HC_NORM_MODEL_PIN as V9,
)
from final_exam.models.attention_score_pin import score_from_attention_pin
from final_exam.models.hyper_connection_pin import HyperConnectionPEBlockPin, HyperConnectionPin
from final_exam.models.lasd_moe_pin import LASDMoEModelPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.tests.test_width_decoder_models_pin import inputs

V5 = 'lasd_graph_layer_headwise_detach_pin_v5'
V7 = 'lasd_graph_layer_headwise_detach_pin_v7'


@pytest.mark.parametrize('distance', [1, 2, 3])
def test_v8_reuses_v5_hc_only_in_performance_pe_and_v7_common_initialization(inputs, distance):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3, 'dropout': .2}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    options = dict(aggregation_distance=distance, hyper_connection_streams=4, hidden_dim=12, dropout=.15)
    models = []
    for mode in (V7, V8):
        torch.manual_seed(17)
        models.append(LASDMoEModelPin(mode, cfg, prep, predictor, options))
    base, model = models
    for name in ('layer_fusion', 'decoder'):
        assert state_digest(getattr(model, name)) == state_digest(getattr(base, name))
    assert state_digest(model.encoder.structure_encoder) == state_digest(base.encoder.structure_encoder)
    for original, wrapped in zip(base.encoder.fusion_blocks, model.encoder.fusion_blocks):
        assert isinstance(wrapped, HyperConnectionPEBlockPin) and wrapped.streams == 4
        assert state_digest(original) == state_digest(wrapped.block)
    assert sum(isinstance(m, HyperConnectionPin) for m in model.modules()) == 3 * cfg['cross_layers']
    for module in (model.layer_fusion, model.decoder, model.encoder.structure_encoder, predictor):
        assert not any(isinstance(m, HyperConnectionPin) for m in module.modules())
    assert model.final_graph_encoder is None and model.decoder_input_norm is None
    spec = model.architecture_contract
    assert spec['hyper_connection'] == lasd_moe_spec_pin(V5, options, cfg)['hyper_connection']
    powers = copy.deepcopy(spec['attention_powers'])
    assert powers.pop('attention_input') == 'hyper_connection_self_attention_read'
    assert powers == base.architecture_contract['attention_powers']
    assert spec['score_detach'] and spec['hidden_state_detach']
    assert spec['final_pe_residual_detach'] is False and spec['encoder_chain_detach'] is False


@pytest.mark.parametrize('streams', [1, 3])
def test_distance_one_is_exact_v5_including_training_dropout_and_gradients(inputs, streams):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3, 'dropout': .2}
    models = []
    for mode in (V5, V8):
        torch.manual_seed(9)
        predictor = DeviceOPPredictorPin(cfg, prep, support)
        models.append(LASDMoEModelPin(mode, cfg, prep, predictor,
            dict(aggregation_distance=1, hyper_connection_streams=streams,
                 dropout=.15, score_chunk_size=3)))
    a, b = models
    assert state_digest(a) == state_digest(b)
    for training in (False, True):
        outputs, rngs = [], []
        for model in models:
            model.train(training)
            model.zero_grad()
            torch.manual_seed(17)
            output = model(batch)
            output.square().sum().backward()
            outputs.append(output)
            rngs.append(torch.get_rng_state())
        torch.testing.assert_close(*outputs, atol=0, rtol=0)
        assert torch.equal(*rngs)
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.grad is None:
                assert q.grad is None, name
            else:
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_one_identity_hc_stream_reproduces_initial_v7_forward(inputs):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dropout': .2}
    models = []
    for mode in (V7, V8):
        torch.manual_seed(29)
        predictor = DeviceOPPredictorPin(cfg, prep, support)
        models.append(LASDMoEModelPin(mode, cfg, prep, predictor,
            dict(aggregation_distance=3, hyper_connection_streams=1, dropout=.1)))
    for training in (False, True):
        results, rngs = [], []
        for model in models:
            model.train(training)
            torch.manual_seed(43)
            results.append(model(batch))
            rngs.append(torch.get_rng_state())
        torch.testing.assert_close(*results, atol=0, rtol=0)
        assert torch.equal(*rngs)


@pytest.mark.parametrize('mode', [V8, V9])
def test_powers_observe_actual_hc_self_attention_input_with_padding(inputs, mode):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3}
    batch = copy.deepcopy(batch)
    batch['device_mask'][1, -2:] = False
    for key in ('owner_index', 'pin_to_device', 'pin_terminal_index'):
        batch[key][1, -2:] = -1
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
        dict(aggregation_distance=3, hyper_connection_streams=3, score_chunk_size=3)).eval()
    # Nontrivial reads ensure we observe the mixed HC vector, not a stream sum or SE tensor.
    with torch.no_grad():
        for block in model.encoder.fusion_blocks:
            block.self_connection.static_alpha[:, 0].fill_(.4)
            block.self_connection.alpha_projection.normal_(0., .15)
    observed, mha_inputs = [], []
    original = model.score_extractor.forward
    def observe(attention, x, mask, valid):
        observed.append((attention, x.clone(), mask.clone(), valid.clone()))
        return original(attention, x, mask, valid)
    handles = [block.self_attn.register_forward_pre_hook(lambda _, args: mha_inputs.append(args[0].clone()))
               for block in model.encoder.fusion_blocks]
    try:
        with torch.no_grad(), patch.object(model.score_extractor, 'forward', side_effect=observe):
            output, info = model(batch, return_scores=True)
    finally:
        for handle in handles: handle.remove()
    assert len(observed) == len(mha_inputs) == cfg['cross_layers']
    assert torch.isfinite(output).all() and not info['power_scores'].requires_grad
    with torch.no_grad():
        for layer, ((attention, x, mask, valid), actual) in enumerate(zip(observed, mha_inputs)):
            assert x.ndim == 3
            torch.testing.assert_close(x, actual, atol=0, rtol=0)
            _, a = attention(x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=False)
            a = a.masked_fill(~(valid[:, None, :, None] & valid[:, None, None, :]), 0.)
            for order in range(3):
                expected = score_from_attention_pin(torch.linalg.matrix_power(a, order+1), valid)
                torch.testing.assert_close(info['power_scores'][:, :, order, layer], expected)
    assert (~info['valid']).any()
    assert torch.count_nonzero(info['power_scores'][~info['valid']]) == 0


@pytest.mark.parametrize('mode', [V8, V9])
def test_hc_and_main_chain_train_but_score_and_fusion_taps_stop_gradient(inputs, mode):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3}
    torch.manual_seed(43)
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
        dict(aggregation_distance=2, hyper_connection_streams=3, hidden_dim=12, score_chunk_size=3)).train()
    memories, raw_states, decoder_inputs = [], [], []
    original = model.encoder.forward_fusion_memories
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        memories.extend(result[0])
        return result
    handles = [block.register_forward_hook(lambda _, args, value: raw_states.append(value))
               for block in model.encoder.fusion_blocks]
    handles.append(model.decoder.register_forward_pre_hook(lambda _, args: decoder_inputs.append(args[0])))
    try:
        with patch.object(model.encoder, 'forward_fusion_memories', side_effect=capture):
            output, info = model(batch, return_scores=True)
    finally:
        for handle in handles: handle.remove()
    assert all(state.ndim == 4 for state in raw_states)
    assert not info['score_features'].requires_grad
    values = torch.stack(memories, 2).unflatten(-1, (cfg['heads'], -1)).transpose(2, 3)
    fused = (values * info['layer_weights'].unsqueeze(-1)).sum(3).flatten(-2)
    actual = decoder_inputs[0]
    residual_sum = memories[-1]+fused
    expected = (model.decoder_input_norm(residual_sum).masked_fill(~info['valid'].unsqueeze(-1), 0.)
                if mode == V9 else residual_sum)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    cotangent = torch.randn_like(actual)
    grads = torch.autograd.grad((actual*cotangent).sum(), memories, allow_unused=True, retain_graph=True)
    assert grads[0] is grads[1] is None
    if mode == V9:
        independent = residual_sum.detach().requires_grad_()
        normalized = model.decoder_input_norm(independent).masked_fill(~info['valid'].unsqueeze(-1), 0.)
        expected_grad = torch.autograd.grad((normalized*cotangent).sum(), independent)[0]
    else:
        expected_grad = cotangent
    torch.testing.assert_close(grads[2], expected_grad, atol=0, rtol=0)
    pe_parameters = [p for block in model.encoder.fusion_blocks for p in block.parameters()]
    score_grads = torch.autograd.grad(info['layer_weights'].square().sum(), pe_parameters,
                                     allow_unused=True, retain_graph=True)
    assert all(g is None for g in score_grads)
    for state in raw_states: state.retain_grad()
    output.square().sum().backward()
    assert all(state.grad is not None and state.grad.abs().sum() > 0 for state in raw_states)
    for block in model.encoder.fusion_blocks:
        assert block.self_attn.in_proj_weight.grad.abs().sum() > 0
        for mixer in (block.self_connection, block.cross_connection, block.ffn_connection):
            for param in (mixer.static_alpha, mixer.static_beta, mixer.alpha_projection, mixer.beta_projection):
                assert param.grad is not None and param.grad.abs().sum() > 0
    assert all(p[0].weight.grad.abs().sum() > 0 for p in model.layer_fusion.order_projections)
    assert all(p.grad is None for p in model.predictor.parameters())
    if mode == V9:
        assert model.decoder_input_norm.weight.grad.abs().sum() > 0
        assert model.decoder_input_norm.bias.grad.abs().sum() > 0


@pytest.mark.parametrize('layers,distance,streams', [(1, 1, 1), (3, 2, 3), (4, 3, 4)])
@pytest.mark.parametrize('mode', [V8, V9])
def test_variable_depth_distance_streams_and_strict_contract(inputs, layers, distance, streams, mode):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': layers+2, 'cross_layers': layers}
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
        dict(aggregation_distance=distance, hyper_connection_streams=streams)).eval()
    with torch.no_grad(): output, info = model(batch, return_scores=True)
    assert torch.isfinite(output).all()
    assert info['power_scores'].shape[2:] == (distance, layers, cfg['heads'], 2)
    assert info['layer_weights'].shape[2:] == (cfg['heads'], layers)
    torch.testing.assert_close(info['layer_weights'][info['valid']].sum(-1),
                               torch.ones_like(info['layer_weights'][info['valid']][..., 0]))
    spec = model.architecture_contract
    validate_lasd_moe_contract_pin(mode, spec)
    validate_lasd_moe_contract_pin(mode, spec, cfg, model.lasd_moe_config)
    for target, field, value in [('hyper_connection', 'streams', streams+1),
                               ('hyper_connection', 'scope', 'all_encoders'),
                               ('attention_powers', 'attention_input', 'raw_se_output')]:
        bad = copy.deepcopy(spec)
        bad[target][field] = value
        with pytest.raises(ValueError): validate_lasd_moe_contract_pin(mode, bad, cfg, model.lasd_moe_config)
    for field, value in [('score_detach', False), ('hidden_state_detach', False),
                          ('final_pe_residual_detach', True), ('final_encoder_layers', 1)]:
        bad = copy.deepcopy(spec); bad[field] = value
        with pytest.raises(ValueError): validate_lasd_moe_contract_pin(mode, bad)


def test_v9_is_exactly_v8_plus_v6_post_sum_feature_norm_and_masks_padding(inputs):
    cfg, prep, support, batch = inputs
    batch = copy.deepcopy(batch)
    batch['device_mask'][1, -2:] = False
    for key in ('owner_index', 'pin_to_device', 'pin_terminal_index'):
        batch[key][1, -2:] = -1
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    options = dict(aggregation_distance=2, hyper_connection_streams=3)
    models = []
    for mode in (V8, V9):
        torch.manual_seed(29)
        models.append(LASDMoEModelPin(mode, cfg, prep, predictor, options).eval())
    base, norm = models
    for name in ('encoder', 'layer_fusion', 'decoder'):
        assert state_digest(getattr(base, name)) == state_digest(getattr(norm, name))
    assert sum(p.numel() for p in norm.parameters()) - sum(p.numel() for p in base.parameters()) == 2*cfg['dim']
    assert norm.architecture_contract['decoder_input_norm'] == lasd_moe_spec_pin(
        'lasd_graph_layer_headwise_detach_pin_v6', options, cfg)['decoder_input_norm']
    assert norm.final_graph_encoder is None
    with torch.no_grad():
        norm.decoder_input_norm.bias.fill_(.7)
        raw, padding, raw_diag = base.memory(batch, return_scores=True)
        actual, norm_padding, norm_diag = norm.memory(batch, return_scores=True)
        assert torch.equal(padding, norm_padding) and padding.any()
        torch.testing.assert_close(actual, norm.decoder_input_norm(raw).masked_fill(padding.unsqueeze(-1), 0.),
                                   atol=0, rtol=0)
        assert torch.count_nonzero(actual[padding]) == 0
        for key in ('power_scores', 'score_features', 'layer_weights'):
            torch.testing.assert_close(raw_diag[key], norm_diag[key], atol=0, rtol=0)
        changed = raw.clone()
        changed[1] *= 100
        changed[0, 1:] += 50
        torch.testing.assert_close(norm.decoder_input_norm(raw)[0, 0],
                                   norm.decoder_input_norm(changed)[0, 0], atol=0, rtol=0)
    bad = copy.deepcopy(norm.architecture_contract)
    bad['decoder_input_normalization'] = 'before_residual_addition'
    with pytest.raises(ValueError): validate_lasd_moe_contract_pin(V9, bad)
