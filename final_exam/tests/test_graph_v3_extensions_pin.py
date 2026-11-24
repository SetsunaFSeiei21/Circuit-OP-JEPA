"""v5: v3 with performance-PE HC; v6: v3 with post-sum LayerNorm."""
import copy
import json
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.lasd_moe_registry_pin import (
    GRAPH_LAYER_V3_EXTENSION_MODELS_PIN, validate_lasd_moe_contract_pin,
)
from final_exam.models.hyper_connection_pin import HyperConnectionPEBlockPin, HyperConnectionPin
from final_exam.models.lasd_moe_pin import LASDMoEModelPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.run_pin import MODEL_NAMES_PIN, load_checkpoint_pin
from final_exam.tests.test_summary_pin import workbook, text_values
from final_exam.tests.test_width_decoder_models_pin import inputs
from final_exam.tests.test_width_decoder_run_pin import setup

V3 = 'lasd_graph_layer_headwise_detach_pin_v3'
V5, V6 = GRAPH_LAYER_V3_EXTENSION_MODELS_PIN


@pytest.mark.parametrize('streams', [1, 4])
def test_hc_is_inside_performance_pe_only_and_common_initialization_is_preserved(inputs, streams):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dropout': .2}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    models = {}
    for mode in (V3, V5, V6):
        torch.manual_seed(17)
        models[mode] = LASDMoEModelPin(mode, cfg, prep, predictor,
                                      {'dropout': .15, 'hyper_connection_streams': streams})
    base, hyper, norm = (models[m] for m in (V3, V5, V6))
    for model in (hyper, norm):
        assert model.final_graph_encoder is None
        assert not any(k.startswith('final_graph_encoder.') for k in model.state_dict())
        assert state_digest(model.layer_fusion) == state_digest(base.layer_fusion)
        assert state_digest(model.decoder) == state_digest(base.decoder)
        assert state_digest(model.encoder.structure_encoder) == state_digest(base.encoder.structure_encoder)
    assert state_digest(norm.encoder) == state_digest(base.encoder)
    assert all(isinstance(block, HyperConnectionPEBlockPin) for block in hyper.encoder.fusion_blocks)
    assert sum(isinstance(m, HyperConnectionPin) for m in hyper.modules()) == 3 * cfg['cross_layers']
    for old, wrapped in zip(base.encoder.fusion_blocks, hyper.encoder.fusion_blocks):
        assert wrapped.streams == streams and state_digest(old) == state_digest(wrapped.block)
    assert not any(isinstance(m, HyperConnectionPin) for m in hyper.layer_fusion.modules())
    assert not any(isinstance(m, HyperConnectionPin) for m in predictor.modules())
    assert hyper.decoder_input_norm is None and norm.decoder_input_norm is not None
    assert sum(p.numel() for p in norm.parameters()) - sum(p.numel() for p in base.parameters()) == 2 * cfg['dim']
    if streams == 1:
        # One identity HC stream reproduces the original native PE residuals.
        for training in (False, True):
            base.train(training)
            hyper.train(training)
            torch.manual_seed(29)
            expected = base(batch)
            rng = torch.get_rng_state()
            torch.manual_seed(29)
            torch.testing.assert_close(hyper(batch), expected, atol=0, rtol=0)
            assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('mode', [V5, V6])
@pytest.mark.parametrize('checkpoint_scores', [False, True])
def test_decoder_equation_and_exact_v3_gradient_boundaries(inputs, mode, checkpoint_scores):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3}
    torch.manual_seed(43)
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
        {'hidden_dim': 12, 'score_chunk_size': 3, 'checkpoint_scores': checkpoint_scores}).train()
    memories, raw_states, routing_inputs, decoder_inputs = [], [], [], []
    original = model.encoder.forward_fusion_memories

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        memories.extend(result[0])
        return result

    handles = [block.register_forward_hook(lambda _, args, value: raw_states.append(value))
               for block in model.encoder.fusion_blocks]
    handles += [model.layer_fusion.register_forward_pre_hook(lambda _, args: routing_inputs.append(args)),
                model.decoder.register_forward_pre_hook(lambda _, args: decoder_inputs.append(args))]
    try:
        with patch.object(model.encoder, 'forward_fusion_memories', side_effect=capture):
            output, info = model(batch, return_scores=True)
    finally:
        for handle in handles:
            handle.remove()
    assert not routing_inputs[0][0].requires_grad
    assert len(memories) == len(raw_states) == 3
    assert all(state.ndim == (4 if mode == V5 else 3) for state in raw_states)
    padding = ~info['valid']
    heads, channels = cfg['heads'], cfg['dim'] // cfg['heads']
    values = torch.stack(memories, dim=2).reshape(*memories[0].shape[:2], 3, heads, channels)
    fused = (values.transpose(2, 3) * info['layer_weights'].unsqueeze(-1)).sum(3).flatten(-2)
    residual_sum = memories[-1] + fused
    expected = (model.decoder_input_norm(residual_sum).masked_fill(padding.unsqueeze(-1), 0.)
                if mode == V6 else residual_sum)
    actual = decoder_inputs[0][0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(decoder_inputs[0][1], padding)
    assert torch.count_nonzero(actual[padding]) == 0

    cotangent = torch.randn_like(actual).masked_fill(padding.unsqueeze(-1), 0.)
    derivatives = torch.autograd.grad((actual * cotangent).sum(),
        (*memories, info['score_features']), allow_unused=True, retain_graph=True)
    assert derivatives[0] is derivatives[1] is derivatives[-1] is None
    if mode == V5:
        torch.testing.assert_close(derivatives[2], cotangent, atol=0, rtol=0)
    else:
        independent = residual_sum.detach().requires_grad_()
        reference = torch.autograd.grad((model.decoder_input_norm(independent) * cotangent).sum(), independent)[0]
        torch.testing.assert_close(derivatives[2], reference, atol=0, rtol=0)
    for state in raw_states:
        state.retain_grad()
    info['layer_weights'].retain_grad()
    output.square().sum().backward()
    assert info['layer_weights'].grad.abs().sum() > 0
    assert all(state.grad is not None and state.grad.abs().sum() > 0 for state in raw_states)
    for block in [*model.encoder.structure_encoder.blocks, *model.encoder.fusion_blocks,
                  *model.layer_fusion.graph_blocks, model.layer_fusion.layer_encoder]:
        assert block.self_attn.in_proj_weight.grad.abs().sum() > 0
    if mode == V5:
        for block in model.encoder.fusion_blocks:
            for mixer in (block.self_connection, block.cross_connection, block.ffn_connection):
                assert mixer.static_alpha.grad.abs().sum() > 0
                assert mixer.static_beta.grad.abs().sum() > 0
                assert mixer.alpha_projection.grad.abs().sum() > 0
                assert mixer.beta_projection.grad.abs().sum() > 0
    else:
        assert model.decoder_input_norm.weight.grad.abs().sum() > 0
        assert model.decoder_input_norm.bias.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.predictor.parameters())


def test_v6_masks_trained_norm_bias_and_does_not_mix_graphs(inputs):
    cfg, prep, support, batch = inputs
    batch = copy.deepcopy(batch)
    batch['device_mask'][1, -2:] = False
    for key in ('owner_index', 'pin_to_device', 'pin_terminal_index'):
        batch[key][1, -2:] = -1
    model = LASDMoEModelPin(V6, cfg, prep, DeviceOPPredictorPin(cfg, prep, support)).eval()
    with torch.no_grad():
        model.decoder_input_norm.bias.fill_(.7)
        memory, padding = model.memory(batch)
        assert padding.any() and torch.count_nonzero(memory[padding]) == 0
        raw = torch.randn(2, 7, cfg['dim'])
        normalized = model.decoder_input_norm(raw)
        changed = raw.clone()
        changed[1] *= 100
        changed[0, 1:] += 50
        torch.testing.assert_close(normalized[0, 0], model.decoder_input_norm(changed)[0, 0], atol=0, rtol=0)


@pytest.mark.parametrize('mode', [V5, V6])
@pytest.mark.parametrize('layers', [1, 3, 4])
def test_variable_pe_depth_strict_contracts_and_hc_stream_replay(inputs, mode, layers):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': layers + 2, 'cross_layers': layers}
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
                           {'hidden_dim': 12, 'hyper_connection_streams': 3}).eval()
    with torch.no_grad():
        output, info = model(batch, return_scores=True)
    assert output.shape == batch['performance'].shape and torch.isfinite(output).all()
    assert info['layer_weights'].shape[2:] == (cfg['heads'], layers)
    spec = model.architecture_contract
    assert spec['score_detach'] and spec['hidden_state_detach']
    assert spec['final_encoder_layers'] == 0 and spec['final_pe_residual_detach'] is False
    assert spec['residual_connection'] == 'last_pe_memory_add'
    assert spec['hidden_state_detach_scope'] == 'fusion_inputs_only'
    assert spec['encoder_chain_detach'] is False
    assert ('hyper_connection' in spec) == (mode == V5)
    if mode == V5:
        assert spec['hyper_connection']['streams'] == 3
        assert spec['hyper_connection']['scope'] == 'performance_pe_only'
        assert spec['decoder_input_normalization'] == 'none'
    else:
        assert spec['decoder_input_normalization'] == 'layer_norm_after_residual_sum'
    validate_lasd_moe_contract_pin(mode, spec)
    validate_lasd_moe_contract_pin(mode, spec, cfg, model.lasd_moe_config)
    for key, value in [('score_detach', False), ('hidden_state_detach', False),
                       ('final_pe_residual_detach', True), ('encoder_chain_detach', True),
                       ('final_encoder_layers', 1), ('decoder_input_normalization', 'invalid')]:
        bad = copy.deepcopy(spec)
        bad[key] = value
        with pytest.raises(ValueError, match='contract'):
            validate_lasd_moe_contract_pin(mode, bad)


def test_extensions_train_export_diagnostics_and_resume(setup):
    _, cfg, _, config, run = setup
    cfg['lasd_moe_pin'] = dict(hidden_dim=12, score_chunk_size=3, diagnostic_samples=2,
                              hyper_connection_streams=3)
    config.write_text(json.dumps(cfg))
    assert {V5, V6} <= set(MODEL_NAMES_PIN)
    out, result = run('graph_extensions', [V3, V5, V6])
    assert len(result['results']) == 3 and len(result['stages']) == 5
    for mode in (V5, V6):
        directory = out / f'seed_17_pin/{mode}'
        model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
        spec = model.architecture_contract
        assert saved['recipe']['architecture'] == saved['input_contract_pin'] == spec
        assert (directory / 'config_pin.json').is_file()
        report = json.loads((directory / 'test_pin.json').read_text())
        assert report['lasd_moe_pin'] == spec and not report['requires_measured_op']
        assert all({'mae', 'rmse', 'r2'} <= set(m) for m in report['metrics'].values())
        diagnostic = report['lasd_moe_diagnostics_pin']
        assert all((directory / diagnostic[k]).is_file() for k in ('scores', 'weights', 'global_scores', 'metadata'))
        history = json.loads((out / f'seed_17_pin/stages_pin/{mode}/history_pin.json').read_text())
        for record in history:
            groups = {g['layer']: g for g in record['encoder_gradients']['layers']}
            assert 'FINAL1' not in groups
            assert ('DECODER_INPUT_NORM1' in groups) == (mode == V6)
            assert all(g['batches_with_grad'] > 0 and g['mean_l2'] > 0 for g in groups.values())
    sheets = workbook(out / 'summary_pin.xlsx')
    assert {'MAE', 'RMSE', 'R2', 'Configuration', 'TrainingCost', 'LASD_MoE'} <= set(sheets)
    texts = text_values(sheets['LASD_MoE'])
    assert {V5, V6, 'Hyper-Connection scope', 'Decoder input normalization',
            'performance_pe_only', 'layer_norm_after_residual_sum'} <= set(texts)
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retraining completed stage')):
        resumed, _ = run('graph_extensions_resume', [V5, V6], out)
    for mode in (V5, V6):
        before, _ = load_checkpoint_pin(out / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        after, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(before) == state_digest(after)
