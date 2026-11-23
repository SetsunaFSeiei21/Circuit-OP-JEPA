"""Graph routing v1/v2/v3: detached fusion taps, optional final block/live residual."""
import json
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.lasd_moe_registry_pin import (
    GRAPH_LAYER_FUSION_DETACH_MODELS_PIN as MODES,
    GRAPH_LAYER_LASD_DETACH_MODEL_PIN as ORIGINAL,
    lasd_moe_spec_pin, validate_lasd_moe_contract_pin,
)
from final_exam.models.lasd_moe_pin import LASDMoEModelPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.run_pin import MODEL_NAMES_PIN, load_checkpoint_pin
from final_exam.tests.test_summary_pin import workbook, text_values
from final_exam.tests.test_width_decoder_models_pin import inputs
from final_exam.tests.test_width_decoder_run_pin import setup

V1, V2, V3 = MODES


def test_only_v1_keeps_final_block_and_original_initialization_is_preserved(inputs):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dropout': .2}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    models = {}
    for mode in (ORIGINAL, *MODES):
        torch.manual_seed(17)
        models[mode] = LASDMoEModelPin(mode, cfg, prep, predictor, {'dropout': .15})
    assert state_digest(models[ORIGINAL]) == state_digest(models[V1])
    assert state_digest(models[V2]) == state_digest(models[V3])
    for mode in MODES:
        assert state_digest(models[mode].encoder) == state_digest(models[ORIGINAL].encoder)
        assert state_digest(models[mode].decoder) == state_digest(models[ORIGINAL].decoder)
        assert state_digest(models[mode].layer_fusion) == state_digest(models[ORIGINAL].layer_fusion)
    removed = sum(p.numel() for p in models[V1].final_graph_encoder.parameters())
    for mode in (V2, V3):
        assert models[mode].final_graph_encoder is None
        assert not any(key.startswith('final_graph_encoder.') for key in models[mode].state_dict())
        assert sum(p.numel() for p in models[V1].parameters()) - sum(
            p.numel() for p in models[mode].parameters()) == removed
    # Detaching copies changes only backward, including in the presence of dropout.
    for training in (False, True):
        models[ORIGINAL].train(training)
        models[V1].train(training)
        torch.manual_seed(29)
        expected = models[ORIGINAL](batch)
        rng = torch.get_rng_state()
        torch.manual_seed(29)
        actual = models[V1](batch)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('checkpoint_scores', [False, True])
def test_fusion_equation_and_exact_gradient_boundaries(inputs, mode, checkpoint_scores):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3}
    torch.manual_seed(43)
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
        {'hidden_dim': 12, 'score_chunk_size': 3, 'checkpoint_scores': checkpoint_scores}).train()
    memories, raw_outputs, routing_inputs, decoder_inputs, final_inputs, masks = [], [], [], [], [], []
    original = model.encoder.forward_fusion_memories

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        memories.extend(result[0])
        masks.extend(result[2])
        return result

    handles = [block.register_forward_hook(lambda _, args, value: raw_outputs.append(value))
               for block in model.encoder.fusion_blocks]
    handles += [model.layer_fusion.register_forward_pre_hook(lambda _, args: routing_inputs.append(args)),
                model.decoder.register_forward_pre_hook(lambda _, args: decoder_inputs.append(args))]
    if model.final_graph_encoder is not None:
        handles.append(model.final_graph_encoder.register_forward_pre_hook(lambda _, args: final_inputs.append(args)))
    try:
        with patch.object(model.encoder, 'forward_fusion_memories', side_effect=capture):
            output, info = model(batch, return_scores=True)
    finally:
        for handle in handles:
            handle.remove()
    assert len(memories) == len(raw_outputs) == 3
    assert not routing_inputs[0][0].requires_grad
    torch.testing.assert_close(routing_inputs[0][0], info['score_features'], atol=0, rtol=0)
    assert routing_inputs[0][2] is masks[0]  # Native mask/role bias has not been detached.
    padding = ~info['valid']
    heads, channels = cfg['heads'], cfg['dim'] // cfg['heads']
    values = torch.stack(memories, dim=2).reshape(*memories[0].shape[:2], 3, heads, channels)
    expected = (values.transpose(2, 3) * info['layer_weights'].unsqueeze(-1)).sum(3).flatten(-2)
    fused = final_inputs[0][0] if mode == V1 else decoder_inputs[0][0]
    if mode == V3:
        expected = memories[-1] + expected
    torch.testing.assert_close(fused, expected, atol=0, rtol=0)
    assert len(final_inputs) == int(mode == V1)
    assert torch.equal(decoder_inputs[0][1], padding)
    assert torch.count_nonzero(fused[padding]) == 0
    if mode == V1:
        assert final_inputs[0][1] is masks[0]

    # The residual is exactly an identity Jacobian to the last PE *tap*.
    # Earlier raw PE states can still get gradients through the live encoder chain.
    cotangent = torch.randn_like(fused).masked_fill(padding.unsqueeze(-1), 0.)
    derivatives = torch.autograd.grad((fused * cotangent).sum(),
        (*memories, info['score_features']), allow_unused=True, retain_graph=True)
    assert derivatives[0] is derivatives[1] is derivatives[-1] is None
    if mode == V3:
        torch.testing.assert_close(derivatives[2], cotangent, atol=0, rtol=0)
    else:
        assert derivatives[2] is None
    for state in raw_outputs:
        state.retain_grad()
    info['layer_weights'].retain_grad()
    output.square().sum().backward()
    assert info['layer_weights'].grad.abs().sum() > 0
    for block in [*model.layer_fusion.graph_blocks, model.layer_fusion.layer_encoder]:
        assert block.self_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.layer_fusion.input_projection[0].weight.grad.abs().sum() > 0
    assert model.layer_fusion.head_projection[-1].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
    for state in raw_outputs:
        if mode == V3:
            assert state.grad is not None and state.grad.abs().sum() > 0
        else:
            assert state.grad is None
    for block in [*model.encoder.structure_encoder.blocks, *model.encoder.fusion_blocks]:
        if mode == V3:
            assert block.self_attn.in_proj_weight.grad.abs().sum() > 0
        else:
            assert all(p.grad is None for p in block.parameters())
    if mode == V1:
        assert model.final_graph_encoder.self_attn.in_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.predictor.parameters())


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('layers', [1, 3, 4])
def test_variable_depth_contract_and_memory_shapes(inputs, mode, layers):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': layers+2, 'cross_layers': layers}
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
                           {'hidden_dim': 12, 'score_chunk_size': 3}).eval()
    with torch.no_grad():
        output, info = model(batch, return_scores=True)
    assert output.shape == batch['performance'].shape and torch.isfinite(output).all()
    assert info['layer_weights'].shape[2:] == (cfg['heads'], layers)
    torch.testing.assert_close(info['layer_weights'][info['valid']].sum(-1),
                               torch.ones(int(info['valid'].sum()), cfg['heads']))
    spec = model.architecture_contract
    assert spec['score_detach'] and spec['hidden_state_detach']
    assert spec['hidden_state_detach_scope'] == 'fusion_inputs_only'
    assert spec['encoder_chain_detach'] is False
    assert spec['final_encoder_layers'] == int(mode == V1)
    assert spec['residual_connection'] == ('last_pe_memory_add' if mode == V3 else 'none')
    assert spec['residual_position'] == ('metric_decoder_input' if mode == V3 else 'none')
    assert spec['final_pe_residual_detach'] is (False if mode == V3 else None)
    validate_lasd_moe_contract_pin(mode, spec)
    validate_lasd_moe_contract_pin(mode, spec, cfg, model.lasd_moe_config)
    # New gradient/final-block policies are part of exact checkpoint identity.
    for key, value in [('hidden_state_detach', False), ('score_detach', False),
                       ('encoder_chain_detach', True), ('final_pe_residual_detach', True),
                       ('final_encoder_layers', 1-spec['final_encoder_layers'])]:
        bad = {**spec, key: value}
        with pytest.raises(ValueError, match='contract'):
            validate_lasd_moe_contract_pin(mode, bad)


def test_three_variants_train_export_and_resume_with_correct_gradient_history(setup):
    root, cfg, _, config, run = setup
    cfg['lasd_moe_pin'] = dict(hidden_dim=12, score_chunk_size=3, diagnostic_samples=2)
    config.write_text(json.dumps(cfg))
    assert set(MODES) <= set(MODEL_NAMES_PIN)
    out, result = run('graph_fusion_variants', [ORIGINAL, *MODES])
    assert len(result['results']) == 4
    assert len(result['stages']) == 6  # One shared OP predictor/cache plus four performance models.
    audits = {}
    for mode in MODES:
        directory = out / f'seed_17_pin/{mode}'
        model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
        spec = model.architecture_contract
        assert saved['recipe']['architecture'] == saved['input_contract_pin'] == spec
        assert (directory / 'config_pin.json').is_file()
        report = json.loads((directory / 'test_pin.json').read_text())
        assert report['lasd_moe_pin'] == spec and not report['requires_measured_op']
        assert all({'mae', 'rmse', 'r2'} <= set(metric) for metric in report['metrics'].values())
        diagnostic = report['lasd_moe_diagnostics_pin']
        for key in ('scores', 'weights', 'global_scores', 'metadata'):
            assert (directory / diagnostic[key]).is_file()
        assert 'experts' not in diagnostic
        history = json.loads((out / f'seed_17_pin/stages_pin/{mode}/history_pin.json').read_text())
        for record in history:
            groups = {v['layer']: v for v in record['encoder_gradients']['layers']}
            assert ('FINAL1' in groups) == (mode == V1)
            for name in ('SCORE_GRAPH1', 'SCORE_GRAPH2', 'SCORE_LAYER1'):
                assert groups[name]['mean_l2'] > 0
            for name in ('SE1', 'SE2', 'PE1', 'PE2'):
                assert (groups[name]['batches_with_grad'] > 0) == (mode == V3)
                assert (groups[name]['mean_l2'] > 0) == (mode == V3)
        audits[mode] = json.loads((directory / 'lasd_moe_audit_pin.json').read_text())
    original_audit = json.loads((out / f'seed_17_pin/{ORIGINAL}/lasd_moe_audit_pin.json').read_text())
    assert audits[V1]['initial_model_digest'] == original_audit['initial_model_digest']
    assert audits[V2]['initial_model_digest'] == audits[V3]['initial_model_digest']
    sheets = workbook(out / 'summary_pin.xlsx')
    assert {'MAE', 'RMSE', 'R2', 'TrainingCost', 'Configuration', 'LASD_MoE'} <= set(sheets)
    texts = text_values(sheets['LASD_MoE'])
    assert set(MODES) <= set(texts)
    assert {'Final encoder layers', 'Final PE residual', 'Fusion memories detached'} <= set(texts)
    contracts = [json.loads(value) for value in texts if value and value.startswith('{"attention_scope"')]
    assert {c['mode'] for c in contracts} == {ORIGINAL, *MODES}
    for contract in contracts:
        validate_lasd_moe_contract_pin(contract['mode'], contract)
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retraining completed stages')):
        resumed, _ = run('graph_fusion_variants_resume', MODES, out)
    for mode in MODES:
        expected, _ = load_checkpoint_pin(out / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        actual, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(expected) == state_digest(actual)
