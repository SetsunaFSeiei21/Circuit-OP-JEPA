"""Graph v4 keeps score gradients and only the deepest fusion-memory gradient."""
import copy
import json
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.lasd_moe_registry_pin import (
    GRAPH_LAYER_LASD_MODEL_PIN as ORIGINAL,
    GRAPH_LAYER_SHALLOW_DETACH_MODEL_PIN as V4,
    validate_lasd_moe_contract_pin,
)
from final_exam.models.lasd_moe_pin import LASDMoEModelPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.run_pin import MODEL_NAMES_PIN, load_checkpoint_pin
from final_exam.tests.test_summary_pin import workbook, text_values
from final_exam.tests.test_width_decoder_models_pin import inputs
from final_exam.tests.test_width_decoder_run_pin import setup


def test_v4_preserves_full_forward_parameters_and_dropout_rng(inputs):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'dropout': .2}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    models = {}
    for mode in (ORIGINAL, V4):
        torch.manual_seed(17)
        models[mode] = LASDMoEModelPin(mode, cfg, prep, predictor, {'dropout': .15})
    assert state_digest(models[ORIGINAL]) == state_digest(models[V4])
    assert models[V4].final_graph_encoder is not None
    for training in (False, True):
        for model in models.values():
            model.train(training)
        torch.manual_seed(29)
        expected = models[ORIGINAL](batch)
        rng = torch.get_rng_state()
        torch.manual_seed(29)
        actual = models[V4](batch)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('checkpoint_scores', [False, True])
def test_v4_exact_shallow_tap_boundary_and_live_score_qk_path(inputs, checkpoint_scores):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3}
    torch.manual_seed(43)
    model = LASDMoEModelPin(V4, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
        {'hidden_dim': 12, 'score_chunk_size': 3, 'checkpoint_scores': checkpoint_scores}).train()
    memories, raw_outputs, router_inputs, final_inputs, masks = [], [], [], [], []
    original = model.encoder.forward_fusion_memories

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        memories.extend(result[0])
        masks.extend(result[2])
        return result

    handles = [block.register_forward_hook(lambda _, args, value: raw_outputs.append(value))
               for block in model.encoder.fusion_blocks]
    handles += [model.layer_fusion.register_forward_pre_hook(lambda _, args: router_inputs.append(args)),
                model.final_graph_encoder.register_forward_pre_hook(lambda _, args: final_inputs.append(args))]
    try:
        with patch.object(model.encoder, 'forward_fusion_memories', side_effect=capture):
            output, info = model(batch, return_scores=True)
    finally:
        for handle in handles:
            handle.remove()

    assert len(memories) == len(raw_outputs) == 3
    assert router_inputs[0][0] is info['score_features']
    assert router_inputs[0][0].requires_grad
    assert router_inputs[0][2] is final_inputs[0][1] is masks[0]
    padding = ~info['valid']
    heads, channels = cfg['heads'], cfg['dim'] // cfg['heads']
    values = torch.stack(memories, dim=2).reshape(*memories[0].shape[:2], 3, heads, channels)
    expected = (values.transpose(2, 3) * info['layer_weights'].unsqueeze(-1)).sum(3).flatten(-2)
    fused = final_inputs[0][0]
    torch.testing.assert_close(fused, expected, atol=0, rtol=0)
    assert torch.count_nonzero(fused[padding]) == 0

    # Projected PE memories are readout taps, not the sequential PE states.
    # Only the last tap has its direct weighted-fusion Jacobian; no extra residual.
    cotangent = torch.randn_like(fused).masked_fill(padding.unsqueeze(-1), 0.)
    derivatives = torch.autograd.grad((fused * cotangent).sum(),
        (*memories, info['score_features']), allow_unused=True, retain_graph=True)
    assert derivatives[0] is derivatives[1] is None
    last_weights = info['layer_weights'][..., -1].repeat_interleave(channels, dim=-1)
    torch.testing.assert_close(derivatives[2], cotangent * last_weights, atol=0, rtol=0)
    assert derivatives[-1] is not None and derivatives[-1].abs().sum() > 0

    # Isolate the score path: this objective cannot use fused hidden states.
    targets = [info[key] for key in ('score_features', 'node_score_features',
               'global_score_features', 'global_statistics', 'scores')]
    targets += [block.self_attn.in_proj_weight for block in model.encoder.fusion_blocks]
    score_grads = torch.autograd.grad(info['layer_weights'][..., 0].square().sum(),
                                     targets, retain_graph=True, allow_unused=True)
    assert all(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
               for grad in score_grads)
    # Independently isolate the ordinary PE chain, without any score/router loss.
    chain_grads = torch.autograd.grad((memories[-1] * cotangent).sum(), raw_outputs,
                                     retain_graph=True, allow_unused=True)
    assert all(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
               for grad in chain_grads)
    for state in raw_outputs:
        state.retain_grad()
    output.square().sum().backward()
    assert all(state.grad is not None and state.grad.abs().sum() > 0 for state in raw_outputs)
    blocks = [*model.encoder.structure_encoder.blocks, *model.encoder.fusion_blocks,
              *model.layer_fusion.graph_blocks, model.layer_fusion.layer_encoder,
              model.final_graph_encoder]
    for block in blocks:
        grad = block.self_attn.in_proj_weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for module in (model.layer_fusion.input_projection, model.layer_fusion.head_projection,
                   model.decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.predictor.parameters())


@pytest.mark.parametrize('layers', [1, 3, 4])
def test_v4_variable_depth_policy_and_checkpoint_contract(inputs, layers):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': layers + 2, 'cross_layers': layers}
    model = LASDMoEModelPin(V4, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
                           {'hidden_dim': 12, 'score_chunk_size': 3}).eval()
    with torch.no_grad():
        output, info = model(batch, return_scores=True)
    assert output.shape == batch['performance'].shape and torch.isfinite(output).all()
    assert info['layer_weights'].shape[2:] == (cfg['heads'], layers)
    torch.testing.assert_close(info['layer_weights'][info['valid']].sum(-1),
                               torch.ones(int(info['valid'].sum()), cfg['heads']))
    spec = model.architecture_contract
    assert spec['score_detach'] is False and spec['score_detach_scope'] == 'none'
    assert spec['hidden_state_detach'] is True
    assert spec['hidden_state_detach_scope'] == 'shallow_fusion_inputs_only'
    assert spec['fusion_memory_detach'] == [True] * (layers - 1) + [False]
    assert spec['last_fusion_memory_detach'] is False
    assert spec['encoder_chain_detach'] is False
    assert spec['final_encoder_layers'] == 1 and spec['residual_connection'] == 'none'
    validate_lasd_moe_contract_pin(V4, spec)
    validate_lasd_moe_contract_pin(V4, spec, cfg, model.lasd_moe_config)
    for key, value in [('score_detach', True), ('hidden_state_detach', False),
                       ('hidden_state_detach_scope', 'fusion_inputs_only'),
                       ('fusion_memory_detach', [True] * layers),
                       ('last_fusion_memory_detach', True), ('encoder_chain_detach', True),
                       ('final_encoder_layers', 0), ('residual_connection', 'last_pe_memory_add')]:
        bad = copy.deepcopy(spec)
        bad[key] = value
        with pytest.raises(ValueError, match='contract'):
            validate_lasd_moe_contract_pin(V4, bad)


def test_v4_train_export_and_completed_resume(setup):
    root, cfg, _, config, run = setup
    cfg['lasd_moe_pin'] = dict(hidden_dim=12, score_chunk_size=3, diagnostic_samples=2)
    config.write_text(json.dumps(cfg))
    assert V4 in MODEL_NAMES_PIN
    out, result = run('graph_shallow_v4', [V4])
    assert len(result['results']) == 1
    assert len(result['stages']) == 3  # Shared OP predictor, frozen cache, performance model.
    directory = out / f'seed_17_pin/{V4}'
    model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
    spec = model.architecture_contract
    assert saved['recipe']['architecture'] == saved['input_contract_pin'] == spec
    report = json.loads((directory / 'test_pin.json').read_text())
    assert report['lasd_moe_pin'] == spec and not report['requires_measured_op']
    assert all({'mae', 'rmse', 'r2'} <= set(metric) for metric in report['metrics'].values())
    diagnostic = report['lasd_moe_diagnostics_pin']
    for key in ('scores', 'weights', 'global_scores', 'metadata'):
        assert (directory / diagnostic[key]).is_file()
    assert 'experts' not in diagnostic
    history = json.loads((out / f'seed_17_pin/stages_pin/{V4}/history_pin.json').read_text())
    for record in history:
        groups = {v['layer']: v for v in record['encoder_gradients']['layers']}
        for name in ('SE1', 'SE2', 'PE1', 'PE2', 'FINAL1', 'SCORE_GRAPH1',
                     'SCORE_GRAPH2', 'SCORE_LAYER1'):
            assert groups[name]['batches_with_grad'] > 0
            assert groups[name]['mean_l2'] > 0
    sheets = workbook(out / 'summary_pin.xlsx')
    assert {'MAE', 'RMSE', 'R2', 'TrainingCost', 'Configuration', 'LASD_MoE'} <= set(sheets)
    texts = text_values(sheets['LASD_MoE'])
    assert V4 in texts
    assert 'Fusion detach by PE (JSON)' in texts
    assert json.dumps(spec['fusion_memory_detach']) in texts
    contracts = [json.loads(value) for value in texts if value and value.startswith('{"attention_scope"')]
    assert len(contracts) == 1 and contracts[0] == spec
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retraining completed stage')):
        resumed, replay = run('graph_shallow_v4_resume', [V4], out)
    actual, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{V4}/checkpoint_pin.pt')
    assert state_digest(model) == state_digest(actual)
    for stage in replay['stages']:
        if stage['epochs']:
            assert stage['resume']['new_optimizer_steps'] == 0
