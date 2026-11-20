"""Graph-context and per-node depth-context masks, alignment, gradients and replay."""
import copy
import csv
import json
from unittest.mock import patch

import pytest
import torch

from device_jepa.model import ModelConfig
from device_jepa.training import state_digest
from final_exam.lasd_moe_registry_pin import (
    GRAPH_LAYER_LASD_MODEL_PIN as MODE, lasd_moe_spec_pin, validate_lasd_moe_contract_pin,
)
from final_exam.models.graph_layer_score_pin import GraphLayerScoreRouterPin
from final_exam.models.lasd_moe_pin import LASDMoEModelPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.run_pin import load_checkpoint_pin
from final_exam.tests.test_width_decoder_models_pin import inputs
from final_exam.tests.test_width_decoder_run_pin import setup
from final_exam.tests.test_summary_pin import workbook, text_values


def router(layers=3, heads=2, dim=8):
    cfg = ModelConfig(dim=dim, heads=heads, encoder_layers=layers+1, cross_layers=layers,
                      ffn_dim=16, dropout=0.)
    return GraphLayerScoreRouterPin(10*layers*heads, cfg, hidden_dim=12).eval()


def mask_for(valid, heads, allowed=None):
    b, n = valid.shape
    if allowed is None:
        allowed = torch.ones(b, n, n, dtype=torch.bool)
    allowed = allowed & valid[:, None, :]
    return torch.zeros(b, heads, n, n).masked_fill(~allowed[:, None], -torch.inf).flatten(0, 1)


@pytest.mark.parametrize('layers,heads,dim', [(1, 2, 8), (2, 2, 8), (3, 8, 128), (4, 4, 16)])
def test_shapes_depth_softmax_and_every_context_slot_preserved(layers, heads, dim):
    torch.manual_seed(11)
    module = router(layers, heads, dim)
    x = torch.randn(2, 4, 10*layers*heads)
    valid = torch.tensor([[True]*4, [True, True, False, False]])
    x[~valid] = float('nan')
    logits, info = module(x, valid, mask_for(valid, heads), return_context=True)
    assert logits.shape == (2, 4, heads, layers)
    assert info['graph_context_states'].shape == info['layer_context_states'].shape == (2, 4, layers, dim)
    assert torch.isfinite(logits).all() and torch.count_nonzero(logits[~valid]) == 0
    weights = logits[valid].softmax(-1)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
    if layers > 1:
        assert weights.std(-1).max() > 1e-5
    # The shared head MLP reads every depth token, without pooling/flattening depth into features.
    expected = module.head_projection(info['layer_context_states']).transpose(2, 3)
    torch.testing.assert_close(logits[valid], expected[valid])
    assert 'router_probs' not in info


def test_graph_mask_blocks_disconnected_context_and_preserves_batch_and_node_order():
    torch.manual_seed(7)
    module = router()
    x = torch.randn(2, 4, module.input_dim, requires_grad=True)
    valid = torch.tensor([[True]*4, [True, True, False, False]])
    # Test the router independently with two disconnected components. The real
    # model additionally has its native graph token, which is globally connected.
    allowed = torch.eye(4, dtype=torch.bool).expand(2, -1, -1).clone()
    allowed[:, :2, :2] = True
    allowed[:, 2:, 2:] = True
    mask = mask_for(valid, 2, allowed)
    logits, info = module(x, valid, mask, return_context=True)
    grad = torch.autograd.grad(logits[0, 0].square().sum(), x)[0]
    assert grad[0, 1].abs().sum() > 0  # Neighbor features affect this node's weights.
    assert torch.count_nonzero(grad[0, 2:]) == torch.count_nonzero(grad[1]) == 0
    for b in range(2):
        separate, _ = module(x[b:b+1], valid[b:b+1], mask[b*2:(b+1)*2])
        torch.testing.assert_close(separate, logits[b:b+1])
    order = torch.tensor([1, 0, 3, 2])
    permuted, _ = module(x[:, order], valid[:, order], mask[:, order][:, :, order])
    torch.testing.assert_close(permuted, logits[:, order])
    x2 = x.detach().clone()
    x2[~valid] = float('nan')
    padded, _ = module(x2, valid, mask)
    torch.testing.assert_close(padded, logits)


def test_layer_encoder_attends_only_across_depth_for_same_node():
    torch.manual_seed(17)
    module = router()
    states = torch.randn(2, 4, 3, 8, requires_grad=True)
    valid = torch.tensor([[True]*4, [True, True, False, False]])
    output = module.encode_layers(states, valid)
    grad = torch.autograd.grad(output[0, 1, 0, 0], states)[0]
    assert all(grad[0, 1, layer].abs().sum() > 0 for layer in range(3))
    assert torch.count_nonzero(grad[1]) == 0
    assert torch.count_nonzero(grad[0, [0, 2, 3]]) == 0
    for b in range(2):
        for node in range(4):
            separate = module.encode_layers(states[b:b+1, node:node+1], valid[b:b+1, node:node+1])
            torch.testing.assert_close(separate, output[b:b+1, node:node+1])


def test_native_mask_used_for_all_graph_blocks_and_score_path_reaches_pe_qk(inputs):
    cfg, prep, support, batch = inputs
    torch.manual_seed(17)
    model = LASDMoEModelPin(MODE, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
                           {'hidden_dim': 12, 'score_chunk_size': 3}).eval()
    masks = []
    handles = [block.register_forward_pre_hook(lambda _, args: masks.append(args[1]))
               for block in model.layer_fusion.graph_blocks]
    handles.append(model.final_graph_encoder.register_forward_pre_hook(lambda _, args: masks.append(args[1])))
    try:
        output, observed = model(batch, return_scores=True)
    finally:
        for handle in handles:
            handle.remove()
    assert len(masks) == cfg['cross_layers'] + 1
    assert all(m is masks[0] for m in masks)  # Includes native role bias, self, GRAPH and padding.
    assert not model.architecture_contract['score_detach']
    assert not model.architecture_contract['hidden_state_detach']
    # No hidden-state path in this loss: its Q/K gradient must come through the scores.
    gradients = torch.autograd.grad(observed['layer_weights'].square().sum(),
        (observed['scores'], model.encoder.fusion_blocks[0].self_attn.in_proj_weight), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    output.square().sum().backward()
    assert all(p.grad is None for p in model.predictor.parameters())


def test_contract_records_two_encoder_axes_and_rejects_changed_architecture():
    cfg = dict(dim=128, heads=8, encoder_layers=6, cross_layers=3, decoder_layers=2,
               ffn_dim=256, dropout=.17)
    spec = lasd_moe_spec_pin(MODE, {'hidden_dim': 32, 'global_std_epsilon': 3e-7}, cfg)
    assert spec['input_dim'] == 240 and spec['routing_output_dim'] == 24
    assert spec['experts'] == spec['top_k'] == 0 and spec['expert'] is None
    assert spec['graph_layer_context']['graph_encoder_layers'] == 3
    assert spec['graph_layer_context']['layer_encoder_layers'] == 1
    assert spec['graph_layer_context']['layer_attention_axes'] == 'B*N,H,L,L'
    validate_lasd_moe_contract_pin(MODE, spec)
    validate_lasd_moe_contract_pin(MODE, spec, cfg, {'hidden_dim': 32, 'global_std_epsilon': 3e-7})
    for key, value in [('layer_encoder_layers', 3), ('layer_attention_mask', 'causal'),
                       ('graph_encoder_layers', 9), ('softmax_axis', 'heads_H')]:
        bad = copy.deepcopy(spec)
        bad['graph_layer_context'][key] = value
        with pytest.raises(ValueError, match='architecture'):
            validate_lasd_moe_contract_pin(MODE, bad)
    for key, value in [('ffn_dim', None), ('ffn_dim', 0), ('block_dropout', float('nan')),
                       ('block_dropout', -1), ('block_dropout', True)]:
        bad = copy.deepcopy(spec)
        bad['graph_layer_context'][key] = value
        with pytest.raises(ValueError):
            validate_lasd_moe_contract_pin(MODE, bad)


def test_training_diagnostics_xlsx_standalone_replay_and_completed_resume(setup):
    root, cfg, data, config, run = setup
    cfg['lasd_moe_pin'] = dict(hidden_dim=12, score_chunk_size=3, diagnostic_samples=2)
    config.write_text(json.dumps(cfg))
    modes = ['device_op_cascade_pin', 'lasd_moe_headwise_global_pin', MODE]
    out, result = run('graph_layer_complete', modes)
    assert len(result['results']) == 3
    directory = out / f'seed_17_pin/{MODE}'
    model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
    assert saved['recipe']['architecture'] == model.architecture_contract
    report = json.loads((directory / 'test_pin.json').read_text())
    diagnostic = report['lasd_moe_diagnostics_pin']
    assert diagnostic['split'] == 'val' and 'experts' not in diagnostic
    assert not (directory / 'lasd_moe_experts_pin.csv').exists()
    for key in ('scores', 'weights', 'global_scores', 'metadata'):
        assert (directory / diagnostic[key]).is_file()
    metadata = json.loads((directory / diagnostic['metadata']).read_text())
    assert metadata['expert_summary'] is None
    with (directory / diagnostic['weights']).open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all('layer_logit' in row for row in rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['sample_id'], row['token_index'], row['head']), []).append(row)
    for values in groups.values():
        logits = torch.tensor([float(r['layer_logit']) for r in values], dtype=torch.float64)
        expected = torch.tensor([float(r['layer_weight']) for r in values], dtype=torch.float64)
        torch.testing.assert_close(logits.softmax(-1), expected)
    history = json.loads((out / f'seed_17_pin/stages_pin/{MODE}/history_pin.json').read_text())
    for record in history:
        layers = {v['layer']: v for v in record['encoder_gradients']['layers']}
        for name in ['SCORE_GRAPH1', 'SCORE_GRAPH2', 'SCORE_LAYER1', 'FINAL1']:
            assert layers[name]['mean_l2'] > 0
    sheets = workbook(out / 'summary_pin.xlsx')
    texts = text_values(sheets['LASD_MoE'])
    assert MODE in texts and 'graph_then_nodewise_layer_encoder' in texts
    assert {'Routing mechanism', 'Graph context blocks', 'Per-node layer context blocks'} <= set(texts)
    assert {'MAE', 'RMSE', 'R2', 'TrainingCost', 'Configuration'} <= set(sheets)
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retrained complete stage')):
        resumed, _ = run('graph_layer_completed_resume', [MODE], out)
    replay, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{MODE}/checkpoint_pin.pt')
    assert state_digest(model) == state_digest(replay)
