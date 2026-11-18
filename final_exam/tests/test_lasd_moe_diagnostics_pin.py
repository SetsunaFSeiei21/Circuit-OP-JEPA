"""Diagnostics preserve training state and exclude padding across graph batches."""
import copy
import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from final_exam.lasd_moe_diagnostics_pin import (
    EncoderGradientNormsPin, export_lasd_moe_diagnostics_pin,
)
from final_exam.models.attention_score_pin import score_from_attention_pin


def _encoder_model():
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.structure_encoder = nn.Module()
    model.encoder.structure_encoder.blocks = nn.ModuleList([nn.Linear(2, 2, bias=False)])
    model.encoder.fusion_blocks = nn.ModuleList([nn.Linear(2, 1, bias=False), nn.Linear(1, 1, bias=False)])
    model.final_graph_encoder = nn.Linear(1, 2, bias=False)
    return model


def test_gradients_are_preclip_per_block_and_epoch_bounded():
    model = _encoder_model()
    observer = EncoderGradientNormsPin(model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    originals = [parameter.grad.clone() for parameter in model.parameters()]
    observer(model, 1)
    for parameter, original in zip(model.parameters(), originals):
        torch.testing.assert_close(parameter.grad, original)
        parameter.grad.mul_(3)
    observer(model, 1)
    report = observer.epoch_record(1)
    assert report['batches'] == 2
    assert report['timing'] == 'after_backward_before_clipping'
    assert [row['layer'] for row in report['layers']] == ['SE1', 'PE1', 'PE2', 'FINAL1']
    for row in report['layers']:
        root_count = row['parameters']**.5
        assert row['mean_l2'] == pytest.approx(2*root_count)
        assert row['max_l2'] == pytest.approx(3*root_count)
        assert row['batches_with_grad'] == 2
    model.zero_grad(set_to_none=True)
    observer(model, 2)
    reset = observer.epoch_record(2)
    assert reset['batches'] == 1
    assert all(row['mean_l2'] == row['max_l2'] == row['batches_with_grad'] == 0
               for row in reset['layers'])
    with pytest.raises(ValueError, match='requested epoch'):
        observer.epoch_record(1)
    with pytest.raises(ValueError, match='different model'):
        observer(_encoder_model(), 3)


class _ObservedModel(nn.Module):
    mode = 'lasd_moe_pin'

    def __init__(self, limit=8):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.dropout = nn.Dropout(.5)
        self.lasd_moe_config = {'diagnostic_samples': limit}
        self.lasd_moe_spec = dict(layer_masks=['global', 'local'], experts=4, top_k=2, heads=2)
        valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
        logits = torch.arange(2*2*4*4, dtype=torch.float32).reshape(2, 2, 4, 4) / 31
        logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
        a = logits.softmax(-1)
        a2 = logits.flip(-2).softmax(-1)
        scores = torch.stack([score_from_attention_pin(a, valid),
                              score_from_attention_pin(a2, valid)], dim=2)
        differences = torch.stack([torch.zeros_like(scores[:, :, 0]),
                                   scores[:, :, 1] - scores[:, :, 0]], dim=2)
        layer_weights = torch.tensor([.2, .8]).expand(2, 4, 2).clone()
        layer_weights[1] = torch.tensor([.9, .1])
        probabilities = torch.tensor([.1, .2, .3, .4]).expand(2, 4, 4).clone()
        choices = torch.tensor([2, 3]).expand(2, 4, 2).clone()
        mixing = torch.tensor([3/7, 4/7]).expand(2, 4, 2).clone()
        # Invalid entries must never leak into statistics or serialization.
        for value in (scores, differences, layer_weights, probabilities, mixing):
            value[~valid] = float('nan')
        choices[~valid] = -1
        self.observed = dict(scores=scores, differences=differences, valid=valid,
                             layer_weights=layer_weights, router_probs=probabilities,
                             selected_experts=choices, selected_weights=mixing)

    def forward(self, batch, *, return_scores):
        assert return_scores and not self.training and not torch.is_grad_enabled()
        torch.rand(3)  # Deliberately consume RNG to test diagnostic isolation.
        return torch.zeros(2, 1), self.observed


class _Context:
    device = torch.device('cpu')

    def __init__(self):
        self.calls = []
        prefix_a, prefix_b = json.dumps(['C1', 'topology_a']), json.dumps(['C2', 'topology_b'])
        self.batch = dict(_sample_ids=['sample_a', 'sample_b'], _group_keys=[
            [prefix_a + ':' + name.encode().hex() for name in ['M1:D', 'M1:G', 'M1:S']],
            [prefix_b + ':' + 'M2:D'.encode().hex()],
        ])

    def batches(self, split, batch_size, seed=None, op=False):
        self.calls.append((split, batch_size, seed, op))
        yield self.batch


def test_exports_validate_only_and_preserve_rng_flags_and_padding(tmp_path):
    model, ctx = _ObservedModel(), _Context()
    model.train()
    model.dropout.eval()
    flags = [module.training for module in model.modules()]
    state = copy.deepcopy(model.state_dict())
    torch.manual_seed(123)
    rng = torch.get_rng_state().clone()
    result = export_lasd_moe_diagnostics_pin(ctx, model, {'batch_size': 16}, tmp_path)
    assert ctx.calls == [('val', 8, None, False)]
    assert result['split'] == 'val' and result['samples'] == 2
    assert flags == [module.training for module in model.modules()]
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    report = json.loads((tmp_path / result['metadata']).read_text())
    assert report['valid_token_counts'] == [4, 2]
    assert report['checkpoint_selection'] is False
    assert report['identities'] == [
        dict(sample_id='sample_a', circuit='C1', topology='topology_a'),
        dict(sample_id='sample_b', circuit='C2', topology='topology_b')]
    expert = report['expert_summary']
    assert expert['selection_counts'] == [0, 0, 6, 6]
    assert expert['selection_fraction_of_routes'] == [0, 0, .5, .5]
    assert sum(expert['mean_selected_mixing_weight']) == pytest.approx(1)
    layer = report['layer_weight_summary']
    assert layer['node_weighted_mean'] == pytest.approx([(4*.2+2*.9)/6, (4*.8+2*.1)/6])
    assert layer['sample_weighted_mean'] == pytest.approx([.55, .45])
    rows = list(csv.DictReader((tmp_path / result['scores']).open()))
    assert len(rows) == 6*2*2
    assert {row['token_id'] for row in rows} == {'GRAPH', 'M1:D', 'M1:G', 'M1:S', 'M2:D'}
    assert all(float(row['delta_receive']) == float(row['delta_transmit']) == 0
               for row in rows if row['pe_layer'] == '1')
    assert all(row['token_id'] in {'GRAPH', 'M2:D'} for row in rows if row['sample_id'] == 'sample_b')
    assert max(item['max_conservation_error'] for item in report['statistics']) < 1e-6
    assert len(list(csv.DictReader((tmp_path / result['weights']).open()))) == 6*2
    assert len(list(csv.DictReader((tmp_path / result['experts']).open()))) == 6*4


def test_sample_cap_and_disabled_diagnostics(tmp_path):
    model, ctx = _ObservedModel(limit=1), _Context()
    result = export_lasd_moe_diagnostics_pin(ctx, model, {'batch_size': 16}, tmp_path)
    report = json.loads((tmp_path / result['metadata']).read_text())
    assert report['sample_ids'] == ['sample_a'] and report['expert_summary']['valid_nodes'] == 4
    assert ctx.calls == [('val', 1, None, False)]
    disabled = tmp_path / 'disabled'
    assert export_lasd_moe_diagnostics_pin(_Context(), _ObservedModel(limit=0),
                                         {'batch_size': 16}, disabled) is None
    assert not disabled.exists()


def test_headwise_export_preserves_individual_pe_weights_and_node_route_counts(tmp_path):
    model = _ObservedModel()
    model.mode = 'lasd_moe_headwise_pin'
    model.lasd_moe_spec['layer_weight_axes'] = 'B,N,H,L'
    old = model.observed['layer_weights']
    model.observed['layer_weights'] = torch.stack((old, old.flip(-1)), dim=2)
    result = export_lasd_moe_diagnostics_pin(_Context(), model, {'batch_size': 16}, tmp_path)
    with (tmp_path / result['weights']).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 * 2 * 2
    for row in rows:
        b = {'sample_a': 0, 'sample_b': 1}[row['sample_id']]
        node, head, layer = int(row['token_index']), int(row['head']), int(row['pe_layer']) - 1
        assert model.observed['valid'][b, node]
        assert float(row['layer_weight']) == pytest.approx(model.observed['layer_weights'][b, node, head, layer].item())
    report = json.loads((tmp_path / result['metadata']).read_text())
    summary = report['layer_weight_summary']
    assert summary['per_head']['axes'] == 'H,L'
    assert summary['node_weighted_mean'] == pytest.approx([.5, .5])
    assert summary['per_head']['sample_weighted_mean'][0] == pytest.approx([.55, .45])
    assert summary['per_head']['node_weighted_mean'][1] == pytest.approx([(4*.8+2*.1)/6, (4*.2+2*.9)/6])
    assert report['expert_summary']['selection_counts'] == [0, 0, 6, 6]


def test_identity_misalignment_is_rejected_and_flags_restored(tmp_path):
    model, ctx = _ObservedModel(), _Context()
    ctx.batch['_group_keys'][0].pop()
    model.train()
    with pytest.raises(ValueError, match='identities'):
        export_lasd_moe_diagnostics_pin(ctx, model, {'batch_size': 16}, tmp_path)
    assert model.training


def test_mixed_topology_identity_is_rejected(tmp_path):
    model, ctx = _ObservedModel(), _Context()
    ctx.batch['_group_keys'][0][0] = json.dumps(['C3', 'other']) + ':' + 'M3:D'.encode().hex()
    with pytest.raises(ValueError, match='mixed circuit/topology'):
        export_lasd_moe_diagnostics_pin(ctx, model, {'batch_size': 16}, tmp_path)
