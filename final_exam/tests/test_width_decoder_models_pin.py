"""Multi-PE readout shapes, real gradients, physical owner alignment and replay."""
import copy

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.op_gate_data_pin import device_input_pin
from final_exam.models.op_gate_pin import DeviceOPGateModelPin, DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin, op_objective_pin
from final_exam.models.width_decoder_pin import (
    WidthDecoderModelPin, WidthDeviceOPPredictorPin, MultiPEReadoutPin, DenseMoEMLPPin,
)
from final_exam.tests.test_op_gate_models_pin import OPAttentionModelsPinTests as Fixtures
from final_exam.width_registry_pin import width_settings_pin, validate_width_contract_pin


@pytest.fixture(scope='module')
def inputs():
    Fixtures.setUpClass()
    return Fixtures.cfg, Fixtures.prep, Fixtures.support, Fixtures.batch


@pytest.mark.parametrize('layers', [1, 2, 3])
def test_taps_preserve_old_encoder_and_run_every_block_once(inputs, layers):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'cross_layers': layers, 'encoder_layers': layers + 2}
    predictor = DeviceOPPredictorPin(cfg, prep, support).eval()
    encoder = predictor.encoder
    device_batch = device_input_pin(batch)
    counts = [0] * layers
    handles = []
    for index, block in enumerate(encoder.fusion_blocks):
        def observe(module, args, output, index=index):
            counts[index] += 1
        handles.append(block.register_forward_hook(observe))
    memories, padding = encoder.forward_fusion_memories(device_batch)
    for handle in handles:
        handle.remove()
    assert counts == [1] * layers
    original, original_padding = encoder(device_batch)
    assert len(memories) == layers
    torch.testing.assert_close(memories[-1], original, atol=0, rtol=0)
    assert torch.equal(padding, original_padding)
    assert all(torch.count_nonzero(memory[padding]) == 0 for memory in memories)


def test_moe_matches_explicit_full_expert_mlps(inputs):
    torch.manual_seed(42)
    model = DenseMoEMLPPin(12, 3, dict(n_experts=4, hidden_dim=8, dropout=0.))
    values = torch.randn(2, 5, 12)
    routing = model.router(values).softmax(-1)
    expected = sum((model.activation(values @ model.weight1[e] + model.bias1[e]) @
                    model.weight2[e] + model.bias2[e]) * routing[..., e, None] for e in range(4))
    torch.testing.assert_close(model(values), expected)
    model(values).square().sum().backward()
    for parameter in (model.router.weight, model.weight1, model.weight2):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('wide_op', [False, True])
def test_all_branches_train_but_frozen_predictor_does_not_and_no_label_leak(inputs, wide_op):
    cfg, prep, support, batch = inputs
    torch.manual_seed(17)
    cls = WidthDeviceOPPredictorPin if wide_op else DeviceOPPredictorPin
    predictor = cls(cfg, prep, support)
    before = state_digest(predictor)
    mode = 'width_decoder_op_pin' if wide_op else 'width_decoder_pin'
    model = WidthDecoderModelPin(mode, cfg, prep, predictor, {'n_experts': 2}).train()
    assert not predictor.training
    assert not any(p.requires_grad for p in predictor.parameters())
    outputs = model(batch)
    assert outputs.shape == batch['performance'].shape
    outputs.square().mean().backward()
    assert all(p.grad is None for p in predictor.parameters())
    for branch in model.decoder.branches:
        grad = branch.decoder.layers[0].multihead_attn.in_proj_weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert state_digest(predictor) == before
    assert model.gate is None
    a, b = model.decoder.branches
    assert a.query.data_ptr() != b.query.data_ptr()
    assert a.decoder.layers[0].linear1.weight.data_ptr() != b.decoder.layers[0].linear1.weight.data_ptr()
    model.eval()
    poison = copy.deepcopy(batch)
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        if key in poison:
            poison[key].fill_(float('nan'))
    for key in ('op_mask', 'owner_op_mask', 'op_target_mask', 'owner_op_target_mask'):
        poison[key] = ~poison[key]
    torch.testing.assert_close(model(poison), model(batch), atol=0, rtol=0)
    validate_width_contract_pin(mode, model.architecture_contract)
    recipe = describe_model_pin(model)
    restored = rebuild_model_pin(recipe, cfg, prep).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(batch), restored(batch), atol=0, rtol=0)
    assert not restored.predictor.training
    bad = copy.deepcopy(recipe)
    bad['architecture']['performance']['pe_order'].reverse()
    with pytest.raises(ValueError, match='architecture'):
        rebuild_model_pin(bad, cfg, prep)


def test_original_and_width_model_encoders_start_identically(inputs):
    cfg, prep, support, _ = inputs
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    torch.manual_seed(29)
    original = DeviceOPGateModelPin('device_op_cascade_pin', cfg, prep, predictor)
    torch.manual_seed(29)
    wide = WidthDecoderModelPin('width_decoder_pin', cfg, prep, predictor)
    assert state_digest(original.encoder) == state_digest(wide.encoder)


def test_wide_op_is_device_aligned_supported_and_trained_without_pin_targets(inputs):
    cfg, prep, support, batch = inputs
    model = WidthDeviceOPPredictorPin(cfg, prep, support, {'n_experts': 2}).eval()
    values, supported = model.predict_devices(batch)
    assert values.shape == batch['owner_op'].shape
    assert values.shape[1] < batch['device_mask'].shape[1]
    assert values.shape[-1] == len(prep['feature_spec']['op_names'])
    assert torch.count_nonzero(values[~supported]) == 0
    poison = copy.deepcopy(batch)
    poison['parameters'].fill_(float('nan'))  # canonical owner parameters remain intact
    poison['op'].fill_(float('nan'))
    poison['owner_op'].fill_(float('nan'))
    torch.testing.assert_close(values, model.predict_devices(poison)[0], atol=0, rtol=0)
    loss, count, (_, field_count) = op_objective_pin(model, batch)
    expected = supported & batch['owner_op_mask'] & batch['owner_op_target_mask'].unsqueeze(-1)
    assert count == int(expected.sum())
    torch.testing.assert_close(field_count, expected.sum((0, 1)))
    loss.backward()
    for branch in model.head.branches:
        assert branch.query is None
        assert branch.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0
    replay = rebuild_model_pin(describe_model_pin(model), cfg, prep).eval()
    replay.load_state_dict(model.state_dict())
    torch.testing.assert_close(values, replay.predict_devices(batch)[0], atol=0, rtol=0)


def test_concatenation_keeps_metric_axis_and_pe_order(inputs):
    cfg, _, _, _ = inputs
    model = MultiPEReadoutPin(cfg, {'n_experts': 2}, output_dim=1, metric_count=3)
    class Marker(torch.nn.Module):
        def __init__(self, i):
            super().__init__(); self.i = i
        def forward(self, memory, padding):
            return torch.arange(3.)[None, :, None].expand(2, 3, cfg['dim']) + 10 * self.i
    model.branches = torch.nn.ModuleList([Marker(1), Marker(2)])
    value = model.features((torch.zeros(2, 5, 8),) * 2, torch.zeros(2, 5, dtype=torch.bool))
    assert value.shape == (2, 3, 16)
    assert torch.equal(value[0, :, 0], torch.tensor([10., 11., 12.]))
    assert torch.equal(value[0, :, 8], torch.tensor([20., 21., 22.]))


@pytest.mark.parametrize('settings', [{'n_experts': 0}, {'n_experts': True}, {'hidden_dim': .5},
                                      {'dropout': float('nan')}, {'dropout': 1}, {'junk': 1}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        width_settings_pin(settings)
