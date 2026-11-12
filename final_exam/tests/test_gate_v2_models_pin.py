"""Per-sample OP contrasts, exact v2 architecture and deployment boundaries."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from device_jepa.training import state_digest
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.gate_version_registry_pin import (
    V2_GATE_MODELS_PIN, V2_JOINT_GATE_MODELS_PIN, V2_STAGED_GATE_MODELS_PIN,
    gate_version_pin,
)
from final_exam.models.gates_pin.v2_pin import OPAttentionGateV2Pin
from final_exam.models.op_gate_pin import (
    DeviceOPGateModelPin, DeviceOPPredictorPin, validate_gate_config_pin,
)
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.staged_op_gate_pin import StagedOPGateModelPin
from final_exam.tests import test_op_gate_models_pin as _fixture_module
from final_exam.tests import test_fusion_staged_op_gate_pin as _scope_module


class GateV2ModelsPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = _fixture_module.OPAttentionModelsPinTests
        fixture.setUpClass()
        cls.cfg = {**fixture.cfg, 'encoder_layers': 6, 'cross_layers': 3}
        cls.prep, cls.support = fixture.prep, fixture.support
        cls.batch = copy.deepcopy(fixture.batch)

    def setUp(self):
        torch.manual_seed(17)

    def model(self, mode=V2_JOINT_GATE_MODELS_PIN[0], gate_config=None):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
        if mode in V2_STAGED_GATE_MODELS_PIN:
            return StagedOPGateModelPin(self.cfg, self.prep, predictor,
                mode=mode, gate_config=gate_config).set_phase(staged_gate_spec_pin(mode)['final_phase']).eval()
        return DeviceOPGateModelPin(mode, self.cfg, self.prep, predictor,
                                    gate_config=gate_config).eval()

    def test_relative_statistics_match_per_sample_per_field_all_mos_reference(self):
        gate = self.model().gate
        values = torch.tensor([
            [[1., 4., 9.], [2., 4., 8.], [3., 4., 7.], [9000., -9999., 5.]],
            [[10., -2., 1.], [30., 0., 3.], [20., 2., 5.], [-5000., 100., 9.]],
            [[4., 1., 5.], [8., 3., 7.], [12., 5., 9.], [0., 0., 0.]],
        ])
        supported = torch.ones_like(values, dtype=torch.bool)
        supported[0, 1:3, 2] = False  # only one eligible MOS supports this field
        relative = gate.relative_op(self.batch, values, supported)
        reference = torch.zeros_like(values)
        for b in range(values.shape[0]):
            for f in range(values.shape[2]):
                eligible = [d for d in range(3) if supported[b, d, f]]
                x = values[b, eligible, f].double()
                if len(eligible) >= 2 and x.max() > x.min():
                    reference[b, eligible, f] = ((x-x.mean()) /
                        torch.sqrt(x.var(unbiased=False)+gate.settings['norm_eps'])).float()
        torch.testing.assert_close(relative, reference, atol=1e-7, rtol=1e-6)
        self.assertEqual(float(relative[0, :, 1:].abs().sum()), 0.)
        # The PMOS at owner index1 participates in the SAME three-device mean.
        self.assertEqual(float(relative[0, 1, 0]), 0.)

    def test_relative_is_batch_independent_excludes_padding_passives_and_unsupported(self):
        gate = self.model().gate
        values, supported = self.model().predicted_devices(self.batch)
        first = gate.relative_op(self.batch, values, supported)
        changed = values.clone()
        changed[1:] *= 1000
        changed[:, 3] = float('nan')  # resistor cannot affect MOS statistics
        torch.testing.assert_close(first[0], gate.relative_op(self.batch, changed, supported)[0], atol=0, rtol=0)
        singleton = {k: v[:1] if isinstance(v, torch.Tensor) and v.shape[:1] == (3,) else v
                     for k, v in self.batch.items()}
        torch.testing.assert_close(first[:1], gate.relative_op(singleton, values[:1], supported[:1]), atol=0, rtol=0)
        fake = dict(self.batch, owner_mask=self.batch['owner_mask'].clone())
        fake['owner_mask'][:, 2] = False
        changed = values.clone()
        changed[:, 2:] = float('nan')
        clean = gate.relative_op(fake, values, supported)
        torch.testing.assert_close(clean, gate.relative_op(fake, changed, supported), atol=0, rtol=0)
        self.assertEqual(float(clean[:, 2:].abs().sum()), 0.)
        hidden = supported.clone()
        hidden[:, :, 0] = False
        changed = values.clone()
        changed[:, :, 0] = float('nan')
        self.assertTrue(bool(torch.isfinite(gate.relative_op(self.batch, changed, hidden)).all()))

    def test_zero_singleton_constant_and_extreme_finite_fields_are_safe(self):
        gate = self.model().gate
        values = torch.zeros(3, 4, 3)
        mask = torch.ones_like(values, dtype=torch.bool)
        torch.testing.assert_close(gate.relative_op(self.batch, values, mask), values)
        torch.testing.assert_close(gate.relative_op(self.batch, values, ~mask), values)
        values[:, :3, 0] = torch.tensor([-3e38, 0., 3e38])
        values[:, :3, 1] = torch.tensor([1., 1.0000001, 1.0000002])
        values[:, :3, 2] = 3e38
        relative = gate.relative_op(self.batch, values, mask)
        self.assertTrue(bool(torch.isfinite(relative).all()))
        self.assertEqual(float(relative[:, :, 2].abs().sum()), 0.)
        torch.testing.assert_close(relative[:, :3, 0],
            torch.tensor([-1.2247449, 0., 1.2247449]).expand(3, -1))
        singleton = mask.clone()
        singleton[:, 1:3] = False
        self.assertEqual(float(gate.relative_op(self.batch, values, singleton).abs().sum()), 0.)

    def test_exact_mlp_input_preserves_original_op_and_ordered_role_one_hots(self):
        model = self.model()
        values, supported = model.predicted_devices(self.batch)
        captured = []
        hook = model.gate.score_head.register_forward_pre_hook(lambda m, a: captured.append(a[0]))
        model.gate(self.batch, values, supported)
        hook.remove()
        fields = values.shape[-1]
        self.assertEqual(captured[0].shape, (*values.shape[:2], 4, 4, 2*fields+8))
        self.assertEqual(model.gate.score_head[0].in_features, 2*fields+8)
        self.assertEqual(model.gate.score_head[0].out_features, self.cfg['dim'])
        self.assertEqual(model.gate.score_head[-1].out_features, self.cfg['heads'])
        self.assertFalse(any(isinstance(m, (nn.Embedding, nn.BatchNorm1d, nn.LayerNorm)) for m in model.gate.modules()))
        safe = torch.where(supported, values, 0.)
        relative = model.gate.relative_op(self.batch, values, supported)
        for query in range(4):
            for key in range(4):
                pair = captured[0][:, :, query, key]
                torch.testing.assert_close(pair[..., :fields], safe, atol=0, rtol=0)
                torch.testing.assert_close(pair[..., fields:2*fields], relative, atol=0, rtol=0)
                torch.testing.assert_close(pair[..., 2*fields:2*fields+4], torch.eye(4)[query].expand(*values.shape[:2], 4))
                torch.testing.assert_close(pair[..., 2*fields+4:], torch.eye(4)[key].expand(*values.shape[:2], 4))

    def test_one_call_shared_by_correct_layers_and_physical_boundary_remains(self):
        for mode in V2_JOINT_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode)
                with patch.object(model.gate, 'forward', wraps=model.gate.forward) as gate_call:
                    before, structure_before, _ = _scope_module.FusionStagedGateModelsPinTests.capture_attention(model, self.batch)
                    self.assertEqual(gate_call.call_count, 1)
                with torch.no_grad():
                    model.gate.score_head[-1].bias.fill_(-2.)
                after, structure_after, _ = _scope_module.FusionStagedGateModelsPinTests.capture_attention(model, self.batch)
                changed = [name for name, value in before.items() if value is not None and not torch.equal(value, after[name])]
                self.assertEqual(len(changed), 3 if '_fusion_' in mode else 6)
                if '_fusion_' in mode:
                    torch.testing.assert_close(structure_before, structure_after, atol=0, rtol=0)
                values, supported = model.predicted_devices(self.batch)
                bias = model.gate(self.batch, values, supported)
                owner, nets = self.batch['owner_index'], self.batch['pin_net_index']
                protected = ((owner[:, :, None] != owner[:, None, :]) | (nets[:, :, None] == nets[:, None, :]) |
                             (owner[:, :, None] < 0) | (owner[:, None, :] < 0) |
                             (owner[:, :, None] == 3) | (owner[:, None, :] == 3))
                inner = bias[:, :, 1:, 1:]
                self.assertEqual(float(inner[protected[:, None].expand_as(inner)].detach().abs().sum()), 0.)
                self.assertEqual(float(bias[:, :, 0].detach().abs().sum()+bias[:, :, :, 0].detach().abs().sum()), 0.)

    def test_gradients_freezing_and_measured_op_independence_for_all_six_models(self):
        poison = copy.deepcopy(self.batch)
        for key in ('op', 'owner_op', 'performance'):
            poison[key].fill_(float('nan'))
        for key in ('op_mask', 'owner_op_mask', 'op_target_mask', 'owner_op_target_mask'):
            poison[key] = ~poison[key]
        for mode in V2_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode)
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), model(poison), atol=0, rtol=0)
                model.train()(self.batch).square().mean().backward()
                self.assertTrue(all(p.grad is None for p in model.predictor.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.gate.parameters() if p.grad is not None), 0.)
                self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.gate.parameters() if p.grad is not None))
                for name in ('encoder', 'decoder'):
                    grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                    if mode in V2_STAGED_GATE_MODELS_PIN and (name == 'encoder' or model.phase == 'gate_only'):
                        self.assertFalse(grads)
                    else:
                        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_recipes_save_normalization_and_reject_version_scope_or_settings_mismatch(self):
        for mode in V2_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode, {'norm_eps': .01})
                recipe = describe_model_pin(model)
                self.assertEqual(recipe['gate_version'], 'v2')
                self.assertEqual(recipe['gate_config']['norm_eps'], .01)
                self.assertEqual(recipe['gate_normalization'], model.gate.normalization_contract)
                rebuilt = rebuild_model_pin(recipe, self.cfg, self.prep)
                rebuilt.load_state_dict(model.state_dict())
                rebuilt.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), rebuilt(self.batch), atol=0, rtol=0)
                for key in ('gate_version', 'attention_scope', 'gate_normalization'):
                    wrong = copy.deepcopy(recipe)
                    wrong.pop(key)
                    with self.assertRaises(ValueError):
                        rebuild_model_pin(wrong, self.cfg, self.prep)
                wrong = copy.deepcopy(recipe)
                wrong['gate_config']['norm_eps'] = .1
                with self.assertRaisesRegex(ValueError, 'normalization'):
                    rebuild_model_pin(wrong, self.cfg, self.prep)

    def test_copying_legacy_base_preserves_weights_and_uses_explicit_norm_epsilon(self):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
        base = StagedOPGateModelPin(self.cfg, self.prep, predictor,
            mode='device_op_distill_gate_pin').set_phase('readout').eval()
        before = {name: state_digest(getattr(base, name)) for name in ('predictor', 'encoder', 'decoder')}
        copies = []
        for mode in V2_STAGED_GATE_MODELS_PIN[:2]:
            item = copy.deepcopy(base)
            torch.manual_seed(29)
            item.set_mode_pin(mode, gate_config={'norm_eps': .02})
            self.assertIsInstance(item.gate, OPAttentionGateV2Pin)
            self.assertEqual(item.gate_config['norm_eps'], .02)
            self.assertEqual(before, {name: state_digest(getattr(item, name)) for name in before})
            self.assertFalse(any(p.requires_grad for p in item.gate.parameters()))
            with patch.object(item.gate, 'forward', side_effect=AssertionError('ungated base invoked gate')):
                with torch.no_grad():
                    torch.testing.assert_close(base(self.batch), item(self.batch), atol=0, rtol=0)
            copies.append(item)
        self.assertEqual(state_digest(copies[0].gate), state_digest(copies[1].gate))

    def test_norm_epsilon_validation_and_legacy_defaults(self):
        self.assertNotIn('norm_eps', validate_gate_config_pin())
        self.assertEqual(validate_gate_config_pin(version='v2')['norm_eps'], 1e-5)
        for value in (0, -1, True, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_gate_config_pin({'norm_eps': value})
        with self.assertRaises(ValueError):
            gate_version_pin('unregistered_pin_v2')


if __name__ == '__main__':
    unittest.main()
