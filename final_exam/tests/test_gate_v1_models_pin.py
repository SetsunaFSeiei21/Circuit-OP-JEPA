"""v1 concatenation contract, physical boundaries, sharing and checkpoint identity."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from device_jepa.training import state_digest
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.gate_version_registry_pin import (
    V1_GATE_MODELS_PIN, V1_JOINT_GATE_MODELS_PIN, V1_STAGED_GATE_MODELS_PIN,
)
from final_exam.models.gates_pin.v1_pin import OPAttentionGateV1Pin
from final_exam.models.op_gate_pin import DeviceOPGateModelPin, DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.staged_op_gate_pin import StagedOPGateModelPin
from final_exam.tests import test_op_gate_models_pin as _fixture_module
from final_exam.tests import test_fusion_staged_op_gate_pin as _scope_module


class GateV1ModelsPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = _fixture_module.OPAttentionModelsPinTests
        fixture.setUpClass()
        cls.cfg = {**fixture.cfg, 'encoder_layers': 6, 'cross_layers': 3}
        cls.prep, cls.support = fixture.prep, fixture.support
        cls.batch = copy.deepcopy(fixture.batch)

    def setUp(self):
        torch.manual_seed(17)

    def model(self, mode=V1_JOINT_GATE_MODELS_PIN[0]):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
        if mode in V1_STAGED_GATE_MODELS_PIN:
            return StagedOPGateModelPin(self.cfg, self.prep, predictor,
                mode=mode).set_phase(staged_gate_spec_pin(mode)['final_phase']).eval()
        return DeviceOPGateModelPin(mode, self.cfg, self.prep, predictor).eval()

    def test_exact_concatenation_has_no_mask_or_learned_role_embeddings(self):
        model = self.model()
        values, supported = model.predicted_devices(self.batch)
        seen = []
        handle = model.gate.score_head.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach()))
        model.gate(self.batch, values, supported)
        handle.remove()
        fields = values.shape[-1]
        self.assertEqual(seen[0].shape, (*values.shape[:2], 4, 4, fields + 8))
        self.assertEqual(model.gate.score_head[0].in_features, fields + 8)
        self.assertEqual(model.gate.score_head[0].out_features, self.cfg['dim'])
        self.assertEqual(model.gate.score_head[-1].out_features, self.cfg['heads'])
        self.assertFalse(any(isinstance(m, nn.Embedding) for m in model.gate.modules()))
        self.assertFalse(hasattr(model.gate, 'op_embedding'))
        safe = torch.where(supported, values, 0.)
        for query in range(4):
            for key in range(4):
                pair = seen[0][:, :, query, key]
                torch.testing.assert_close(pair[..., :fields], safe)
                torch.testing.assert_close(pair[..., fields:fields + 4],
                    torch.eye(4)[query].expand(*values.shape[:2], 4))
                torch.testing.assert_close(pair[..., fields + 4:],
                    torch.eye(4)[key].expand(*values.shape[:2], 4))

    def test_support_sanitizes_nan_but_is_not_an_extra_gate_feature(self):
        model = self.model()
        values, supported = model.predicted_devices(self.batch)
        self.assertGreater(values.shape[-1], 1)
        values = values.clone()
        values[..., 0] = 0.
        first = model.gate(self.batch, values, supported)
        hidden = supported.clone()
        hidden[..., 0] = False
        # A zero input field has the same MLP representation regardless of its
        # support bit, provided remaining fields keep this MOS eligible.
        second = model.gate(self.batch, values, hidden)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        values[..., 0] = float('nan')
        torch.testing.assert_close(second, model.gate(self.batch, values, hidden), atol=0, rtol=0)

    def test_v1_still_excludes_rc_other_devices_ports_and_real_shorts(self):
        model = self.model()
        values, supported = model.predicted_devices(self.batch)
        bias = model.gate(self.batch, values, supported)
        owner, nets = self.batch['owner_index'], self.batch['pin_net_index']
        protected = ((owner[:, :, None] != owner[:, None, :]) |
                     (nets[:, :, None] == nets[:, None, :]) |
                     (owner[:, :, None] < 0) | (owner[:, None, :] < 0) |
                     (owner[:, :, None] == 3) | (owner[:, None, :] == 3))
        inner = bias[:, :, 1:, 1:]
        self.assertEqual(float(inner[protected[:, None].expand_as(inner)].detach().abs().sum()), 0.)
        self.assertEqual(float(bias[:, :, 0].detach().abs().sum() + bias[:, :, :, 0].detach().abs().sum()), 0.)
        self.assertTrue(bool(torch.isfinite(bias).all()))
        self.assertTrue(bool((bias <= 0).all()))
        self.assertTrue(bool((bias < 0).any()))
        # Even a false claim of supported MOS-like fields cannot gate R/C.
        forged = torch.ones_like(supported)
        forged_bias = model.gate(self.batch, torch.zeros_like(values), forged)[:, :, 1:, 1:]
        passive = (owner[:, :, None] == 3) | (owner[:, None, :] == 3)
        self.assertEqual(float(forged_bias[passive[:, None].expand_as(forged_bias)].detach().abs().sum()), 0.)

    def test_one_gate_call_and_scope_for_both_joint_variants(self):
        for mode in V1_JOINT_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode)
                with patch.object(model.gate, 'forward', wraps=model.gate.forward) as gate_call:
                    before, structure_before, _ = _scope_module.FusionStagedGateModelsPinTests.capture_attention(model, self.batch)
                    self.assertEqual(gate_call.call_count, 1)
                with torch.no_grad():
                    model.gate.score_head[-1].bias.fill_(-2.)
                after, structure_after, _ = _scope_module.FusionStagedGateModelsPinTests.capture_attention(model, self.batch)
                changed = []
                for name, value in before.items():
                    if value is not None and not torch.equal(value, after[name]):
                        changed.append(name)
                expected = 3 if '_fusion_' in mode else 6
                self.assertEqual(len(changed), expected)
                self.assertTrue(all(name.startswith('encoder.') and not name.endswith('cross_attn') for name in changed))
                if '_fusion_' in mode:
                    torch.testing.assert_close(structure_before, structure_after, atol=0, rtol=0)
                    self.assertTrue(all(name.startswith('encoder.fusion_blocks.') for name in changed))
                else:
                    self.assertFalse(torch.equal(structure_before, structure_after))

    def test_joint_gradients_and_gate_only_gradients(self):
        for mode in V1_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode).train()
                model(self.batch).square().mean().backward()
                self.assertFalse(model.predictor.training)
                self.assertTrue(all(p.grad is None for p in model.predictor.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.gate.parameters() if p.grad is not None), 0.)
                self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.gate.parameters() if p.grad is not None))
                for name in ('encoder', 'decoder'):
                    grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                    if mode in V1_STAGED_GATE_MODELS_PIN and (name == 'encoder' or model.phase == 'gate_only'):
                        self.assertFalse(grads)
                        self.assertFalse(getattr(model, name).training)
                    else:
                        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_versioned_recipes_roundtrip_and_reject_missing_or_wrong_version(self):
        for mode in V1_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode)
                recipe = describe_model_pin(model)
                self.assertEqual(recipe['gate_version'], 'v1')
                rebuilt = rebuild_model_pin(recipe, self.cfg, self.prep)
                rebuilt.load_state_dict(model.state_dict())
                rebuilt.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), rebuilt(self.batch), atol=0, rtol=0)
                for replacement in (None, 'v0', 'v2'):
                    wrong = copy.deepcopy(recipe)
                    if replacement is None:
                        wrong.pop('gate_version')
                    else:
                        wrong['gate_version'] = replacement
                    with self.assertRaises(ValueError):
                        rebuild_model_pin(wrong, self.cfg, self.prep)

    def test_copying_ungated_legacy_base_replaces_only_gate_and_preserves_phase(self):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
        base = StagedOPGateModelPin(self.cfg, self.prep, predictor,
                                   mode='device_op_distill_gate_pin').set_phase('readout').eval()
        before = {name: state_digest(getattr(base, name)) for name in ('predictor', 'encoder', 'decoder')}
        copies = []
        for mode in V1_STAGED_GATE_MODELS_PIN[:2]:
            item = copy.deepcopy(base)
            torch.manual_seed(29)
            item.set_mode_pin(mode)
            self.assertIsInstance(item.gate, OPAttentionGateV1Pin)
            self.assertTrue(all(not p.requires_grad for p in item.gate.parameters()))
            self.assertTrue(all(p.requires_grad for p in item.decoder.parameters()))
            self.assertEqual(item.phase, 'readout')
            self.assertEqual(before, {name: state_digest(getattr(item, name)) for name in before})
            with patch.object(item.gate, 'forward', side_effect=AssertionError('ungated base invoked gate')):
                with torch.no_grad():
                    torch.testing.assert_close(base(self.batch), item(self.batch), atol=0, rtol=0)
            copies.append(item)
        self.assertEqual(state_digest(copies[0].gate), state_digest(copies[1].gate))
        copies[0].set_phase('gate_only')
        with self.assertRaises(ValueError):
            copies[0].set_mode_pin('device_op_distill_gate_pin')


if __name__ == '__main__':
    unittest.main()
