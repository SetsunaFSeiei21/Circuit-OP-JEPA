"""Dense expert routing, gate boundaries, stage isolation and v3 recipes."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from device_jepa.training import state_digest
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.gate_version_registry_pin import (
    V3_GATE_MODELS_PIN, V3_JOINT_GATE_MODELS_PIN, V3_STAGED_GATE_MODELS_PIN,
    gate_version_pin,
)
from final_exam.models.gates_pin.v3_pin import MoLinearPin, OPAttentionGateV3Pin
from final_exam.models.op_gate_pin import (
    DeviceOPGateModelPin, DeviceOPPredictorPin, gate_class_pin, validate_gate_config_pin,
)
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.staged_op_gate_pin import StagedOPGateModelPin
from final_exam.tests import test_op_gate_models_pin as fixtures
from final_exam.tests import test_fusion_staged_op_gate_pin as scope_fixtures


class MoLinearPinTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_dense_mixture_and_all_derivatives_match_independent_linear_reference(self):
        layer = MoLinearPin(5, 3, 4).double()
        reference = copy.deepcopy(layer)
        x = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        actual = layer(x)
        # Independent expression using E ordinary affine layers; includes
        # input-dependent soft routing and does not reuse the einsum path.
        route = F.softmax(F.linear(xr, reference.router.weight, reference.router.bias), dim=-1)
        expected = sum(route[..., e:e+1] * F.linear(xr, reference.weight[e].T, reference.bias[e])
                       for e in range(reference.n_experts))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        grad = torch.autograd.grad(actual.square().sum(), (x, *layer.parameters()))
        grad_ref = torch.autograd.grad(expected.square().sum(), (xr, *reference.parameters()))
        for value, other in zip(grad, grad_ref):
            torch.testing.assert_close(value, other, atol=1e-11, rtol=1e-11)
            self.assertGreater(float(value.abs().sum()), 0.)
        self.assertTrue(bool((grad[-1].abs().sum(-1) > 0).all()))

    def test_one_expert_reduces_to_affine_and_leading_positions_do_not_mix(self):
        layer = MoLinearPin(5, 3, 1)
        x = torch.randn(3, 4, 5)
        torch.testing.assert_close(layer(x), F.linear(x, layer.weight[0].T, layer.bias[0]))
        layer = MoLinearPin(5, 3, 4)
        baseline = layer(x)
        changed = x.clone()
        changed[1:] *= 100
        torch.testing.assert_close(baseline[0], layer(changed)[0], atol=0, rtol=0)
        with torch.no_grad():
            layer.router.weight.zero_()
            layer.router.bias.copy_(torch.tensor([-2., -1., 0., 2.]))
        route = layer.router(x).softmax(-1)
        self.assertTrue(bool((route > 0).all()))
        self.assertFalse(torch.equal(route[..., 0], route[..., 3]))

    def test_cpu_autocast_forward_and_backward_are_finite(self):
        layer = MoLinearPin(5, 3, 4)
        x = torch.randn(2, 4, 5, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = layer(x)
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(output).all()))
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_autocast_and_device_consistency(self):
        layer = MoLinearPin(5, 3, 4).cuda()
        x = torch.randn(2, 4, 5, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.float16):
            output = layer(x)
        output.float().square().mean().backward()
        self.assertEqual(output.device, x.device)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in layer.parameters()))


class GateV3ModelsPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OPAttentionModelsPinTests.setUpClass()
        source = fixtures.OPAttentionModelsPinTests
        cls.cfg = {**source.cfg, 'encoder_layers': 6, 'cross_layers': 3}
        cls.prep, cls.support, cls.batch = source.prep, source.support, copy.deepcopy(source.batch)

    def setUp(self):
        torch.manual_seed(17)

    def model(self, mode=V3_JOINT_GATE_MODELS_PIN[0], gate_config=None):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
        config = {'n_experts': 3, 'mlp_dropout': .1, **(gate_config or {})}
        if mode in V3_STAGED_GATE_MODELS_PIN:
            return StagedOPGateModelPin(self.cfg, self.prep, predictor,
                mode=mode, gate_config=config).set_phase(staged_gate_spec_pin(mode)['final_phase']).eval()
        return DeviceOPGateModelPin(mode, self.cfg, self.prep, predictor,
                                   gate_config=config).eval()

    def test_v2_input_and_statistics_preserved_with_three_independent_mixture_layers(self):
        model = self.model()
        gate = model.gate
        values, support = model.predicted_devices(self.batch)
        capture = []
        handle = gate.score_head.register_forward_pre_hook(lambda m, a: capture.append(a[0].detach()))
        gate(self.batch, values, support)
        handle.remove()
        fields = values.shape[-1]
        self.assertEqual(capture[0].shape, (*values.shape[:2], 4, 4, 2*fields+8))
        relative = gate.relative_op(self.batch, values, support)
        for q in range(4):
            for k in range(4):
                pairs = capture[0][:, :, q, k]
                torch.testing.assert_close(pairs[..., :fields], torch.where(support, values, 0.))
                torch.testing.assert_close(pairs[..., fields:2*fields], relative)
                torch.testing.assert_close(pairs[..., 2*fields:2*fields+4], torch.eye(4)[q].expand(*values.shape[:2], 4))
                torch.testing.assert_close(pairs[..., 2*fields+4:], torch.eye(4)[k].expand(*values.shape[:2], 4))
        self.assertEqual([type(x) for x in gate.score_head],
                         [MoLinearPin, nn.Dropout, nn.GELU, MoLinearPin, nn.Dropout, nn.GELU, MoLinearPin])
        mixtures = [x for x in gate.score_head if isinstance(x, MoLinearPin)]
        self.assertEqual(len({id(layer.router) for layer in mixtures}), 3)
        self.assertEqual(gate.architecture_contract['layer_dims'], [2*fields+8, 8, 8, 2])
        self.assertFalse(any(isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.Embedding)) for m in gate.modules()))
        self.assertEqual(gate.normalization_contract, gate_class_pin('v2')(self.cfg, self.prep).normalization_contract)

    def test_initialization_is_near_open_but_router_and_op_derivatives_are_nonzero(self):
        gate = self.model().gate
        x = torch.randn(2, 4, 4, 4, gate.input_dim, requires_grad=True)
        logits = gate.score_head(x)
        scores = logits.sigmoid()
        torch.testing.assert_close(scores, torch.full_like(scores, .99), atol=1e-3, rtol=0.)
        self.assertGreater(float((scores.max()-scores.min()).detach()), 0.)
        self.assertGreater(float((scores[:, :, 0, 2]-scores[:, :, 2, 0]).detach().abs().max()), 0.)
        logits.square().mean().backward()
        self.assertGreater(float(x.grad[..., :2*len(gate.field_names)].abs().sum()), 0.)
        for layer in gate.score_head:
            if isinstance(layer, MoLinearPin):
                for parameter in layer.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_one_gate_call_reused_by_selected_attention_layers_and_boundary_protected(self):
        for mode in V3_JOINT_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode)
                with patch.object(model.gate, 'forward', wraps=model.gate.forward) as call:
                    before, structure, _ = scope_fixtures.FusionStagedGateModelsPinTests.capture_attention(model, self.batch)
                    self.assertEqual(call.call_count, 1)
                with torch.no_grad():
                    model.gate.score_head[-1].bias.fill_(-2.)
                after, structure_after, _ = scope_fixtures.FusionStagedGateModelsPinTests.capture_attention(model, self.batch)
                changed = [key for key, value in before.items() if value is not None and not torch.equal(value, after[key])]
                self.assertEqual(len(changed), 3 if '_fusion_' in mode else 6)
                if '_fusion_' in mode:
                    torch.testing.assert_close(structure, structure_after, atol=0, rtol=0)
                values, support = model.predicted_devices(self.batch)
                bias = model.gate(self.batch, values, support)
                owners, roles, eligible = model.gate._layout(self.batch, values, support)
                self.assertEqual(float(bias[:, :, 1:, 1:][(~eligible[:, None]).expand_as(bias[:, :, 1:, 1:])].detach().abs().sum()), 0.)
                self.assertEqual(float(bias[:, :, 0].detach().abs().sum()+bias[:, :, :, 0].detach().abs().sum()), 0.)
                self.assertTrue(bool((bias <= 0).all()))
                self.assertTrue(bool((bias < 0).any()))

    def test_all_six_modes_freeze_predictor_and_staged_bases_and_ignore_measured_op(self):
        poison = copy.deepcopy(self.batch)
        for key in ('op', 'owner_op', 'performance'):
            poison[key].fill_(float('nan'))
        for key in ('op_mask', 'owner_op_mask', 'op_target_mask', 'owner_op_target_mask'):
            poison[key] = ~poison[key]
        for mode in V3_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode)
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), model(poison), atol=0, rtol=0)
                model.train()(self.batch).square().mean().backward()
                self.assertFalse(model.predictor.training)
                self.assertTrue(all(p.grad is None for p in model.predictor.parameters()))
                for layer in model.gate.score_head:
                    if isinstance(layer, MoLinearPin):
                        self.assertGreater(float(layer.router.weight.grad.abs().sum()), 0.)
                        self.assertGreater(float(layer.weight.grad.abs().sum()), 0.)
                for key in ('encoder', 'decoder'):
                    grads = [p.grad for p in getattr(model, key).parameters() if p.grad is not None]
                    if mode in V3_STAGED_GATE_MODELS_PIN and (key == 'encoder' or model.phase == 'gate_only'):
                        self.assertFalse(grads)
                        self.assertFalse(getattr(model, key).training)
                    else:
                        self.assertGreater(sum(float(x.abs().sum()) for x in grads), 0.)

    def test_recipe_roundtrip_and_incompatible_architectures_rejected(self):
        for mode in V3_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode, {'norm_eps': .02})
                recipe = describe_model_pin(model)
                self.assertEqual(recipe['gate_architecture'], model.gate.architecture_contract)
                self.assertEqual(recipe['gate_normalization'], model.gate.normalization_contract)
                restored = rebuild_model_pin(recipe, self.cfg, self.prep)
                restored.load_state_dict(model.state_dict())
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), restored(self.batch), atol=0, rtol=0)
                for key in ('gate_version', 'attention_scope', 'gate_architecture', 'gate_normalization'):
                    wrong = copy.deepcopy(recipe)
                    wrong.pop(key)
                    with self.assertRaises(ValueError):
                        rebuild_model_pin(wrong, self.cfg, self.prep)
                for key, value in (('n_experts', 4), ('mlp_dropout', .2), ('norm_eps', .03)):
                    wrong = copy.deepcopy(recipe)
                    wrong['gate_config'][key] = value
                    with self.assertRaises(ValueError):
                        rebuild_model_pin(wrong, self.cfg, self.prep)

    def test_copy_ungated_base_leaves_downstream_identical_and_copies_settings(self):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
        base = StagedOPGateModelPin(self.cfg, self.prep, predictor,
            mode='device_op_distill_gate_pin').set_phase('readout').eval()
        before = {name: state_digest(getattr(base, name)) for name in ('predictor', 'encoder', 'decoder')}
        copies = []
        for mode in V3_STAGED_GATE_MODELS_PIN[:2]:
            item = copy.deepcopy(base)
            torch.manual_seed(29)
            item.set_mode_pin(mode, gate_config={'norm_eps': .02, 'n_experts': 2, 'mlp_dropout': .15})
            self.assertIsInstance(item.gate, OPAttentionGateV3Pin)
            self.assertEqual(item.gate_config['n_experts'], 2)
            self.assertEqual(item.gate.settings['mlp_dropout'], .15)
            self.assertEqual(before, {name: state_digest(getattr(item, name)) for name in before})
            self.assertFalse(any(p.requires_grad for p in item.gate.parameters()))
            with patch.object(item.gate, 'forward', side_effect=AssertionError('ungated base used gate')):
                with torch.no_grad():
                    torch.testing.assert_close(base(self.batch), item(self.batch), atol=0, rtol=0)
            copies.append(item)
        self.assertEqual(state_digest(copies[0].gate), state_digest(copies[1].gate))

    def test_settings_defaults_and_explicit_invalid_values(self):
        self.assertEqual(validate_gate_config_pin(), dict(epsilon=1e-4, strength=1., initial_gate=.99))
        for version in ('v0', 'v1', 'v2'):
            self.assertNotIn('n_experts', validate_gate_config_pin(version=version))
            self.assertNotIn('mlp_dropout', validate_gate_config_pin(version=version))
        settings = validate_gate_config_pin(version='v3')
        self.assertIs(type(settings['n_experts']), int)
        self.assertEqual(settings['n_experts'], 16)
        self.assertEqual(settings['norm_eps'], 1e-5)
        self.assertEqual(settings['mlp_dropout'], .1)
        for key, invalid in (('n_experts', (0, -1, 2., True, None, '4')),
                             ('mlp_dropout', (-.1, 1., True, float('inf'), float('nan')))):
            for value in invalid:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_gate_config_pin({key: value}, version='v3')
        with self.assertRaises(ValueError):
            gate_version_pin('unregistered_gate_pin_v3')


if __name__ == '__main__':
    unittest.main()
