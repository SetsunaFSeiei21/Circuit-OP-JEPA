"""Freeze, gradient, exact-bypass and replay contracts of post-trained OP gates."""
import copy
import unittest
from unittest.mock import patch

import torch

from device_jepa.training import state_digest
from experiments.op_transfer.losses import LatentScaler
from final_exam.data_pin import group_keys_pin
from final_exam.models.op_gate_pin import DeviceOPGateModelPin, DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import (
    ReadoutModelPin, decoder_pin, describe_model_pin, op_encoder_pin,
    op_input_pin, rebuild_model_pin,
)
from final_exam.models.staged_op_gate_pin import (
    STAGED_GATE_MODELS_PIN, StagedOPEncoderDistillationPin, StagedOPGateModelPin,
)
from final_exam.tests import test_op_gate_models_pin as gate_test_helpers


class StagedOPGateModelsPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = gate_test_helpers.OPAttentionModelsPinTests
        fixture.setUpClass()
        cls.cfg, cls.prep, cls.support = fixture.cfg, fixture.prep, fixture.support
        cls.batch = copy.deepcopy(fixture.batch)
        cls.batch['_group_keys'] = [group_keys_pin(row, cls.prep['feature_spec']['terminal_names'])
                                     for row in fixture.rows[:3]]

    def setUp(self):
        torch.manual_seed(17)

    def model(self, mode=STAGED_GATE_MODELS_PIN[0], *, dropout=0.):
        cfg = {**self.cfg, 'dropout': dropout}
        predictor = DeviceOPPredictorPin(cfg, self.prep, self.support)
        return StagedOPGateModelPin(cfg, self.prep, predictor, mode=mode)

    def teacher(self):
        return ReadoutModelPin(op_encoder_pin(self.cfg, self.prep, use_design=True),
                               decoder_pin(self.cfg, self.prep), source='design_op').eval()

    def scaler(self, teacher):
        with torch.no_grad():
            memory, padding = teacher.encoder(op_input_pin(self.batch, use_design=True))
        return LatentScaler().update(memory, padding, self.batch['_group_keys']).finalize()

    def test_gate_is_exactly_bypassed_during_all_preparatory_phases(self):
        model = self.model().eval()
        baseline = DeviceOPGateModelPin('device_op_cascade_pin', self.cfg, self.prep,
                                       copy.deepcopy(model.predictor)).eval()
        baseline.encoder.load_state_dict(model.encoder.state_dict())
        baseline.decoder.load_state_dict(model.decoder.state_dict())
        with torch.no_grad():
            # Even pathological unused gate weights cannot contaminate the base.
            for parameter in model.gate.parameters():
                parameter.fill_(float('nan'))
            for phase in ('supervised', 'distill_encoder', 'readout'):
                model.set_phase(phase).eval()
                # Match grad flags too: PyTorch may choose a different numeric
                # attention kernel for a wholly frozen block.
                for name in ('encoder', 'decoder'):
                    for target, source in zip(getattr(baseline, name).parameters(),
                                              getattr(model, name).parameters()):
                        target.requires_grad_(source.requires_grad)
                expected = baseline(self.batch)
                with patch.object(model.gate, 'forward', side_effect=AssertionError('gate called')):
                    torch.testing.assert_close(expected, model(self.batch), atol=0, rtol=0)

    def test_each_phase_has_exact_gradient_and_train_eval_policy(self):
        model = self.model(dropout=.3)
        expected = {
            'supervised': {'encoder', 'decoder'}, 'distill_encoder': {'encoder'},
            'readout': {'decoder'}, 'gate_only': {'gate'},
            'gate_decoder': {'gate', 'decoder'},
        }
        for phase, active in expected.items():
            model.set_phase(phase).train()
            for name in ('predictor', 'encoder', 'decoder', 'gate'):
                module = getattr(model, name)
                self.assertEqual(module.training, name in active, (phase, name))
                self.assertTrue(all(p.requires_grad == (name in active) for p in module.parameters()))
            self.assertEqual(model.gate_enabled, phase.startswith('gate_'))
        with self.assertRaisesRegex(ValueError, 'phase'):
            model.set_phase('typo')

    def test_gate_only_backpropagates_through_frozen_encoder_and_decoder(self):
        model = self.model(dropout=.3).set_phase('gate_only').train()
        before = {name: state_digest(getattr(model, name)) for name in ('predictor', 'encoder', 'decoder', 'gate')}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        prediction = model(self.batch)
        prediction.square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.gate.op_embedding.parameters()), 0.)
        self.assertTrue(bool((model.gate.score_head[-1].weight.grad.abs().sum(-1) > 0).all()))
        for name in ('predictor', 'encoder', 'decoder'):
            self.assertTrue(all(p.grad is None for p in getattr(model, name).parameters()))
            self.assertFalse(getattr(model, name).training)
        optimizer.step()
        for name in ('predictor', 'encoder', 'decoder'):
            self.assertEqual(before[name], state_digest(getattr(model, name)), name)
        self.assertNotEqual(before['gate'], state_digest(model.gate))
        self.assertGreater(float((model(self.batch)-prediction).detach().abs().max()), 0.)

    def test_gate_decoder_updates_only_the_two_requested_modules(self):
        model = self.model(STAGED_GATE_MODELS_PIN[1]).set_phase('gate_decoder').train()
        before = {name: state_digest(getattr(model, name)) for name in ('predictor', 'encoder', 'decoder', 'gate')}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        model(self.batch).square().mean().backward()
        optimizer.step()
        for name in ('predictor', 'encoder'):
            self.assertEqual(before[name], state_digest(getattr(model, name)), name)
        for name in ('decoder', 'gate'):
            self.assertNotEqual(before[name], state_digest(getattr(model, name)), name)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for staged gate regression')
    def test_cuda_staged_gate_steps_keep_frozen_modules_unchanged(self):
        batch = {key: value.cuda() if torch.is_tensor(value) else copy.deepcopy(value)
                 for key, value in self.batch.items()}
        for mode in STAGED_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                phase = 'gate_decoder' if mode == STAGED_GATE_MODELS_PIN[1] else 'gate_only'
                model = self.model(mode, dropout=.3).cuda().set_phase(phase).train()
                names = ('predictor', 'encoder', 'decoder', 'gate')
                before = {name: state_digest(getattr(model, name)) for name in names}
                frozen = ('predictor', 'encoder') if phase == 'gate_decoder' else names[:3]
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                              lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(batch)
                    self.assertTrue(prediction.is_cuda)
                    loss = (prediction - torch.randn_like(prediction)).square().mean()
                    self.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    gradients = [p.grad for p in model.gate.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(gradient.is_cuda and bool(torch.isfinite(gradient).all())
                                        for gradient in gradients))
                    self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.)
                    for name in frozen:
                        module = getattr(model, name)
                        self.assertFalse(module.training)
                        self.assertTrue(all(p.grad is None for p in module.parameters()))
                    optimizer.step()
                    torch.cuda.synchronize()
                    for name in frozen:
                        self.assertEqual(before[name], state_digest(getattr(model, name)), name)
                self.assertNotEqual(before['gate'], state_digest(model.gate))
                if phase == 'gate_decoder':
                    self.assertNotEqual(before['decoder'], state_digest(model.decoder))
                model.eval()
                with torch.no_grad():
                    self.assertTrue(bool(torch.isfinite(model(batch)).all()))

    def test_encoder_distillation_is_label_free_and_gate_decoder_teacher_stay_frozen(self):
        teacher, student = self.teacher(), self.model()
        student.initialize_from_teacher(teacher)
        self.assertEqual(state_digest(student.encoder), state_digest(teacher.encoder.core))
        self.assertEqual(state_digest(student.decoder), state_digest(teacher.decoder))
        self.assertFalse({p.data_ptr() for p in student.parameters()} & {p.data_ptr() for p in teacher.parameters()})
        pair = StagedOPEncoderDistillationPin(student, teacher, self.scaler(teacher)).train()
        before = {name: state_digest(getattr(student, name)) for name in ('predictor', 'decoder', 'gate')}
        teacher_digest = state_digest(teacher)
        clean = pair.objective(self.batch)
        poison = copy.deepcopy(self.batch)
        poison['performance'].fill_(float('nan'))
        poison['performance_mask'].zero_()
        with patch.object(student.decoder, 'forward', side_effect=AssertionError('decoder called')), \
                patch.object(student.gate, 'forward', side_effect=AssertionError('gate called')):
            torch.testing.assert_close(clean[0], pair.objective(poison)[0], atol=0, rtol=0)
        self.assertIn('pin_alignment_mse', clean[2])
        clean[0].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in student.encoder.parameters() if p.grad is not None), 0.)
        for name in before:
            self.assertEqual(before[name], state_digest(getattr(student, name)))
            self.assertTrue(all(p.grad is None for p in getattr(student, name).parameters()))
        self.assertEqual(teacher_digest, state_digest(teacher))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertFalse(teacher.training)
        # Matches the engine's cached teacher branch used by both train and val.
        target, padding = pair.teacher_memory(self.batch)
        cached = {**self.batch, '_teacher_memory': target, '_teacher_padding': padding}
        torch.testing.assert_close(pair.objective(cached)[0], clean[0], atol=0, rtol=0)

    def test_standalone_recipe_restores_phase_and_ignores_measured_inputs(self):
        for mode in STAGED_GATE_MODELS_PIN:
            phase = 'gate_decoder' if mode == STAGED_GATE_MODELS_PIN[1] else 'gate_only'
            model = self.model(mode).set_phase(phase).eval()
            recipe = describe_model_pin(model)
            restored = rebuild_model_pin(recipe, self.cfg, self.prep)
            restored.load_state_dict(model.state_dict())
            restored.eval()
            self.assertEqual(restored.phase, phase)
            self.assertEqual({n for n,p in model.named_parameters() if p.requires_grad},
                             {n for n,p in restored.named_parameters() if p.requires_grad})
            poison = copy.deepcopy(self.batch)
            for key in ('op', 'owner_op', 'performance'):
                poison[key].fill_(float('nan'))
            for key in ('op_mask', 'owner_op_mask', 'op_target_mask', 'owner_op_target_mask', 'performance_mask'):
                poison[key].logical_not_()
            with torch.no_grad():
                expected = model(self.batch)
                torch.testing.assert_close(expected, restored(self.batch), atol=0, rtol=0)
                torch.testing.assert_close(expected, restored(poison), atol=0, rtol=0)
            bad = {**recipe, 'gate_enabled': False}
            with self.assertRaisesRegex(ValueError, 'metadata'):
                rebuild_model_pin(bad, self.cfg, self.prep)

    def test_shared_freeze_and_phase_transition_clear_stale_gradients(self):
        model = self.model().set_phase('supervised').train()
        model(self.batch).square().mean().backward()
        model.requires_grad_(False).eval()
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.decoder.training)
        model.set_phase('gate_only')
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.gate.parameters()))


if __name__ == '__main__':
    unittest.main()
