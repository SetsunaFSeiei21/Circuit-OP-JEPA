"""Teacher-guided cascade: gradients, input boundaries and alignment masks."""
import copy
import unittest
from unittest.mock import patch

import torch

from device_jepa.data import build_manifest, fit_preprocessor, samples_for_split, synthetic_dataset
from device_jepa.model import ModelConfig, masked_mse
from device_jepa.tensors import collate_samples
from device_jepa.training import seed_everything, state_digest
from experiments.op_transfer.distillation import PredictedOPDistillation, PredictedOPEncoderDistillation
from experiments.op_transfer.losses import LatentScaler
from experiments.op_transfer.models import (
    OPPredictor, PredictedOPCascade, ReadoutModel, decoder, design_encoder,
    fit_op_support, freeze, op_encoder, op_input,
)


class PredictedOPDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(17)
        cls.data = synthetic_dataset(8)
        cls.manifest = build_manifest(cls.data, ood_circuit='synthetic_4')
        cls.prep = fit_preprocessor(cls.data, cls.manifest, include_performance=True)
        cls.rows = samples_for_split(cls.data, cls.manifest, 'train')
        cls.support = fit_op_support(cls.rows, cls.prep)
        cls.cfg = ModelConfig(dim=8, heads=2, encoder_layers=2, cross_layers=1,
                              ffn_dim=16, decoder_layers=1, decoder_ffn_dim=16, dropout=0.)

    def make(self, *, dropout=0.):
        seed_everything(17)
        cfg = ModelConfig(**{**self.cfg.to_dict(), 'dropout': dropout})
        batch = collate_samples(self.rows, self.prep, include_performance=True)
        batch['_group_keys'] = [[f"{row['circuit_id']}:{d['device_id']}" for d in
                                sorted(row['devices'], key=lambda d: d['device_id'])] for row in self.rows]
        teacher = ReadoutModel(op_encoder(cfg, self.prep, use_design=True),
                               decoder(cfg, self.prep), source='design_op').eval()
        with torch.no_grad():
            memory, padding = teacher.encoder(op_input(batch, use_design=True))
        scaler = LatentScaler(.01).update(memory, padding, batch['_group_keys']).finalize()
        predictor = OPPredictor(design_encoder(cfg, self.prep), self.prep, self.support)
        student = PredictedOPCascade(predictor, copy.deepcopy(teacher))
        return PredictedOPDistillation(student, teacher, scaler), batch

    def test_only_student_encoder_and_head_receive_gradients_and_updates(self):
        model, batch = self.make(dropout=.3)
        teacher_before = state_digest(model.teacher)
        predictor_before = state_digest(model.student.predictor)
        student_before = state_digest(model.student.downstream)
        model.train()
        self.assertFalse(model.teacher.training)
        self.assertFalse(model.teacher.encoder.training)
        self.assertFalse(model.student.predictor.training)
        self.assertTrue(model.student.downstream.training)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        loss, count, detail = model.objective(batch)
        self.assertGreater(count, 0)
        self.assertGreater(float(detail['alignment_mse']), 0.)
        torch.testing.assert_close(loss.detach(), detail['weighted_performance'] + detail['weighted_alignment'])
        loss.backward()
        for frozen in (model.teacher, model.student.predictor):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in frozen.parameters()))
        for trained in (model.student.downstream.encoder, model.student.downstream.decoder):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in trained.parameters()))
        optimizer.step()
        self.assertEqual(teacher_before, state_digest(model.teacher))
        self.assertEqual(predictor_before, state_digest(model.student.predictor))
        self.assertNotEqual(student_before, state_digest(model.student.downstream))

    def test_validation_uses_only_unweighted_performance_and_no_true_op(self):
        model, batch = self.make()
        model.eval()
        polluted = copy.deepcopy(batch)
        for key in ('op', 'op_mask', 'op_target_mask', '_group_keys'):
            polluted.pop(key, None)
        polluted.update(op_state=torch.tensor(float('nan')), simulation_status='failed')
        with patch.object(model, 'teacher_memory', side_effect=AssertionError('teacher accessed')):
            a = model.objective(batch, alignment_weight=2., performance_weight=3.)
            b = model.objective(polluted, alignment_weight=7., performance_weight=9.)
        expected = masked_mse(model(batch), batch['performance'], batch['performance_mask'])
        torch.testing.assert_close(a[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        polluted['performance'] = torch.full_like(batch['performance'], float('nan'))
        with torch.no_grad():
            torch.testing.assert_close(model(batch), model(polluted), rtol=0, atol=0)

    def test_cached_teacher_targets_match_direct_targets(self):
        model, batch = self.make()
        model.train()
        direct = model.objective(batch)
        cached = dict(batch)
        cached['_teacher_memory'], cached['_teacher_padding'] = model.teacher_memory(batch)
        self.assertFalse(cached['_teacher_memory'].requires_grad)
        with patch.object(model, 'teacher_memory', side_effect=AssertionError('cache bypassed')):
            replay = model.objective(cached)
        torch.testing.assert_close(direct[0], replay[0], rtol=0, atol=0)
        for name, value in direct[2].items():
            torch.testing.assert_close(value, replay[2][name], rtol=0, atol=0)

    def test_ineligible_and_padding_teacher_targets_are_excluded_before_arithmetic(self):
        model, batch = self.make()
        model.train()
        cached = copy.deepcopy(batch)
        cached['_teacher_memory'], cached['_teacher_padding'] = model.teacher_memory(batch)
        cached['op_mask'][0] = False
        cached['op_target_mask'][0] = False
        prepared = model.student.predicted_batch(cached)
        valid = (cached['device_mask'] & cached['op_target_mask']
                 & (cached['op_mask'] & prepared['op_mask']).any(-1))
        self.assertTrue(bool(valid.any()))
        baseline = model.objective(cached)
        polluted = copy.deepcopy(cached)
        polluted['_teacher_memory'][:, 1:][~valid] = float('nan')
        polluted['_teacher_memory'][:, 0][~valid.any(-1)] = float('nan')
        replay = model.objective(polluted)
        torch.testing.assert_close(baseline[0], replay[0], rtol=0, atol=0)
        replay[0].backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_zero_alignment_is_performance_only_without_teacher_or_op_targets(self):
        model, batch = self.make()
        model.train()
        stripped = {k: v for k, v in batch.items() if k not in ('op', 'op_mask', 'op_target_mask', '_group_keys')}
        with patch.object(model, 'teacher_memory', side_effect=AssertionError('teacher accessed')):
            loss, _, detail = model.objective(stripped, alignment_weight=0., performance_weight=2.)
        expected = masked_mse(model(stripped), batch['performance'], batch['performance_mask'])
        torch.testing.assert_close(loss, 2 * expected, rtol=0, atol=0)
        self.assertEqual(float(detail['alignment_mse']), 0.)
        self.assertEqual(float(detail['weighted_alignment']), 0.)

    def test_missing_op_devices_keep_supervised_loss_without_teacher_call(self):
        model, batch = self.make()
        model.train()
        batch['op_mask'].zero_()
        with patch.object(model, 'teacher_memory', side_effect=AssertionError('teacher accessed')):
            loss, _, detail = model.objective(batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(detail['alignment_mse']), 0.)
        torch.testing.assert_close(loss.detach(), detail['perf_mse'])
        batch['performance_mask'].zero_()
        self.assertIsNone(model.objective(batch))

    def test_same_teacher_scaler_preserves_absolute_alignment_and_raw_diagnostic(self):
        model, batch = self.make()
        model.train()
        with torch.no_grad():
            raw, padding = model.teacher.encoder(op_input(batch, use_design=True))
        same = raw.clone().requires_grad_()
        with patch.object(model.student.downstream.encoder, 'forward', return_value=(same, padding)):
            _, _, zero = model.objective(batch)
        self.assertEqual(float(zero['alignment_mse']), 0.)
        offset = .125
        shifted = (raw + offset).requires_grad_()
        with patch.object(model.student.downstream.encoder, 'forward', return_value=(shifted, padding)):
            _, _, different = model.objective(batch)
        self.assertGreater(float(different['alignment_mse']), 0.)
        self.assertAlmostEqual(float(different['raw_device_alignment_mse']), offset ** 2, places=7)
        self.assertEqual(model.teacher_scaler.state_dict()['fit_partition'], 'train')
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            model.teacher_scaler.update(raw, padding, batch['_group_keys'])

    def test_rejects_joint_predictor_wrong_source_shared_modules_and_unfitted_scaler(self):
        model, _ = self.make()
        with self.assertRaisesRegex(ValueError, 'frozen OP predictor'):
            PredictedOPDistillation(PredictedOPCascade(copy.deepcopy(model.student.predictor),
                copy.deepcopy(model.student.downstream), joint=True), copy.deepcopy(model.teacher), model.teacher_scaler)
        wrong_teacher = copy.deepcopy(model.teacher)
        wrong_teacher.source = 'op'
        with self.assertRaisesRegex(ValueError, 'topology, design and OP'):
            PredictedOPDistillation(copy.deepcopy(model.student), wrong_teacher, model.teacher_scaler)
        with self.assertRaisesRegex(ValueError, 'share parameter storage'):
            PredictedOPDistillation(model.student, model.student.downstream, model.teacher_scaler)
        with self.assertRaisesRegex(ValueError, 'finalized'):
            PredictedOPDistillation(copy.deepcopy(model.student), copy.deepcopy(model.teacher), LatentScaler())

    def test_rejects_misaligned_cached_padding_and_invalid_loss_weights(self):
        model, batch = self.make()
        model.train()
        batch['_teacher_memory'], batch['_teacher_padding'] = model.teacher_memory(batch)
        batch['_teacher_padding'] = ~batch['_teacher_padding']
        with self.assertRaisesRegex(ValueError, 'tokens must match'):
            model.objective(batch)
        for key, value in (('alignment_weight', -1.), ('alignment_weight', float('nan')),
                           ('performance_weight', 0.), ('performance_weight', float('inf'))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                model.objective(batch, **{key: value})


class EncoderOnlyDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PredictedOPDistillationTests.setUpClass()

    def make(self, *, dropout=0.):
        pair, batch = PredictedOPDistillationTests().make(dropout=dropout)
        return PredictedOPEncoderDistillation(pair.student, pair.teacher, pair.teacher_scaler), batch

    def test_encoder_stage_only_updates_encoder_without_calling_decoder(self):
        model, batch = self.make(dropout=.3)
        frozen = (model.teacher, model.student.predictor, model.student.downstream.decoder)
        before = [state_digest(part) for part in frozen]
        encoder_before = state_digest(model.student.downstream.encoder)
        batch.pop('performance')
        batch.pop('performance_mask')
        model.train()
        self.assertTrue(model.student.downstream.encoder.training)
        self.assertTrue(all(not part.training for part in frozen))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        with patch.object(model.student.downstream.decoder, 'forward', side_effect=AssertionError('decoder called')):
            loss, count, detail = model.objective(batch)
        self.assertGreater(count, 0)
        torch.testing.assert_close(loss.detach(), (detail['device_alignment_mse']+detail['graph_alignment_mse'])/2)
        loss.backward()
        for part in frozen:
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in part.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum())
                            for p in model.student.downstream.encoder.parameters()))
        optimizer.step()
        self.assertEqual(before, [state_digest(part) for part in frozen])
        self.assertNotEqual(encoder_before, state_digest(model.student.downstream.encoder))

    def test_validation_aligns_teacher_without_any_performance_labels(self):
        model, batch = self.make()
        model.eval()
        expected = model.objective(batch)
        batch['performance'] = torch.full_like(batch['performance'], float('nan'))
        batch['performance_mask'].zero_()
        with patch.object(model.student.downstream.decoder, 'forward', side_effect=AssertionError('decoder called')):
            actual = model.objective(batch)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assertNotIn('perf_mse', actual[2])
        batch['_teacher_memory'], batch['_teacher_padding'] = model.teacher_memory(batch)
        with patch.object(model, 'teacher_memory', side_effect=AssertionError('cache bypassed')):
            cached = model.objective(batch)
            torch.testing.assert_close(cached[0], expected[0], rtol=0, atol=0)
            batch['_teacher_memory'] = batch['_teacher_memory'] + 100
            changed = model.objective(batch)
        self.assertNotEqual(float(changed[0]), float(expected[0]))

    def test_missing_alignment_targets_skip_without_supervised_fallback(self):
        model, batch = self.make()
        batch['op_mask'].zero_()
        for train in (True, False):
            model.train(train)
            with patch.object(model, 'teacher_memory', side_effect=AssertionError('no target')):
                self.assertIsNone(model.objective(batch))

    def test_decoder_stage_only_updates_head_and_cached_features_match(self):
        from experiments.op_transfer.engine import performance_objective
        pair, batch = self.make(dropout=.3)
        model = pair.student
        model.downstream.frozen = True
        freeze(model.downstream.encoder)
        model.downstream.decoder.requires_grad_(True)
        # Stage boundary clears old gradients; only the head is optimized now.
        model.zero_grad(set_to_none=True)
        frozen = (model.predictor, model.downstream.encoder)
        before = [state_digest(part) for part in frozen]
        head_before = state_digest(model.downstream.decoder)
        model.train()
        self.assertTrue(model.downstream.decoder.training)
        self.assertTrue(all(not part.training for part in frozen))
        batch.pop('op'); batch.pop('op_mask'); batch.pop('op_target_mask')
        a = model.memory(batch)
        b = model.memory(batch)
        self.assertFalse(a[0].requires_grad)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        cached = dict(batch, _frozen_memory=a[0], _frozen_padding=a[1])
        model.eval()
        torch.testing.assert_close(performance_objective(model,batch)[0],
                                   performance_objective(model,cached)[0], rtol=0, atol=0)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        performance_objective(model,cached)[0].backward()
        optimizer.step()
        self.assertEqual(before, [state_digest(part) for part in frozen])
        self.assertNotEqual(head_before, state_digest(model.downstream.decoder))


if __name__ == '__main__':
    unittest.main()
