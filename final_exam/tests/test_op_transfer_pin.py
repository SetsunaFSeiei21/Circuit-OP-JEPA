"""Pin encoder / physical-device OP boundary and staged distillation contracts."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from device_jepa.data import build_manifest, fit_preprocessor, samples_for_split, synthetic_dataset
from device_jepa.training import state_digest
from device_jepa.tensors import collate_samples
from experiments.op_transfer.losses import LatentScaler
from final_exam.data_pin import PinResidentDataset, group_keys_pin, make_preprocessing_pin
from final_exam.models.models_pin import build_encoder_pin
from final_exam.models.op_transfer_pin import (
    OPPredictorPin, ReadoutModelPin, PredictedOPCascadePin,
    PredictedOPDistillationPin, PredictedOPEncoderDistillationPin,
    broadcast_owner_op_pin, decoder_pin, describe_model_pin, design_encoder_pin,
    design_input_pin, fit_op_support_pin, op_encoder_pin, op_input_pin,
    op_objective_pin, pool_owner_memory_pin, rebuild_model_pin,
)


class PinTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.dataset = synthetic_dataset(12)
        cls.manifest = build_manifest(cls.dataset, ood_circuit='synthetic_4')
        original = fit_preprocessor(cls.dataset, cls.manifest, include_performance=True)
        cls.rows = samples_for_split(cls.dataset, cls.manifest, 'train')
        cls.prep = make_preprocessing_pin(original, cls.rows)
        cls.support = fit_op_support_pin(cls.rows, cls.prep)
        cls.cfg = dict(dim=8, heads=2, encoder_layers=2, cross_layers=1, ffn_dim=16,
                       decoder_layers=1, decoder_ffn_dim=16, dropout=0., max_devices=64,
                       terminal_bias=True)
        cls.selected = cls.rows[:3]
        store = PinResidentDataset(cls.selected, cls.prep, 'cpu', include_op=True,
                                   include_performance=True, max_tokens=64)
        _, cls.batch = next(store.batches(3))
        cls.batch['_group_keys'] = [group_keys_pin(row, cls.prep['feature_spec']['terminal_names'])
                                    for row in cls.selected]

    def predictor(self, encoder_name='zerosim_supervised_pin'):
        return OPPredictorPin(build_encoder_pin(encoder_name, self.cfg, self.prep),
                              self.prep, self.support, encoder_name=encoder_name)

    def pair(self, sequential=False):
        teacher = ReadoutModelPin(op_encoder_pin(self.cfg, self.prep, use_design=True),
                                   decoder_pin(self.cfg, self.prep), source='design_op').eval()
        with torch.no_grad():
            memory, padding = teacher.encoder(op_input_pin(self.batch, use_design=True))
        scaler = LatentScaler(.01).update(memory, padding, self.batch['_group_keys']).finalize()
        student = PredictedOPCascadePin(self.predictor(), copy.deepcopy(teacher))
        cls = PredictedOPEncoderDistillationPin if sequential else PredictedOPDistillationPin
        return cls(student, teacher, scaler)

    def test_one_physical_op_vector_is_broadcast_to_all_owner_pins(self):
        predictor = self.predictor().eval()
        with torch.no_grad():
            owners, owner_support = predictor.predict_devices(self.batch)
            pins, pin_support = predictor(self.batch)
        self.assertEqual(owners.shape[:2], self.batch['owner_mask'].shape)
        self.assertEqual(pins.shape[:2], self.batch['device_mask'].shape)
        self.assertGreater(pins.shape[1], owners.shape[1])
        for row in range(pins.shape[0]):
            for owner in range(owners.shape[1]):
                selected = self.batch['owner_index'][row] == owner
                if selected.any():
                    torch.testing.assert_close(pins[row, selected], owners[row, owner].expand(selected.sum(), -1))
                    torch.testing.assert_close(pin_support[row, selected], owner_support[row, owner].expand(selected.sum(), -1))
        ports = self.batch['device_mask'] & (self.batch['owner_index'] == -1)
        self.assertTrue(ports.any())
        self.assertFalse(pin_support[ports].any())
        self.assertEqual(float(pins[ports].abs().sum()), 0.)

    def test_op_objective_counts_physical_devices_not_replicated_pin_targets(self):
        predictor = self.predictor()
        loss, count, (sse, counts) = op_objective_pin(predictor, self.batch)
        expected = (self.batch['owner_op_mask'] & predictor.topology_mask(self.batch)
                    & self.batch['owner_op_target_mask'].unsqueeze(-1))
        self.assertEqual(count, int(expected.sum()))
        torch.testing.assert_close(counts, expected.sum((0, 1)))
        self.assertLess(count, int(self.batch['op_mask'].sum()))
        self.assertTrue(torch.isfinite(sse).all())
        poisoned = copy.deepcopy(self.batch)
        poisoned['op'].fill_(float('nan'))
        poisoned['op_mask'].zero_()
        poisoned['op_target_mask'].zero_()
        poisoned['owner_op'][~expected] = float('nan')
        replay, replay_count, _ = op_objective_pin(predictor, poisoned)
        torch.testing.assert_close(loss, replay, rtol=0, atol=0)
        self.assertEqual(count, replay_count)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.encoder.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in predictor.parameters()))

    def test_pooling_excludes_ports_and_preserves_equal_owner_loss_weight(self):
        batch = dict(device_mask=torch.tensor([[True]*7]),
            owner_index=torch.tensor([[-1,0,0,1,1,1,1]]),
            pin_terminal_index=torch.tensor([[-1,0,1,0,1,2,3]]),
            owner_mask=torch.tensor([[True,True]]))
        memory = torch.tensor([[[100.],[999.],[2.],[4.],[1.],[3.],[5.],[7.]]], requires_grad=True)
        padding = torch.zeros((1,8),dtype=torch.bool)
        pooled, counts = pool_owner_memory_pin(memory,padding,batch)
        torch.testing.assert_close(pooled,torch.tensor([[[3.],[4.]]]))
        torch.testing.assert_close(counts,torch.tensor([[2.,4.]]))
        pooled.sum().backward()
        self.assertEqual(float(memory.grad[0,0]),0.)
        self.assertEqual(float(memory.grad[0,1]),0.)
        self.assertAlmostEqual(float(memory.grad[0,2]),.5)
        self.assertAlmostEqual(float(memory.grad[0,4]),.25)
        values = torch.tensor([[[3.],[4.]]])
        op, mask = broadcast_owner_op_pin(values,torch.ones_like(values,dtype=torch.bool),batch)
        self.assertEqual(op.flatten().tolist(),[0.,3.,3.,4.,4.,4.,4.])
        self.assertFalse(mask[0,0].any())

    def test_cascade_and_predictor_ignore_all_measured_inputs(self):
        pair = self.pair().eval()
        poisoned = copy.deepcopy(self.batch)
        for key in ('op','owner_op','performance'):
            poisoned[key].fill_(float('nan'))
        for key in ('op_mask','owner_op_mask','op_target_mask','owner_op_target_mask','performance_mask'):
            poisoned[key] = ~poisoned[key]
        for key in ('op_state','simulation_status','feasible'):
            poisoned[key] = 'invalid'
        with torch.no_grad():
            torch.testing.assert_close(pair(self.batch),pair(poisoned),rtol=0,atol=0)
            first = pair.student.predictor(self.batch)
            second = pair.student.predictor(poisoned)
            torch.testing.assert_close(first[0],second[0],rtol=0,atol=0)
        allowed = design_input_pin(poisoned)
        self.assertFalse({'owner_op','op','op_mask','owner_op_mask','performance'} & allowed.keys())
        prepared = pair.student.predicted_batch(poisoned)
        self.assertNotIn('owner_op',prepared)
        self.assertIn('owner_index',prepared)

    def test_joint_student_only_updates_student_encoder_and_decoder(self):
        pair = self.pair().train()
        frozen = [state_digest(pair.teacher),state_digest(pair.student.predictor)]
        old = state_digest(pair.student.downstream)
        optimizer = torch.optim.AdamW([p for p in pair.parameters() if p.requires_grad],lr=.001)
        loss,count,detail = pair.objective(self.batch)
        self.assertGreater(count,0)
        self.assertIn('pin_alignment_mse',detail)
        self.assertNotIn('device_alignment_mse',detail)
        loss.backward()
        optimizer.step()
        self.assertEqual(frozen,[state_digest(pair.teacher),state_digest(pair.student.predictor)])
        self.assertNotEqual(old,state_digest(pair.student.downstream))
        for model in (pair.teacher,pair.student.predictor):
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        pair.eval()
        without_op = {key:value for key,value in self.batch.items()
                      if key not in ('op','owner_op','op_mask','owner_op_mask','op_target_mask','owner_op_target_mask')}
        with patch.object(pair,'teacher_memory',side_effect=AssertionError('teacher read during selection')):
            selected = pair.objective(without_op)
        self.assertTrue(torch.isfinite(selected[0]))

    def test_encoder_only_stage_ignores_labels_and_keeps_decoder_frozen(self):
        pair = self.pair(sequential=True).train()
        before = state_digest(pair.student.downstream.decoder)
        poisoned = copy.deepcopy(self.batch)
        poisoned['performance'].fill_(float('nan'))
        poisoned['performance_mask'].zero_()
        loss,_,detail = pair.objective(self.batch)
        replay,_,_ = pair.objective(poisoned)
        torch.testing.assert_close(loss,replay,rtol=0,atol=0)
        self.assertIn('pin_alignment_mse',detail)
        loss.backward()
        optimizer = torch.optim.AdamW([p for p in pair.parameters() if p.requires_grad],lr=.001)
        optimizer.step()
        self.assertEqual(before,state_digest(pair.student.downstream.decoder))
        self.assertTrue(all(p.grad is None for p in pair.student.downstream.decoder.parameters()))

    def test_pin_recipes_rebuild_cascade_oracle_and_deepgen_predictor(self):
        models = [self.pair().student,
                  ReadoutModelPin(op_encoder_pin(self.cfg,self.prep),decoder_pin(self.cfg,self.prep),source='op'),
                  self.predictor('deepgen_supervised_pin')]
        for model in models:
            model.eval()
            recipe = describe_model_pin(model)
            self.assertTrue(recipe['kind'].endswith('_pin'))
            restored = rebuild_model_pin(recipe,self.cfg,self.prep).eval()
            restored.load_state_dict(model.state_dict(),strict=True)
            with torch.no_grad():
                first,second=model(self.batch),restored(self.batch)
                if isinstance(first,tuple):
                    torch.testing.assert_close(first[0],second[0],rtol=0,atol=0)
                    torch.testing.assert_close(first[1],second[1],rtol=0,atol=0)
                else:
                    torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_invalid_owner_mapping_and_device_preprocessing_are_rejected(self):
        predictor=self.predictor()
        batch=copy.deepcopy(self.batch)
        port=(batch['owner_index'][0] == -1).nonzero()[0,0]
        batch['owner_index'][0,port]=0
        with self.assertRaisesRegex(ValueError,'External ports'):
            predictor(batch)
        with self.assertRaisesRegex(ValueError,'pin preprocessing'):
            OPPredictorPin(design_encoder_pin(self.cfg,self.prep),self.prep['device_preprocessing'],self.support)
        broken=copy.deepcopy(self.support)
        broken['field_names']=list(reversed(broken['field_names']))
        with self.assertRaisesRegex(ValueError,'field order'):
            OPPredictorPin(design_encoder_pin(self.cfg,self.prep),self.prep,broken)

    def test_pin_oracles_reject_device_graphs_or_missing_pin_identity(self):
        device_batch = collate_samples(self.selected,self.prep['device_preprocessing'],include_op=True)
        for use_design in (False,True):
            model = ReadoutModelPin(op_encoder_pin(self.cfg,self.prep,use_design=use_design),
                decoder_pin(self.cfg,self.prep),source='design_op' if use_design else 'op').eval()
            with self.assertRaisesRegex(ValueError,'Pin models require'):
                model(device_batch)
            for missing in ('pin_to_device','pin_terminal_index','pin_net_index'):
                partial = dict(self.batch)
                partial.pop(missing)
                with self.assertRaisesRegex(ValueError,'Pin models require'):
                    model(partial)


if __name__ == '__main__':
    unittest.main()
