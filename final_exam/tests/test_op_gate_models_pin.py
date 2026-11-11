"""Physics-boundary, architecture and deployment tests for OP-guided pin gates."""
import copy
import unittest
from unittest.mock import patch

import torch

from device_jepa.data import build_manifest, fit_preprocessor, samples_for_split, synthetic_dataset
from device_jepa.training import state_digest
from final_exam.gate_version_registry_pin import V1_JOINT_GATE_MODELS_PIN
from final_exam.data_pin import PinResidentDataset, make_preprocessing_pin
from final_exam.op_gate_data_pin import ShuffleBankPin, device_input_pin
from final_exam.models.op_gate_pin import (
    GATE_MODELS_PIN, GATE_ORACLES_PIN, DeviceOPPredictorPin, DeviceOPGateModelPin,
    OPAttentionGatePin, describe_gate_model_pin, rebuild_gate_model_pin, validate_gate_config_pin,
)
from final_exam.models.op_transfer_pin import fit_op_support_pin, op_objective_pin


class OPAttentionModelsPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        data = synthetic_dataset(20)
        data['samples'] = [row for row in data['samples'] if row['circuit_id'] == 'synthetic_0']
        for row in data['samples']:
            row['devices'] = row['devices'][:4]
        manifest = build_manifest(data, config=dict(train_circuits=['synthetic_0'],ood_circuits=[],
                                                   split_ratios=[.6,.2,.2],split_seed=17))
        original = fit_preprocessor(data,manifest,include_performance=True)
        cls.rows = samples_for_split(data,manifest,'train')
        cls.prep = make_preprocessing_pin(original,cls.rows)
        cls.support = fit_op_support_pin(cls.rows,cls.prep)
        cls.cfg = dict(dim=8,heads=2,encoder_layers=4,cross_layers=2,ffn_dim=16,
                       decoder_layers=1,decoder_ffn_dim=16,dropout=0.,max_devices=32,terminal_bias=True)
        store = PinResidentDataset(cls.rows[:3],cls.prep,'cpu',include_op=True,
                                   include_performance=True,max_tokens=32,verbose=False)
        _,cls.batch = next(store.batches(3,metadata=True))
        cls.batch['_sample_ids'] = [row['sample_id'] for row in cls.rows[:3]]

    def setUp(self):
        torch.manual_seed(17)

    def predictor(self):
        return DeviceOPPredictorPin(self.cfg,self.prep,self.support)

    def model(self,mode='device_op_gate_pin',**kwargs):
        predictor = None if mode == 'device_op_gate_only_base_pin' else self.predictor()
        bank = None
        if '_shuffle_' in mode:
            predictor.eval()
            with torch.no_grad():
                values,support = predictor.predict_devices(self.batch)
            bank = ShuffleBankPin(values,support,self.batch['_sample_ids'],17)
        return DeviceOPGateModelPin(mode,self.cfg,self.prep,predictor,
                                   shuffle_bank=bank,**kwargs).eval()

    def test_predictor_encodes_canonical_devices_and_counts_physical_targets(self):
        predictor = self.predictor().eval()
        seen = []
        handle = predictor.encoder.register_forward_pre_hook(lambda module,args: seen.append(args[0]))
        values,support = predictor.predict_devices(self.batch)
        handle.remove()
        self.assertEqual(seen[0]['device_mask'].shape,self.batch['owner_mask'].shape)
        self.assertLess(values.shape[1],self.batch['device_mask'].shape[1])
        self.assertEqual(predictor.encoder.structure_encoder.type_embedding.num_embeddings,
                         len(self.prep['device_preprocessing']['device_vocab']))
        # Changing expanded pin design tensors cannot change this D-token model.
        poison = copy.deepcopy(self.batch)
        poison['parameters'].fill_(1000.)
        poison['op'].fill_(float('nan'))
        poison['owner_op'].fill_(float('nan'))
        torch.testing.assert_close(values,predictor.predict_devices(poison)[0],atol=0,rtol=0)
        _,count,(_,counts) = op_objective_pin(predictor,self.batch)
        expected = (support & self.batch['owner_op_mask']
                    & self.batch['owner_op_target_mask'].unsqueeze(-1))
        self.assertEqual(count,int(expected.sum()))
        torch.testing.assert_close(counts,expected.sum((0,1)))

    def test_shared_model_width_embedding_and_individual_directed_heads(self):
        predictor = self.predictor().eval()
        values,support = predictor.predict_devices(self.batch)
        gate = OPAttentionGatePin(self.cfg,self.prep)
        embedded,scored = [],[]
        h1 = gate.op_embedding.register_forward_hook(lambda m,a,out: embedded.append(out))
        h2 = gate.score_head.register_forward_hook(lambda m,a,out: scored.append(out))
        bias = gate(self.batch,values,support)
        h1.remove(); h2.remove()
        self.assertEqual(embedded[0].shape,(*values.shape[:2],self.cfg['dim']))
        self.assertEqual(scored[0].shape,(*values.shape[:2],4,4,self.cfg['heads']))
        self.assertEqual(bias.shape,(values.shape[0],self.cfg['heads'],
                                    self.batch['device_mask'].shape[1]+1,self.batch['device_mask'].shape[1]+1))
        self.assertGreater(float((scored[0][:,:,0,1]-scored[0][:,:,1,0]).detach().abs().max()),0.)
        self.assertGreater(float((scored[0][...,0]-scored[0][...,1]).detach().abs().max()),0.)

    def test_gate_protects_net_shorts_self_external_other_devices_and_nonmos(self):
        predictor = self.predictor().eval()
        values,support = predictor.predict_devices(self.batch)
        gate = OPAttentionGatePin(self.cfg,self.prep)
        bias = gate(self.batch,values,support).detach()
        owner,net = self.batch['owner_index'],self.batch['pin_net_index']
        protected = ((owner[:,:,None] != owner[:,None,:]) | (net[:,:,None] == net[:,None,:])
                     | (owner[:,:,None] < 0) | (owner[:,None,:] < 0))
        self.assertEqual(float(bias[:,:,1:,1:][protected[:,None].expand_as(bias[:,:,1:,1:])].abs().sum()),0.)
        self.assertEqual(float(bias[:,:,0,:].abs().sum()+bias[:,:,:,0].abs().sum()),0.)
        self.assertTrue(bool((bias <= 0).all()))
        self.assertTrue(bool((bias < 0).any()))
        # Synthetic owner 3 is a resistor; no p/n gate is allowed.
        passive = owner == 3
        passive_pairs = passive[:,:,None] | passive[:,None,:]
        self.assertEqual(float(bias[:,:,1:,1:][passive_pairs[:,None].expand_as(bias[:,:,1:,1:])].abs().sum()),0.)
        changed = copy.deepcopy(self.batch)
        # Force every terminal of owner zero to the same actual net.
        changed['pin_net_index'][owner == 0] = 99
        replay = gate(changed,values,support).detach()[:,:,1:,1:]
        same_owner = (owner[:,:,None] == 0) & (owner[:,None,:] == 0)
        self.assertEqual(float(replay[same_owner[:,None].expand_as(replay)].abs().sum()),0.)

    def test_bias_reaches_every_structure_and_fusion_self_attention_layer(self):
        model = self.model('device_op_gate_only_pin')
        blocks = list(model.encoder.structure_encoder.blocks)+list(model.encoder.fusion_blocks)
        received = []
        handles = [block.register_forward_pre_hook(lambda m,args: received.append(args[1].detach().clone()))
                   for block in blocks]
        with torch.no_grad():
            values,support = model.predicted_devices(self.batch)
            expected = model.gate(self.batch,values,support).reshape(
                self.batch['device_mask'].shape[0]*self.cfg['heads'],
                self.batch['device_mask'].shape[1]+1,self.batch['device_mask'].shape[1]+1)
            model(self.batch)
            gated = received[:]
            received.clear()
            model.gate.settings['strength'] = 0.
            model(self.batch)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(gated),self.cfg['encoder_layers'])
        for with_gate,without_gate in zip(gated,received):
            finite = torch.isfinite(with_gate) & torch.isfinite(without_gate)
            torch.testing.assert_close((with_gate-without_gate)[finite],expected[finite],atol=3e-8,rtol=1e-5)

    def test_gate_and_op_embedding_learn_while_device_predictor_stays_frozen(self):
        model = self.model().train()
        before = state_digest(model.predictor)
        model(self.batch).square().mean().backward()
        self.assertFalse(model.predictor.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.predictor.parameters()))
        self.assertEqual(state_digest(model.predictor),before)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.gate.op_embedding.parameters()),0.)
        self.assertTrue(bool((model.gate.score_head[-1].weight.grad.abs().sum(-1) > 0).all()))

    def test_all_nonoracle_modes_ignore_measured_op_and_performance(self):
        poison = copy.deepcopy(self.batch)
        for name in ('op','owner_op','performance'):
            poison[name].fill_(float('nan'))
        for name in ('op_mask','owner_op_mask','op_target_mask','owner_op_target_mask','performance_mask'):
            poison[name] = ~poison[name]
        for name in GATE_MODELS_PIN:
            if name in GATE_ORACLES_PIN:
                continue
            model = self.model(name)
            with torch.no_grad():
                torch.testing.assert_close(model(self.batch),model(poison),atol=0,rtol=0,msg=name)

    def test_gate_only_omits_feature_tokens_and_features_remain_correct_in_controls(self):
        values,support = self.predictor().eval().predict_devices(self.batch)
        for name in ('device_op_cascade_pin','device_op_gate_only_pin'):
            model = self.model(name,gate_config={'strength':0.})
            self.assertEqual(hasattr(model.encoder,'op_names'),name == 'device_op_cascade_pin')
            with torch.no_grad(),patch.object(model,'predicted_devices',return_value=(values,support)):
                expected = model(self.batch)
            with torch.no_grad(),patch.object(model,'predicted_devices',return_value=(values+support*5.,support)):
                changed = model(self.batch)
            if model.use_op_features:
                self.assertGreater(float((expected-changed).abs().max()),1e-6)
            else:
                torch.testing.assert_close(expected,changed,atol=0,rtol=0)

    def test_oracle_fallback_and_no_missingness_side_channel(self):
        model = self.model('device_op_gate_oracle_pin')
        values,support = model.predicted_devices(self.batch)
        batch = copy.deepcopy(self.batch)
        batch['owner_op_mask'].zero_()
        batch['owner_op'].fill_(float('nan'))
        chosen,mask = model.gate_values(batch,values,support)
        torch.testing.assert_close(chosen,values,atol=0,rtol=0)
        torch.testing.assert_close(mask,support)
        self.assertEqual(model.last_oracle_coverage['observed'],0)
        batch['owner_op_mask'] = support.clone()
        with self.assertRaisesRegex(ValueError,'finite'):
            model.gate_values(batch,values,support)
        with self.assertRaisesRegex(ValueError,'requires measured'):
            model({key:value for key,value in self.batch.items() if key != 'owner_op'})

    def test_initialization_matched_within_each_family_and_zero_gate_matches_baseline(self):
        predictor = self.predictor()
        for names in (GATE_MODELS_PIN[:4] + V1_JOINT_GATE_MODELS_PIN, GATE_MODELS_PIN[4:8]):
            baseline = None
            for name in names:
                if '_shuffle_' in name:
                    continue
                torch.manual_seed(29)
                model = DeviceOPGateModelPin(name,self.cfg,self.prep,
                    None if name == 'device_op_gate_only_base_pin' else copy.deepcopy(predictor),
                    gate_config={'strength':0.}).eval()
                if baseline is None:
                    baseline = model
                else:
                    self.assertEqual(state_digest(model.encoder),state_digest(baseline.encoder))
                    self.assertEqual(state_digest(model.decoder),state_digest(baseline.decoder))
                    with torch.no_grad():
                        torch.testing.assert_close(model(self.batch),baseline(self.batch),atol=0,rtol=0)

    def test_standalone_recipes_restore_all_modes_and_predictor_without_external_files(self):
        for name in GATE_MODELS_PIN:
            model = self.model(name)
            recipe = describe_gate_model_pin(model)
            restored = rebuild_gate_model_pin(recipe,self.cfg,self.prep)
            restored.load_state_dict(model.state_dict())
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(self.batch),restored(self.batch),atol=0,rtol=0,msg=name)
            if restored.predictor is not None:
                self.assertFalse(any(p.requires_grad for p in restored.predictor.parameters()))
                self.assertEqual(restored._predictor_digest_pin,state_digest(restored.predictor))
        predictor = self.predictor().eval()
        restored = rebuild_gate_model_pin(describe_gate_model_pin(predictor),self.cfg,self.prep)
        restored.load_state_dict(predictor.state_dict())
        restored.eval()
        torch.testing.assert_close(predictor(self.batch)[0],restored(self.batch)[0],atol=0,rtol=0)

    def test_cache_requires_checkpoint_identity_valid_support_and_detaches_values(self):
        model = self.model()
        values,support = model.predicted_devices(self.batch)
        cached = dict(self.batch,_predicted_device_op=values.clone().requires_grad_(),
                      _predicted_device_op_mask=support,_predicted_device_op_digest=state_digest(model.predictor))
        with torch.no_grad():
            torch.testing.assert_close(model(self.batch),model(cached),atol=0,rtol=0)
        self.assertFalse(model.predicted_devices(cached)[0].requires_grad)
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            model(dict(cached,_predicted_device_op_digest='other-checkpoint'))
        with self.assertRaisesRegex(ValueError,'support'):
            model(dict(cached,_predicted_device_op_mask=~support))

    def test_gate_config_rejects_invalid_ranges_and_unknown_options(self):
        for value in ({'epsilon':0},{'epsilon':1},{'strength':-1},{'initial_gate':1},
                      {'strength':True},{'strength':float('inf')},{'hidden_dim':64}):
            with self.assertRaises(ValueError):
                validate_gate_config_pin(value)


if __name__ == '__main__':
    unittest.main()
