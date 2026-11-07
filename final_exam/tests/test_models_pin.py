"""Pin adapters must use pin memory, preserve readouts and exclude measured data."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from device_jepa.baselines import _pin_tokens,collate_pin_samples,fit_pin_vocabulary
from device_jepa.data import synthetic_dataset,build_manifest,fit_preprocessor
from device_jepa.tensors import collate_samples
from final_exam.data_pin import PinResidentDataset,make_preprocessing_pin
from final_exam.models.models_pin import TORCH_BASELINES_PIN,build_encoder_pin,build_model_pin
from final_exam.models.gtn_pin import build_loop_membership_pin


def make_pin_batch(rows,device_prep,pin_vocab):
    batch=collate_pin_samples(rows,device_prep,pin_vocab,include_performance=True,include_op=True)
    batch['pin_to_device']=torch.full_like(batch['device_types'],-1)
    batch['pin_terminal_index']=torch.full_like(batch['device_types'],-1)
    batch['pin_net_index']=torch.full_like(batch['device_types'],-1)
    tn=device_prep['feature_spec']['terminal_names']
    for b,row in enumerate(rows):
        device_ids={d['device_id']:i for i,d in enumerate(sorted(row['devices'],key=lambda x:x['device_id']))}
        descriptors=_pin_tokens(row,tn,device_prep.get('input_contract_version',1))
        nets={name:i for i,name in enumerate(sorted({d['net'] for d in descriptors}))}
        for p,descriptor in enumerate(descriptors):
            batch['pin_net_index'][b,p]=nets[descriptor['net']]
            if descriptor['device_id'] is not None:
                batch['pin_to_device'][b,p]=device_ids[descriptor['device_id']]
                batch['pin_terminal_index'][b,p]=tn.index(descriptor['terminal'])
    return batch


def pad_pin(batch,extra=3):
    output={}
    n=batch['device_mask'].shape[1]
    mappings={'pin_to_device','pin_terminal_index','pin_net_index'}
    graph_keys={'adjacency','edge_roles'}
    pin_keys={'device_types','device_categories','device_models','device_mask','port_roles',
        'named_port_roles','parameters','parameter_mask','parameter_applicability','op','op_mask',
        'op_target_mask'}|mappings
    for key,value in batch.items():
        if key in graph_keys:
            shape=list(value.shape);shape[1]=shape[2]=n+extra
            new=value.new_zeros(shape);new[:,:n,:n]=value
        elif key in pin_keys:
            shape=list(value.shape);shape[1]=n+extra
            new=value.new_full(shape,-1 if key in mappings else 0);new[:,:n]=value
        else:
            new=value
        output[key]=new
    return output


class PinBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.data=synthetic_dataset(10)
        cls.manifest=build_manifest(cls.data,'synthetic_4')
        cls.device_prep=fit_preprocessor(cls.data,cls.manifest,True)
        cls.pin_vocab=fit_pin_vocabulary(cls.data,cls.manifest,cls.device_prep['input_contract_version'])
        train=[r for r in cls.data['samples'] if r['sample_id'] in cls.manifest['train']]
        cls.prep=make_preprocessing_pin(cls.device_prep,train)
        cls.rows=cls.data['samples'][::9][:3]
        cls.batch=make_pin_batch(cls.rows,cls.device_prep,cls.pin_vocab)
        cls.cfg=dict(dim=8,heads=2,encoder_layers=2,cross_layers=1,ffn_dim=16,decoder_layers=1,
                     decoder_ffn_dim=16,dropout=0.,max_devices=64,terminal_bias=True)

    def test_all_pin_models_forward_backward_and_reload(self):
        for name in TORCH_BASELINES_PIN:
            with self.subTest(model=name):
                model=build_model_pin(name,self.cfg,self.prep)
                pred=model(self.batch)
                self.assertEqual(pred.shape,(len(self.rows),len(self.prep['feature_spec']['metric_names'])))
                self.assertTrue(torch.isfinite(pred).all())
                pred.square().mean().backward()
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                model.eval()
                restored=build_model_pin(name,self.cfg,self.prep).eval()
                restored.load_state_dict(model.state_dict(),strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch),restored(self.batch),atol=0,rtol=0)

    def test_encoder_memory_has_terminal_and_port_tokens(self):
        for name in TORCH_BASELINES_PIN:
            if name=='mlp_pin':
                continue
            with self.subTest(model=name):
                memory,padding=build_encoder_pin(name,self.cfg,self.prep)(self.batch)
                self.assertEqual(memory.shape[1],self.batch['device_mask'].shape[1]+1)
                self.assertGreater(memory.shape[1],max(len(r['devices']) for r in self.rows)+1)
                torch.testing.assert_close(~padding[:,1:],self.batch['device_mask'])
                self.assertTrue((self.batch['pin_to_device'][self.batch['device_mask']]<0).any())

    def test_device_graph_is_not_accepted_under_pin_model_name(self):
        device_batch=collate_samples(self.rows,self.device_prep)
        for name in TORCH_BASELINES_PIN:
            with self.subTest(model=name),self.assertRaisesRegex(ValueError,'Pin models require'):
                build_model_pin(name,self.cfg,self.prep)(device_batch)

    def test_real_op_and_labels_do_not_affect_any_pin_baseline(self):
        poisoned=copy.deepcopy(self.batch)
        poisoned['op'].fill_(float('nan'));poisoned['performance'].fill_(float('nan'))
        poisoned['op_mask']=~poisoned['op_mask']
        poisoned['performance_mask']=~poisoned['performance_mask']
        poisoned['owner_op']=torch.full((3,6,2),float('nan'))
        poisoned['owner_op_mask']=torch.zeros((3,6,2),dtype=torch.bool)
        for name in TORCH_BASELINES_PIN:
            with self.subTest(model=name),torch.no_grad():
                model=build_model_pin(name,self.cfg,self.prep).eval()
                torch.testing.assert_close(model(self.batch),model(poisoned),atol=0,rtol=0)

    def test_padding_and_design_gradient_path(self):
        for name in TORCH_BASELINES_PIN:
            with self.subTest(model=name):
                torch.manual_seed(29)
                model=build_model_pin(name,self.cfg,self.prep).eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch),model(pad_pin(self.batch)),atol=2e-6,rtol=2e-6)
                batch=dict(self.batch)
                batch['parameters']=batch['parameters'].clone().requires_grad_(True)
                gradient=torch.autograd.grad(model(batch).square().sum(),batch['parameters'])[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()),0.)

    def test_padding_excluded_from_native_batchnorm(self):
        for name in ('deepgen_supervised_pin','accformer_pin'):
            with self.subTest(model=name):
                first=build_model_pin(name,self.cfg,self.prep).train()
                second=copy.deepcopy(first)
                torch.testing.assert_close(first(self.batch),second(pad_pin(self.batch)),atol=2e-5,rtol=2e-5)
                for key,value in first.state_dict().items():
                    if 'running_mean' in key or 'running_var' in key:
                        torch.testing.assert_close(value,second.state_dict()[key],atol=1e-7,rtol=1e-6)

    def test_pin_capacity_is_enforced(self):
        for name in TORCH_BASELINES_PIN:
            with self.subTest(model=name),self.assertRaises(ValueError):
                build_model_pin(name,{**self.cfg,'max_devices':2},self.prep)(self.batch)

    def test_gtn_true_loop_endpoint_projection_and_false_cliques(self):
        live=torch.ones(1,7,dtype=torch.bool)
        owner=torch.tensor([[0,0,1,1,2,2,-1]])
        # No cycle: every owner touches the same net; port is an endpoint only.
        self.assertFalse(build_loop_membership_pin(owner,torch.zeros_like(owner),live).any())
        nets=torch.tensor([[0,1,1,2,2,0,0]])
        membership=build_loop_membership_pin(owner,nets,live)
        torch.testing.assert_close(membership,torch.tensor([[[1,1,1,1,1,1,0]]],dtype=torch.bool))
        # The side pin of an on-loop owner does not automatically join the loop.
        expanded_owner=torch.tensor([[0,0,1,1,2,2,-1,0]])
        expanded_nets=torch.tensor([[0,1,1,2,2,0,0,9]])
        expanded=build_loop_membership_pin(expanded_owner,expanded_nets,torch.ones_like(expanded_owner,dtype=torch.bool))
        self.assertFalse(expanded[0,0,-1])
        self.assertFalse(expanded[0,0,-2])

    def test_gtn_prepared_forward_never_discovers_cycles(self):
        model=build_model_pin('gtn_pin',self.cfg,self.prep).eval()
        store=PinResidentDataset(self.rows,self.prep,'cpu',verbose=False)
        model.encoder.prepare_resident_pin(store)
        with patch('final_exam.models.gtn_pin.build_loop_membership_pin',side_effect=AssertionError('per-batch graph discovery')):
            for _,batch in store.batches(2):
                self.assertIn('gtn_loop_membership_by_pin',batch)
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(model(batch)).all())

    def test_pin_factories_reject_device_preprocessing_and_unsuffixed_name(self):
        with self.assertRaisesRegex(ValueError,'pin preprocessing'):
            build_model_pin('gat_pin',self.cfg,self.device_prep)
        with self.assertRaisesRegex(ValueError,'Unknown pin'):
            build_model_pin('gat',self.cfg,self.prep)


if __name__=='__main__':
    unittest.main()
