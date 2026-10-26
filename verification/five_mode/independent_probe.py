"""Independent temporary review probes; analytic fixtures only, no source data."""
import copy, json, sys, tempfile, unittest
from pathlib import Path
import torch

from device_jepa.data import build_manifest, fit_preprocessor, parameter_applicability, validate_dataset
from device_jepa.model import DeviceEncoder, OperatingPointDeviceEncoder, ModelConfig, make_decoder
from device_jepa.baselines import SupervisedPerformanceModel, collate_pin_samples, fit_pin_vocabulary
from device_jepa.tensors import collate_samples
from device_jepa.training import encoder_from_metadata, seed_everything, state_digest


def fixture():
    spec = {'parameter_names':['W_m','L_m','m','R_ohm','C_F','dc_current_A','dc_voltage_V','custom'],
            'op_names':['id_a','gm_s','vds_v'], 'metric_names':['gain'],
            'terminal_names':['d','g','s','b','p','n']}
    data = {'schema_version':1,'feature_spec':spec,'metadata':{'synthetic':True},'samples':[]}
    templates = [
      ('M1','NMOS',{'W_m':1e-6,'L_m':None,'m':0.0},{'id_a':0.0,'gm_s':None,'vds_v':0.5}),
      ('M2','PMOS',{'W_m':2e-6,'L_m':1e-7,'m':1.0},{'id_a':-1e-5,'gm_s':2e-4,'vds_v':-0.5}),
      ('R1','resistor',{'R_ohm':1000.0},{}),
      ('C1','capacitor',{'C_F':1e-12},{}),
      ('I1','independent_current_source',{'dc_current_A':0.0},{}),
      ('V1','independent_voltage_source',{'dc_voltage_V':0.0},{}),
      ('X1','opaque_custom',{'custom':None},{})]
    for circuit in ['A','B']:
      for index in range(6):
        devices=[]
        for name,kind,params,op in templates:
          params,op=copy.deepcopy(params),copy.deepcopy(op)
          if kind=='NMOS': params['W_m'] *= (1+index/10)
          if kind=='PMOS': op['id_a'] *= (1+index/10)
          terminals={'d':'out','g':'in','s':'0','b':'0'} if kind in ['NMOS','PMOS'] else {'p':'out','n':'0'}
          devices.append({'device_id':name,'device_type':kind,'parameters':params,'op':op,'terminals':terminals})
        data['samples'].append({'sample_id':f'{circuit}{index}','design_id':str(index),'circuit_id':circuit,
                               'ports':{'IN':'in','OUT':'out'},'devices':devices,'performance':{'gain':float(index)}})
    # Cross-circuit designs differ in parameters to avoid exact duplicate leakage.
    for sample in data['samples']:
      if sample['circuit_id']=='B': sample['devices'][0]['parameters']['W_m'] *= 3
    validate_dataset(data)
    return data


class IndependentReview(unittest.TestCase):
    def setUp(self):
      seed_everything(23)
      self.data=fixture()
      self.manifest=build_manifest(self.data,config={'train_circuits':['A'],'ood_circuits':['B'],'split_ratios':[.6,.2,.2]})
      self.prep=fit_preprocessor(self.data,self.manifest,True)
      self.sample=copy.deepcopy(self.data['samples'][0])
      self.cfg=ModelConfig(dim=8,heads=2,encoder_layers=2,cross_layers=1,ffn_dim=16,decoder_layers=1,decoder_ffn_dim=16,dropout=0,max_devices=64)
    def model(self, op=True, legacy=False):
      p=self.prep
      enc=(OperatingPointDeviceEncoder(self.cfg,len(p['device_vocab']),8,6,3,
           input_contract_version=p.get('input_contract_version',1),
           num_categories=len(p.get('device_category_vocab',{})),
           num_models=len(p.get('model_vocab',{})),
           num_port_roles=len(p.get('port_role_vocab',{})))
           if op else encoder_from_metadata(self.cfg.to_dict(),p))
      return SupervisedPerformanceModel(enc,make_decoder(self.cfg,1)).eval()
    def test_three_parameter_states_are_distinct(self):
      b=collate_samples([self.sample],self.prep,include_op=False)
      i=b['device_ids'][0].index('M1')
      # L missing applicable, m real zero observed, R not applicable.
      self.assertEqual((b['parameter_applicability'][0,i,1].item(),b['parameter_mask'][0,i,1].item()),(True,False))
      self.assertEqual((b['parameter_applicability'][0,i,2].item(),b['parameter_mask'][0,i,2].item()),(True,True))
      self.assertEqual((b['parameter_applicability'][0,i,3].item(),b['parameter_mask'][0,i,3].item()),(False,False))
      model=self.model(False)
      tokens,live=model.encoder.input_tokens(b)
      self.assertTrue(live[0,i,1] and live[0,i,2]); self.assertFalse(live[0,i,3])
      explicit=copy.deepcopy(self.sample)
      next(d for d in explicit['devices'] if d['device_id']=='M1')['parameter_applicability']={'L_m':False}
      self.assertFalse(torch.equal(model(b),model(collate_samples([explicit],self.prep,include_op=False))))
    def test_no_op_model_invariance_to_raw_op_and_states(self):
      b=collate_samples([self.sample],self.prep,include_op=False)
      changed=copy.deepcopy(self.sample)
      for d in changed['devices']: d.update(op={'id_a':float('nan')},op_target=True,op_state={'bad':object()})
      other=collate_samples([changed],self.prep,include_op=False)
      self.assertFalse(any(k.startswith('op') for k in other))
      model=self.model(False)
      torch.testing.assert_close(model(b),model(other),rtol=0,atol=0)
    def test_op_model_reads_signed_continuous_values_and_not_states(self):
      b=collate_samples([self.sample],self.prep,include_op=True,include_op_states=False)
      model=self.model()
      p=model(b)
      changed=copy.deepcopy(self.sample)
      next(d for d in changed['devices'] if d['device_id']=='M2')['op']['id_a']=1e-5
      for d in changed['devices']: d['op_state']={'bad':object()}
      other=collate_samples([changed],self.prep,include_op=True,include_op_states=False)
      self.assertFalse(any(k.startswith('op_state') for k in other))
      self.assertFalse(torch.equal(p,model(other)))
    def test_masked_payload_and_padding_do_not_change_predictions(self):
      b=collate_samples([self.sample],self.prep,include_op=True,include_op_states=False)
      model=self.model()
      p=model(b)
      poison=copy.deepcopy(b)
      poison['parameters'][~poison['parameter_mask']]=float('nan')
      poison['op'][~poison['op_mask']]=float('nan')
      torch.testing.assert_close(p,model(poison),rtol=0,atol=0)
      short=copy.deepcopy(self.sample); short['devices']=short['devices'][:2]
      short_b=collate_samples([short],self.prep,include_op=True,include_op_states=False)
      padded=collate_samples([short,self.sample],self.prep,include_op=True,include_op_states=False)
      torch.testing.assert_close(model(short_b)[0],model(padded)[0],rtol=1e-5,atol=1e-7)
    def test_heldout_does_not_fit_numeric_or_vocab_metadata(self):
      changed=copy.deepcopy(self.data)
      train=set(self.manifest['train'])
      for row_index, row in enumerate(changed['samples']):
        if row['sample_id'] not in train:
          row['performance']['gain']=1e100
          for d in row['devices']:
            d['device_type']='heldout_custom_type'
            d['parameters']={name:1e100*(row_index+1) for name in self.prep['feature_spec']['parameter_names']}
            d['op']={name:1e100 for name in self.prep['feature_spec']['op_names']}
      # Preserve membership in a refreshed immutable manifest, without refitting split grouping.
      from device_jepa.data import dataset_fingerprint
      manifest=copy.deepcopy(self.manifest); manifest['dataset_fingerprint']=dataset_fingerprint(changed)
      self.assertEqual(self.prep,fit_preprocessor(changed,manifest,True))
    def test_legacy_checkpoint_architecture_reconstructs(self):
      legacy=copy.deepcopy(self.prep)
      legacy.pop('input_contract_version'); legacy.pop('device_category_vocab')
      legacy.pop('model_vocab'); legacy.pop('port_role_vocab')
      old=DeviceEncoder(self.cfg,len(legacy['device_vocab']),8,6).eval()
      new=encoder_from_metadata(self.cfg.to_dict(),legacy).eval()
      new.load_state_dict(old.state_dict(),strict=True)
      batch=collate_samples([self.sample],legacy,include_op=False)
      torch.testing.assert_close(old(batch)[0],new(batch)[0],rtol=0,atol=0)
    def test_unknown_subtype_stays_unknown_not_padding(self):
      changed=copy.deepcopy(self.sample)
      changed['devices'][0]['device_type']='new_variant'
      b=collate_samples([changed],self.prep,include_op=False)
      i=b['device_ids'][0].index('M1')
      self.assertEqual(b['device_types'][0,i].item(),1)
      self.assertEqual(b['device_categories'][0,i].item(),self.prep['device_category_vocab']['unknown'])
      self.assertTrue(b['device_mask'][0,i])

if __name__=='__main__': unittest.main(verbosity=2)
