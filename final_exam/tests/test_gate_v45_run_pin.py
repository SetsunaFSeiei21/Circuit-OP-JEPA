"""Signed, delayed-fusion gates preserve deployment, export and recovery contracts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.gate_version_registry_pin import V4_GATE_MODELS_PIN, V5_GATE_MODELS_PIN
from final_exam.run_pin import run_experiment_pin, load_checkpoint_pin, validate_request_pin
from final_exam.tests.test_run_pin import tiny_inputs


class GateV45RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        _,self.cfg,self.data,self.path=tiny_inputs(self.root)
        self.cfg['model'].update(encoder_layers=6,cross_layers=3)
        self.cfg['op_gate_pin']=dict(norm_eps=1e-4,n_experts=2,mlp_dropout=.2,strength=1.)
        opts=engine.settings(self.cfg)
        opts['epochs']=dict.fromkeys(opts['epochs'],1)
        opts['epochs'].update(readout=2,student_predicted_op=2,gate=2)
        opts.update(patience=0,batch_size=8)
        self.cfg['op_transfer']=opts
        self.path.write_text(json.dumps(self.cfg))
        torch.set_num_threads(1)

    def run_models(self,name,models,source=None):
        output=self.root/(name+'_pin')
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=run_experiment_pin(self.data,self.path,output,list(models),[17],'cpu',resume_from=source)
        return output,result

    def test_all_eight_variants_export_restore_and_share_dependencies(self):
        models=('device_op_cascade_pin',)+V4_GATE_MODELS_PIN+V5_GATE_MODELS_PIN
        output,result=self.run_models('complete',models)
        self.assertEqual([r['mode'] for r in result['results']],list(models))
        names=[s['stage_id'].split('/')[-1] for s in result['stages']]
        for prerequisite in ('device_op_predictor_pin','design_op_oracle_pin',
                             'device_op_distill_encoder_pin','device_op_distill_readout_pin'):
            self.assertEqual(names.count(prerequisite),1)
        audits={}
        for mode in models[1:]:
            d=output/'seed_17_pin'/mode
            model,saved=load_checkpoint_pin(d/'checkpoint_pin.pt')
            report=json.loads((d/'test_pin.json').read_text())['op_gate_pin']
            self.assertEqual(report['attention_scope'],'fusion_after_first_self_attention')
            self.assertEqual(report['gate_architecture'],model.gate.architecture_contract)
            self.assertEqual(report['gate_network_parameters'],sum(p.numel() for p in model.gate.parameters()))
            self.assertEqual(saved['recipe']['gate_architecture'],model.gate.architecture_contract)
            audit=json.loads((d/'op_gate_audit_pin.json').read_text());audits[mode]=audit
            if '_distill_' in mode or '_supervised_' in mode:
                for frozen in ('predictor','encoder'):
                    self.assertEqual(audit[f'initial_{frozen}_digest'],audit[f'final_{frozen}_digest'])
                if '_gate_decoder_' in mode:
                    self.assertNotEqual(audit['initial_decoder_digest'],audit['final_decoder_digest'])
                else:
                    self.assertEqual(audit['initial_decoder_digest'],audit['final_decoder_digest'])
                self.assertNotEqual(audit['initial_gate_digest'],audit['final_gate_digest'])
        self.assertEqual(audits[V4_GATE_MODELS_PIN[1]]['base_state_digest'],audits[V5_GATE_MODELS_PIN[1]]['base_state_digest'])
        with zipfile.ZipFile(output/'summary_pin.xlsx') as z:
            xml=''.join(z.read(f).decode() for f in z.namelist() if f.endswith('.xml'))
            for mode in models:self.assertIn(mode,xml)
        resumed,res=self.run_models('completed_resume',models,output)
        for stage in res['stages']:
            if stage['epochs']:self.assertEqual(stage['resume']['new_optimizer_steps'],0)
        for mode in models:
            a,_=load_checkpoint_pin(output/'seed_17_pin'/mode/'checkpoint_pin.pt')
            b,_=load_checkpoint_pin(resumed/'seed_17_pin'/mode/'checkpoint_pin.pt')
            self.assertEqual(state_digest(a),state_digest(b))
        subset,_=self.run_models('baseline_subset',['device_op_cascade_pin'],output)
        self.assertNotIn('initial_gate',json.loads((subset/'config_pin.json').read_text())['op_gate_pin'])

    def test_interruption_restores_layer_specific_optimizer_and_rng(self):
        original=engine.fit_stage
        for v in (4,5):
            mode=f'device_op_distill_gate_decoder_fusion_pin_v{v}'
            reference,_=self.run_models(f'reference{v}',[mode])
            def interrupted(ctx,model,name,*args,**kwargs):
                if name==mode:
                    calls=0
                    def stop(_):
                        nonlocal calls
                        calls+=1
                        if calls==2:raise RuntimeError('test interruption')
                    kwargs['before_epoch']=stop
                return original(ctx,model,name,*args,**kwargs)
            with patch.object(engine,'fit_stage',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'test interruption'):
                    self.run_models(f'failed{v}',[mode])
            recovered,result=self.run_models(f'recovered{v}',[mode],self.root/f'failed{v}_pin')
            stage=next(s for s in result['stages'] if s['stage_id'].endswith('/'+mode))
            self.assertTrue(stage['resume']['optimizer_state_restored'])
            self.assertEqual(stage['resume']['restored_epoch'],1)
            a,_=load_checkpoint_pin(reference/'seed_17_pin'/mode/'checkpoint_pin.pt')
            b,_=load_checkpoint_pin(recovered/'seed_17_pin'/mode/'checkpoint_pin.pt')
            self.assertEqual(state_digest(a),state_digest(b))

    def test_invalid_depth_and_legacy_initialization_fail_before_training(self):
        for version in (4,5):
            mode=f'device_op_gate_fusion_pin_v{version}'
            c=copy.deepcopy(self.cfg);c['model'].update(encoder_layers=2,cross_layers=1)
            with self.assertRaisesRegex(ValueError,'at least two'):
                validate_request_pin(c,[mode],[17],'cpu')
            c['final_exam_pin']['model_overrides']['zerosim_supervised_pin'] = dict(encoder_layers=4,cross_layers=3)
            validate_request_pin(c,[mode],[17],'cpu')
            c=copy.deepcopy(self.cfg)
            c['final_exam_pin']['model_overrides']['zerosim_supervised_pin'] = dict(cross_layers=1)
            with self.assertRaisesRegex(ValueError,'at least two'):
                validate_request_pin(c,[mode],[17],'cpu')
            for key in ('initial_gate','epsilon'):
                c=copy.deepcopy(self.cfg);c['op_gate_pin'][key]=.99
                with self.assertRaises(ValueError):validate_request_pin(c,[mode],[17],'cpu')


if __name__=='__main__':unittest.main()
