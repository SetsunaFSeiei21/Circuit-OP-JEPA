"""End-to-end artifact replay and deployment boundary checks on a tiny circuit."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from device_jepa.data import build_manifest, synthetic_dataset
from final_exam.data_pin import PinDataContext
from final_exam.run_pin import MODEL_NAMES_PIN, REFERENCES_PIN, load_checkpoint_pin, run_experiment_pin, validate_request_pin

ROOT = Path(__file__).resolve().parents[2]


def tiny_inputs(directory):
    data = synthetic_dataset(24)
    data['samples'] = [r for r in data['samples'] if r['circuit_id'] == 'synthetic_0']
    for row in data['samples']:
        row['devices'] = row['devices'][:4]
    config = json.loads((ROOT / 'final_exam/configs/default_pin.json').read_text())
    config['dataset'].update(train_circuits=['synthetic_0'], ood_circuits=[], split_ratios=[.6,.2,.2])
    config['model'].update(dim=8, heads=2, encoder_layers=2, cross_layers=1, ffn_dim=16,
                           decoder_ffn_dim=16, decoder_layers=1, max_devices=32, dropout=0.)
    config['op_transfer'].update(batch_size=8, progress_every=100)
    config['final_exam_pin']['xgboost'].update(max_depth=2)
    config['final_exam_pin']['model_overrides'] = {'gat_pin': {'ffn_dim': 24}}
    data_path, cfg_path = directory/'data.json', directory/'input.json'
    data_path.write_text(json.dumps(data))
    cfg_path.write_text(json.dumps(config))
    return data, config, data_path, cfg_path


class RunnerPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.config, self.data_path, self.cfg_path = tiny_inputs(self.root)
        torch.set_num_threads(1)

    def test_all_models_two_seeds_save_replay_and_boundaries(self):
        output = self.root/'complete_pin'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(self.data_path,self.cfg_path,output,list(MODEL_NAMES_PIN),[17,29],
                                    'cpu',smoke=True)
        self.assertEqual(len(result['results']), len(MODEL_NAMES_PIN) * 2)
        self.assertTrue(all(p.stem.endswith('_pin') for p in output.rglob('*') if p.is_file()))
        self.assertEqual(json.loads((output/'run_pin.json').read_text())['status'], 'complete')
        with zipfile.ZipFile(output/'summary_pin.xlsx') as book:
            self.assertIn('xl/workbook.xml',book.namelist())
        cfg = json.loads((output/'config_pin.json').read_text())
        manifest = json.loads((output/'manifest_pin.json').read_text())
        (self.root/'replay').mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = PinDataContext(self.data,manifest,cfg,torch.device('cpu'),{},self.root/'replay')
            batch = next(ctx.batches('test',32,op=True))
        poison = copy.deepcopy(batch)
        for key in ('op','owner_op','performance','physical_performance'):
            poison[key].fill_(float('nan'))
        poison['op_mask'] = ~poison['op_mask']
        poison['owner_op_mask'] = ~poison['owner_op_mask']
        poison['performance_mask'] = ~poison['performance_mask']
        for seed in (17,29):
            stages = [s['stage_id'] for s in result['stages'] if s['seed'] == seed]
            self.assertEqual(stages.count(f'seed_{seed}_pin/op_predictor_pin'),1)
            self.assertEqual(stages.count(f'seed_{seed}_pin/design_op_oracle_pin'),1)
            for name in MODEL_NAMES_PIN:
                directory = output/f'seed_{seed}_pin'/name
                checkpoint = directory/('checkpoint_pin.json' if name == 'xgboost_pin' else 'checkpoint_pin.pt')
                model, metadata = load_checkpoint_pin(checkpoint)
                with torch.no_grad():
                    prediction = model(batch)
                    self.assertTrue(torch.isfinite(prediction).all(),name)
                    if name not in REFERENCES_PIN:
                        torch.testing.assert_close(prediction,model(poison),atol=0,rtol=0,msg=name)
                self.assertTrue((directory/'config_pin.json').is_file())
                if name == 'gat_pin':
                    self.assertEqual(metadata['model_config']['ffn_dim'],24)
                if name == 'student_predicted_op_distill_pin':
                    audit=json.loads((directory/'distillation_pin.json').read_text())
                    self.assertTrue(audit['decoder_unchanged_during_encoder'])
                    self.assertTrue(audit['encoder_unchanged_during_decoder'])
                restored, _=load_checkpoint_pin(checkpoint)
                with torch.no_grad():
                    torch.testing.assert_close(prediction,restored(batch),atol=0,rtol=0)
            for dependency in ('op_predictor_pin','design_op_oracle_pin','deepgen_op_predictor_pin'):
                model, metadata=load_checkpoint_pin(output/f'seed_{seed}_pin'/'shared_pin'/dependency/'checkpoint_pin.pt')
                self.assertEqual(metadata['mode'],dependency)
                self.assertIn('preprocessing',metadata)
                if dependency != 'deepgen_op_predictor_pin':
                    self.assertFalse(any(p.requires_grad for p in model.parameters()))
        # Every model-level log and shared train-stage log is linked to the run.
        self.assertTrue((output/'logs_pin/seed_17_pin/shared_pin/design_op_oracle_pin/train_pin.log').is_file())
        self.assertTrue((output/'logs_pin/seed_17_pin/student_predicted_op_distill_pin/decoder_pin/train_pin.log').is_file())

    def test_selected_stages_only_and_prewrite_checks(self):
        output=self.root/'selected_pin'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result=run_experiment_pin(self.data_path,self.cfg_path,output,['mlp_pin'],[17],'cpu',smoke=True)
        self.assertEqual([s['stage_id'] for s in result['stages']],['seed_17_pin/mlp_pin'])
        self.assertFalse((output/'seed_17_pin/shared_pin').exists())
        student_output = self.root/'student_only_pin'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            student_result = run_experiment_pin(self.data_path,self.cfg_path,student_output,
                ['student_predicted_op_distill_pin'],[17],'cpu',smoke=True)
        self.assertEqual({s['stage_id'] for s in student_result['stages']}, {
            'seed_17_pin/op_predictor_pin','seed_17_pin/design_op_oracle_pin',
            'seed_17_pin/predicted_op_distill_encoder_pin','seed_17_pin/predicted_op_distill_decoder_pin'})
        self.assertTrue((student_output/'seed_17_pin/shared_pin/design_op_oracle_pin/checkpoint_pin.pt').is_file())
        self.assertFalse((student_output/'seed_17_pin/design_op_oracle_pin').exists())
        self.assertFalse((student_output/'seed_17_pin/predicted_op_pin').exists())
        before=(output/'run_pin.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'empty'):
            run_experiment_pin(self.data_path,self.cfg_path,output,['mlp_pin'],[17],'cpu',smoke=True)
        self.assertEqual((output/'run_pin.json').read_bytes(),before)
        self.config['op_transfer']['epochs']['readout']=61
        self.cfg_path.write_text(json.dumps(self.config))
        bad=self.root/'invalid_pin'
        with self.assertRaisesRegex(ValueError,'Matched'):
            run_experiment_pin(self.data_path,self.cfg_path,bad,['mlp_pin'],[17],'cpu')
        self.assertFalse(bad.exists())

    def test_invalid_model_seed_or_topology(self):
        for models,seeds in ((['typo'],[17]),(['mlp_pin','mlp_pin'],[17]),(['mlp_pin'],[17,17])):
            with self.assertRaises(ValueError):
                validate_request_pin(self.config,models,seeds,'cpu')
        terminals = self.data['samples'][0]['devices'][0]['terminals']
        terminals[next(iter(terminals))]='changed_net'
        self.data_path.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError,'topology'):
            run_experiment_pin(self.data_path,self.cfg_path,self.root/'badtopology_pin',['mlp_pin'],[17],'cpu')
        self.assertFalse((self.root/'badtopology_pin').exists())

    def test_manifest_configuration_mismatch_is_rejected_before_write(self):
        dataset_config = {**self.config['dataset'], 'split_seed': 43}
        path = self.root/'different_manifest_pin.json'
        path.write_text(json.dumps(build_manifest(self.data,config=dataset_config)))
        with self.assertRaisesRegex(ValueError,'split_seed'):
            run_experiment_pin(self.data_path,self.cfg_path,self.root/'mismatch_pin',['mlp_pin'],[17],
                           'cpu',manifest_path=path)
        self.assertFalse((self.root/'mismatch_pin').exists())

    def test_log_acquisition_failure_records_failed_run(self):
        logs = self.root/'file_instead_of_directory_pin'
        logs.write_text('preserve')
        output = self.root/'log_failure_pin'
        with self.assertRaises(FileExistsError):
            run_experiment_pin(self.data_path,self.cfg_path,output,['mlp_pin'],[17],'cpu',log_dir=logs,smoke=True)
        self.assertEqual(json.loads((output/'run_pin.json').read_text())['status'],'failed')
        self.assertEqual(logs.read_text(),'preserve')

    def test_device_names_configs_and_checkpoint_formats_are_distinct(self):
        with self.assertRaises(ValueError):
            validate_request_pin(self.config,['mlp'],[17],'cpu')
        wrong = copy.deepcopy(self.config)
        wrong['final_exam'] = wrong.pop('final_exam_pin')
        with self.assertRaisesRegex(ValueError,'not interchangeable'):
            validate_request_pin(wrong,['mlp_pin'],[17],'cpu')
        path = self.root/'device_checkpoint.pt'
        torch.save({'format':'final_exam_v1'},path)
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            load_checkpoint_pin(path)
        from final_exam.run import load_checkpoint
        torch.save({'format':'final_exam_pin_v1'},path)
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            load_checkpoint(path)

    def test_pin_capacity_is_checked_before_output_and_override_remains_usable(self):
        self.config['model']['max_devices'] = 8
        self.cfg_path.write_text(json.dumps(self.config))
        output = self.root/'capacity_pin'
        with self.assertRaisesRegex(ValueError,'pin/port tokens'):
            run_experiment_pin(self.data_path,self.cfg_path,output,['mlp_pin'],[17],'cpu',smoke=True)
        self.assertFalse(output.exists())
        self.config['final_exam_pin']['model_overrides']['mlp_pin'] = {'max_devices':32}
        self.cfg_path.write_text(json.dumps(self.config))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_experiment_pin(self.data_path,self.cfg_path,output,['mlp_pin'],[17],'cpu',smoke=True)
        self.assertEqual(json.loads((output/'run_pin.json').read_text())['status'],'complete')


if __name__ == '__main__':
    unittest.main()
