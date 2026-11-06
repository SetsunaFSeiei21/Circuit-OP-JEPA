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
from experiments.op_transfer.engine import DataContext
from final_exam.run import MODEL_NAMES, REFERENCES, load_checkpoint, run_experiment, validate_request

ROOT = Path(__file__).resolve().parents[2]


def tiny_inputs(directory):
    data = synthetic_dataset(24)
    data['samples'] = [r for r in data['samples'] if r['circuit_id'] == 'synthetic_0']
    for row in data['samples']:
        row['devices'] = row['devices'][:4]
    config = json.loads((ROOT / 'final_exam/configs/default.json').read_text())
    config['dataset'].update(train_circuits=['synthetic_0'], ood_circuits=[], split_ratios=[.6,.2,.2])
    config['model'].update(dim=8, heads=2, encoder_layers=2, cross_layers=1, ffn_dim=16,
                           decoder_ffn_dim=16, decoder_layers=1, max_devices=8, dropout=0.)
    config['op_transfer'].update(batch_size=8, progress_every=100)
    config['final_exam']['xgboost'].update(max_depth=2)
    config['final_exam']['model_overrides'] = {'gat': {'ffn_dim': 24}}
    data_path, cfg_path = directory/'data.json', directory/'input.json'
    data_path.write_text(json.dumps(data))
    cfg_path.write_text(json.dumps(config))
    return data, config, data_path, cfg_path


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.config, self.data_path, self.cfg_path = tiny_inputs(self.root)
        torch.set_num_threads(1)

    def test_all_models_two_seeds_save_replay_and_boundaries(self):
        output = self.root/'complete'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment(self.data_path,self.cfg_path,output,list(MODEL_NAMES),[17,29],
                                    'cpu',smoke=True)
        self.assertEqual(len(result['results']), 26)
        self.assertEqual(json.loads((output/'run.json').read_text())['status'], 'complete')
        with zipfile.ZipFile(output/'summary.xlsx') as book:
            self.assertIn('xl/workbook.xml',book.namelist())
        cfg = json.loads((output/'config.json').read_text())
        manifest = json.loads((output/'manifest.json').read_text())
        (self.root/'replay').mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = DataContext(self.data,manifest,cfg,torch.device('cpu'),{},self.root/'replay')
            batch = next(ctx.batches('test',32,op=True))
        poison = copy.deepcopy(batch)
        for key in ('op','performance','physical_performance'):
            poison[key].fill_(float('nan'))
        poison['op_mask'] = ~poison['op_mask']
        poison['performance_mask'] = ~poison['performance_mask']
        for seed in (17,29):
            stages = [s['stage_id'] for s in result['stages'] if s['seed'] == seed]
            self.assertEqual(stages.count(f'seed_{seed}/op_predictor'),1)
            self.assertEqual(stages.count(f'seed_{seed}/design_op_oracle'),1)
            for name in MODEL_NAMES:
                directory = output/f'seed_{seed}'/name
                checkpoint = directory/('checkpoint.json' if name == 'xgboost' else 'checkpoint.pt')
                model, metadata = load_checkpoint(checkpoint)
                with torch.no_grad():
                    prediction = model(batch)
                    self.assertTrue(torch.isfinite(prediction).all(),name)
                    if name not in REFERENCES:
                        torch.testing.assert_close(prediction,model(poison),atol=0,rtol=0,msg=name)
                self.assertTrue((directory/'config.json').is_file())
                if name == 'gat':
                    self.assertEqual(metadata['model_config']['ffn_dim'],24)
                if name == 'student_predicted_op_distill':
                    audit=json.loads((directory/'distillation.json').read_text())
                    self.assertTrue(audit['decoder_unchanged_during_encoder'])
                    self.assertTrue(audit['encoder_unchanged_during_decoder'])
                restored, _=load_checkpoint(checkpoint)
                with torch.no_grad():
                    torch.testing.assert_close(prediction,restored(batch),atol=0,rtol=0)
            for dependency in ('op_predictor','design_op_oracle','deepgen_op_predictor'):
                model, metadata=load_checkpoint(output/f'seed_{seed}'/'shared'/dependency/'checkpoint.pt')
                self.assertEqual(metadata['mode'],dependency)
                self.assertIn('preprocessing',metadata)
                if dependency != 'deepgen_op_predictor':
                    self.assertFalse(any(p.requires_grad for p in model.parameters()))
        # Every model-level log and shared train-stage log is linked to the run.
        self.assertTrue((output/'logs/seed_17/shared/design_op_oracle/train.log').is_file())
        self.assertTrue((output/'logs/seed_17/student_predicted_op_distill/decoder/train.log').is_file())

    def test_selected_stages_only_and_prewrite_checks(self):
        output=self.root/'selected'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result=run_experiment(self.data_path,self.cfg_path,output,['mlp'],[17],'cpu',smoke=True)
        self.assertEqual([s['stage_id'] for s in result['stages']],['seed_17/mlp'])
        self.assertFalse((output/'seed_17/shared').exists())
        student_output = self.root/'student_only'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            student_result = run_experiment(self.data_path,self.cfg_path,student_output,
                ['student_predicted_op_distill'],[17],'cpu',smoke=True)
        self.assertEqual({s['stage_id'] for s in student_result['stages']}, {
            'seed_17/op_predictor','seed_17/design_op_oracle',
            'seed_17/predicted_op_distill_encoder','seed_17/predicted_op_distill_decoder'})
        self.assertTrue((student_output/'seed_17/shared/design_op_oracle/checkpoint.pt').is_file())
        self.assertFalse((student_output/'seed_17/design_op_oracle').exists())
        self.assertFalse((student_output/'seed_17/predicted_op').exists())
        before=(output/'run.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'empty'):
            run_experiment(self.data_path,self.cfg_path,output,['mlp'],[17],'cpu',smoke=True)
        self.assertEqual((output/'run.json').read_bytes(),before)
        self.config['op_transfer']['epochs']['readout']=61
        self.cfg_path.write_text(json.dumps(self.config))
        bad=self.root/'invalid'
        with self.assertRaisesRegex(ValueError,'Matched'):
            run_experiment(self.data_path,self.cfg_path,bad,['mlp'],[17],'cpu')
        self.assertFalse(bad.exists())

    def test_invalid_model_seed_or_topology(self):
        for models,seeds in ((['typo'],[17]),(['mlp','mlp'],[17]),(['mlp'],[17,17])):
            with self.assertRaises(ValueError):
                validate_request(self.config,models,seeds,'cpu')
        terminals = self.data['samples'][0]['devices'][0]['terminals']
        terminals[next(iter(terminals))]='changed_net'
        self.data_path.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError,'topology'):
            run_experiment(self.data_path,self.cfg_path,self.root/'badtopology',['mlp'],[17],'cpu')
        self.assertFalse((self.root/'badtopology').exists())

    def test_manifest_configuration_mismatch_is_rejected_before_write(self):
        dataset_config = {**self.config['dataset'], 'split_seed': 43}
        path = self.root/'different_manifest.json'
        path.write_text(json.dumps(build_manifest(self.data,config=dataset_config)))
        with self.assertRaisesRegex(ValueError,'split_seed'):
            run_experiment(self.data_path,self.cfg_path,self.root/'mismatch',['mlp'],[17],
                           'cpu',manifest_path=path)
        self.assertFalse((self.root/'mismatch').exists())

    def test_log_acquisition_failure_records_failed_run(self):
        logs = self.root/'file_instead_of_directory'
        logs.write_text('preserve')
        output = self.root/'log_failure'
        with self.assertRaises(FileExistsError):
            run_experiment(self.data_path,self.cfg_path,output,['mlp'],[17],'cpu',log_dir=logs,smoke=True)
        self.assertEqual(json.loads((output/'run.json').read_text())['status'],'failed')
        self.assertEqual(logs.read_text(),'preserve')


if __name__ == '__main__':
    unittest.main()
