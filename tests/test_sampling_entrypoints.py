"""Public entrypoints must retain the same capped cohort across stages/seeds."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch

from device_jepa.cli import main
from device_jepa.data import synthetic_dataset, load_dataset


class SamplingEntrypointsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = synthetic_dataset(30, 17)
        circuits = sorted({row['circuit_id'] for row in self.source['samples']})
        self.config = dict(training=dict(seed=17), dataset=dict(
            train_circuits=circuits[:-1], ood_circuits=circuits[-1:],
            split_ratios=[.8, .1, .1], split_seed=17,
            finetune_label_fraction=1., max_samples_per_circuit=20, sampling_seed=29))
        self.data = self.root / 'full.json'
        self.cfg = self.root / 'config.json'
        self.manifest = self.root / 'manifest.json'
        self.data.write_text(json.dumps(self.source))
        self.save_config()

    def save_config(self):
        self.cfg.write_text(json.dumps(self.config))

    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            main(list(map(str, args)))

    def split(self):
        self.invoke('split', '--data', self.data, '--config', self.cfg, '--output', self.manifest)
        return json.loads(self.manifest.read_text())

    def test_split_caps_each_id_and_ood_circuit_before_partition(self):
        manifest = self.split()
        self.assertEqual(len(manifest['ood']), 20)
        id_count = len(self.config['dataset']['train_circuits'])
        self.assertEqual(len(manifest['train']), 16 * id_count)
        self.assertEqual(len(manifest['val']), 2 * id_count)
        self.assertEqual(len(manifest['test']), 2 * id_count)
        self.assertEqual(manifest['sampling']['sampling_seed'], 29)
        for record in manifest['sampling']['circuits'].values():
            self.assertEqual((record['source_count'], record['selected_count']), (30, 20))
        before = self.manifest.read_bytes()
        # No config on verification: recover saved selection, not all 30 rows.
        with patch('device_jepa.sampling.select_sample_ids', side_effect=AssertionError('must replay saved IDs')):
            self.invoke('split', '--data', self.data, '--output', self.manifest)
            self.invoke('split', '--data', self.data, '--config', self.cfg, '--output', self.manifest)
        self.assertEqual(before, self.manifest.read_bytes())
        self.assertEqual(len(load_dataset(self.data)['samples']), len(self.source['samples']))

    def test_training_seed_changes_do_not_change_selection(self):
        manifest = self.split()
        seen = []
        def pretrain(dataset, saved, config, *args, **kwargs):
            self.assertEqual(saved, manifest)
            seen.append((config['training']['seed'],
                         sorted(row['sample_id'] for row in dataset['samples'])))
            return 'mock_checkpoint.pt'
        with patch('device_jepa.training.pretrain', side_effect=pretrain), \
             patch('device_jepa.sampling.select_sample_ids', side_effect=AssertionError('must replay saved IDs')):
            for seed in (17, 43):
                self.config['training']['seed'] = seed
                self.save_config()
                self.invoke('pretrain', '--data', self.data, '--config', self.cfg,
                            '--manifest', self.manifest, '--output', self.root / f'seed_{seed}')
        self.assertNotEqual(seen[0][0], seen[1][0])
        self.assertEqual(seen[0][1], seen[1][1])
        self.assertEqual(set(seen[0][1]), set().union(*(set(manifest[s]) for s in ('train', 'val', 'test', 'ood'))))

    def test_evaluation_recovers_checkpoint_selection_from_full_canonical(self):
        manifest = self.split()
        checkpoint = dict(stage='baseline', options=self.config, manifest=manifest)
        selected = []
        def evaluate(dataset, *args):
            selected.extend(row['sample_id'] for row in dataset['samples'])
            return {'split': 'test', 'metrics': {}}
        with patch('device_jepa.training.load_checkpoint', return_value=checkpoint), \
             patch('device_jepa.baselines.evaluate_baseline', side_effect=evaluate), \
             patch('device_jepa.sampling.select_sample_ids', side_effect=AssertionError('must replay saved IDs')):
            self.invoke('evaluate', '--data', self.data, '--checkpoint', 'mock.pt',
                        '--output', self.root / 'test_report.json')
        self.assertEqual(set(selected), set().union(*(set(manifest[s]) for s in ('train', 'val', 'test', 'ood'))))

    def test_invalid_import_budget_fails_before_opening_raw_source(self):
        self.config['dataset']['max_samples_per_circuit'] = 0
        self.save_config()
        with patch('device_jepa.adapter.adapt_analog_agent') as adapt:
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                self.invoke('import_analog_agent', '--input', 'missing_raw', '--config', self.cfg,
                            '--output', self.root / 'new.json')
            adapt.assert_not_called()
        self.assertFalse((self.root / 'new.json').exists())

    def test_sample_command_keeps_input_and_refuses_output_overwrite(self):
        before = self.data.read_bytes()
        output = self.root / 'selected.json'
        self.invoke('sample', '--data', self.data, '--config', self.cfg, '--output', output)
        selected = load_dataset(output)
        self.assertEqual(len(selected['samples']), 20 * len(selected['sampling']['circuits']))
        saved = output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            self.invoke('sample', '--data', self.data, '--config', self.cfg, '--output', output)
        self.assertEqual(saved, output.read_bytes())
        self.assertEqual(before, self.data.read_bytes())

    def test_sample_command_has_no_training_runtime_dependency(self):
        output = self.root / 'selected_without_torch.json'
        with patch.dict('sys.modules', {'torch': None}):
            self.invoke('sample', '--data', self.data, '--config', self.cfg, '--output', output)
        self.assertEqual(len(load_dataset(output)['samples']), 20 * len(self.config['dataset']['train_circuits'] +
                         self.config['dataset']['ood_circuits']))

    def test_canonical_comparison_materializes_one_cohort_for_multiple_seeds(self):
        from scripts.run import run_comparison
        from device_jepa.states import apply_op_states
        from device_jepa.training import load_checkpoint
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'configs/smoke.json').read_text())
        config['dataset'] = dict(self.config['dataset'], max_samples_per_circuit=10)
        self.cfg.write_text(json.dumps(config))
        self.data.write_text(json.dumps(apply_op_states(self.source)))
        output = self.root / 'comparison'
        def invoke(command, **kwargs):
            main(command[3:])
            return subprocess.CompletedProcess(command, 0)
        with patch.object(run_comparison.subprocess, 'run', side_effect=invoke), \
             contextlib.redirect_stdout(io.StringIO()):
            run_comparison.main(['--data', str(self.data), '--config', str(self.cfg),
                '--output', str(output), '--seeds', '17', '43', '--models', 'zerosim_device', '--smoke'])
        selected = load_dataset(output / 'sampled_data.json')
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(len(selected['samples']), 10 * len(manifest['sampling']['circuits']))
        self.assertEqual(len(manifest['ood']), 10)
        for seed in (17, 43):
            saved = load_checkpoint(output / 'zerosim_device' / f'seed_{seed}' / 'zerosim_device_best.pt')
            self.assertEqual(saved['manifest'], manifest)
            self.assertEqual(saved['preprocessing']['fitted_sample_ids'], manifest['train'])


if __name__ == '__main__':
    unittest.main()
