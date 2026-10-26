"""Selection controls actual training, best-checkpoint evaluation and summaries."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from device_jepa.cli import main as cli
from device_jepa.data import synthetic_dataset
from device_jepa.methods import (METHODS, DEFAULT_MODELS, Method, add_model_arguments,
                                 select_models)
from device_jepa.states import apply_op_states

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selected_runner', ROOT / 'scripts/run/run_comparison.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / 'configs/smoke.json').read_text())
        self.config['model'].update(dim=8, heads=2, ffn_dim=16, decoder_ffn_dim=16,
                                    dropout=0., max_devices=128)
        self.config['dataset'].update(train_circuits=['synthetic_0'], ood_circuits=[])
        data = apply_op_states(synthetic_dataset(12))
        data['samples'] = [s for s in data['samples'] if s['circuit_id'] == 'synthetic_0']
        self.data_path = self.root / 'data.json'
        self.data_path.write_text(json.dumps(data))

    def run_experiment(self, selected, *, from_config=False, seeds=(17,)):
        config = copy.deepcopy(self.config)
        config['experiment'] = {'models': selected if from_config else ['zerosim_pin']}
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(config))
        output = self.root / ('_'.join(selected) + ('_configured' if from_config else '_cli'))
        args = ['--data', str(self.data_path), '--config', str(config_path), '--output', str(output),
                '--smoke', '--seeds', *map(str, seeds)]
        if not from_config:
            args += ['--models', *selected]
        calls = []
        def invoke(command, **kwargs):
            calls.append(command[3:])
            cli(command[3:])
            return subprocess.CompletedProcess(command, 0)
        with patch.object(runner.subprocess, 'run', side_effect=invoke), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main(args)
        result = json.loads((output / 'comparison/comparison.json').read_text())
        self.assertTrue(result['complete'])
        self.assertEqual(result['expected_modes'], selected)
        self.assertEqual({r['mode'] for r in result['per_seed']}, set(selected))
        self.assertEqual({r['mode'] for r in result['run_costs']}, set(selected))
        self.assertEqual({r['mode'] for r in result['evaluation_costs']}, set(selected))
        self.assertEqual({r['split'] for r in result['per_seed']}, {'test'})
        self.assertEqual(len(result['run_costs']), len(selected) * len(seeds))
        self.assertEqual(json.loads((output / 'config.json').read_text())['experiment']['models'], selected)
        self.assertEqual(json.loads((output / 'experiment.json').read_text()),
                         dict(models=selected, seeds=list(seeds), evaluation_splits=['test'], complete=True))
        for name in METHODS:
            self.assertEqual((output / name).exists(), name in selected, name)
        baseline_calls = [c for c in calls if c[0] == 'train_baseline']
        self.assertEqual([c[c.index('--mode') + 1] for c in baseline_calls],
                         [m for _ in seeds for m in selected if METHODS[m].workflow == 'baseline'])
        self.assertEqual(sum(c[0] == 'pretrain' for c in calls), len(seeds) if 'jepa_zerosim' in selected else 0)
        evaluations = [c for c in calls if c[0] == 'evaluate']
        self.assertEqual(len(evaluations), len(seeds) * len(selected))
        for c in evaluations:
            checkpoint = Path(c[c.index('--checkpoint') + 1])
            mode = checkpoint.parent.parent.name
            self.assertIn(mode, selected)
            self.assertEqual(checkpoint.name, METHODS[mode].best_checkpoint(mode))
        return result

    def test_each_model_alone_runs_only_its_own_stages_and_test(self):
        for name in DEFAULT_MODELS:
            with self.subTest(model=name):
                result = self.run_experiment([name])
                self.assertEqual(result['paired_summary'], [])

    def test_cli_subset_overrides_config_across_multiple_seeds(self):
        self.run_experiment(['jepa_zerosim', 'zerosim_device'], seeds=(17, 29))

    def test_configured_op_pair_generates_only_its_own_pair_table(self):
        result = self.run_experiment(['zerosim_device', 'zerosim_device_op'], from_config=True)
        self.assertEqual({r['node_granularity'] for r in result['paired_summary']}, {'device'})

    def test_cli_alias_validation_and_explicit_all(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        self.assertEqual(parser.parse_args(['--modes', 'zerosim_device']).models, ['zerosim_device'])
        self.assertEqual(parser.parse_args(['--models', 'all']).models, list(METHODS))
        for args in (['--models'], ['--models', 'unknown'],
                     ['--models', 'zerosim_pin', 'zerosim_pin'],
                     ['--models', 'all', 'zerosim_pin'],
                     ['--models', 'zerosim_pin', '--modes', 'zerosim_device']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(args)

    def test_new_registration_does_not_expand_legacy_or_explicit_selections(self):
        extra = Method('baseline', 'device', False, 'test-only future registration')
        with patch.dict(METHODS, {'future_ablation': extra}):
            self.assertEqual(select_models({}), list(DEFAULT_MODELS))
            options = {'experiment': {'models': ['zerosim_device']}}
            self.assertEqual(select_models(options), ['zerosim_device'])
            parser = argparse.ArgumentParser()
            add_model_arguments(parser)
            self.assertIn('future_ablation', parser.parse_args(['--models', 'all']).models)

    def test_invalid_config_selection_fails_before_output_or_children(self):
        for invalid in ([], 'zerosim_device', ['unknown'], ['zerosim_pin', 'zerosim_pin']):
            config = copy.deepcopy(self.config)
            config['experiment'] = {'models': invalid}
            path = self.root / 'bad-config.json'
            path.write_text(json.dumps(config))
            output = self.root / 'bad-output'
            with patch.object(runner.subprocess, 'run') as invoke, self.assertRaises(ValueError):
                runner.main(['--data', str(self.data_path), '--config', str(path), '--output', str(output)])
            invoke.assert_not_called()
            self.assertFalse(output.exists())

    def test_direct_smoke_honors_selection_without_pretraining(self):
        config = copy.deepcopy(self.config)
        config['experiment'] = {'models': ['zerosim_device_op']}
        path = self.root / 'smoke-config.json'
        path.write_text(json.dumps(config))
        output = self.root / 'selected-smoke'
        data = json.loads(self.data_path.read_text())
        with patch('device_jepa.cli.synthetic_dataset', return_value=data), \
             patch('device_jepa.training.pretrain', side_effect=AssertionError('unselected JEPA ran')), \
             contextlib.redirect_stdout(io.StringIO()):
            cli(['smoke', '--config', str(path), '--output', str(output)])
        result = json.loads((output / 'comparison/comparison.json').read_text())
        self.assertTrue(result['complete'])
        self.assertEqual(result['expected_modes'], ['zerosim_device_op'])
        self.assertFalse(list(output.rglob('pretrain_best.pt')))
        self.assertFalse(list(output.rglob('decoder_best.pt')))


if __name__ == '__main__':
    unittest.main()
