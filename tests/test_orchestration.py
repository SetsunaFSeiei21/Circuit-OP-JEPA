"""Exercise the actual orchestration contract, with CLI calls in-process for speed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from device_jepa.cli import main as cli
from device_jepa.data import synthetic_dataset
from device_jepa.states import apply_op_states
from device_jepa.comparison import MODES, LEGACY_MODES, OP_MODES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison_runner', ROOT/'scripts/run/run_comparison.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OrchestrationTests(unittest.TestCase):
    def test_five_modes_automatically_finish_with_paired_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data.json'
            data.write_text(json.dumps(apply_op_states(synthetic_dataset(12))))
            output = root/'experiment'
            calls = []
            def invoke(command, **kwargs):
                self.assertEqual(command[1:3], ['-m', 'device_jepa'])
                calls.append(command[3:])
                cli(command[3:])
                return subprocess.CompletedProcess(command, 0)
            argv = ['run_comparison.py', '--data', str(data), '--config', str(ROOT/'configs/smoke.json'),
                    '--output', str(output), '--seeds', '17', '--smoke']
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', side_effect=invoke):
                runner.main()
            result = json.loads((output/'comparison/comparison.json').read_text())
            self.assertEqual({r['mode'] for r in result['per_seed']}, set(MODES))
            self.assertEqual({r['split'] for r in result['per_seed']}, {'test', 'ood'})
            self.assertEqual(sum(c[0] == 'train_baseline' for c in calls), 4)
            self.assertEqual(sum(c[0] == 'evaluate' for c in calls), 10)
            self.assertEqual({r['node_granularity'] for r in result['paired_summary']}, {'pin', 'device'})
            self.assertEqual({r['split'] for r in result['paired_summary']}, {'test', 'ood'})
            self.assertTrue(result['complete'])
            self.assertEqual(result['epoch_budget_policy'], 'equal_total')
            self.assertEqual(result['total_epoch_budget'], 2)
            self.assertEqual(len(result['run_costs']), 5)
            self.assertEqual(len(result['evaluation_costs']), 10)
            for cost in result['run_costs']:
                self.assertEqual(cost['total_epochs'], 2)
                mode_dir = output / cost['mode'] / 'seed_17'
                if cost['mode'] == 'jepa_zerosim':
                    self.assertEqual(len(json.loads((mode_dir/'pretrain_history.json').read_text())), 1)
                    self.assertEqual(len(json.loads((mode_dir/'decoder_history.json').read_text())), 1)
                    self.assertEqual(cost['supervised_epochs'], 1)
                else:
                    self.assertEqual(len(json.loads((mode_dir/f"{cost['mode']}_history.json").read_text())), 2)
                    self.assertEqual(cost['supervised_epochs'], 2)
                self.assertEqual(cost['requires_op_at_inference'], cost['mode'] in OP_MODES)
                self.assertEqual(set(cost['evaluation_splits']), {'test', 'ood'})
                self.assertGreater(cost['supervised_optimizer_steps'], 0)
            for cost in result['evaluation_costs']:
                self.assertIsNone(cost['op_acquisition_cost_seconds'])
            for filename in ['summary.csv', 'per_seed.csv', 'costs.csv', 'evaluation_costs.csv', 'paired_per_seed.csv', 'paired_summary.csv', 'README.txt']:
                self.assertTrue((output/'comparison'/filename).is_file())
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                runner.main()

    def test_explicit_legacy_three_modes_remain_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data.json'
            data.write_text(json.dumps(apply_op_states(synthetic_dataset(12))))
            output = root/'experiment'
            def invoke(command, **kwargs):
                cli(command[3:])
                return subprocess.CompletedProcess(command, 0)
            argv = ['run_comparison.py', '--data', str(data), '--config', str(ROOT/'configs/smoke.json'),
                    '--output', str(output), '--seeds', '17', '--smoke', '--modes', *LEGACY_MODES]
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', side_effect=invoke):
                runner.main()
            result = json.loads((output/'comparison/comparison.json').read_text())
            self.assertEqual(set(result['compared_modes']), set(LEGACY_MODES))
            self.assertEqual(result['paired_per_seed'], [])
            self.assertTrue(result['complete'])
            self.assertFalse((output/'comparison/paired_summary.csv').exists())

    def test_duplicate_modes_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['run_comparison.py', '--data', str(Path(tmp)/'data.json'),
                    '--config', str(ROOT/'configs/smoke.json'), '--output', str(Path(tmp)/'experiment'),
                    '--modes', 'zerosim_pin', 'zerosim_pin']
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run') as invoke:
                with self.assertRaises(SystemExit):
                    runner.main()
                invoke.assert_not_called()

    def test_failure_is_not_reported_as_partial_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'experiment'
            argv = ['run_comparison.py', '--data', str(Path(tmp)/'data.json'),
                    '--config', str(ROOT/'configs/smoke.json'), '--output', str(output), '--smoke']
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run',
                    side_effect=subprocess.CalledProcessError(1, ['device_jepa'])):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.main()
            self.assertFalse((output/'comparison/comparison.json').exists())


if __name__ == '__main__':
    unittest.main()
