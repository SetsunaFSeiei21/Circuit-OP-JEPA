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

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison_runner', ROOT/'scripts/run/run_comparison.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OrchestrationTests(unittest.TestCase):
    def test_three_modes_automatically_finish_with_comparison(self):
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
            self.assertEqual({r['mode'] for r in result['per_seed']}, {'zerosim_pin', 'zerosim_device', 'jepa_zerosim'})
            self.assertEqual({r['split'] for r in result['per_seed']}, {'test', 'ood'})
            self.assertEqual(sum(c[0] == 'train_baseline' for c in calls), 2)
            self.assertEqual(sum(c[0] == 'evaluate' for c in calls), 6)
            for filename in ['summary.csv', 'per_seed.csv', 'costs.csv', 'README.txt']:
                self.assertTrue((output/'comparison'/filename).is_file())
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                runner.main()

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
