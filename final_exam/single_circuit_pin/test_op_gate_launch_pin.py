"""The OP-gate launcher selects eight controls and snapshots the chosen config."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import unittest

from final_exam.single_circuit_pin import test_launch_pin as launch_tests


class GateLauncherPinTests(unittest.TestCase):
    setUp = launch_tests.LauncherTests.setUp

    # Reuse fixture construction without collecting the existing launcher tests.
    def make_submission_fixture(self, sbatch_body='echo 98765'):
        script, task, environment = launch_tests.LauncherTests.make_submission_fixture(self, sbatch_body)
        script.unlink()
        script = task/'train_op_gate_pin.sh'
        shutil.copyfile(Path(__file__).with_name('train_op_gate_pin.sh'), script)
        config = self.root/'final_exam/configs/op_gate_pin.json'
        config.write_text('{"op_gate_pin":{"epsilon":0.0001}}\n')
        return script, task, environment

    def test_dry_run_creates_nothing_and_selects_gpu_smoke_resources(self):
        script, task, environment = self.make_submission_fixture('exit 99')
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        result = subprocess.run(['bash', str(script), '--dry-run', '--smoke'],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))
        command = shlex.split(result.stdout)
        models = command[command.index('--models')+1:command.index('--seeds')]
        self.assertEqual(models, [
            'device_op_cascade_pin', 'device_op_gate_pin', 'device_op_gate_oracle_pin',
            'device_op_gate_shuffle_pin', 'device_op_gate_only_base_pin', 'device_op_gate_only_pin',
            'device_op_gate_only_oracle_pin', 'device_op_gate_only_shuffle_pin'])
        self.assertEqual(command[command.index('--seeds')+1:command.index('--device')], ['17','29','43'])
        self.assertIn('--cpus-per-task=8', command)
        self.assertIn('--gres=gpu:1', command)
        self.assertIn('--time=00:30:00', command)
        self.assertIn('--account=b_phzhwu', command)
        self.assertIn('--partition=ex01A800', command)
        run_id = command[command.index('--run-id')+1]
        self.assertIn('_op_gate_', run_id)
        self.assertTrue(run_id.endswith('_pin'))
        self.assertIn(str(task/'hpc_logs_pin'/run_id), result.stdout)

    def test_submission_records_job_and_shared_run_id(self):
        script, task, environment = self.make_submission_fixture("echo '98765;cluster'")
        result = subprocess.run(['bash', str(script)], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = list(task.glob('hpc_logs_pin/*/submission_pin.txt'))
        self.assertEqual(len(records), 1)
        self.assertIn('job_id=98765;cluster', records[0].read_text())
        saved = json.loads((records[0].parent/'submitted_config_pin.json').read_text())
        self.assertEqual(saved, {'op_gate_pin': {'epsilon': .0001}})
        self.assertIn('op_gate_pin.json', records[0].read_text())
        run_id = records[0].parent.name
        self.assertIn(str(task/'logs_pin'/run_id), result.stdout)
        self.assertIn(str(task/'results_pin'/run_id), result.stdout)
