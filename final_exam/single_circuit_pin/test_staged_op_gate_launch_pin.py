"""New staged gate submission stays in-project and snapshots its independent budget."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import unittest

from final_exam.single_circuit_pin import test_launch_pin as launch_tests


MODELS = ['device_op_distill_gate_pin', 'device_op_distill_gate_decoder_pin',
          'device_op_supervised_gate_pin']


class StagedGateLauncherPinTests(unittest.TestCase):
    setUp = launch_tests.LauncherTests.setUp
    MODELS = MODELS
    SCRIPT_NAME = "train_staged_op_gate_pin.sh"
    CONFIG_NAME = "staged_op_gate_pin.json"
    RUN_TAG = "_staged_op_gate_"

    def fixture(self, body='echo 98765'):
        script, task, environment = launch_tests.LauncherTests.make_submission_fixture(self, body)
        script.unlink()
        script = task / self.SCRIPT_NAME
        shutil.copyfile(Path(__file__).with_name(script.name), script)
        config = self.root / "final_exam/configs" / self.CONFIG_NAME
        source = Path(__file__).resolve().parents[1] / "configs" / self.CONFIG_NAME
        shutil.copyfile(source, config)
        raw = self.root / 'runs/raw/full_data/two_stage_folded_opamp/sampling_history.sqlite3'
        raw.parent.mkdir()
        raw.touch()
        return script, task, environment, config

    def test_dry_run_has_three_options_and_no_files_created(self):
        script, task, environment, _ = self.fixture('exit 90')
        before = sorted(str(path) for path in self.root.rglob('*'))
        run = subprocess.run(['bash', str(script), '--dry-run', '--smoke'],
                             env=environment, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))
        command = shlex.split(run.stdout)
        self.assertEqual(command[command.index('--models')+1:command.index('--seeds')], self.MODELS)
        self.assertEqual(command[command.index('--seeds')+1:command.index('--device')], ['17','29','43'])
        self.assertEqual(command[command.index('--circuit')+1], 'two_stage_folded_opamp')
        self.assertIn('--cpus-per-task=8', command)
        self.assertIn('--gres=gpu:1', command)
        self.assertIn('--time=00:30:00', command)
        self.assertIn(str(task / 'run_pin.sh'), command)
        run_id = command[command.index('--run-id')+1]
        self.assertIn(self.RUN_TAG, run_id)
        self.assertTrue(run_id.endswith('_pin'))
        self.assertIn(str(task / 'hpc_logs_pin' / run_id), run.stdout)

    def test_submission_freezes_config_and_exports_project_root(self):
        script, task, environment, config = self.fixture(
            'test "$JEPA_PROJECT_ROOT" = "$PWD" || exit 21\necho "98765;cluster"')
        run = subprocess.run(['bash', str(script)], env=environment, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        records = list(task.glob('hpc_logs_pin/*/submission_pin.txt'))
        self.assertEqual(len(records), 1)
        saved = records[0].parent / 'submitted_config_pin.json'
        self.assertEqual(saved.read_bytes(), config.read_bytes())
        contents = json.loads(saved.read_text())
        self.assertEqual(contents['op_transfer']['epochs']['gate'], 30)
        self.assertEqual(contents['final_exam_pin']['models'], self.MODELS)
        config.write_text('{}\n')
        self.assertEqual(json.loads(saved.read_text()), contents)
        self.assertIn('job_id=98765;cluster', records[0].read_text())
        run_id = records[0].parent.name
        self.assertIn(str(task / 'logs_pin' / run_id), run.stdout)
        self.assertIn(str(task / 'results_pin' / run_id), run.stdout)

    def test_rejected_submission_preserves_exit_status(self):
        script, task, environment, _ = self.fixture('echo rejected >&2\nexit 9')
        run = subprocess.run(['bash', str(script)], env=environment, text=True, capture_output=True)
        self.assertEqual(run.returncode, 9)
        self.assertIn('rejected', run.stderr)
        self.assertNotIn('Submitted job', run.stdout)
        self.assertFalse((task / 'results_pin').exists())


if __name__ == '__main__':
    unittest.main()
