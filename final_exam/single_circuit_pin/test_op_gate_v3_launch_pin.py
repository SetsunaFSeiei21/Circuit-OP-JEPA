"""v3 gate launcher preserves budgets, selected IDs and immutable config copies."""
import json
from pathlib import Path
import subprocess
import shutil
import sys

from final_exam.single_circuit_pin import test_staged_op_gate_launch_pin as staged_tests


class GateV3LauncherPinTests(staged_tests.StagedGateLauncherPinTests):
    MODELS = [
        'device_op_cascade_pin',
        'device_op_gate_pin_v3', 'device_op_gate_fusion_pin_v3',
        'device_op_distill_gate_pin_v3', 'device_op_distill_gate_fusion_pin_v3',
        'device_op_supervised_gate_pin_v3', 'device_op_supervised_gate_fusion_pin_v3',
    ]
    SCRIPT_NAME = 'train_op_gate_pin_v3.sh'
    CONFIG_NAME = 'op_gate_pin_v3.json'
    RUN_TAG = '_op_gate_v3_'

    def test_template_preserves_v2_budgets_and_data_selection(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        old = json.loads((configs / 'op_gate_pin_v2.json').read_text())
        new = json.loads((configs / self.CONFIG_NAME).read_text())
        self.assertEqual(new['final_exam_pin'].pop('models'), self.MODELS)
        old['final_exam_pin'].pop('models')
        self.assertEqual(new['op_gate_pin'].pop('n_experts'), 16)
        self.assertEqual(new['op_gate_pin'].pop('mlp_dropout'), 0.1)
        self.assertEqual(old, new)

    def test_help_has_no_submission_or_output_side_effects(self):
        script, _, environment, _ = self.fixture('exit 90')
        before = sorted(str(path) for path in self.root.rglob('*'))
        result = subprocess.run(['bash', str(script), '--help'], env=environment,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(self.SCRIPT_NAME, result.stdout)
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))

    def test_unsupported_option_fails_before_submission(self):
        script, _, environment, _ = self.fixture('exit 90')
        before = sorted(str(path) for path in self.root.rglob('*'))
        result = subprocess.run(['bash', str(script), '--unknown'], env=environment,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Unknown option', result.stderr)
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))

    def test_submitted_wrapper_resolves_project_from_slurm_spool(self):
        # Execute the actual shared wrapper from an unrelated spool directory.
        # The fake sbatch passes the submitter environment and the exact v3 args.
        body = r"""while [[ "$1" == --* ]]; do shift; done
wrapper="$1"
shift
mkdir -p "$JEPA_PROJECT_ROOT/spool/job123"
cp "$wrapper" "$JEPA_PROJECT_ROOT/spool/job123/slurm_script"
cd "$JEPA_PROJECT_ROOT/spool/job123"
JEPA_CONDA_ENV='' bash ./slurm_script "$@" >/dev/null
echo 98765"""
        script, task, environment, config = self.fixture(body)
        shutil.copyfile(Path(__file__).with_name('run_pin.sh'), task / 'run_pin.sh')
        common = self.root / 'final_exam/common'
        common.mkdir()
        (common / 'launch_pin.py').write_text(
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['JEPA_PROJECT_ROOT'], 'captured.json').write_text("
            "json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'script': __file__}))\n")
        environment['JEPA_PYTHON'] = sys.executable
        run = subprocess.run(['bash', str(script)], env=environment, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        captured = json.loads((self.root / 'captured.json').read_text())
        self.assertEqual(captured['cwd'], str(self.root))
        self.assertEqual(captured['script'], str(common / 'launch_pin.py'))
        args = captured['args']
        self.assertEqual(args[args.index('--models')+1:args.index('--seeds')], self.MODELS)
        submitted = Path(args[args.index('--config')+1])
        self.assertEqual(submitted.read_bytes(), config.read_bytes())
        self.assertTrue(submitted.is_relative_to(task / 'hpc_logs_pin'))
        self.assertIn('_op_gate_v3_', args[args.index('--run-id')+1])
