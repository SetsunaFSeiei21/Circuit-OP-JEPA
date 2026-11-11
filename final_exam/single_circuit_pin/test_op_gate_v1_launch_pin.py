"""Versioned gate study retains safe Slurm submission and config snapshots."""
import json
from pathlib import Path
import subprocess

from final_exam.single_circuit_pin import test_staged_op_gate_launch_pin as staged_tests


class GateV1LauncherPinTests(staged_tests.StagedGateLauncherPinTests):
    MODELS = [
        'device_op_cascade_pin',
        'device_op_gate_pin_v1', 'device_op_gate_fusion_pin_v1',
        'device_op_distill_gate_pin_v1', 'device_op_distill_gate_fusion_pin_v1',
        'device_op_supervised_gate_pin_v1', 'device_op_supervised_gate_fusion_pin_v1',
    ]
    SCRIPT_NAME = 'train_op_gate_pin_v1.sh'
    CONFIG_NAME = 'op_gate_pin_v1.json'
    RUN_TAG = '_op_gate_v1_'

    def test_template_preserves_common_budgets(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        old = json.loads((configs / 'staged_op_gate_pin.json').read_text())
        new = json.loads((configs / self.CONFIG_NAME).read_text())
        self.assertEqual(new['final_exam_pin'].pop('models'), self.MODELS)
        old['final_exam_pin'].pop('models')
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
