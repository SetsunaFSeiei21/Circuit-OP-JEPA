"""v2 gate launcher preserves budgets, selected IDs and immutable config copies."""
import json
from pathlib import Path
import subprocess

from final_exam.single_circuit_pin import test_staged_op_gate_launch_pin as staged_tests


class GateV2LauncherPinTests(staged_tests.StagedGateLauncherPinTests):
    MODELS = [
        'device_op_cascade_pin',
        'device_op_gate_pin_v2', 'device_op_gate_fusion_pin_v2',
        'device_op_distill_gate_pin_v2', 'device_op_distill_gate_fusion_pin_v2',
        'device_op_supervised_gate_pin_v2', 'device_op_supervised_gate_fusion_pin_v2',
    ]
    SCRIPT_NAME = 'train_op_gate_pin_v2.sh'
    CONFIG_NAME = 'op_gate_pin_v2.json'
    RUN_TAG = '_op_gate_v2_'

    def test_template_preserves_v1_budgets_and_data_selection(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        old = json.loads((configs / 'op_gate_pin_v1.json').read_text())
        new = json.loads((configs / self.CONFIG_NAME).read_text())
        self.assertEqual(new['final_exam_pin'].pop('models'), self.MODELS)
        old['final_exam_pin'].pop('models')
        self.assertEqual(new['op_gate_pin'].pop('norm_eps'), 1e-5)
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
