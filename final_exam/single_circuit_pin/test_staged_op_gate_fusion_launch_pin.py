"""Fusion-only gate launcher preserves resource, snapshot and path contracts."""
import json
from pathlib import Path

from final_exam.single_circuit_pin import test_staged_op_gate_launch_pin as staged_tests


class FusionStagedGateLauncherPinTests(staged_tests.StagedGateLauncherPinTests):
    MODELS = ['device_op_distill_gate_fusion_pin', 'device_op_distill_gate_decoder_fusion_pin',
              'device_op_supervised_gate_fusion_pin']
    SCRIPT_NAME = 'train_staged_op_gate_fusion_pin.sh'
    CONFIG_NAME = 'staged_op_gate_fusion_pin.json'
    RUN_TAG = '_staged_op_gate_fusion_'

    def test_template_changes_model_selection_only(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        old = json.loads((configs / 'staged_op_gate_pin.json').read_text())
        new = json.loads((configs / self.CONFIG_NAME).read_text())
        self.assertEqual(new['final_exam_pin'].pop('models'), self.MODELS)
        old['final_exam_pin'].pop('models')
        self.assertEqual(old, new)
