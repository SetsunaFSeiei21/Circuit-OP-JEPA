"""Typed signed gates keep the existing submission, sampling and budget contracts."""
import json
from pathlib import Path

from final_exam.single_circuit_pin import test_op_gate_v4_v5_launch_pin as signed_tests


class GateV6LauncherPinTests(signed_tests.GateV4LauncherPinTests):
    VERSION = 'v6'
    BASE_VERSION = 'v4'
    MODELS = [
        'device_op_cascade_pin',
        'device_op_gate_fusion_pin_v6',
        'device_op_distill_gate_fusion_pin_v6',
        'device_op_supervised_gate_fusion_pin_v6',
        'device_op_distill_gate_decoder_fusion_pin_v6',
    ]
    SCRIPT_NAME = 'train_op_gate_pin_v6.sh'
    CONFIG_NAME = 'op_gate_pin_v6.json'
    RUN_TAG = '_op_gate_v6_'

    def test_only_model_names_change_from_corresponding_parent_config(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        parent = json.loads((configs / f'op_gate_pin_{self.BASE_VERSION}.json').read_text())
        current = json.loads((configs / self.CONFIG_NAME).read_text())
        expected = [name.replace('_' + self.BASE_VERSION, '_' + self.VERSION)
                    for name in parent['final_exam_pin']['models']]
        self.assertEqual(current['final_exam_pin']['models'], expected)
        parent['final_exam_pin']['models'] = expected
        self.assertEqual(parent, current)


class GateV7LauncherPinTests(GateV6LauncherPinTests):
    VERSION = 'v7'
    BASE_VERSION = 'v5'
    MODELS = [
        'device_op_cascade_pin',
        'device_op_gate_fusion_pin_v7',
        'device_op_distill_gate_fusion_pin_v7',
        'device_op_supervised_gate_fusion_pin_v7',
        'device_op_distill_gate_decoder_fusion_pin_v7',
    ]
    SCRIPT_NAME = 'train_op_gate_pin_v7.sh'
    CONFIG_NAME = 'op_gate_pin_v7.json'
    RUN_TAG = '_op_gate_v7_'
