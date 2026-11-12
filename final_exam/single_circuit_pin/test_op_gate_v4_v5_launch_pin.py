"""Signed-gate launchers retain sampling, budgets, snapshots and HPC isolation."""
import json
from pathlib import Path
import subprocess

from final_exam.single_circuit_pin import test_staged_op_gate_launch_pin as staged_tests


class GateV4LauncherPinTests(staged_tests.StagedGateLauncherPinTests):
    VERSION = 'v4'
    MODELS = [
        'device_op_cascade_pin',
        'device_op_gate_fusion_pin_v4',
        'device_op_distill_gate_fusion_pin_v4',
        'device_op_supervised_gate_fusion_pin_v4',
        'device_op_distill_gate_decoder_fusion_pin_v4',
    ]
    SCRIPT_NAME = 'train_op_gate_pin_v4.sh'
    CONFIG_NAME = 'op_gate_pin_v4.json'
    RUN_TAG = '_op_gate_v4_'

    def test_template_preserves_v3_budgets_sampling_and_mixture_size(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        old = json.loads((configs / 'op_gate_pin_v3.json').read_text())
        new = json.loads((configs / self.CONFIG_NAME).read_text())
        self.assertEqual(new['final_exam_pin'].pop('models'), self.MODELS)
        old['final_exam_pin'].pop('models')
        old['op_gate_pin'].pop('epsilon')
        old['op_gate_pin'].pop('initial_gate')
        self.assertEqual(old, new)
        self.assertEqual(set(new['op_gate_pin']),
                         {'strength', 'norm_eps', 'n_experts', 'mlp_dropout'})
        self.assertEqual(new['op_gate_pin']['strength'], 1.)
        self.assertGreaterEqual(new['model']['cross_layers'], 2)

    def test_help_and_rejected_arguments_create_no_files(self):
        script, _, environment, _ = self.fixture('exit 90')
        before = sorted(str(p) for p in self.root.rglob('*'))
        for argument, code in (('--help', 0), ('--unknown', 2)):
            result = subprocess.run(['bash', str(script), argument], env=environment,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))


class GateV5LauncherPinTests(GateV4LauncherPinTests):
    VERSION = 'v5'
    MODELS = [
        'device_op_cascade_pin',
        'device_op_gate_fusion_pin_v5',
        'device_op_distill_gate_fusion_pin_v5',
        'device_op_supervised_gate_fusion_pin_v5',
        'device_op_distill_gate_decoder_fusion_pin_v5',
    ]
    SCRIPT_NAME = 'train_op_gate_pin_v5.sh'
    CONFIG_NAME = 'op_gate_pin_v5.json'
    RUN_TAG = '_op_gate_v5_'
