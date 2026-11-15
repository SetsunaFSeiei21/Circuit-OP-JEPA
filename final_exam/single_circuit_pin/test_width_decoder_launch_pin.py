"""Wide readout submission preserves resources, snapshots and result paths."""
import json
from pathlib import Path

from final_exam.single_circuit_pin import test_staged_op_gate_launch_pin as launcher_tests


class WidthDecoderLauncherPinTests(launcher_tests.StagedGateLauncherPinTests):
    MODELS = ['device_op_cascade_pin', 'width_decoder_pin', 'width_decoder_op_pin']
    SCRIPT_NAME = 'train_width_decoder_pin.sh'
    CONFIG_NAME = 'width_decoder_pin.json'
    RUN_TAG = '_width_decoder_'

    def test_template_keeps_stage_budgets_and_explicit_mixture(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs' / self.CONFIG_NAME).read_text())
        self.assertEqual(config['final_exam_pin']['models'], self.MODELS)
        self.assertEqual(config['op_transfer']['epochs']['op_predictor'], 30)
        self.assertEqual(config['op_transfer']['epochs']['readout'], 60)
        self.assertEqual(config['width_decoder_pin'], dict(n_experts=4, hidden_dim=128, dropout=.1))
        self.assertNotIn('op_gate_pin', config)
