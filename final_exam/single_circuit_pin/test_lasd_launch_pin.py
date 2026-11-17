"""LASD uses the same Slurm snapshots, roots and matched training budgets."""
import json
from pathlib import Path

from final_exam.single_circuit_pin.test_staged_op_gate_launch_pin import StagedGateLauncherPinTests


class LASDLauncherPinTests(StagedGateLauncherPinTests):
    MODELS = ['width_decoder_pin', 'width_decoder_score_pin', 'width_decoder_lasd_pin',
              'width_decoder_score_detach_pin', 'width_decoder_lasd_detach_pin']
    SCRIPT_NAME = 'train_lasd_pin.sh'
    CONFIG_NAME = 'lasd_pin.json'
    RUN_TAG = '_lasd_'

    def test_template_controls_scores_without_changing_old_architecture(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs' / self.CONFIG_NAME).read_text())
        assert cfg['final_exam_pin']['models'] == self.MODELS
        assert cfg['model']['decoder_ffn_dim'] == 2048
        assert cfg['model']['decoder_layers'] == 2
        assert cfg['op_transfer']['epochs']['op_predictor'] == 30
        assert cfg['op_transfer']['epochs']['readout'] == 60
        assert cfg['lasd_pin']['enabled'] is True
        assert cfg['lasd_pin']['checkpoint_scores'] is True
