"""Low-rank ablations keep shared resources, config snapshots and distinct paths."""
import json
from pathlib import Path
import subprocess

from final_exam.single_circuit_pin import test_staged_op_gate_launch_pin as launcher_tests


class LowRankDecoderLauncherPinTests(launcher_tests.StagedGateLauncherPinTests):
    MODELS = ['width_decoder_pin', 'width_decoder_lowrank_attention_pin',
              'width_decoder_lowrank_ffn_pin', 'width_decoder_lowrank_both_pin']
    SCRIPT_NAME = 'train_low_rank_decoder_pin.sh'
    CONFIG_NAME = 'low_rank_decoder_pin.json'
    RUN_TAG = '_low_rank_decoder_'

    def test_template_keeps_matched_budgets_and_rank_controls(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs' / self.CONFIG_NAME).read_text())
        self.assertEqual(config['final_exam_pin']['models'], self.MODELS)
        self.assertEqual(config['model']['decoder_layers'], 1)
        self.assertEqual(config['model']['dim'], 128)
        self.assertEqual(config['model']['decoder_ffn_dim'], 2048)
        self.assertEqual(config['op_transfer']['epochs']['op_predictor'], 30)
        self.assertEqual(config['op_transfer']['epochs']['readout'], 60)
        self.assertEqual(config['op_transfer']['patience'], 10)
        self.assertEqual(config['low_rank_decoder_pin'], dict(attention_rank=32, ffn_rank=64))
        self.assertEqual(config['width_decoder_pin'], dict(n_experts=4, hidden_dim=128, dropout=.1))
        self.assertNotIn('op_gate_pin', config)

    def test_resume_help_is_login_node_safe(self):
        script = Path(__file__).with_name('resume_low_rank_decoder_pin.sh')
        run = subprocess.run(['bash', str(script), '--help'], text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('resume_low_rank_decoder_pin.sh', run.stdout)
        self.assertIn('Original models, seeds, config and split are retained', run.stdout)
        syntax = subprocess.run(['bash', '-n', str(script)], text=True, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
