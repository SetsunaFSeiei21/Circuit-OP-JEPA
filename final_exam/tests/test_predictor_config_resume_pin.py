"""Independent OP architectures/options remain deployable and resumable."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from device_jepa.training import state_digest
from experiments.op_transfer.engine import settings
from final_exam.run_pin import load_checkpoint_pin, run_experiment_pin
from final_exam.tests.test_run_pin import tiny_inputs


class IndependentPredictorResumeTests(unittest.TestCase):
    def test_pre_separation_stage_with_sparse_legacy_config_resumes_exactly(self):
        from final_exam import run_pin as runner

        # Reproduce the pre-separation resolver: no OP block or inserted defaults
        # in the exact config stored inside the old epoch-resume identity.
        def legacy_effective(config, mode, seed):
            value = copy.deepcopy(config)
            shared_names = (runner.CASCADE_MODELS_PIN | runner.REFERENCES_PIN
                            | set(runner.ALL_GATE_MODELS_PIN) | set(runner.WIDTH_MODELS_PIN)
                            | set(runner.LASD_MOE_MODELS_PIN)
                            | {runner.WIDTH_PREDICTOR_PIN, 'op_predictor', 'op_predictor_pin',
                               'device_op_predictor_pin'})
            key = 'zerosim_supervised_pin' if mode in shared_names else mode
            override = value.get('final_exam_pin', {}).get('model_overrides', {}).get(key, {})
            value['model'].update(override)
            value.setdefault('training', {})['seed'] = seed
            return value

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config, data_path, config_path = tiny_inputs(root)
            config['op_transfer'] = dict(
                epochs=dict(op_predictor=2, readout=2, student_predicted_op=2,
                            predicted_op_distill_encoder=1, predicted_op_distill_decoder=1),
                batch_size=8, lr=0.001, patience=0)
            config_path.write_text(json.dumps(config))
            modes = ['device_op_cascade_pin']
            source, recovered = root / 'legacy_pin', root / 'recovered_pin'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(runner, 'effective_config_pin', side_effect=legacy_effective):
                    run_experiment_pin(data_path, config_path, source, modes, [17], 'cpu')
                with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('unexpected retraining')):
                    result = run_experiment_pin(data_path, config_path, recovered, modes, [17], 'cpu',
                                                resume_from=source)
            saved = json.loads((source / 'seed_17_pin/stages_pin/device_op_predictor_pin/config_pin.json').read_text())
            self.assertNotIn('op_predictor', saved)
            self.assertNotIn('grad_clip', saved['op_transfer'])
            for stage in result['stages']:
                if stage['epochs']:
                    self.assertEqual(stage['resume']['new_optimizer_steps'], 0)
            original, _ = load_checkpoint_pin(source / 'seed_17_pin/device_op_cascade_pin/checkpoint_pin.pt')
            resumed, _ = load_checkpoint_pin(recovered / 'seed_17_pin/device_op_cascade_pin/checkpoint_pin.pt')
            self.assertEqual(state_digest(original), state_digest(resumed))

    def test_independent_device_and_wide_op_completed_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config, data_path, config_path = tiny_inputs(root)
            options = settings(config)
            options['epochs'] = dict.fromkeys(options['epochs'], 1)
            options['epochs'].update(readout=2, student_predicted_op=2)
            options.update(patience=0, batch_size=8, lr=0.001)
            config['op_transfer'] = options
            config['width_decoder_pin'] = dict(n_experts=2, hidden_dim=8, dropout=0.)
            config['op_predictor'] = dict(
                model=dict(dim=12, heads=3, encoder_layers=4, cross_layers=2,
                           ffn_dim=24, decoder_ffn_dim=24),
                training=dict(epochs=2, batch_size=4, lr=0.002, patience=0),
                width_decoder_pin=dict(n_experts=3, hidden_dim=12, dropout=0.))
            config_path.write_text(json.dumps(config))
            modes = ['device_op_cascade_pin', 'width_decoder_op_pin']
            source = root / 'source_pin'
            recovered = root / 'recovered_pin'
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run_experiment_pin(data_path, config_path, source, modes, [17], 'cpu')
                with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('unexpected retraining')):
                    report = run_experiment_pin(data_path, config_path, recovered, modes, [17], 'cpu',
                                                resume_from=source)
            for stage in report['stages']:
                if stage['epochs']:
                    self.assertEqual(stage['resume']['new_optimizer_steps'], 0)
            for mode in modes:
                model, metadata = load_checkpoint_pin(source / f'seed_17_pin/{mode}/checkpoint_pin.pt')
                replay, _ = load_checkpoint_pin(recovered / f'seed_17_pin/{mode}/checkpoint_pin.pt')
                self.assertEqual(model.encoder.cfg.dim, 8)
                self.assertEqual(model.predictor.encoder.cfg.dim, 12)
                self.assertEqual(model.predictor.encoder.cfg.cross_layers, 2)
                self.assertEqual(metadata['options']['lr'], 0.001)
                self.assertEqual(state_digest(model), state_digest(replay))
            for mode in ('device_op_predictor_pin', 'width_device_op_predictor_pin'):
                model, metadata = load_checkpoint_pin(source / f'seed_17_pin/shared_pin/{mode}/checkpoint_pin.pt')
                self.assertEqual(metadata['options']['lr'], 0.002)
                self.assertEqual(metadata['options']['batch_size'], 4)
                self.assertEqual(metadata['options']['epochs']['op_predictor'], 2)
                self.assertEqual(metadata['model_config']['dim'], 12)
                if mode.startswith('width'):
                    self.assertEqual(model.width_config['n_experts'], 3)
            config['op_predictor']['training']['lr'] = 0.004
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'configuration must match'):
                run_experiment_pin(data_path, config_path, root / 'invalid_pin', modes, [17], 'cpu',
                                   resume_from=source)


if __name__ == '__main__':
    unittest.main()
