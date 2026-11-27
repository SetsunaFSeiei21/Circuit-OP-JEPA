"""Independent OP architecture/optimizer controls and legacy replay contracts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.op_transfer.predictor_config import predictor_config, validate_encoder_transfer


def config():
    return dict(model=dict(dim=8, heads=2, encoder_layers=2, cross_layers=1,
                           ffn_dim=16, decoder_ffn_dim=16, dropout=0.),
                op_transfer=dict(epochs=dict(op_predictor=2, readout=3),
                                 lr=.001, batch_size=4, patience=2))


class PredictorConfigTests(unittest.TestCase):
    def test_legacy_inheritance(self):
        from experiments.op_transfer.engine import settings
        value = config()
        before = copy.deepcopy(value)
        resolved = predictor_config(value)
        self.assertEqual(resolved['model']['dim'], 8)
        self.assertEqual(settings(resolved), settings(value))
        self.assertEqual(resolved, value)
        self.assertEqual(value, before)
        self.assertNotIn('op_predictor', resolved)

    def test_legacy_stage_config_identity_is_preserved(self):
        import hashlib
        value = config()
        before = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        resolved = predictor_config(value)
        after = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
        self.assertEqual(before, after)
        self.assertNotIn('torch_threads', resolved['op_transfer'])
        self.assertNotIn('terminal_bias', resolved['model'])

    def test_separate_model_training_and_width(self):
        value = config()
        value['width_decoder_pin'] = dict(n_experts=4, hidden_dim=8, dropout=.1)
        value['op_predictor'] = dict(
            model=dict(dim=16, encoder_layers=4, cross_layers=2),
            training=dict(epochs=7, lr=.0003, batch_size=2, weight_decay=.02,
                          grad_clip=.5, patience=4, progress_every=5, torch_threads=1),
            width_decoder_pin=dict(n_experts=2, hidden_dim=12, dropout=.2))
        before = copy.deepcopy(value)
        resolved = predictor_config(value)
        self.assertEqual(resolved['model']['dim'], 16)
        self.assertEqual(resolved['model']['cross_layers'], 2)
        self.assertEqual(resolved['model']['ffn_dim'], 16)
        self.assertEqual(resolved['op_transfer']['epochs']['op_predictor'], 7)
        self.assertEqual(resolved['op_transfer']['epochs']['readout'], 3)
        self.assertEqual(resolved['op_transfer']['lr'], .0003)
        self.assertEqual(resolved['op_transfer']['batch_size'], 2)
        self.assertEqual(resolved['width_decoder_pin']['hidden_dim'], 12)
        self.assertEqual(value, before)
        resolved['model']['dim'] = 32
        self.assertEqual(value, before)

    def test_invalid_fields_and_values_fail(self):
        for section in (None, {'lr': .1}, {'model': []}, {'model': {'typo': 2}},
                        {'model': {'dim': 9}}, {'training': {'momentum': .1}},
                        {'training': {'epochs': 0}}, {'training': {'batch_size': True}},
                        {'training': {'lr': float('nan')}},
                        {'width_decoder_pin': {'hidden_dim': 0}}):
            with self.subTest(section=section):
                value = config()
                value['op_predictor'] = section
                with self.assertRaises(ValueError):
                    predictor_config(value)

    def test_smoke_caps_explicit_epochs_and_patience(self):
        value = config()
        value['op_predictor'] = dict(training=dict(epochs=99, patience=15))
        resolved = predictor_config(value, smoke=True)
        self.assertEqual(resolved['op_transfer']['epochs']['op_predictor'], 1)
        self.assertEqual(resolved['op_transfer']['patience'], 0)
        self.assertEqual(predictor_config(resolved)['op_transfer']['epochs']['op_predictor'], 1)
        self.assertEqual(value['op_predictor']['training']['epochs'], 99)

    def test_deepgen_transfer_rejects_incompatible_encoder(self):
        value = predictor_config(config())
        other = copy.deepcopy(value)
        validate_encoder_transfer(value, other)
        other['model']['dim'] = 16
        with self.assertRaisesRegex(ValueError, 'transfers the encoder directly'):
            validate_encoder_transfer(value, other)

    def test_legacy_runner_trains_and_replays_distinct_predictor(self):
        import torch
        from experiments.op_transfer import engine
        from experiments.op_transfer.test_engine import fixed_topology_data, small_config
        torch.set_num_threads(1)
        value = small_config()
        value['op_predictor'] = dict(model=dict(dim=16, heads=4, encoder_layers=3,
                                               cross_layers=1, ffn_dim=24),
                                    training=dict(epochs=2, lr=.0003, batch_size=3,
                                                  patience=0, torch_threads=1))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'config.json').write_text(json.dumps(value))
            (root/'data.json').write_text(json.dumps(fixed_topology_data()))
            with contextlib.redirect_stdout(io.StringIO()):
                engine.run_experiment(root/'data.json', root/'config.json', root/'out',
                                      ['predicted_op'], [17], 'cpu')
            directory = root/'out'/'seed_17'
            predictor, metadata = engine.load_model(directory/'op_predictor'/'checkpoint.pt')
            cascade, _ = engine.load_model(directory/'predicted_op'/'checkpoint.pt')
            self.assertEqual(predictor.encoder.cfg.dim, 16)
            self.assertEqual(cascade.predictor.encoder.cfg.dim, 16)
            self.assertEqual(cascade.downstream.encoder.cfg.dim, 8)
            self.assertEqual(metadata['options']['lr'], .0003)
            self.assertEqual(metadata['options']['batch_size'], 3)
            cost = json.loads((root/'out'/'stages.json').read_text())['unique_stages']
            op_stage = next(row for row in cost if row['stage_id'].endswith('/op_predictor'))
            self.assertEqual(op_stage['epochs'], 2)

    def test_device_final_runner_saves_effective_op_options(self):
        import torch
        from final_exam.run import load_checkpoint, run_experiment
        from final_exam.tests.test_run import tiny_inputs
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, value, data_path, cfg_path = tiny_inputs(root)
            value['op_predictor'] = dict(model=dict(dim=16, heads=4, encoder_layers=3,
                                                   cross_layers=1, ffn_dim=24),
                                        training=dict(epochs=9, lr=.0003, batch_size=3,
                                                      patience=5, torch_threads=1))
            cfg_path.write_text(json.dumps(value))
            with contextlib.redirect_stdout(io.StringIO()):
                run_experiment(data_path, cfg_path, root/'out', ['predicted_op'], [17],
                               'cpu', smoke=True)
            directory = root/'out'/'seed_17'
            predictor, metadata = load_checkpoint(directory/'shared'/'op_predictor'/'checkpoint.pt')
            cascade, _ = load_checkpoint(directory/'predicted_op'/'checkpoint.pt')
            self.assertEqual(predictor.encoder.cfg.dim, 16)
            self.assertEqual(cascade.predictor.encoder.cfg.dim, 16)
            self.assertEqual(cascade.downstream.encoder.cfg.dim, 8)
            self.assertEqual(metadata['options']['lr'], .0003)
            self.assertEqual(metadata['options']['batch_size'], 3)
            self.assertEqual(metadata['options']['epochs']['op_predictor'], 1)
            self.assertEqual(metadata['options']['patience'], 0)


if __name__ == '__main__':
    unittest.main()
