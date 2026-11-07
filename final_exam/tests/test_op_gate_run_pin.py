"""Artifact, budget, source-boundary and paired-initialization gate checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from final_exam.data_pin import PinDataContext
from final_exam.run_pin import (GATE_MODELS_PIN, REFERENCES_PIN, load_checkpoint_pin,
                                run_experiment_pin, validate_request_pin)
from final_exam.tests.test_run_pin import tiny_inputs


class GateRunnerPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.config, self.data_path, self.cfg_path = tiny_inputs(self.root)
        torch.set_num_threads(1)

    def test_eight_controls_share_device_pretraining_and_replay_independently(self):
        output = self.root / 'gate_study_pin'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(self.data_path, self.cfg_path, output,
                                       list(GATE_MODELS_PIN), [17], 'cpu', smoke=True)
        stages = result['stages']
        self.assertEqual(len(stages), 10)  # one predictor, one cache, eight downstream stages
        self.assertEqual(sum(s['stage_id'].endswith('/device_op_predictor_pin') for s in stages), 1)
        self.assertEqual(sum(s['stage_id'].endswith('/frozen_device_op_cache_pin') for s in stages), 1)
        config = json.loads((output / 'config_pin.json').read_text())
        manifest = json.loads((output / 'manifest_pin.json').read_text())
        (self.root / 'replay_pin').mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = PinDataContext(self.data, manifest, config, 'cpu', {}, self.root / 'replay_pin')
            batch = next(ctx.batches('test', 8, op=True))
        self.assertNotIn('_predicted_device_op', batch)  # replay uses the saved predictor itself
        audits = {}
        for record in result['results']:
            mode = record['mode']
            directory = output / 'seed_17_pin' / mode
            model, metadata = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
            audits[mode] = json.loads((directory / 'op_gate_audit_pin.json').read_text())
            self.assertEqual(metadata['requires_measured_op'], mode in REFERENCES_PIN)
            self.assertEqual(record['report']['requires_measured_op'], mode in REFERENCES_PIN)
            self.assertEqual(record['report']['op_gate_pin']['group'],
                             'A_features_and_gate' if mode in GATE_MODELS_PIN[:4] else 'B_gate_only')
            if mode == 'device_op_gate_only_base_pin':
                self.assertEqual(len(record['stage_ids']), 1)
                self.assertIsNone(model.predictor)
            else:
                self.assertEqual(len(record['stage_ids']), 3)
                self.assertTrue((directory / audits[mode]['predictor_checkpoint']).is_file())
                self.assertFalse(any(p.requires_grad for p in model.predictor.parameters()))
                self.assertEqual(metadata['recipe']['predictor']['encoder_granularity'], 'device')
            with torch.no_grad():
                expected = model(batch)
                restored, _ = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
                torch.testing.assert_close(expected, restored(batch), atol=0, rtol=0)
                if mode not in REFERENCES_PIN:
                    poisoned = copy.deepcopy(batch)
                    for field in ('op', 'owner_op', 'performance', 'physical_performance'):
                        poisoned[field].fill_(float('nan'))
                    poisoned['op_mask'].logical_not_()
                    poisoned['owner_op_mask'].logical_not_()
                    torch.testing.assert_close(expected, model(poisoned), atol=0, rtol=0)
                else:
                    coverage = record['report']['oracle_gate_coverage_pin']
                    self.assertGreater(coverage['supported_fields'], 0)
                    self.assertLessEqual(coverage['observed_supported_fields'], coverage['supported_fields'])
        for family in (GATE_MODELS_PIN[:4], GATE_MODELS_PIN[4:]):
            for name in ('initial_encoder_digest', 'initial_decoder_digest'):
                self.assertEqual(len({audits[mode][name] for mode in family}), 1, name)
        dependency = output / 'seed_17_pin/shared_pin/device_op_predictor_pin/checkpoint_pin.pt'
        op_model, saved = load_checkpoint_pin(dependency)
        self.assertEqual(saved['recipe']['encoder_granularity'], 'device')
        self.assertFalse(any(p.requires_grad for p in op_model.parameters()))
        with torch.no_grad():
            values, _ = op_model.predict_devices(batch)
        self.assertEqual(values.shape[:2], batch['owner_mask'].shape)
        with zipfile.ZipFile(output / 'summary_pin.xlsx') as book:
            self.assertIn('ModelInputs', book.read('xl/workbook.xml').decode())

    def test_gate_only_baseline_does_not_train_or_save_an_op_predictor(self):
        output = self.root / 'baseline_pin'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(self.data_path, self.cfg_path, output,
                ['device_op_gate_only_base_pin'], [17], 'cpu', smoke=True)
        self.assertEqual(len(result['stages']), 1)
        self.assertFalse((output / 'seed_17_pin/shared_pin').exists())

    def test_invalid_gate_configuration_fails_before_writing_output(self):
        for settings in ({'epsilon': 0}, {'strength': -1}, {'initial_gate': 1}, {'dim': 512}):
            config = copy.deepcopy(self.config)
            config['op_gate_pin'] = settings
            self.cfg_path.write_text(json.dumps(config))
            output = self.root / 'invalid_pin'
            with self.assertRaises(ValueError):
                run_experiment_pin(self.data_path, self.cfg_path, output,
                                   ['device_op_gate_pin'], [17], 'cpu', smoke=True)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
