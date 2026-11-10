"""Run the three staged gates and audit shared weights, costs and deployment."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import torch

from device_jepa.training import state_digest
from final_exam.data_pin import PinDataContext
from final_exam.run_pin import (
    STAGED_GATE_MODELS_PIN, load_checkpoint_pin, run_experiment_pin,
)
from final_exam.tests.test_run_pin import tiny_inputs


class StagedGateRunnerPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.config, self.data_path, self.cfg_path = tiny_inputs(self.root)
        torch.set_num_threads(1)

    def run_models(self, models, name, *, smoke=True):
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(self.data_path, self.cfg_path, output,
                                        list(models), [17], 'cpu', smoke=smoke)
        return output, result

    def replay_batch(self, output):
        config = json.loads((output / 'config_pin.json').read_text())
        manifest = json.loads((output / 'manifest_pin.json').read_text())
        destination = self.root / 'replay_pin'
        destination.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            context = PinDataContext(self.data, manifest, config, 'cpu', {}, destination)
            return next(context.batches('test', 8, op=True))

    def test_three_variants_share_bases_and_save_replayable_artifacts(self):
        output, result = self.run_models(STAGED_GATE_MODELS_PIN, 'complete_pin')
        ids = [stage['stage_id'] for stage in result['stages']]
        self.assertEqual(len(ids), 9)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual({name.rsplit('/', 1)[-1] for name in ids}, {
            'device_op_predictor_pin', 'frozen_device_op_cache_pin',
            'design_op_oracle_pin', 'device_op_distill_encoder_pin',
            'device_op_distill_readout_pin', 'device_op_supervised_readout_pin',
            *STAGED_GATE_MODELS_PIN,
        })
        self.assertEqual(len(result['results']), 3)
        self.assertEqual(json.loads((output / 'run_pin.json').read_text())['status'], 'complete')
        self.assertTrue(all(path.stem.endswith('_pin') for path in output.rglob('*') if path.is_file()))
        batch = self.replay_batch(output)
        self.assertNotIn('_predicted_device_op', batch)
        poison = copy.deepcopy(batch)
        for name in ('op', 'owner_op', 'performance', 'physical_performance'):
            poison[name].fill_(float('nan'))
        for name in ('op_mask', 'owner_op_mask', 'op_target_mask',
                     'owner_op_target_mask', 'performance_mask'):
            poison[name].logical_not_()
        audits = {}
        for record in result['results']:
            mode = record['mode']
            directory = output / 'seed_17_pin' / mode
            audit = json.loads((directory / 'op_gate_audit_pin.json').read_text())
            audits[mode] = audit
            model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
            base_path = directory / audit['base_checkpoint']
            base, base_saved = load_checkpoint_pin(base_path)
            gate_decoder = mode == 'device_op_distill_gate_decoder_pin'
            self.assertTrue(audit['gate_was_disabled_during_base'])
            self.assertFalse(base.gate_enabled)
            self.assertTrue(model.gate_enabled)
            self.assertEqual(model.phase, 'gate_decoder' if gate_decoder else 'gate_only')
            self.assertTrue(audit['encoder_frozen'])
            self.assertTrue(audit['predictor_frozen'])
            self.assertEqual(audit['decoder_frozen'], not gate_decoder)
            self.assertEqual(state_digest(model.encoder), state_digest(base.encoder))
            self.assertEqual(state_digest(model.predictor), state_digest(base.predictor))
            self.assertEqual(audit['initial_encoder_digest'], state_digest(base.encoder))
            self.assertEqual(audit['initial_decoder_digest'], state_digest(base.decoder))
            self.assertEqual(audit['initial_gate_digest'], state_digest(base.gate))
            self.assertEqual(audit['final_encoder_digest'], state_digest(model.encoder))
            self.assertEqual(audit['final_decoder_digest'], state_digest(model.decoder))
            self.assertEqual(audit['final_gate_digest'], state_digest(model.gate))
            self.assertNotEqual(state_digest(model.gate), state_digest(base.gate))
            if not gate_decoder:
                self.assertEqual(state_digest(model.decoder), state_digest(base.decoder))
            else:
                self.assertNotEqual(state_digest(model.decoder), state_digest(base.decoder))
            trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            self.assertTrue(any(name.startswith('gate.') for name in trainable))
            self.assertEqual(any(name.startswith('decoder.') for name in trainable), gate_decoder)
            self.assertTrue(all(name.startswith('gate.') or
                                (gate_decoder and name.startswith('decoder.')) for name in trainable))
            self.assertFalse(saved['requires_measured_op'])
            self.assertFalse(record['report']['requires_measured_op'])
            self.assertEqual(saved['recipe']['predictor']['encoder_granularity'], 'device')
            self.assertEqual(set(saved['stage_ids']), set(record['stage_ids']))
            self.assertEqual(len(record['stage_ids']), 4 if 'supervised' in mode else 6)
            self.assertEqual(len(set(record['stage_ids'])), len(record['stage_ids']))
            self.assertTrue(set(record['stage_ids']) <= set(ids))
            self.assertEqual(saved['config'], json.loads((directory / 'config_pin.json').read_text()))
            comparison = json.loads((directory / 'gate_comparison_pin.json').read_text())
            self.assertEqual(comparison['base_checkpoint'], audit['base_checkpoint'])
            self.assertEqual(comparison['final_phase'], model.phase)
            self.assertEqual(comparison['post_gate_test'], record['report'])
            self.assertEqual(comparison['base_test']['split'], 'test')
            self.assertEqual(comparison['base_test']['count'], record['report']['count'])
            with torch.no_grad():
                expected = model(batch)
                self.assertTrue(torch.isfinite(expected).all())
                torch.testing.assert_close(expected, model(poison), atol=0, rtol=0)
            # A checkpoint copied away from its teacher, shared bases and data
            # directory must rebuild and run with fresh (uncached) OP inputs.
            detached = self.root / f'{mode}_checkpoint_pin.pt'
            shutil.copyfile(directory / 'checkpoint_pin.pt', detached)
            restored, _ = load_checkpoint_pin(detached)
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(batch), atol=0, rtol=0)
            self.assertTrue((output / 'logs_pin' / 'seed_17_pin' / mode / 'train_pin.log').is_file())
        first, second = (audits[name] for name in STAGED_GATE_MODELS_PIN[:2])
        for key in ('base_checkpoint', 'base_state_digest', 'initial_encoder_digest',
                    'initial_decoder_digest', 'initial_gate_digest'):
            self.assertEqual(first[key], second[key], key)
        teacher = output / 'seed_17_pin/shared_pin/design_op_oracle_pin/checkpoint_pin.pt'
        self.assertTrue(teacher.is_file())
        for mode in STAGED_GATE_MODELS_PIN[:2]:
            self.assertTrue((output / 'seed_17_pin' / mode / audits[mode]['teacher_checkpoint']).is_file())
        with zipfile.ZipFile(output / 'summary_pin.xlsx') as book:
            workbook = book.read('xl/workbook.xml').decode()
            self.assertIn('ModelInputs', workbook)
            self.assertIn('GateBeforeAfter', workbook)
            cells = ''.join(book.read(name).decode() for name in book.namelist()
                            if name.startswith('xl/worksheets/') and name.endswith('.xml'))
            for mode in STAGED_GATE_MODELS_PIN:
                self.assertIn(mode, cells)

    def test_supervised_route_has_no_teacher_or_distillation_costs(self):
        mode = 'device_op_supervised_gate_pin'
        output, result = self.run_models([mode], 'supervised_only_pin')
        ids = {stage['stage_id'].rsplit('/', 1)[-1] for stage in result['stages']}
        self.assertEqual(ids, {'device_op_predictor_pin', 'frozen_device_op_cache_pin',
                               'device_op_supervised_readout_pin', mode})
        shared = output / 'seed_17_pin/shared_pin'
        self.assertFalse((shared / 'design_op_oracle_pin').exists())
        self.assertFalse((shared / 'device_op_distilled_base_pin').exists())
        self.assertTrue((shared / 'device_op_supervised_base_pin/checkpoint_pin.pt').is_file())
        audit = json.loads((output / 'seed_17_pin' / mode / 'op_gate_audit_pin.json').read_text())
        self.assertIsNone(audit['teacher_checkpoint'])

    def test_invalid_final_gate_budget_fails_before_any_output(self):
        for value in (0, -1, True, 1.5, '30'):
            with self.subTest(value=value):
                config = copy.deepcopy(self.config)
                config['op_transfer']['epochs']['gate'] = value
                self.cfg_path.write_text(json.dumps(config))
                output = self.root / 'invalid_pin'
                with self.assertRaises(ValueError):
                    self.run_models(['device_op_supervised_gate_pin'], output.name, smoke=False)
                self.assertFalse(output.exists())

    def test_model_order_does_not_change_shared_bases_or_final_weights(self):
        forward, one = self.run_models(STAGED_GATE_MODELS_PIN, 'forward_pin')
        reverse, two = self.run_models(reversed(STAGED_GATE_MODELS_PIN), 'reverse_pin')
        self.assertEqual({item['stage_id'] for item in one['stages']},
                         {item['stage_id'] for item in two['stages']})
        for name in STAGED_GATE_MODELS_PIN:
            models = [load_checkpoint_pin(root / 'seed_17_pin' / name / 'checkpoint_pin.pt')[0]
                      for root in (forward, reverse)]
            self.assertEqual(state_digest(models[0]), state_digest(models[1]), name)
        for name in ('device_op_distilled_base_pin', 'device_op_supervised_base_pin'):
            models = [load_checkpoint_pin(root / 'seed_17_pin/shared_pin' / name / 'checkpoint_pin.pt')[0]
                      for root in (forward, reverse)]
            self.assertEqual(state_digest(models[0]), state_digest(models[1]), name)


if __name__ == '__main__':
    unittest.main()
