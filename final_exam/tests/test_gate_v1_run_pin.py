"""Versioned gate names, shared controls, deployment and interruption recovery."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from device_jepa.training import state_digest
from final_exam.gate_version_registry_pin import V1_GATE_MODELS_PIN, V1_JOINT_GATE_MODELS_PIN
from final_exam.run_pin import run_experiment_pin, load_checkpoint_pin
from final_exam.tests.test_run_pin import tiny_inputs


class GateV1RunnerPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.config, self.data_path, self.cfg_path = tiny_inputs(self.root)
        torch.set_num_threads(1)

    def run_models(self, name, models, resume_from=None):
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(self.data_path, self.cfg_path, output, list(models),
                [17], 'cpu', smoke=True, resume_from=resume_from)
        return output, result

    def test_complete_nine_models_share_controls_replay_and_resume(self):
        from final_exam.data_pin import PinDataContext
        models = ('device_op_cascade_pin',) + V1_GATE_MODELS_PIN
        output, result = self.run_models('study_pin', models)
        self.assertEqual([r['mode'] for r in result['results']], list(models))
        ids = [s['stage_id'].split('/')[-1] for s in result['stages']]
        self.assertEqual(len(ids), 14)  # eight fits plus six unique prerequisites/cache
        self.assertEqual(len(set(ids)), len(ids))
        self.assertNotIn('device_op_cascade_pin', ids)
        self.assertFalse(any('_v1_pin' in name for name in ids))
        seed_dir = output / 'seed_17_pin'
        base, _ = load_checkpoint_pin(seed_dir / 'shared_pin/device_op_supervised_base_pin/checkpoint_pin.pt')
        cascade, saved = load_checkpoint_pin(seed_dir / 'device_op_cascade_pin/checkpoint_pin.pt')
        self.assertEqual(state_digest(base), state_digest(cascade))
        self.assertEqual(saved['mode'], 'device_op_cascade_pin')
        replay = self.root / 'replay_pin'
        replay.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = PinDataContext(self.data, json.loads((output / 'manifest_pin.json').read_text()),
                json.loads((output / 'config_pin.json').read_text()), 'cpu', {}, replay)
        batch = next(ctx.batches('test', 8, op=True))
        poisoned = copy.deepcopy(batch)
        for field in ('op', 'owner_op', 'performance', 'physical_performance'):
            poisoned[field].fill_(float('nan'))
        audits = {}
        for mode in V1_GATE_MODELS_PIN:
            path = seed_dir / mode
            model, metadata = load_checkpoint_pin(path / 'checkpoint_pin.pt')
            self.assertEqual(metadata['mode'], mode)
            self.assertEqual(metadata['recipe']['gate_version'], 'v1')
            self.assertEqual(metadata['input_contract_pin']['gate_version'], 'v1')
            self.assertTrue((output / 'logs_pin/seed_17_pin' / mode / 'train_pin.log').is_file())
            self.assertTrue((seed_dir / 'stages_pin' / mode / 'best_pin.pt').is_file())
            with torch.no_grad():
                torch.testing.assert_close(model(batch), model(poisoned), atol=0, rtol=0)
            audits[mode] = json.loads((path / 'op_gate_audit_pin.json').read_text())
        for family in ('distill', 'supervised'):
            all_layers = audits[f'device_op_{family}_gate_pin_v1']
            fusion = audits[f'device_op_{family}_gate_fusion_pin_v1']
            for key in ('base_state_digest', 'initial_gate_digest', 'initial_encoder_digest', 'initial_decoder_digest'):
                self.assertEqual(all_layers[key], fusion[key])
        with zipfile.ZipFile(output / 'summary_pin.xlsx') as book:
            xml = ''.join(book.read(name).decode() for name in book.namelist() if name.endswith('.xml'))
            for mode in models:
                self.assertIn(mode, xml)
            self.assertIn('GateStudy', xml)
        hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in output.rglob('*') if p.is_file()}
        recovered, resumed = self.run_models('resumed_pin', models, output)
        for stage in resumed['stages']:
            if stage['epochs']:
                self.assertEqual(stage['resume']['policy'], 'completed_stage')
                self.assertEqual(stage['resume']['new_optimizer_steps'], 0)
        for mode in models:
            old, _ = load_checkpoint_pin(seed_dir / mode / 'checkpoint_pin.pt')
            new, _ = load_checkpoint_pin(recovered / 'seed_17_pin' / mode / 'checkpoint_pin.pt')
            self.assertEqual(state_digest(old), state_digest(new))
        self.assertEqual(hashes, {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in output.rglob('*') if p.is_file()})
        subset, subset_result = self.run_models('baseline_subset_pin', ['device_op_cascade_pin'], output)
        subset_model, _ = load_checkpoint_pin(subset / 'seed_17_pin/device_op_cascade_pin/checkpoint_pin.pt')
        self.assertEqual(state_digest(cascade), state_digest(subset_model))
        self.assertEqual(len(subset_result['stages']), 3)
        for stage in subset_result['stages']:
            if stage['epochs']:
                self.assertEqual(stage['resume']['new_optimizer_steps'], 0)

    def test_joint_gate_resumes_completed_epoch_after_interruption(self):
        from experiments.op_transfer import engine
        mode = V1_JOINT_GATE_MODELS_PIN[0]
        reference, _ = self.run_models('reference_pin', [mode])
        original_fit = engine.fit_stage

        def interrupted(ctx, model, stage_id, *args, **kwargs):
            if stage_id == mode:
                calls = 0
                def stop_before_epoch(model):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError('simulated interruption')
                kwargs['before_epoch'] = stop_before_epoch
            return original_fit(ctx, model, stage_id, *args, **kwargs)

        with patch.object(engine, 'fit_stage', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                self.run_models('failed_pin', [mode])
        recovered, result = self.run_models('recovered_pin', [mode], self.root / 'failed_pin')
        cost = next(s for s in result['stages'] if s['stage_id'].endswith('/' + mode))
        self.assertEqual(cost['resume']['policy'], 'epoch_resume')
        self.assertTrue(cost['resume']['optimizer_state_restored'])
        expected, _ = load_checkpoint_pin(reference / 'seed_17_pin' / mode / 'checkpoint_pin.pt')
        actual, _ = load_checkpoint_pin(recovered / 'seed_17_pin' / mode / 'checkpoint_pin.pt')
        self.assertEqual(state_digest(expected), state_digest(actual))


if __name__ == '__main__':
    unittest.main()
