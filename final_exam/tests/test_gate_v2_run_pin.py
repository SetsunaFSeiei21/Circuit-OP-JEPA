"""Sample-relative gates retain stage, inference and recovery contracts."""
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

from device_jepa.data import load_dataset
from device_jepa.training import state_digest
from final_exam.gate_version_registry_pin import V2_GATE_MODELS_PIN, V2_JOINT_GATE_MODELS_PIN
from final_exam.run_pin import load_checkpoint_pin, run_experiment_pin
from final_exam.tests.test_run_pin import tiny_inputs


class GateV2RunnerPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.config, self.data_path, self.cfg_path = tiny_inputs(self.root)
        # A nondefault value detects lost settings when a legacy ungated base
        # is copied into a v2 model or reconstructed from a saved recipe.
        self.config['op_gate_pin'] = dict(norm_eps=1e-4)
        self.config['dataset'].update(max_samples_per_circuit=18, sampling_seed=29)
        self.cfg_path.write_text(json.dumps(self.config))
        torch.set_num_threads(1)

    def run_models(self, name, models, resume_from=None, config_path=None):
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(self.data_path, config_path or self.cfg_path,
                output, list(models), [17], 'cpu', smoke=True, resume_from=resume_from)
        return output, result

    @staticmethod
    def snapshot(directory):
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob('*') if path.is_file()}

    def test_nine_models_preserve_cohort_controls_inference_and_completed_resume(self):
        from final_exam.data_pin import PinDataContext
        models = ('device_op_cascade_pin',) + V2_GATE_MODELS_PIN
        output, result = self.run_models('study_pin', models)
        self.assertEqual([row['mode'] for row in result['results']], list(models))
        stage_names = [row['stage_id'].split('/')[-1] for row in result['stages']]
        self.assertEqual(len(stage_names), 14)  # eight fits, six shared prerequisites/cache
        self.assertEqual(len(set(stage_names)), len(stage_names))
        self.assertNotIn('device_op_cascade_pin', stage_names)
        self.assertFalse(any('_v2_pin' in name for name in stage_names))
        seed_dir = output / 'seed_17_pin'
        base, _ = load_checkpoint_pin(seed_dir / 'shared_pin/device_op_supervised_base_pin/checkpoint_pin.pt')
        cascade, _ = load_checkpoint_pin(seed_dir / 'device_op_cascade_pin/checkpoint_pin.pt')
        self.assertEqual(state_digest(base), state_digest(cascade))
        manifest = json.loads((output / 'manifest_pin.json').read_text())
        self.assertEqual(manifest['sampling']['circuits']['synthetic_0']['selected_count'], 18)
        selected = load_dataset(self.data_path, sampling_config=manifest['sampling'])
        replay = self.root / 'replay_pin'
        replay.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = PinDataContext(selected, manifest,
                json.loads((output / 'config_pin.json').read_text()), 'cpu', {}, replay)
        batch = next(ctx.batches('test', 8, op=True))
        poisoned = copy.deepcopy(batch)
        for field in ('op', 'owner_op', 'performance', 'physical_performance'):
            poisoned[field].fill_(float('nan'))
        for field in ('op_mask', 'owner_op_mask', 'performance_mask'):
            poisoned[field] = ~poisoned[field]
        audits = {}
        for mode in V2_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                directory = seed_dir / mode
                model, metadata = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
                self.assertEqual(metadata['mode'], mode)
                self.assertEqual(metadata['recipe']['gate_version'], 'v2')
                self.assertEqual(metadata['recipe']['gate_config']['norm_eps'], 1e-4)
                self.assertEqual(metadata['input_contract_pin']['gate_version'], 'v2')
                self.assertEqual(model.gate.settings['norm_eps'], 1e-4)
                self.assertTrue((output / 'logs_pin/seed_17_pin' / mode / 'train_pin.log').is_file())
                self.assertTrue((seed_dir / 'stages_pin' / mode / 'best_pin.pt').is_file())
                with torch.no_grad():
                    prediction = model(batch)
                    self.assertTrue(torch.isfinite(prediction).all())
                    torch.testing.assert_close(prediction, model(poisoned), atol=0, rtol=0)
                    restored, _ = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
                    torch.testing.assert_close(prediction, restored(batch), atol=0, rtol=0)
                audits[mode] = json.loads((directory / 'op_gate_audit_pin.json').read_text())
                report = json.loads((directory / 'test_pin.json').read_text())
                for contract in (audits[mode], report['op_gate_pin']):
                    self.assertEqual(contract['gate_input_dim'], 2 * len(model.gate.field_names) + 8)
                    self.assertEqual(contract['gate_hidden_dim'], self.config['model']['dim'])
                    self.assertEqual(contract['norm_eps'], 1e-4)
                    self.assertEqual(contract['gate_relative_normalization'], 'per_sample_per_field_all_valid_mos')
                if mode not in V2_JOINT_GATE_MODELS_PIN:
                    audit = audits[mode]
                    for key in ('predictor', 'encoder'):
                        self.assertEqual(audit[f'initial_{key}_digest'], audit[f'final_{key}_digest'])
                    if '_gate_decoder_' in mode:
                        self.assertEqual(audit['final_trainable'], 'gate_and_decoder')
                        self.assertFalse(audit['decoder_frozen'])
                        self.assertNotEqual(audit['initial_decoder_digest'], audit['final_decoder_digest'])
                    else:
                        self.assertTrue(audit['decoder_frozen'])
                        self.assertEqual(audit['initial_decoder_digest'], audit['final_decoder_digest'])
                    comparison = json.loads((directory / 'gate_comparison_pin.json').read_text())
                    self.assertEqual(comparison['gate_version'], 'v2')
                    self.assertEqual(comparison['post_gate_test'], report)
        for family in ('distill', 'supervised'):
            all_layers = audits[f'device_op_{family}_gate_pin_v2']
            fusion = audits[f'device_op_{family}_gate_fusion_pin_v2']
            for key in ('base_state_digest', 'initial_gate_digest', 'initial_encoder_digest', 'initial_decoder_digest'):
                self.assertEqual(all_layers[key], fusion[key])
        with zipfile.ZipFile(output / 'summary_pin.xlsx') as book:
            xml = ''.join(book.read(name).decode() for name in book.namelist() if name.endswith('.xml'))
            for mode in models:
                self.assertIn(mode, xml)
            self.assertIn('GateStudy', xml)
            self.assertIn('per_sample_per_field_all_valid_mos', xml)
        before = self.snapshot(output)
        # Even if the original input is a complete JSON, recovery owns its
        # previously saved sample IDs and cannot draw another cohort.
        with patch('device_jepa.sampling.select_sample_ids', side_effect=AssertionError('redrew samples')):
            recovered, resumed = self.run_models('resumed_pin', models, output)
        self.assertEqual(manifest, json.loads((recovered / 'manifest_pin.json').read_text()))
        for stage in resumed['stages']:
            if stage['epochs']:
                self.assertEqual(stage['resume']['policy'], 'completed_stage')
                self.assertEqual(stage['resume']['new_optimizer_steps'], 0)
        for mode in models:
            old, _ = load_checkpoint_pin(seed_dir / mode / 'checkpoint_pin.pt')
            new, _ = load_checkpoint_pin(recovered / 'seed_17_pin' / mode / 'checkpoint_pin.pt')
            self.assertEqual(state_digest(old), state_digest(new))
        self.assertEqual(self.snapshot(output), before)
        subset, subset_result = self.run_models('baseline_subset_pin', ['device_op_cascade_pin'], output)
        subset_model, _ = load_checkpoint_pin(subset / 'seed_17_pin/device_op_cascade_pin/checkpoint_pin.pt')
        self.assertEqual(state_digest(cascade), state_digest(subset_model))
        self.assertEqual(len(subset_result['stages']), 3)
        for stage in subset_result['stages']:
            if stage['epochs']:
                self.assertEqual(stage['resume']['new_optimizer_steps'], 0)

    def test_joint_gate_resumes_completed_epoch_after_interruption(self):
        from experiments.op_transfer import engine
        mode = V2_JOINT_GATE_MODELS_PIN[0]
        reference, _ = self.run_models('reference_pin', [mode])
        original_fit = engine.fit_stage

        def interrupted(ctx, model, stage_id, *args, **kwargs):
            if stage_id == mode:
                calls = 0
                def stop_before_epoch(model):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError('simulated v2 interruption')
                kwargs['before_epoch'] = stop_before_epoch
            return original_fit(ctx, model, stage_id, *args, **kwargs)

        with patch.object(engine, 'fit_stage', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'simulated v2 interruption'):
                self.run_models('failed_pin', [mode])
        recovered, result = self.run_models('recovered_pin', [mode], self.root / 'failed_pin')
        cost = next(row for row in result['stages'] if row['stage_id'].endswith('/' + mode))
        self.assertEqual(cost['resume']['policy'], 'epoch_resume')
        self.assertTrue(cost['resume']['optimizer_state_restored'])
        expected, _ = load_checkpoint_pin(reference / 'seed_17_pin' / mode / 'checkpoint_pin.pt')
        actual, _ = load_checkpoint_pin(recovered / 'seed_17_pin' / mode / 'checkpoint_pin.pt')
        self.assertEqual(state_digest(expected), state_digest(actual))

    def test_changed_normalization_epsilon_rejected_before_resume_writes(self):
        mode = V2_JOINT_GATE_MODELS_PIN[0]
        source, _ = self.run_models('source_pin', [mode])
        before = self.snapshot(source)
        changed = copy.deepcopy(self.config)
        changed['op_gate_pin']['norm_eps'] = 1e-3
        path = self.root / 'changed.json'
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            self.run_models('invalid_resume_pin', [mode], source, config_path=path)
        self.assertFalse((self.root / 'invalid_resume_pin').exists())
        self.assertEqual(self.snapshot(source), before)

    def test_baseline_only_resume_inherits_default_norm_eps_from_v2_source(self):
        # User configuration can predate v2 and omit all gate settings. The
        # initial mixed study resolves v2's epsilon; a baseline-only recovery
        # must recover that default without requiring users to edit the input.
        self.config.pop('op_gate_pin')
        self.cfg_path.write_text(json.dumps(self.config))
        models = ('device_op_cascade_pin', V2_JOINT_GATE_MODELS_PIN[0])
        source, _ = self.run_models('default_source_pin', models)
        saved_config = json.loads((source / 'config_pin.json').read_text())
        self.assertEqual(saved_config['op_gate_pin']['norm_eps'], 1e-5)
        before = self.snapshot(source)
        checkpoint = Path('seed_17_pin/device_op_cascade_pin/checkpoint_pin.pt')
        expected, _ = load_checkpoint_pin(source / checkpoint)
        recovered, result = self.run_models('default_baseline_resume_pin', ['device_op_cascade_pin'], source)
        actual, _ = load_checkpoint_pin(recovered / checkpoint)
        self.assertEqual(state_digest(expected), state_digest(actual))
        for stage in result['stages']:
            if stage['epochs']:
                self.assertEqual(stage['resume']['new_optimizer_steps'], 0)
        changed = copy.deepcopy(self.config)
        changed['op_gate_pin'] = dict(norm_eps=1e-3)
        path = self.root / 'changed_baseline_config.json'
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            self.run_models('changed_baseline_resume_pin', ['device_op_cascade_pin'],
                            source, config_path=path)
        self.assertFalse((self.root / 'changed_baseline_resume_pin').exists())
        self.assertEqual(self.snapshot(source), before)


if __name__ == '__main__':
    unittest.main()
