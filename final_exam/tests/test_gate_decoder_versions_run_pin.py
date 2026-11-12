"""Versioned gate/decoder fine-tuning reuses bases and resumes both optimizers' parameters."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.run_pin import load_checkpoint_pin, run_experiment_pin
from final_exam.tests.test_run_pin import tiny_inputs


class GateDecoderVersionsRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        _, self.config, self.data, self.path = tiny_inputs(self.root)
        options = engine.settings(self.config)
        options['epochs'] = dict.fromkeys(options['epochs'], 1)
        options['epochs'].update(readout=2, student_predicted_op=2, gate=2)
        options.update(patience=0, batch_size=8)
        self.config['op_transfer'] = options
        self.config['op_gate_pin'] = dict(n_experts=2, mlp_dropout=0.2)
        self.path.write_text(json.dumps(self.config))
        torch.set_num_threads(1)

    def run_models(self, name, modes, source=None):
        destination = self.root / (name + '_pin')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(self.data, self.path, destination, modes, [17], 'cpu',
                                        resume_from=source)
        return destination, result

    @staticmethod
    def snapshot(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file()}

    def test_six_variants_share_distilled_base_and_record_actual_updates(self):
        modes = [f'device_op_distill_gate_decoder_{scope}_v{version}'
                 for version in (1, 2, 3) for scope in ('pin', 'fusion_pin')]
        modes += [f'device_op_distill_gate_pin_v{version}' for version in (1, 2, 3)]
        output, result = self.run_models('mixed_versions', modes)
        names = [stage['stage_id'].split('/')[-1] for stage in result['stages']]
        for prerequisite in ('device_op_predictor_pin', 'design_op_oracle_pin',
                             'device_op_distill_encoder_pin', 'device_op_distill_readout_pin'):
            self.assertEqual(names.count(prerequisite), 1)
        audits = {}
        for mode in modes:
            directory = output / 'seed_17_pin' / mode
            audit = json.loads((directory / 'op_gate_audit_pin.json').read_text())
            audits[mode] = audit
            for frozen in ('predictor', 'encoder'):
                self.assertEqual(audit[f'initial_{frozen}_digest'], audit[f'final_{frozen}_digest'])
            decoder_active = '_gate_decoder_' in mode
            self.assertEqual(audit['decoder_frozen'], not decoder_active)
            self.assertEqual(audit['final_phase'], 'gate_decoder' if decoder_active else 'gate_only')
            if decoder_active:
                self.assertNotEqual(audit['initial_decoder_digest'], audit['final_decoder_digest'])
            else:
                self.assertEqual(audit['initial_decoder_digest'], audit['final_decoder_digest'])
            self.assertNotEqual(audit['initial_gate_digest'], audit['final_gate_digest'])
            model, payload = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
            self.assertEqual(payload['recipe']['phase'], audit['final_phase'])
            self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
            self.assertEqual(any(p.requires_grad for p in model.decoder.parameters()), decoder_active)
            cost = next(s for s in result['stages'] if s['stage_id'].endswith('/' + mode))
            self.assertEqual(cost['epochs'], 2)
        self.assertEqual(len({a['base_state_digest'] for a in audits.values()}), 1)
        for version in (1, 2, 3):
            paired = [audits[m] for m in modes if m.endswith(f'_v{version}')]
            self.assertEqual(len({a['initial_gate_digest'] for a in paired}), 1)

    def test_interrupted_gate_decoder_restores_epoch_optimizer_and_dropout_rng(self):
        original = engine.fit_stage
        for version in (1, 2, 3):
            with self.subTest(version=version):
                mode = f'device_op_distill_gate_decoder_fusion_pin_v{version}'
                reference, _ = self.run_models(f'reference_v{version}', [mode])

                def interrupted(ctx, model, stage_id, *args, **kwargs):
                    if stage_id == mode:
                        calls = 0
                        def before_epoch(_):
                            nonlocal calls
                            calls += 1
                            if calls == 2:
                                raise RuntimeError('simulated gate decoder interruption')
                        kwargs['before_epoch'] = before_epoch
                    return original(ctx, model, stage_id, *args, **kwargs)

                with patch.object(engine, 'fit_stage', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'simulated gate decoder interruption'):
                        self.run_models(f'failed_v{version}', [mode])
                source = self.root / f'failed_v{version}_pin'
                hashes = self.snapshot(source)
                resumed, result = self.run_models(f'resumed_v{version}', [mode], source)
                cost = next(s for s in result['stages'] if s['stage_id'].endswith('/' + mode))
                self.assertEqual(cost['resume']['policy'], 'epoch_resume')
                self.assertTrue(cost['resume']['optimizer_state_restored'])
                self.assertEqual(cost['resume']['restored_epoch'], 1)
                path = Path('seed_17_pin') / mode / 'checkpoint_pin.pt'
                expected, _ = load_checkpoint_pin(reference / path)
                actual, _ = load_checkpoint_pin(resumed / path)
                self.assertEqual(state_digest(expected), state_digest(actual))
                self.assertEqual(hashes, self.snapshot(source))


if __name__ == '__main__':
    unittest.main()
