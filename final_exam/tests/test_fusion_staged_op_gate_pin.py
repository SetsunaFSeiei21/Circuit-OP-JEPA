"""Fusion-only OP gating: scope, frozen gradients, saved recipes and recovery."""
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
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.staged_op_gate_pin import StagedOPGateModelPin
from final_exam.staged_gate_registry_pin import (
    ALL_STAGED_GATE_MODELS_PIN, FUSION_STAGED_GATE_MODELS_PIN,
    STAGED_GATE_MODELS_PIN, staged_gate_spec_pin,
)
from final_exam.run_pin import load_checkpoint_pin, run_experiment_pin
from final_exam.tests import test_op_gate_models_pin as gate_helpers
from final_exam.tests.test_run_pin import tiny_inputs


class FusionStagedGateModelsPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = gate_helpers.OPAttentionModelsPinTests
        fixture.setUpClass()
        cls.prep = fixture.prep
        cls.support = fixture.support
        cls.batch = copy.deepcopy(fixture.batch)
        # Match the user's 3 structure + 3 fusion architecture at a small width.
        cls.cfg = {**fixture.cfg, 'encoder_layers': 6, 'cross_layers': 3}

    def setUp(self):
        torch.manual_seed(17)

    def model(self, mode=FUSION_STAGED_GATE_MODELS_PIN[0]):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
        return StagedOPGateModelPin(self.cfg, self.prep, predictor, mode=mode)

    @staticmethod
    def capture_attention(model, batch):
        masks, structures, handles = {}, [], []

        def save_mask(name):
            def capture(module, args, kwargs):
                value = kwargs.get('attn_mask')
                masks[name] = None if value is None else value.detach().clone()
            return capture

        for name, module in model.encoder.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                handles.append(module.register_forward_pre_hook(save_mask('encoder.' + name), with_kwargs=True))
        for name, module in model.decoder.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                handles.append(module.register_forward_pre_hook(save_mask('decoder.' + name), with_kwargs=True))
        handles.append(model.encoder.structure_encoder.register_forward_hook(
            lambda module, args, output: structures.append(output['tokens'].detach().clone())))
        try:
            with torch.no_grad():
                prediction = model(batch)
        finally:
            for handle in handles:
                handle.remove()
        return masks, structures[0], prediction

    def test_gate_changes_only_fusion_self_attention_masks(self):
        model = self.model().set_phase('gate_only').eval()
        before, structure_before, prediction_before = self.capture_attention(model, self.batch)
        with torch.no_grad():
            model.gate.score_head[-1].bias.fill_(-2.)
        after, structure_after, prediction_after = self.capture_attention(model, self.batch)
        torch.testing.assert_close(structure_before, structure_after, atol=0, rtol=0)
        changed = []
        for name, value in before.items():
            with self.subTest(attention=name):
                other = after[name]
                if name.startswith('encoder.fusion_blocks.') and name.endswith('.self_attn'):
                    self.assertIsNotNone(value)
                    self.assertTrue(torch.equal(torch.isneginf(value), torch.isneginf(other)))
                    allowed = torch.isfinite(value)
                    self.assertGreater(float((value[allowed] - other[allowed]).abs().max()), 0.)
                    changed.append(name)
                elif value is None:
                    self.assertIsNone(other)
                else:
                    torch.testing.assert_close(value, other, atol=0, rtol=0)
        self.assertEqual(len(changed), 3)
        self.assertEqual(sum(name.startswith('encoder.structure_encoder.blocks.') for name in before), 3)
        self.assertEqual(sum(name.endswith('.cross_attn') for name in before), 3)
        self.assertTrue(any(name.startswith('decoder.') for name in before))
        self.assertGreater(float((prediction_before - prediction_after).abs().max()), 0.)

    def test_original_scope_still_changes_structure_attention(self):
        model = self.model(STAGED_GATE_MODELS_PIN[0]).set_phase('gate_only').eval()
        before, structure_before, _ = self.capture_attention(model, self.batch)
        with torch.no_grad():
            model.gate.score_head[-1].bias.fill_(-2.)
        after, structure_after, _ = self.capture_attention(model, self.batch)
        for name in before:
            if name.startswith('encoder.structure_encoder.blocks.'):
                finite = torch.isfinite(before[name])
                self.assertGreater(float((before[name][finite] - after[name][finite]).abs().max()), 0.)
        self.assertGreater(float((structure_before - structure_after).abs().max()), 0.)

    def test_all_preparatory_phases_bypass_new_gate_exactly(self):
        for mode in FUSION_STAGED_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode).eval()
                for phase in ('supervised', 'distill_encoder', 'readout'):
                    model.set_phase(phase).eval()
                    with torch.no_grad():
                        prepared = model.prepared_batch(self.batch)
                        memory, padding = model.encoder(prepared)
                        expected = model.decoder(memory, padding)
                        with patch.object(model.gate, 'forward', side_effect=AssertionError('gate called during base')):
                            torch.testing.assert_close(expected, model(self.batch), atol=0, rtol=0)

    def test_final_phases_update_only_requested_modules_with_finite_gate_gradients(self):
        for mode in FUSION_STAGED_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                phase = staged_gate_spec_pin(mode)['final_phase']
                model = self.model(mode).set_phase(phase).train()
                active = {'gate', 'decoder'} if phase == 'gate_decoder' else {'gate'}
                names = ('predictor', 'encoder', 'decoder', 'gate')
                before = {name: state_digest(getattr(model, name)) for name in names}
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                prediction = model(self.batch)
                prediction.square().mean().backward()
                gradients = [p.grad for p in model.gate.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
                self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.)
                self.assertTrue(bool((model.gate.score_head[-1].weight.grad.abs().sum(-1) > 0).all()))
                for name in set(names) - active:
                    self.assertTrue(all(p.grad is None for p in getattr(model, name).parameters()), name)
                    self.assertFalse(getattr(model, name).training)
                optimizer.step()
                for name in names:
                    if name in active:
                        self.assertNotEqual(before[name], state_digest(getattr(model, name)), name)
                    else:
                        self.assertEqual(before[name], state_digest(getattr(model, name)), name)

    def test_recipes_preserve_scope_and_reject_scope_mismatch(self):
        for mode in ALL_STAGED_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode).set_phase(staged_gate_spec_pin(mode)['final_phase']).eval()
                recipe = describe_model_pin(model)
                expected_scope = staged_gate_spec_pin(mode)['attention_scope']
                self.assertEqual(recipe['attention_scope'], expected_scope)
                restored = rebuild_model_pin(recipe, self.cfg, self.prep)
                restored.load_state_dict(model.state_dict())
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.batch), restored(self.batch), atol=0, rtol=0)
                wrong_scope = ('fusion_self_attention' if expected_scope == 'all_encoder_self_attention'
                               else 'all_encoder_self_attention')
                with self.assertRaises(ValueError):
                    rebuild_model_pin({**recipe, 'attention_scope': wrong_scope}, self.cfg, self.prep)
                missing_scope = copy.deepcopy(recipe)
                missing_scope.pop('attention_scope')
                if mode in STAGED_GATE_MODELS_PIN:
                    legacy = rebuild_model_pin(missing_scope, self.cfg, self.prep)
                    self.assertEqual(legacy.attention_scope, 'all_encoder_self_attention')
                else:
                    with self.assertRaises(ValueError):
                        rebuild_model_pin(missing_scope, self.cfg, self.prep)


class FusionStagedGateRunnerPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.config, self.data_path, self.cfg_path = tiny_inputs(self.root)
        torch.set_num_threads(1)

    def run_models(self, name, models=FUSION_STAGED_GATE_MODELS_PIN, source=None):
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(self.data_path, self.cfg_path, output, list(models), [17],
                                        'cpu', smoke=True, resume_from=source)
        return output, result

    @staticmethod
    def hashes(root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob('*') if path.is_file()}

    def test_six_variants_share_identical_bases_and_export_distinct_scope(self):
        output, result = self.run_models('all_six_pin', STAGED_GATE_MODELS_PIN + FUSION_STAGED_GATE_MODELS_PIN)
        self.assertEqual(len(result['results']), 6)
        ids = [record['stage_id'] for record in result['stages']]
        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)
        for name in ('device_op_predictor_pin', 'frozen_device_op_cache_pin', 'design_op_oracle_pin',
                     'device_op_distill_encoder_pin', 'device_op_distill_readout_pin',
                     'device_op_supervised_readout_pin'):
            self.assertEqual(ids.count('seed_17_pin/' + name), 1)
        audits = {}
        for record in result['results']:
            mode = record['mode']
            scope = staged_gate_spec_pin(mode)['attention_scope']
            directory = output / 'seed_17_pin' / mode
            saved = torch.load(directory / 'checkpoint_pin.pt', map_location='cpu', weights_only=True)
            self.assertEqual(saved['recipe']['attention_scope'], scope)
            self.assertEqual(saved['input_contract_pin']['attention_scope'], scope)
            self.assertEqual(record['report']['op_gate_pin']['attention_scope'], scope)
            audits[mode] = json.loads((directory / 'op_gate_audit_pin.json').read_text())
            self.assertEqual(audits[mode]['attention_scope'], scope)
            self.assertEqual(len(record['stage_ids']), 4 if 'supervised' in mode else 6)
        for original, fusion in zip(STAGED_GATE_MODELS_PIN, FUSION_STAGED_GATE_MODELS_PIN):
            for key in ('base_checkpoint', 'base_state_digest', 'initial_encoder_digest',
                        'initial_decoder_digest', 'initial_gate_digest'):
                self.assertEqual(audits[original][key], audits[fusion][key], (original, key))
        with zipfile.ZipFile(output / 'summary_pin.xlsx') as book:
            cells = ''.join(book.read(path).decode() for path in book.namelist()
                            if path.startswith('xl/worksheets/') and path.endswith('.xml'))
            self.assertIn('fusion_self_attention', cells)
            self.assertIn('all_encoder_self_attention', cells)
            for mode in STAGED_GATE_MODELS_PIN + FUSION_STAGED_GATE_MODELS_PIN:
                self.assertIn(mode, cells)

    def test_new_three_completed_run_recovery_never_optimizes_again(self):
        source, original = self.run_models('complete_pin')
        before = self.hashes(source)
        with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('completed stage trained again')):
            output, resumed = self.run_models('resumed_pin', source=source)
        self.assertEqual(self.hashes(source), before)
        self.assertEqual(original['results'], resumed['results'])
        self.assertTrue((output / 'summary_pin.xlsx').is_file())
        for stage in resumed['stages']:
            self.assertEqual(stage['resume']['new_optimizer_steps'], 0)
        for mode in FUSION_STAGED_GATE_MODELS_PIN:
            first, _ = load_checkpoint_pin(source / 'seed_17_pin' / mode / 'checkpoint_pin.pt')
            second, _ = load_checkpoint_pin(output / 'seed_17_pin' / mode / 'checkpoint_pin.pt')
            self.assertEqual(state_digest(first), state_digest(second))
            self.assertEqual(second.attention_scope, 'fusion_self_attention')

    def test_supervised_fusion_only_run_does_not_train_teacher(self):
        mode = FUSION_STAGED_GATE_MODELS_PIN[2]
        output, result = self.run_models('supervised_pin', [mode])
        self.assertEqual({record['stage_id'].split('/')[-1] for record in result['stages']},
                         {'device_op_predictor_pin', 'frozen_device_op_cache_pin',
                          'device_op_supervised_readout_pin', mode})
        self.assertFalse((output / 'seed_17_pin/shared_pin/design_op_oracle_pin').exists())


if __name__ == '__main__':
    unittest.main()
