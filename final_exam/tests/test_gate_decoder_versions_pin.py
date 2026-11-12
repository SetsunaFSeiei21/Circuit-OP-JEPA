"""Versioned post-distillation gates jointly tune only the gate and decoder."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from device_jepa.training import state_digest
from final_exam.gate_version_registry_pin import (
    VERSIONED_STAGED_GATE_MODELS_PIN, gate_version_pin,
)
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.staged_op_gate_pin import StagedOPGateModelPin
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.tests import test_op_gate_models_pin as fixtures
from final_exam.tests import test_fusion_staged_op_gate_pin as scope_fixtures


DECODER_MODES = tuple(
    f'device_op_distill_gate_decoder{suffix}_pin_v{version}'
    for version in (1, 2, 3) for suffix in ('', '_fusion')
)


class VersionedGateDecoderPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OPAttentionModelsPinTests.setUpClass()
        source = fixtures.OPAttentionModelsPinTests
        cls.cfg = {**source.cfg, 'encoder_layers': 6, 'cross_layers': 3}
        cls.prep, cls.support = source.prep, source.support
        cls.batch = copy.deepcopy(source.batch)

    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def model(self, mode, *, dropout=0.):
        cfg = {**self.cfg, 'dropout': dropout}
        predictor = DeviceOPPredictorPin(cfg, self.prep, self.support)
        gate_config = {'n_experts': 2, 'mlp_dropout': .2} if mode.endswith('_v3') else None
        return StagedOPGateModelPin(cfg, self.prep, predictor,
                                   mode=mode, gate_config=gate_config)

    def test_all_six_names_resolve_to_distilled_gate_decoder_without_changing_existing_modes(self):
        self.assertEqual(len(set(DECODER_MODES)), 6)
        for mode in DECODER_MODES:
            with self.subTest(mode=mode):
                self.assertIn(mode, VERSIONED_STAGED_GATE_MODELS_PIN)
                self.assertEqual(gate_version_pin(mode), mode.rsplit('_', 1)[-1])
                spec = staged_gate_spec_pin(mode)
                self.assertEqual(spec['base_training'], 'distilled')
                self.assertEqual(spec['final_phase'], 'gate_decoder')
                self.assertEqual(spec['attention_scope'], 'fusion_self_attention' if '_fusion_' in mode
                                 else 'all_encoder_self_attention')
        for mode in set(VERSIONED_STAGED_GATE_MODELS_PIN) - set(DECODER_MODES):
            if gate_version_pin(mode) not in ('v1', 'v2', 'v3'):
                continue  # Later version strategies have their own contract tests.
            self.assertEqual(staged_gate_spec_pin(mode)['final_phase'], 'gate_only')

    def exercise_step(self, mode, device):
        batch = {key: value.to(device) if torch.is_tensor(value) else copy.deepcopy(value)
                 for key, value in self.batch.items()}
        model = self.model(mode, dropout=.3).to(device).set_phase('gate_decoder').train()
        before = {name: state_digest(getattr(model, name))
                  for name in ('predictor', 'encoder', 'decoder', 'gate')}
        for name in before:
            active = name in ('gate', 'decoder')
            module = getattr(model, name)
            self.assertEqual(module.training, active)
            self.assertTrue(all(p.requires_grad == active for p in module.parameters()))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            self.assertTrue(bool(torch.isfinite(prediction).all()))
            prediction.square().mean().backward()
            for name in ('encoder', 'predictor'):
                self.assertTrue(all(p.grad is None for p in getattr(model, name).parameters()))
            for name in ('gate', 'decoder'):
                grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in grads))
                self.assertGreater(sum(float(grad.abs().sum()) for grad in grads), 0.)
            optimizer.step()
        for name in ('encoder', 'predictor'):
            self.assertEqual(before[name], state_digest(getattr(model, name)), name)
        for name in ('gate', 'decoder'):
            self.assertNotEqual(before[name], state_digest(getattr(model, name)), name)

    def test_optimizer_steps_update_gate_and_decoder_while_encoder_and_predictor_stay_identical(self):
        for mode in DECODER_MODES:
            with self.subTest(mode=mode):
                self.exercise_step(mode, 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_gate_decoder_steps_keep_frozen_modules_identical(self):
        for mode in DECODER_MODES:
            with self.subTest(mode=mode):
                self.exercise_step(mode, 'cuda')

    def test_gate_remains_disabled_until_final_phase_and_transition_clears_gradients(self):
        for mode in DECODER_MODES:
            with self.subTest(mode=mode):
                model = self.model(mode)
                for phase in ('supervised', 'distill_encoder', 'readout'):
                    model.set_phase(phase).train()
                    with patch.object(model.gate, 'forward', side_effect=AssertionError('premature gate')):
                        model(self.batch).square().mean().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                model.set_phase('gate_decoder')
                self.assertTrue(model.gate_enabled)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))

    def test_scope_changes_only_requested_encoder_attention_and_gate_is_shared(self):
        for mode in DECODER_MODES:
            with self.subTest(mode=mode):
                model = self.model(mode).set_phase('gate_decoder').eval()
                with patch.object(model.gate, 'forward', wraps=model.gate.forward) as call:
                    before, structure, _ = scope_fixtures.FusionStagedGateModelsPinTests.capture_attention(model, self.batch)
                    self.assertEqual(call.call_count, 1)
                with torch.no_grad():
                    model.gate.score_head[-1].bias.fill_(-2.)
                after, structure_after, _ = scope_fixtures.FusionStagedGateModelsPinTests.capture_attention(model, self.batch)
                changed = [name for name, value in before.items()
                           if value is not None and not torch.equal(value, after[name])]
                self.assertEqual(len(changed), 3 if '_fusion_' in mode else 6)
                self.assertTrue(all(name.startswith('encoder.') and not name.endswith('cross_attn')
                                    for name in changed))
                if '_fusion_' in mode:
                    self.assertTrue(all(name.startswith('encoder.fusion_blocks.') for name in changed))
                    torch.testing.assert_close(structure, structure_after, atol=0, rtol=0)

    def test_gate_only_and_decoder_variants_start_from_identical_shared_base_and_gate(self):
        base = self.model('device_op_distill_gate_pin').set_phase('readout').eval()
        before = {name: state_digest(getattr(base, name))
                  for name in ('predictor', 'encoder', 'decoder')}
        for version in (1, 2, 3):
            copies = []
            for suffix in ('', '_fusion'):
                for decoder in ('', '_decoder'):
                    mode = f'device_op_distill_gate{decoder}{suffix}_pin_v{version}'
                    item = copy.deepcopy(base)
                    torch.manual_seed(29)
                    settings = {'n_experts': 2, 'mlp_dropout': .2} if version == 3 else None
                    item.set_mode_pin(mode, gate_config=settings)
                    self.assertEqual(before, {name: state_digest(getattr(item, name)) for name in before})
                    with torch.no_grad():
                        torch.testing.assert_close(base(self.batch), item(self.batch), atol=0, rtol=0)
                    item.set_phase(staged_gate_spec_pin(mode)['final_phase']).eval()
                    copies.append(item)
            self.assertEqual(len({state_digest(item.gate) for item in copies}), 1)
            # Policies alter trainability, not weights. PyTorch may pick a
            # different attention kernel when all decoder weights are frozen.
            with torch.no_grad():
                torch.testing.assert_close(copies[0](self.batch), copies[1](self.batch), atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(copies[2](self.batch), copies[3](self.batch), atol=1e-7, rtol=1e-6)
        self.assertEqual(before, {name: state_digest(getattr(base, name)) for name in before})

    def test_checkpoint_replay_preserves_phase_and_rejects_gate_only_renaming(self):
        poison = copy.deepcopy(self.batch)
        for key in ('op', 'owner_op', 'performance'):
            poison[key].fill_(float('nan'))
        for key in ('op_mask', 'owner_op_mask', 'op_target_mask', 'owner_op_target_mask', 'performance_mask'):
            poison[key].logical_not_()
        for mode in DECODER_MODES:
            with self.subTest(mode=mode):
                model = self.model(mode).set_phase('gate_decoder').eval()
                stream = io.BytesIO()
                torch.save({'recipe': describe_model_pin(model), 'state_dict': model.state_dict()}, stream)
                stream.seek(0)
                saved = torch.load(stream, weights_only=False)
                restored = rebuild_model_pin(saved['recipe'], self.cfg, self.prep)
                restored.load_state_dict(saved['state_dict'])
                restored.eval()
                self.assertEqual(restored.phase, 'gate_decoder')
                self.assertTrue(all(p.requires_grad for p in restored.decoder.parameters()))
                with torch.no_grad():
                    expected = model(self.batch)
                    torch.testing.assert_close(expected, restored(self.batch), atol=0, rtol=0)
                    torch.testing.assert_close(expected, restored(poison), atol=0, rtol=0)
                wrong_phase = {**saved['recipe'], 'phase': 'gate_only'}
                with self.assertRaisesRegex(ValueError, 'phase'):
                    rebuild_model_pin(wrong_phase, self.cfg, self.prep)
                wrong_name = {**saved['recipe'], 'mode': mode.replace('_gate_decoder_', '_gate_')}
                with self.assertRaisesRegex(ValueError, 'phase'):
                    rebuild_model_pin(wrong_name, self.cfg, self.prep)
                wrong_scope = {**saved['recipe'], 'attention_scope': 'none'}
                with self.assertRaisesRegex(ValueError, 'scope'):
                    rebuild_model_pin(wrong_scope, self.cfg, self.prep)


if __name__ == '__main__':
    unittest.main()
