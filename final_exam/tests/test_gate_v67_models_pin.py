"""MOS polarity inputs preserve signed late-fusion gate and deployment contracts."""
import copy
import math
import unittest
from unittest.mock import patch

import torch

from device_jepa.training import state_digest
from final_exam.gate_version_registry_pin import (
    V6_GATE_MODELS_PIN, V7_GATE_MODELS_PIN, VERSIONED_STAGED_GATE_MODELS_PIN,
)
from final_exam.models.op_gate_pin import DeviceOPGateModelPin, DeviceOPPredictorPin, gate_class_pin, validate_gate_config_pin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.staged_op_gate_pin import StagedOPGateModelPin
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.tests import test_op_gate_models_pin as fixtures
from final_exam.tests import test_fusion_staged_op_gate_pin as scope_helpers


class GateV67ModelsPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OPAttentionModelsPinTests.setUpClass()
        base = fixtures.OPAttentionModelsPinTests
        cls.cfg = {**base.cfg, 'encoder_layers': 6, 'cross_layers': 3}
        cls.prep, cls.support, cls.batch = base.prep, base.support, copy.deepcopy(base.batch)

    def setUp(self):
        torch.manual_seed(17)

    def model(self, mode='device_op_gate_fusion_pin_v6', config=None):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
        settings = dict(n_experts=3, mlp_dropout=0., **(config or {}))
        if mode in VERSIONED_STAGED_GATE_MODELS_PIN:
            return StagedOPGateModelPin(self.cfg, self.prep, predictor, mode=mode,
                gate_config=settings).set_phase(staged_gate_spec_pin(mode)['final_phase']).eval()
        return DeviceOPGateModelPin(mode, self.cfg, self.prep, predictor, gate_config=settings).eval()

    def test_pair_inputs_only_append_fixed_type_one_hot_to_the_old_inputs(self):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support).eval()
        values, support = predictor(self.batch)
        vocab = self.prep['device_preprocessing']['device_vocab']
        expected = torch.zeros(*values.shape[:2], 2)
        for kind, index in vocab.items():
            if kind.lower() in ('nmos', 'pmos'):
                expected[..., ('nmos', 'pmos').index(kind.lower())] = (
                    self.batch['owner_device_types'] == index).float()
        for old_version, version in (('v4', 'v6'), ('v5', 'v7')):
            with self.subTest(version=version):
                gate = gate_class_pin(version)(self.cfg, self.prep, {'n_experts': 2}).eval()
                old = gate_class_pin(old_version)(self.cfg, self.prep, {'n_experts': 2}).eval()
                previous = old.pair_inputs(self.batch, values, support)
                inputs = gate.pair_inputs(self.batch, values, support)
                self.assertEqual(inputs.shape, (*values.shape[:2], 4, 4, 2*values.shape[-1]+10))
                torch.testing.assert_close(inputs[..., :-2], previous, atol=0, rtol=0)
                torch.testing.assert_close(inputs[..., -2:],
                    expected[:, :, None, None].expand(*values.shape[:2], 4, 4, 2), atol=0, rtol=0)
                self.assertEqual(gate.normalization_contract, old.normalization_contract)
                self.assertEqual(gate.architecture_contract['device_type_encoding'], dict(
                    kind='one_hot', categories=['nmos', 'pmos'], width=2,
                    placement='after_terminal_roles', normalization='none'))
                # The type code is not transformed by OP magnitude or support.
                changed = gate.pair_inputs(self.batch, values*100., ~support)
                torch.testing.assert_close(changed[..., -2:], inputs[..., -2:], atol=0, rtol=0)
                for _, network in gate.iter_layer_score_heads():
                    self.assertEqual(network[0].in_features, gate.input_dim)
                    self.assertEqual(network[0].out_features, self.cfg['dim'])

    def test_equal_op_values_can_produce_different_scores_from_type_alone(self):
        values = torch.zeros(*self.batch['owner_mask'].shape,
                             len(self.prep['feature_spec']['op_names']))
        supported = torch.ones_like(values, dtype=torch.bool)
        vocab = self.prep['device_preprocessing']['device_vocab']
        n = next(i for i, kind in enumerate(self.batch['owner_device_types'][0].tolist())
                 if kind == vocab['nmos'])
        p = next(i for i, kind in enumerate(self.batch['owner_device_types'][0].tolist())
                 if kind == vocab['pmos'])
        for version in ('v6', 'v7'):
            gate = gate_class_pin(version)(self.cfg, self.prep,
                {'n_experts': 2, 'mlp_dropout': 0.}).eval()
            inputs = gate.pair_inputs(self.batch, values, supported)
            torch.testing.assert_close(inputs[0, n, ..., :-2], inputs[0, p, ..., :-2], atol=0, rtol=0)
            # Construct a scorer depending only on the two added channels.
            # This checks a functional path through routers/experts and every
            # gated layer, rather than relying on random initial differences.
            seen = set()
            for _, network in gate.iter_layer_score_heads():
                if id(network) in seen:
                    continue
                seen.add(id(network))
                with torch.no_grad():
                    for param in network.parameters():
                        param.zero_()
                    network[0].weight[:, -2, 0] = 1.
                    network[0].weight[:, -1, 0] = -1.
                    network[3].weight[:, 0, 0] = 1.
                    network[6].weight[:, 0, 0] = 1.
                scores = network(inputs)
                self.assertGreater(float(scores[0, n, 0, 2, 0].detach()), 0.)
                self.assertLess(float(scores[0, p, 0, 2, 0].detach()), 0.)
                x = inputs.detach().clone().requires_grad_()
                network(x)[..., 0].sum().backward()
                self.assertGreater(float(x.grad[..., -2:].abs().sum()), 0.)

    def test_type_encoding_uses_saved_vocab_names_not_fixed_numeric_ids(self):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support).eval()
        values, support = predictor(self.batch)
        vocab = self.prep['device_preprocessing']['device_vocab']
        permutation = torch.arange(len(vocab)-1, -1, -1)
        prep = copy.deepcopy(self.prep)
        prep['device_preprocessing']['device_vocab'] = {
            name.upper(): int(permutation[index]) for name, index in vocab.items()}
        batch = dict(self.batch,
                     owner_device_types=permutation[self.batch['owner_device_types']])
        for version in ('v6', 'v7'):
            original = gate_class_pin(version)(self.cfg, self.prep, {'n_experts': 2}).eval()
            remapped = gate_class_pin(version)(self.cfg, prep, {'n_experts': 2}).eval()
            torch.testing.assert_close(original.pair_inputs(self.batch, values, support),
                remapped.pair_inputs(batch, values, support), atol=0, rtol=0)

    def test_type_encoding_excludes_passives_padding_and_has_requested_dtype(self):
        for version in ('v6', 'v7'):
            gate = gate_class_pin(version)(self.cfg, self.prep, {'n_experts': 2}).eval()
            batch = dict(self.batch, owner_mask=self.batch['owner_mask'].clone())
            batch['owner_mask'][:, 0] = False
            one_hot = gate.device_type_one_hot(batch, dtype=torch.float64)
            self.assertEqual(one_hot.dtype, torch.float64)
            self.assertEqual(one_hot.device, batch['owner_device_types'].device)
            torch.testing.assert_close(one_hot[:, 0], torch.zeros(3, 2, dtype=torch.float64))
            # The fixture's last physical owner is a resistor.
            torch.testing.assert_close(one_hot[:, 3], torch.zeros(3, 2, dtype=torch.float64))
            torch.testing.assert_close(one_hot[:, 1], torch.tensor([0., 1.], dtype=torch.float64).expand(3, 2))

    def test_physical_owner_reordering_preserves_pin_biases(self):
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support).eval()
        values, support = predictor(self.batch)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        owner = self.batch['owner_index']
        remapped_owner = torch.where(owner >= 0, inverse[owner.clamp_min(0)], -1)
        batch = dict(self.batch,
            owner_device_types=self.batch['owner_device_types'][:, permutation],
            owner_mask=self.batch['owner_mask'][:, permutation],
            owner_index=remapped_owner)
        if 'pin_to_device' in batch:
            batch['pin_to_device'] = remapped_owner
        for version in ('v6', 'v7'):
            gate = gate_class_pin(version)(self.cfg, self.prep,
                {'n_experts': 2, 'mlp_dropout': 0.}).eval()
            before = gate(self.batch, values, support)
            after = gate(batch, values[:, permutation], support[:, permutation])
            self.assertIsNone(before[0]); self.assertIsNone(after[0])
            for a, b in zip(before[1:], after[1:]):
                torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-5)

    def test_legacy_layout_conversion_keeps_type_mapping_and_rejects_recipe_substitution(self):
        from final_exam.models.op_gate_pin import OPAttentionGatePin
        predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support).eval()
        values, support = predictor(self.batch)
        source = OPAttentionGatePin(self.cfg, self.prep).eval()
        for version in ('v6', 'v7'):
            gate = gate_class_pin(version).from_gate_layout(source, {'n_experts': 2}).eval()
            fresh = gate_class_pin(version)(self.cfg, self.prep, {'n_experts': 2}).eval()
            torch.testing.assert_close(gate.pair_inputs(self.batch, values, support),
                fresh.pair_inputs(self.batch, values, support), atol=0, rtol=0)
            model = self.model('device_op_gate_fusion_pin_'+version)
            recipe = describe_model_pin(model)
            wrong = copy.deepcopy(recipe)
            wrong['gate_architecture'].pop('device_type_encoding')
            with self.assertRaises(ValueError):
                rebuild_model_pin(wrong, self.cfg, self.prep)
            wrong = copy.deepcopy(recipe)
            wrong['gate_architecture']['device_type_encoding']['categories'].reverse()
            with self.assertRaises(ValueError):
                rebuild_model_pin(wrong, self.cfg, self.prep)
            # A prior version's checkpoint is not a same-architecture resume.
            previous = self.model('device_op_gate_fusion_pin_'+{'v6':'v4', 'v7':'v5'}[version])
            with self.assertRaises(RuntimeError):
                model.load_state_dict(previous.state_dict())

    def test_both_signs_bounded_bias_and_protected_edges(self):
        for version in ('v6', 'v7'):
            gate = gate_class_pin(version)(self.cfg, self.prep, {'strength': .7, 'n_experts': 2}).eval()
            predictor = DeviceOPPredictorPin(self.cfg, self.prep, self.support)
            with torch.no_grad():
                values, support = predictor(self.batch)
                for _, head in gate.iter_layer_score_heads():
                    head[-1].weight.zero_()
                    head[-1].bias[:, 0] = 2.
                    head[-1].bias[:, 1] = -2.
                biases = gate(self.batch, values, support)
            self.assertIsNone(biases[0])
            _, _, eligible = gate._layout(self.batch, values, support)
            for bias in biases[1:]:
                self.assertTrue(bool((bias.abs() <= .7).all()))
                self.assertTrue(bool((bias[:, 0, 1:, 1:][eligible] > 0).all()))
                self.assertTrue(bool((bias[:, 1, 1:, 1:][eligible] < 0).all()))
                torch.testing.assert_close(bias[:, 0, 1:, 1:][eligible],
                    torch.full_like(bias[:, 0, 1:, 1:][eligible], .7 * math.tanh(2)))
                self.assertEqual(float(bias[:, :, 1:, 1:][(~eligible[:, None]).expand_as(bias[:, :, 1:, 1:])].abs().sum()), 0.)
                self.assertEqual(float(bias[:, :, 0].abs().sum()+bias[:, :, :, 0].abs().sum()), 0.)

    def test_only_pe2_plus_masks_change_for_every_training_strategy(self):
        capture = scope_helpers.FusionStagedGateModelsPinTests.capture_attention
        for mode in V6_GATE_MODELS_PIN + V7_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model = self.model(mode)
                before, structure, _ = capture(model, self.batch)
                with torch.no_grad():
                    for _, network in model.gate.iter_layer_score_heads():
                        network[-1].bias.fill_(.8)
                after, structure_after, _ = capture(model, self.batch)
                torch.testing.assert_close(structure, structure_after, atol=0, rtol=0)
                changed = []
                for key, old in before.items():
                    if old is None:
                        self.assertIsNone(after[key]); continue
                    self.assertTrue(torch.equal(torch.isneginf(old), torch.isneginf(after[key])))
                    if not torch.equal(old, after[key]):
                        changed.append(key)
                self.assertEqual(changed, ['encoder.fusion_blocks.1.self_attn', 'encoder.fusion_blocks.2.self_attn'])

    def test_v6_reuses_one_bias_v7_clones_equal_initial_weights_without_aliasing(self):
        torch.manual_seed(29)
        shared = self.model('device_op_gate_fusion_pin_v6')
        torch.manual_seed(29)
        independent = self.model('device_op_gate_fusion_pin_v7')
        values, support = shared.predicted_devices(self.batch)
        a = shared.gate(self.batch, values, support)
        b = independent.gate(self.batch, values, support)
        self.assertIs(a[1], a[2])
        self.assertIsNot(b[1], b[2])
        for actual, expected in zip(b[1:], a[1:]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        p0, p1 = independent.gate.score_heads
        for left, right in zip(p0.parameters(), p1.parameters()):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in independent.gate.parameters()),
                         2*sum(p.numel() for p in shared.gate.parameters()))
        with torch.no_grad():
            p0[-1].bias.add_(.6)
        changed = independent.gate(self.batch, values, support)
        self.assertFalse(torch.equal(changed[1], b[1]))
        torch.testing.assert_close(changed[2], b[2], atol=0, rtol=0)

    def test_neutral_initialization_retains_input_and_expert_gradients(self):
        for version in ('v6', 'v7'):
            gate = gate_class_pin(version)(self.cfg, self.prep, {'n_experts': 3, 'mlp_dropout': 0.})
            seen=set()
            for _, network in gate.iter_layer_score_heads():
                if id(network) in seen: continue
                seen.add(id(network))
                x=torch.randn(2,3,4,4,gate.input_dim,requires_grad=True)
                z=network(x)
                self.assertLess(float(z.detach().abs().max()), .02)
                z.tanh().square().mean().backward()
                self.assertGreater(float(x.grad.abs().sum()), 0.)
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                    and float(p.grad.abs().sum()) > 0 for p in network.parameters()))

    def test_frozen_phase_gradients_and_ungated_preparation(self):
        for mode in V6_GATE_MODELS_PIN + V7_GATE_MODELS_PIN:
            with self.subTest(mode=mode):
                model=self.model(mode)
                if mode in VERSIONED_STAGED_GATE_MODELS_PIN:
                    phase=model.phase
                    for preparation in ('supervised','distill_encoder','readout'):
                        model.set_phase(preparation).eval()
                        with patch.object(model.gate,'forward',side_effect=AssertionError('gate in base stage')):
                            model(self.batch)
                    model.set_phase(phase)
                poison=copy.deepcopy(self.batch)
                for key in ('op','owner_op','performance'):poison[key].fill_(float('nan'))
                with torch.no_grad():
                    model.eval()
                    torch.testing.assert_close(model(self.batch),model(poison),atol=0,rtol=0)
                model.train()(self.batch).square().mean().backward()
                active={'gate','encoder','decoder'}
                if mode in VERSIONED_STAGED_GATE_MODELS_PIN:
                    active={'gate','decoder'} if model.phase=='gate_decoder' else {'gate'}
                for name in ('gate','predictor','encoder','decoder'):
                    grads=[p.grad for p in getattr(model,name).parameters() if p.grad is not None]
                    if name in active:
                        self.assertTrue(grads,name)
                        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
                        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.,name)
                    else:self.assertFalse(grads,name)
                for _, network in model.gate.iter_layer_score_heads():
                    self.assertTrue(all(p.grad is not None for p in network.parameters()))

    def test_recipe_roundtrip_rejects_sharing_scope_and_transform_substitution(self):
        for mode in V6_GATE_MODELS_PIN + V7_GATE_MODELS_PIN:
            model=self.model(mode)
            recipe=describe_model_pin(model)
            restored=rebuild_model_pin(recipe,self.cfg,self.prep).eval()
            restored.load_state_dict(model.state_dict())
            with torch.no_grad():torch.testing.assert_close(model(self.batch),restored(self.batch),atol=0,rtol=0)
            for key in ('gate_normalization','gate_architecture','gate_version','attention_scope'):
                wrong=copy.deepcopy(recipe);wrong.pop(key)
                with self.assertRaises(ValueError):rebuild_model_pin(wrong,self.cfg,self.prep)
            for key,value in [('shared_across_layers',not recipe['gate_architecture']['shared_across_layers']),
                              ('gated_pe_layers',[1,2,3]),('bias_transform','sigmoid')]:
                wrong=copy.deepcopy(recipe);wrong['gate_architecture'][key]=value
                with self.assertRaises(ValueError):rebuild_model_pin(wrong,self.cfg,self.prep)
            wrong=copy.deepcopy(recipe);wrong['gate_config']['strength']=.5
            with self.assertRaises(ValueError):rebuild_model_pin(wrong,self.cfg,self.prep)

    def test_base_conversion_keeps_trained_modules_and_rejects_too_few_layers(self):
        predictor=DeviceOPPredictorPin(self.cfg,self.prep,self.support)
        base=StagedOPGateModelPin(self.cfg,self.prep,predictor,mode='device_op_distill_gate_pin').set_phase('readout').eval()
        before={key:state_digest(getattr(base,key)) for key in ('predictor','encoder','decoder')}
        for mode in ('device_op_distill_gate_fusion_pin_v6','device_op_distill_gate_fusion_pin_v7'):
            copied=copy.deepcopy(base).set_mode_pin(mode,{'n_experts':2})
            self.assertEqual(before,{key:state_digest(getattr(copied,key)) for key in before})
            self.assertNotIn('initial_gate',copied.gate_config)
            self.assertNotIn('epsilon',copied.gate_config)
            with torch.no_grad():torch.testing.assert_close(base(self.batch),copied(self.batch),atol=0,rtol=0)
        for version in ('v6','v7'):
            with self.assertRaisesRegex(ValueError,'at least two fusion'):
                gate_class_pin(version)({**self.cfg,'cross_layers':1},self.prep)
            for key in ('initial_gate','epsilon'):
                with self.assertRaisesRegex(ValueError,'Unknown'):
                    validate_gate_config_pin({key:.5},version=version)

    def test_variable_fusion_depth_and_invalid_bias_sequences(self):
        for version in ('v6','v7'):
            for depth in (2,4):
                cfg={**self.cfg,'cross_layers':depth,'encoder_layers':3+depth}
                predictor=DeviceOPPredictorPin(cfg,self.prep,self.support)
                model=StagedOPGateModelPin(cfg,self.prep,predictor,
                    mode='device_op_distill_gate_fusion_pin_'+version,
                    gate_config={'n_experts':2,'mlp_dropout':0.}).set_phase('gate_only').eval()
                values,support=model.predicted_devices(self.batch)
                biases=model.gate(self.batch,values,support)
                self.assertEqual(len(biases),depth)
                self.assertEqual(model.gate.architecture_contract['gated_pe_layers'],list(range(2,depth+1)))
                inputs=model.prepared_batch(self.batch)
                structure=model.encoder.structure_encoder(inputs)
                with self.assertRaisesRegex(ValueError,'one entry per fusion'):
                    model.encoder.forward_from_structure(structure,inputs,fusion_attention_bias=biases[1:])
                wrong=list(biases);wrong[-1]=torch.full_like(biases[-1],float('inf'))
                with self.assertRaisesRegex(ValueError,'must be finite'):
                    model.encoder.forward_from_structure(structure,inputs,fusion_attention_bias=wrong)
                wrong[-1]=biases[-1][:,:1]
                with self.assertRaisesRegex(ValueError,'must match'):
                    model.encoder.forward_from_structure(structure,inputs,fusion_attention_bias=wrong)
                with torch.no_grad():
                    model.gate.settings['strength']=0.
                    zero_output=model(self.batch)
                    model.set_phase('readout').eval()
                    # Freezing policy can select a different native attention
                    # kernel; zero bias must agree within float32 roundoff.
                    torch.testing.assert_close(zero_output,model(self.batch),atol=1e-7,rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_gate_only_backward_through_frozen_encoder(self):
        batch={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in self.batch.items()}
        for version in ('v6','v7'):
            model=self.model('device_op_distill_gate_fusion_pin_'+version).cuda().train()
            model(batch).square().mean().backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.gate.parameters()))


if __name__=='__main__':unittest.main()
