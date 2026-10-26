"""Five-mode input contracts: heterogeneous schemas and direct continuous OP."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from device_jepa.adapter import adapt_analog_agent
from device_jepa.baselines import (_collate, _make_model, collate_pin_samples,
    evaluate_baseline, fit_pin_vocabulary, train_baseline)
from device_jepa.data import (build_manifest, device_category, fit_preprocessor,
    parameter_applicability, samples_for_split, synthetic_dataset, validate_dataset)
from device_jepa.model import FrozenPerformanceModel, JEPAModel, make_decoder
from device_jepa.tensors import collate_samples
from device_jepa.training import encoder_from_metadata, load_checkpoint, seed_everything
from device_jepa.training import evaluate, pretrain, report_provenance, state_digest, train_decoder
from test_adapter import fixture
from test_baselines import tiny_options


class HeterogeneousInputsTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)
        self.temp = tempfile.TemporaryDirectory()
        fixture(Path(self.temp.name))
        self.data = adapt_analog_agent(self.temp.name)
        self.manifest = build_manifest(self.data, config={
            'train_circuits': ['Tiny'], 'ood_circuits': [], 'split_ratios': [.34, .33, .33],
            'split_seed': 17, 'finetune_label_fraction': 1.})
        self.prep = fit_preprocessor(self.data, self.manifest, True)
        self.vocab = fit_pin_vocabulary(self.data, self.manifest)
        self.row = copy.deepcopy(self.data['samples'][0])
        self.cfg = tiny_options()['model']

    def tearDown(self):
        self.temp.cleanup()

    def test_adapter_preserves_real_parameter_fields_and_subtypes(self):
        by_id = {d['device_id']: d for d in self.row['devices']}
        self.assertEqual(by_id['XN']['parameters'], {'W_m': 2e-6, 'L_m': .5e-6, 'm': 6.})
        self.assertEqual(by_id['RLOAD']['parameters'], {'R_ohm': 10000.})
        self.assertEqual(by_id['CLOAD']['parameters'], {'C_F': 1e-12})
        self.assertEqual(by_id['IBIAS']['parameters'], {'dc_current_A': 1e-6})
        self.assertEqual(by_id['VBIAS']['parameters'], {'dc_voltage_V': .7})
        categories = {name: device_category(d) for name, d in by_id.items()}
        self.assertEqual(categories, {'XN': 'active', 'XP': 'active', 'RLOAD': 'passive',
            'CLOAD': 'passive', 'IBIAS': 'source', 'VBIAS': 'source'})
        self.assertNotEqual(self.prep['device_vocab']['NMOS'], self.prep['device_vocab']['PMOS'])

    def test_device_and_pin_masks_categories_and_parent_op_agree(self):
        device = collate_samples([self.row], self.prep, include_op=True)
        pin = collate_pin_samples([self.row], self.prep, self.vocab, include_performance=False, include_op=True)
        for i, name in enumerate(pin['device_ids'][0]):
            if name.startswith('port:'):
                self.assertFalse(pin['parameter_applicability'][0, i].any())
                self.assertFalse(pin['op_target_mask'][0, i])
                self.assertEqual(pin['device_categories'][0, i], self.prep['device_category_vocab']['port'])
                continue
            parent = name.split(':')[1]
            j = device['device_ids'][0].index(parent)
            for key in ('parameters', 'parameter_mask', 'parameter_applicability',
                        'device_categories', 'op', 'op_mask', 'op_target_mask'):
                torch.testing.assert_close(pin[key][0, i], device[key][0, j])
        for j, name in enumerate(device['device_ids'][0]):
            if name not in ('XN', 'XP'):
                self.assertFalse(device['op_target_mask'][0, j])
                self.assertFalse(device['op_mask'][0, j].any())
                self.assertFalse(device['op'][0, j].any())

    def test_zero_missing_not_applicable_and_padding_are_distinct(self):
        row = copy.deepcopy(self.row)
        for d in row['devices']:
            if d['device_id'] == 'XN':
                d['parameters'].pop('W_m')
            if d['device_id'] == 'IBIAS':
                d['parameters']['dc_current_A'] = 0.
        batch = collate_samples([row, {**row, 'devices': row['devices'][:1]}], self.prep, include_op=False)
        names = self.prep['feature_spec']['parameter_names']
        x, r, i = [batch['device_ids'][0].index(k) for k in ('XN', 'RLOAD', 'IBIAS')]
        w, current = names.index('W_m'), names.index('dc_current_A')
        self.assertTrue(batch['parameter_applicability'][0, x, w])
        self.assertFalse(batch['parameter_mask'][0, x, w])
        self.assertFalse(batch['parameter_applicability'][0, r, w])
        self.assertTrue(batch['parameter_applicability'][0, i, current])
        self.assertTrue(batch['parameter_mask'][0, i, current])
        self.assertFalse(batch['parameter_applicability'][1, 1:].any())
        self.assertFalse(batch['device_mask'][1, 1:].any())
        encoder = encoder_from_metadata(self.cfg, self.prep).eval()
        tokens, live = encoder.input_tokens(batch)
        self.assertTrue(live[0, x, w])
        self.assertFalse(live[0, r, w])
        altered = copy.deepcopy(batch)
        altered['parameter_mask'][0, x, w] = True
        new_tokens, _ = encoder.input_tokens(altered)
        self.assertGreater(float((tokens[0, x, w] - new_tokens[0, x, w]).abs().max().detach()), 0.)
        not_applicable = copy.deepcopy(batch)
        not_applicable['parameter_applicability'][0, x, w] = False
        self.assertFalse(encoder.input_tokens(not_applicable)[1][0, x, w])
        self.assertGreater(float((encoder(batch)[0] - encoder(not_applicable)[0]).abs().max().detach()), 1e-7)

    def test_coarse_category_is_consumed_but_subtype_retained(self):
        batch = collate_samples([self.row], self.prep, include_op=False)
        model = encoder_from_metadata(self.cfg, self.prep).eval()
        changed = copy.deepcopy(batch)
        changed['device_categories'][0, 0] = self.prep['device_category_vocab']['unknown']
        self.assertTrue(torch.equal(batch['device_types'], changed['device_types']))
        self.assertGreater(float((model(batch)[0] - model(changed)[0]).abs().max().detach()), 1e-7)

    def test_custom_applicability_and_legacy_null_padding(self):
        self.assertEqual(parameter_applicability({'device_type': 'resistor', 'parameters': {'W': None, 'R': 0}}, ['W', 'R']), [False, True])
        unknown = {'device_type': 'custom', 'parameters': {'q': None}, 'parameter_applicability': {'r': True}}
        self.assertEqual(parameter_applicability(unknown, ['q', 'r', 's']), [True, True, False])
        self.assertEqual(device_category(unknown), 'unknown')
        bad = copy.deepcopy(self.data)
        bad['samples'][0]['devices'][0]['parameter_applicability'] = {'dc_current_A': 'yes'}
        with self.assertRaisesRegex(ValueError, 'applicability'):
            validate_dataset(bad)


class DirectOPInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(17)
        cls.data = synthetic_dataset(6)
        cls.options = tiny_options(.5)
        cls.options['training']['decoder_epochs'] = 1
        cls.manifest = build_manifest(cls.data, config=cls.options['dataset'])
        cls.prep = fit_preprocessor(cls.data, cls.manifest, True, cls.options['normalization'])
        cls.rows = samples_for_split(cls.data, cls.manifest, 'train')[:2]
        cls.vocab = fit_pin_vocabulary(cls.data, cls.manifest)

    def batch_and_model(self, mode):
        cfg = copy.deepcopy(self.options['model'])
        if 'pin' in mode:
            cfg['terminal_bias'] = False
        return (_collate(self.rows, self.prep, mode, cfg, self.vocab),
                _make_model(mode, cfg, self.prep, self.vocab).eval())

    def test_op_values_and_masks_affect_both_new_predictions(self):
        for mode in ('zerosim_pin_op', 'zerosim_device_op'):
            with self.subTest(mode=mode):
                batch, model = self.batch_and_model(mode)
                original = model(batch)
                changed = copy.deepcopy(batch)
                changed['op'][changed['op_mask']] += 7.
                self.assertGreater(float((original-model(changed)).detach().abs().max()), 1e-7)
                changed = copy.deepcopy(batch)
                changed['op_mask'].zero_()
                self.assertGreater(float((original-model(changed)).detach().abs().max()), 1e-7)
                loss = original.square().sum()
                loss.backward()
                self.assertGreater(float(model.encoder.op_value.weight.grad.abs().sum()), 0.)
                self.assertGreater(float(model.encoder.structure_encoder.category_embedding.weight.grad.abs().sum()), 0.)

    def test_original_baselines_and_frozen_jepa_ignore_op(self):
        for mode in ('zerosim_pin', 'zerosim_device'):
            batch, model = self.batch_and_model(mode)
            changed = copy.deepcopy(batch)
            changed.update(op=torch.full((1,), float('nan')), op_mask=object(), op_state=object(), op_target_mask=object())
            torch.testing.assert_close(model(batch), model(changed), rtol=0, atol=0)
        batch = collate_samples(self.rows, self.prep, include_op=True)
        encoder = encoder_from_metadata(self.options['model'], self.prep)
        model = FrozenPerformanceModel(encoder, make_decoder(encoder.cfg, 2)).eval()
        original = model(batch)
        batch['op'].fill_(float('nan'))
        batch['op_mask'].zero_()
        batch['op_state'] = object()
        torch.testing.assert_close(original, model(batch), rtol=0, atol=0)

    def test_missing_op_keys_raise_and_masked_values_never_poison(self):
        for mode in ('zerosim_pin_op', 'zerosim_device_op'):
            batch, model = self.batch_and_model(mode)
            original = model(batch)
            poison = copy.deepcopy(batch)
            poison['op'][~poison['op_mask']] = float('nan')
            poison['parameters'][~poison['parameter_mask']] = float('nan')
            torch.testing.assert_close(original, model(poison), rtol=0, atol=0)
            del batch['op']
            with self.assertRaisesRegex(ValueError, 'requires DC OP'):
                model(batch)

    def test_no_states_or_noneligible_op_are_consulted(self):
        prep = copy.deepcopy(self.prep)
        prep['op_state_features'] = ['would_raise_if_read']
        rows = copy.deepcopy(self.rows)
        for row in rows:
            for d in row['devices']:
                d['op_state'] = {'would_raise_if_read': object()}
                if d['device_type'] == 'resistor':
                    d['op'] = {'id': float('nan')}
        for mode in ('zerosim_pin_op', 'zerosim_device_op'):
            batch = _collate(rows, prep, mode, self.options['model'], self.vocab)
            self.assertNotIn('op_state', batch)
            self.assertFalse(batch['op_mask'][~batch['op_target_mask']].any())
            self.assertTrue(torch.isfinite(batch['op']).all())

    def test_train_only_op_scaler_ignores_held_out_extremes(self):
        changed = copy.deepcopy(self.data)
        for row in changed['samples']:
            if row['sample_id'] not in self.manifest['train']:
                for d in row['devices']:
                    d['op'] = {k: 1e30 for k in changed['feature_spec']['op_names']}
        manifest = build_manifest(changed, config=self.options['dataset'])
        self.assertEqual(manifest['train'], self.manifest['train'])
        new = fit_preprocessor(changed, manifest, True, self.options['normalization'])
        self.assertEqual(new['op'], self.prep['op'])
        self.assertEqual(new['fit_ids']['op'], self.manifest['train'])

    def test_jepa_design_branch_keeps_single_shared_no_predictor_architecture(self):
        encoder = encoder_from_metadata(self.options['model'], self.prep)
        model = JEPAModel(encoder, len(self.prep['feature_spec']['op_names']))
        self.assertIs(model.op_encoder.structure_encoder, encoder.structure_encoder)
        self.assertFalse(any('predictor' in n or 'teacher' in n for n, _ in model.named_parameters()))
        self.assertFalse(hasattr(encoder, 'op_value'))

    def test_report_state_config_comes_from_checkpoint_only_for_jepa(self):
        config = {'features': ['current_pass'], 'mode': 'collector_checked',
                  'thresholds': {'min_signal_current_a': 1e-12}}
        source = {'preprocessing': self.prep, 'manifest': self.manifest,
                  'mode': 'jepa_zerosim', 'model_config': self.options['model'],
                  'options': {**self.options, 'op_state': config},
                  'partition_fingerprint': 'partition', 'input_op_fingerprint': 'input'}
        info = report_provenance(source, self.data, self.rows)['op_information']
        self.assertEqual(info['state_config'], config)
        info['state_config']['thresholds']['min_signal_current_a'] = 99.
        self.assertEqual(source['options']['op_state'], config)
        for mode in ('zerosim_device', 'zerosim_pin', 'zerosim_device_op', 'zerosim_pin_op'):
            source['mode'] = mode
            self.assertEqual(report_provenance(source, self.data, self.rows)['op_information']['state_config'], {})

    def test_direct_op_end_to_end_reload_budget_metadata_and_fair_scalers(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoints = {}
            for mode in ('zerosim_pin_op', 'zerosim_device_op'):
                path = train_baseline(self.data, self.manifest, self.options, Path(folder) / mode, mode)
                cp = load_checkpoint(path)
                checkpoints[mode] = cp
                self.assertTrue(cp['requires_op_at_inference'])
                self.assertTrue(cp['op_inputs_used'])
                self.assertEqual(cp['input_contract_version'], 3)
                self.assertEqual(cp['preprocessing'], self.prep)
                self.assertEqual(cp['manifest'], self.manifest)
                self.assertNotEqual(cp['encoder_digest'], cp['initial_encoder_digest'])
                self.assertNotEqual(cp['decoder_digest'], cp['initial_decoder_digest'])
                report = evaluate_baseline(self.data, path)
                self.assertTrue(report['requires_op_at_inference'])
                self.assertEqual(report['op_information']['training_use'], 'input')
                self.assertEqual(report['op_information']['state_features'], [])
                self.assertIsNone(report['op_information']['acquisition_cost_seconds'])
                self.assertEqual(report['op_information']['acquisition_cost_status'], 'not_measured')
                self.assertEqual(report['evaluation_ids'], self.manifest['test'])
                self.assertEqual(report, evaluate_baseline(self.data, path))
                cp['requires_op_at_inference'] = False
                invalid = Path(folder) / 'invalid.pt'
                torch.save(cp, invalid)
                with self.assertRaisesRegex(ValueError, 'OP requirement'):
                    load_checkpoint(invalid)
            self.assertEqual(checkpoints['zerosim_pin_op']['preprocessing']['parameters'],
                             checkpoints['zerosim_device_op']['preprocessing']['parameters'])

    def test_legacy_input_metadata_reconstructs_original_architecture(self):
        prep = copy.deepcopy(self.prep)
        prep.pop('input_contract_version')
        prep.pop('device_category_vocab')
        prep.pop('model_vocab')
        prep.pop('port_role_vocab')
        prep['fit_ids'].pop('model_vocabulary')
        prep['fit_ids'].pop('port_role_vocabulary')
        old = encoder_from_metadata(self.options['model'], prep)
        self.assertEqual(old.parameter_value.in_features, 1)
        self.assertIsNone(old.structure_encoder.category_embedding)
        fresh = encoder_from_metadata(self.options['model'], self.prep)
        self.assertEqual(fresh.parameter_value.in_features, 2)
        self.assertIsNotNone(fresh.structure_encoder.category_embedding)

    def test_legacy_stage_one_can_train_decoder_without_contract_mislabel(self):
        options = copy.deepcopy(self.options)
        options['op_state'] = {'features': []}
        with tempfile.TemporaryDirectory() as folder:
            path = pretrain(self.data, self.manifest, options, folder)
            source = load_checkpoint(path)
            source.pop('input_contract_version')
            source['preprocessing'].pop('input_contract_version')
            source['preprocessing'].pop('device_category_vocab')
            source['preprocessing'].pop('model_vocab')
            source['preprocessing'].pop('port_role_vocab')
            source['preprocessing']['fit_ids'].pop('model_vocabulary')
            source['preprocessing']['fit_ids'].pop('port_role_vocabulary')
            legacy_encoder = encoder_from_metadata(options['model'], source['preprocessing'])
            source['encoder_state'] = legacy_encoder.state_dict()
            source['encoder_digest'] = state_digest(legacy_encoder)
            legacy_path = Path(folder) / 'legacy_pretrain.pt'
            torch.save(source, legacy_path)
            decoded = train_decoder(self.data, legacy_path, options, Path(folder) / 'decoder')
            checkpoint = load_checkpoint(decoded)
            self.assertEqual(checkpoint['input_contract_version'], 1)
            self.assertNotIn('input_contract_version', checkpoint['preprocessing'])
            self.assertEqual(checkpoint['encoder_digest'], source['encoder_digest'])
            self.assertFalse(evaluate(self.data, decoded)['requires_op_at_inference'])

    def test_all_train_op_missing_fails_only_new_baselines(self):
        data = copy.deepcopy(self.data)
        for row in data['samples']:
            for d in row['devices']:
                d['op'] = {}
        manifest = build_manifest(data, config=self.options['dataset'])
        with tempfile.TemporaryDirectory() as folder:
            for mode in ('zerosim_pin_op', 'zerosim_device_op'):
                with self.assertRaisesRegex(ValueError, 'observed eligible ID-train DC OP'):
                    train_baseline(data, manifest, self.options, folder, mode)


if __name__ == '__main__':
    unittest.main()
