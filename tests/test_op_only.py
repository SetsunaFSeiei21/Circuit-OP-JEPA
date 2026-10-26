"""Topology + DC OP ablations must never consume design values or masks."""
import copy
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from device_jepa.adapter import adapt_analog_agent
from device_jepa.baselines import (_collate, _make_model, evaluate_baseline,
                                   fit_pin_vocabulary, train_baseline)
from device_jepa.comparison import compare_reports
from device_jepa.data import build_manifest, fit_preprocessor, synthetic_dataset
from device_jepa.methods import DEFAULT_MODELS, OP_ONLY_MODES
from device_jepa.resident import ResidentDataset
from device_jepa.training import load_checkpoint, seed_everything
from test_adapter import fixture
from test_baselines import tiny_options
from test_comparison import report


DESIGN_KEYS = {'parameters', 'parameter_mask', 'parameter_applicability'}


class OPOnlyInputsTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.data = adapt_analog_agent(self.root)
        self.options = tiny_options()
        self.options['model'].update(dim=8, heads=2, ffn_dim=16, decoder_ffn_dim=16, dropout=0.)
        self.options['dataset'].update(train_circuits=['Tiny'], ood_circuits=[], split_ratios=[.34, .33, .33])
        self.manifest = build_manifest(self.data, config=self.options['dataset'])
        self.prep = fit_preprocessor(self.data, self.manifest, True)
        self.vocab = fit_pin_vocabulary(self.data, self.manifest)
        self.rows = self.data['samples'][:2]

    def make(self, mode, device='cpu'):
        cfg = copy.deepcopy(self.options['model'])
        if 'pin' in mode:
            cfg['terminal_bias'] = False
        model = _make_model(mode, cfg, self.prep, self.vocab).to(device).eval()
        return cfg, model

    def test_corresponding_backbone_and_initial_weights_are_identical(self):
        for mode in OP_ONLY_MODES:
            seed_everything(17)
            _, full = self.make(mode.removesuffix('_only'))
            seed_everything(17)
            _, only = self.make(mode)
            old, new = full.state_dict(), only.state_dict()
            self.assertEqual(set(old) - set(new), {'encoder.parameter_names.weight',
                'encoder.parameter_value.weight', 'encoder.parameter_value.bias'})
            self.assertFalse(set(new) - set(old))
            for name in new:
                torch.testing.assert_close(new[name], old[name], rtol=0, atol=0)
            self.assertEqual(only.encoder.cfg, full.encoder.cfg)
            self.assertFalse(hasattr(only.encoder, 'parameter_value'))
            self.assertGreater(sum(p.numel() for p in full.parameters()), sum(p.numel() for p in only.parameters()))

    def test_scalar_and_resident_batches_never_read_unused_design_values(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            for device in row['devices']:
                device['parameters'] = {name: object() for name in self.prep['feature_spec']['parameter_names']}
                device['parameter_applicability'] = object()
        for mode in OP_ONLY_MODES:
            cfg, model = self.make(mode)
            expected = _collate(self.rows, self.prep, mode, cfg, self.vocab)
            altered = _collate(changed, self.prep, mode, cfg, self.vocab)
            self.assertFalse(DESIGN_KEYS & expected.keys())
            for key in expected:
                if torch.is_tensor(expected[key]):
                    torch.testing.assert_close(expected[key], altered[key], rtol=0, atol=0)
            cache = ResidentDataset(changed, self.prep, 'cpu', include_op=True,
                include_performance=True, include_parameters=False, verbose=False,
                pin_vocab=self.vocab if 'pin' in mode else None, max_tokens=cfg['max_devices'])
            self.assertFalse(DESIGN_KEYS & cache.tables.keys())
            _, resident = next(cache.batches(2))
            self.assertFalse(DESIGN_KEYS & resident.keys())
            torch.testing.assert_close(model(expected), model(resident), rtol=1e-5, atol=1e-6)

    def test_predictions_ignore_design_values_masks_and_states_but_use_op_and_topology(self):
        for mode in OP_ONLY_MODES:
            cfg, model = self.make(mode)
            batch = _collate(self.rows, self.prep, mode, cfg, self.vocab)
            initial = model(batch)
            poison = copy.deepcopy(batch)
            poison.update({key: object() for key in DESIGN_KEYS})
            poison.update(op_state=object(), op_state_mask=object())
            torch.testing.assert_close(initial, model(poison), rtol=0, atol=0)
            for key in ('op', 'op_mask', 'adjacency'):
                changed = copy.deepcopy(batch)
                if key == 'op':
                    changed[key][changed['op_mask']] += 7
                elif key == 'op_mask':
                    changed[key].zero_()
                else:
                    changed[key].fill_(True)
                self.assertGreater(float((initial - model(changed)).detach().abs().max()), 1e-7, (mode, key))
            initial.square().sum().backward()
            self.assertGreater(float(model.encoder.op_value.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(model.encoder.structure_encoder.category_embedding.weight.grad.abs().sum()), 0.)
            self.assertTrue(all('parameter_value' not in n and 'parameter_names' not in n for n, _ in model.named_parameters()))
            del batch['op']
            with self.assertRaisesRegex(ValueError, 'requires DC OP'):
                model(batch)

    def test_original_op_models_still_use_design_inputs(self):
        for mode in OP_ONLY_MODES:
            original = mode.removesuffix('_only')
            cfg, model = self.make(original)
            batch = _collate(self.rows, self.prep, original, cfg, self.vocab)
            changed = copy.deepcopy(batch)
            changed['parameters'][changed['parameter_mask']] += 5
            self.assertGreater(float((model(batch) - model(changed)).detach().abs().max()), 1e-7)

    def test_explicit_geometry_op_fields_fail_before_output(self):
        for mode in OP_ONLY_MODES:
            options = copy.deepcopy(self.options)
            options['normalization']['op_fields'] = self.data['feature_spec']['op_names']
            destination = self.root / mode
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'geometry'):
                train_baseline(self.data, self.manifest, options, destination, mode)
            self.assertFalse(destination.exists())
        self.assertFalse(set(OP_ONLY_MODES) & set(DEFAULT_MODELS))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware required')
    def test_cuda_op_only_resident_forward_backward(self):
        for mode in OP_ONLY_MODES:
            cfg, model = self.make(mode, 'cuda')
            cache = ResidentDataset(self.rows, self.prep, 'cuda', include_op=True,
                include_performance=True, include_parameters=False, verbose=False,
                pin_vocab=self.vocab if 'pin' in mode else None, max_tokens=cfg['max_devices'])
            _, batch = next(cache.batches(2))
            self.assertFalse(DESIGN_KEYS & batch.keys())
            model(batch).square().sum().backward()
            self.assertTrue(torch.isfinite(model.encoder.op_value.weight.grad).all())


class OPOnlyProvenanceTests(unittest.TestCase):
    def test_design_ablation_requires_matched_backbone_configuration(self):
        for mode in OP_ONLY_MODES:
            original = mode.removesuffix('_only')
            reports = [report(original), report(mode)]
            for item in reports:
                item['provenance']['model_config'].update(max_devices=256, terminal_bias=False)
            self.assertTrue(compare_reports(reports, expected_modes=[original, mode])['complete'])
            # These settings may differ between pin/device families, but must
            # not differ within a corresponding design-input ablation.
            for key, value in [('max_devices', 512), ('terminal_bias', True)]:
                broken = copy.deepcopy(reports)
                broken[1]['provenance']['model_config'][key] = value
                with self.assertRaisesRegex(ValueError, 'design-input ablation'):
                    compare_reports(broken, expected_modes=[original, mode])

    def test_reports_cannot_hide_design_inputs_or_geometry(self):
        for mode in OP_ONLY_MODES:
            good = report(mode)
            result = compare_reports([good], expected_modes=[mode])
            self.assertTrue(result['complete'])
            self.assertFalse(result['run_costs'][0]['uses_design_parameters'])
            for value in (True, None, 0):
                bad = copy.deepcopy(good)
                if value is None:
                    del bad['provenance']['uses_design_parameters']
                else:
                    bad['provenance']['uses_design_parameters'] = value
                with self.assertRaisesRegex(ValueError, 'uses_design_parameters'):
                    compare_reports([bad], expected_modes=[mode])
            bad = copy.deepcopy(good)
            bad['provenance']['op_information']['feature_names'].append('w_um')
            with self.assertRaisesRegex(ValueError, 'geometry'):
                compare_reports([bad], expected_modes=[mode])

    def test_checkpoints_reload_with_explicit_contract_and_reject_tampering(self):
        data = synthetic_dataset(6)
        options = tiny_options()
        options['training']['baseline_epochs'] = 2
        manifest = build_manifest(data, config=options['dataset'])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for mode in OP_ONLY_MODES:
                path = train_baseline(data, manifest, options, Path(tmp) / mode, mode)
                checkpoint = load_checkpoint(path)
                self.assertIs(checkpoint['uses_design_parameters'], False)
                evaluation = evaluate_baseline(data, path)
                self.assertIs(evaluation['provenance']['uses_design_parameters'], False)
                self.assertTrue(evaluation['requires_op_at_inference'])
                self.assertTrue(compare_reports([evaluation], expected_modes=[mode])['complete'])
                for value in (True, None):
                    broken = copy.deepcopy(checkpoint)
                    if value is None:
                        del broken['uses_design_parameters']
                    else:
                        broken['uses_design_parameters'] = value
                    target = Path(tmp) / 'broken.pt'
                    torch.save(broken, target)
                    with self.assertRaisesRegex(ValueError, 'uses_design_parameters'):
                        load_checkpoint(target)


if __name__ == '__main__':
    unittest.main()
