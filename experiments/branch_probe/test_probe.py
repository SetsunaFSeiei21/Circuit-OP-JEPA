"""CPU regression checks for matched frozen-branch readouts.

Run from the repository root:
  python -m unittest discover -s experiments/branch_probe -p 'test_*.py'
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch

from device_jepa.data import build_manifest, is_op_target, synthetic_dataset
from device_jepa.states import apply_op_states
from device_jepa.tensors import collate_samples
from device_jepa.training import load_checkpoint, pretrain, state_digest
from experiments.branch_probe import probe


ROOT = Path(__file__).resolve().parents[2]
MODES = ('jepa_zerosim', 'jepa_zerosim_ema')


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BranchProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.options = json.loads((ROOT / 'configs/smoke.json').read_text())
        cls.options['model'].update(dim=16, heads=4, ffn_dim=32, decoder_ffn_dim=32, dropout=.2)
        cls.options['training'].update(pretrain_epochs=1, decoder_epochs=2, batch_size=8)
        data = synthetic_dataset(6, 17)
        original = build_manifest(data, config=cls.options['dataset'])
        # Missing OP records stay in BOTH branches and all evaluation cohorts.
        cls.missing_ids = {original[split][0] for split in ('train', 'val', 'test', 'ood')}
        for row in data['samples']:
            if row['sample_id'] in cls.missing_ids:
                for device in row['devices']:
                    device['op'] = {key: None for key in device['op']}
        cls.data = apply_op_states(data, cls.options['op_state'])
        cls.manifest = build_manifest(cls.data, config=cls.options['dataset'])
        cls.paths, cls.contexts, cls.reports, cls.outputs = {}, {}, {}, {}
        for mode in MODES:
            cls.paths[mode] = quiet(pretrain, cls.data, cls.manifest, cls.options,
                                    cls.root / ('source_' + mode), mode=mode)
            cls.contexts[mode] = quiet(probe.prepare_probe, cls.data, cls.paths[mode])
            cls.outputs[mode] = cls.root / ('probe_' + mode)
            cls.reports[mode] = quiet(probe.run_pair, cls.contexts[mode], cls.outputs[mode],
                                      seed=29, device='cpu', eval_splits=('val', 'test', 'ood'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_both_modes_produce_matching_pairs_and_restore_source(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                source = load_checkpoint(self.paths[mode])
                model = probe.reconstruct_model(source).eval()
                self.assertEqual(state_digest(model.encoder), source['encoder_digest'])
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, source['model_state'][key], rtol=0, atol=0)
                reports = self.reports[mode]
                self.assertEqual(len(reports), 6)
                self.assertEqual({item['branch'] for item in reports}, {'design', 'op'})
                self.assertEqual(len({item['decoder_initial_digest'] for item in reports}), 1)
                self.assertEqual(len({item['target_scaler_digest'] for item in reports}), 1)
                self.assertEqual(len({item['training_cost']['decoder_optimizer_steps'] for item in reports}), 1)
                for split in ('val', 'test', 'ood'):
                    pair = [item for item in reports if item['split'] == split]
                    self.assertEqual(len(pair), 2)
                    self.assertEqual(set(pair[0]['evaluation_ids']), set(self.manifest[split]))
                    self.assertEqual(pair[0]['evaluation_ids'], pair[1]['evaluation_ids'])
                    for item in pair:
                        self.assertEqual(item['source_mode'], mode)
                        self.assertEqual(item['decoder_seed'], 29)
                        self.assertTrue(item['encoder_frozen_verified'])
                        self.assertEqual(item['requires_op_at_inference'], item['branch'] == 'op')
                        self.assertEqual(item['uses_design_parameters'], item['branch'] == 'design')
                        self.assertEqual(item['coverage']['excluded_samples'], 0)
                        self.assertGreaterEqual(item['coverage']['all_op_missing_samples'], 1)

    def test_branch_inputs_are_separated_and_masks_do_not_use_observations(self):
        rows = [self.data['samples'][i] for i in (0, 1, 2)]
        for mode in MODES:
            with self.subTest(mode=mode):
                context = self.contexts[mode]
                model = probe.reconstruct_model(context['source']).eval().requires_grad_(False)
                batch = collate_samples(rows, context['prep'], include_op=True)
                eligible = torch.zeros_like(batch['device_mask'])
                for i, row in enumerate(rows):
                    eligible[i, :len(row['devices'])] = torch.tensor([is_op_target(d) for d in row['devices']])
                design = {key: value for key, value in batch.items() if not key.startswith('op')}
                op = {key: value for key, value in batch.items() if not key.startswith('parameter')}
                with torch.no_grad():
                    a, apad = probe.branch_memory(model, batch, 'design', eligible)
                    b, bpad = probe.branch_memory(model, batch, 'op', eligible)
                    no_op, _ = probe.branch_memory(model, design, 'design', eligible)
                    no_design, _ = probe.branch_memory(model, op, 'op', eligible)
                    missing = copy.deepcopy(batch)
                    missing['op'].fill_(float('nan'))
                    missing['op_mask'].zero_()
                    missing_op, missing_padding = probe.branch_memory(model, missing, 'op', eligible)
                torch.testing.assert_close(a, no_op, rtol=0, atol=0)
                torch.testing.assert_close(b, no_design, rtol=0, atol=0)
                torch.testing.assert_close(apad, bpad, rtol=0, atol=0)
                torch.testing.assert_close(bpad, missing_padding, rtol=0, atol=0)
                torch.testing.assert_close(apad, ~eligible, rtol=0, atol=0)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.shape[1], batch['device_mask'].shape[1])  # No graph token.
                self.assertTrue(torch.isfinite(missing_op).all())

    def test_performance_scalers_fit_only_labeled_train_and_source_remains_unchanged(self):
        mode = MODES[0]
        context = self.contexts[mode]
        before = copy.deepcopy(context['source']['preprocessing'])
        changed = copy.deepcopy(self.data)
        held_out = set(self.manifest['val'] + self.manifest['test'] + self.manifest['ood'])
        for row in changed['samples']:
            if row['sample_id'] in held_out:
                row['performance'] = {name: 1e30 for name in row['performance']}
        changed_context = quiet(probe.prepare_probe, changed, self.paths[mode])
        a = quiet(probe.fit_probe_preprocessor, context, 'cpu')
        b = quiet(probe.fit_probe_preprocessor, changed_context, 'cpu')
        self.assertEqual(a['performance'], b['performance'])
        self.assertEqual(a['performance_fitted_sample_ids'], sorted(self.manifest['labeled_train']))
        self.assertEqual(a['parameters'], before['parameters'])
        self.assertEqual(a['op'], before['op'])
        self.assertEqual(context['source']['preprocessing'], before)
        self.assertNotIn('performance', before)

    def test_reloaded_decoders_reproduce_predictions(self):
        for mode in MODES:
            for branch in ('design', 'op'):
                with self.subTest(mode=mode, branch=branch):
                    old = next(r for r in self.reports[mode] if r['branch'] == branch and r['split'] == 'test')
                    path = self.outputs[mode] / branch / 'decoder_best.pt'
                    actual = quiet(probe.evaluate_probe, self.contexts[mode], path, split='test', device='cpu')
                    self.assertEqual(actual['evaluation_ids'], old['evaluation_ids'])
                    self.assertEqual(actual['predictions'], old['predictions'])
                    self.assertEqual(actual['metrics'], old['metrics'])

    def test_checkpoint_replay_uses_saved_scalers_without_refitting(self):
        mode = MODES[0]
        context = quiet(probe.prepare_probe, self.data, self.paths[mode])
        path = self.outputs[mode] / 'design/decoder_best.pt'
        old = next(r for r in self.reports[mode] if r['branch'] == 'design' and r['split'] == 'test')
        with patch.object(probe, '_statistics', side_effect=AssertionError('CPU refit during replay')), \
                patch('device_jepa.resident.device_statistics', side_effect=AssertionError('GPU refit during replay')):
            actual = quiet(probe.evaluate_probe, context, path, split='test', device='cpu')
        self.assertEqual(actual['predictions'], old['predictions'])
        checkpoint = torch.load(path, weights_only=True)
        checkpoint['preprocessing']['performance']['mean'][0] += 1.0
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder, 'tampered.pt')
            torch.save(checkpoint, bad)
            with self.assertRaises(ValueError):
                quiet(probe.evaluate_probe, context, bad, split='test', device='cpu')

    def test_equal_features_receive_identical_optimizer_noise_and_updates(self):
        context = self.contexts[MODES[0]]
        captured = []
        reconstruct = probe.reconstruct_model

        def record_model(source):
            model = reconstruct(source)
            captured.append(model)
            return model

        def constant_features(model, batch, branch, eligible):
            dim = model.encoder.cfg.dim
            memory = torch.arange(dim, device=eligible.device, dtype=torch.float32).view(1, 1, dim)
            return memory.expand(*eligible.shape, dim).clone() / dim, ~eligible

        before = file_hash(self.paths[MODES[0]])
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(probe, 'branch_memory', constant_features), \
                patch.object(probe, 'reconstruct_model', record_model):
            quiet(probe.run_pair, context, folder, seed=43, device='cpu',
                  training_overrides={'decoder_epochs': 2, 'batch_size': 4})
            a = torch.load(Path(folder, 'design/decoder_best.pt'), weights_only=True)
            b = torch.load(Path(folder, 'op/decoder_best.pt'), weights_only=True)
            for key in a['decoder_state']:
                torch.testing.assert_close(a['decoder_state'][key], b['decoder_state'][key], rtol=0, atol=0)
            self.assertTrue(captured)
            for model in captured:
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                self.assertEqual(state_digest(model), context['source_model_digest'])
        self.assertEqual(file_hash(self.paths[MODES[0]]), before)

    def test_existing_output_is_rejected_without_changes(self):
        output = self.outputs[MODES[0]]
        files = {p.relative_to(output): file_hash(p) for p in output.rglob('*') if p.is_file()}
        with self.assertRaises(ValueError):
            quiet(probe.run_pair, self.contexts[MODES[0]], output, seed=17, device='cpu')
        self.assertEqual(files, {p.relative_to(output): file_hash(p) for p in output.rglob('*') if p.is_file()})

    def test_invalid_probe_configuration_fails_before_creating_output(self):
        for overrides in ({'decoder_epochs': 0}, {'batch_size': 0}, {'decoder_lr': -1.}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as parent:
                output = Path(parent, 'new')
                with self.assertRaises(ValueError):
                    quiet(probe.run_pair, self.contexts[MODES[0]], output, seed=17,
                          training_overrides=overrides, device='cpu')
                self.assertFalse(output.exists())

    def test_inactive_extreme_validation_labels_do_not_enter_training_normalization(self):
        changed = copy.deepcopy(self.data)
        train = set(self.manifest['labeled_train'])
        for row in changed['samples']:
            row['performance']['power'] = None if row['sample_id'] in train else 1e100
        context = quiet(probe.prepare_probe, changed, self.paths[MODES[0]])
        with tempfile.TemporaryDirectory() as folder:
            reports = quiet(probe.run_pair, context, folder, seed=29, device='cpu',
                            training_overrides={'decoder_epochs': 1})
        self.assertEqual(len(reports), 2)
        self.assertEqual(context['prep'], self.contexts[MODES[0]]['prep'])

    def test_changed_immutable_design_or_op_is_rejected(self):
        for field in ('parameters', 'op'):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.data)
                item = changed['samples'][0]['devices'][0][field]
                key = next(iter(item))
                item[key] = 123.0
                with self.assertRaisesRegex(ValueError, 'immutable|fingerprint|provenance'):
                    quiet(probe.prepare_probe, changed, self.paths[MODES[0]])

    def test_missing_op_branch_or_inconsistent_export_cannot_reconstruct(self):
        for mode in MODES:
            for mutation in ('missing_branch', 'exported_encoder', 'wrong_stage'):
                with self.subTest(mode=mode, mutation=mutation):
                    source = load_checkpoint(self.paths[mode])
                    if mutation == 'missing_branch':
                        source['model_state'] = {key: value for key, value in source['model_state'].items()
                                                 if not key.startswith('op_encoder.')}
                    elif mutation == 'exported_encoder':
                        key = next(iter(source['encoder_state']))
                        source['encoder_state'][key] = source['encoder_state'][key].clone() + 1
                    else:
                        source['stage'] = 'decoder'
                    with self.assertRaises((ValueError, RuntimeError)):
                        probe.reconstruct_model(source)

    def test_zero_eligible_sample_fails_explicitly_instead_of_silent_filtering(self):
        data = copy.deepcopy(self.data)
        sid = self.manifest['ood'][0]
        for row in data['samples']:
            if row['sample_id'] == sid:
                for dev in row['devices']:
                    dev['op_target'] = False
        data = apply_op_states(data, self.options['op_state'])
        manifest = build_manifest(data, config=self.options['dataset'])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = quiet(pretrain, data, manifest, self.options, folder)
            with self.assertRaisesRegex(ValueError, 'eligible'):
                quiet(probe.prepare_probe, data, checkpoint)

    def test_summary_and_excel_keep_decoder_seed_scope_and_pair_identity(self):
        from experiments.branch_probe.summarize import build_summary, summarize_reports
        reports = self.reports[MODES[0]]
        result = build_summary(reports)
        self.assertEqual({row['num_seeds'] for row in result['summary']}, {1})
        self.assertTrue(all(row['mae_std'] is None for row in result['summary']))
        with tempfile.TemporaryDirectory() as folder:
            summarize_reports(reports, folder)
            self.assertTrue(Path(folder, 'summary.json').is_file())
            self.assertTrue(Path(folder, 'summary.csv').is_file())
            self.assertTrue(Path(folder, 'paired_deltas.csv').is_file())
            with ZipFile(Path(folder, 'model_comparison.xlsx')) as workbook:
                self.assertIsNone(workbook.testzip())
                strings = '\n'.join(workbook.read(name).decode() for name in workbook.namelist()
                                    if name.endswith('.xml'))
                self.assertIn('decoder', strings.lower())
                self.assertIn('fixed', strings.lower())

    def test_summary_rejects_mixed_sources_unpaired_or_mismatched_heads(self):
        from experiments.branch_probe.summarize import build_summary
        for field, value in (('source_checkpoint_sha256', 'different'),
                             ('data_fingerprint', 'different_labels'),
                             ('decoder_initial_digest', 'different'),
                             ('target_scaler_digest', 'different'),
                             ('readout_policy', 'all_device_and_graph_tokens')):
            with self.subTest(field=field):
                reports = copy.deepcopy(self.reports[MODES[0]])
                reports[1][field] = value
                with self.assertRaises(ValueError):
                    build_summary(reports)
        with self.assertRaises(ValueError):
            build_summary([r for r in self.reports[MODES[0]] if r['branch'] == 'design'])

    def test_summary_rejects_different_selected_decoder_across_evaluation_splits(self):
        from experiments.branch_probe.summarize import build_summary
        for field, value in (('decoder_digest', 'different_snapshot'), ('epoch', 999),
                             ('selection', {'split': 'val', 'objective': 'normalized_performance_mse', 'value': 1e9}),
                             ('training_cost', {'decoder_optimizer_steps': 999})):
            with self.subTest(field=field):
                reports = copy.deepcopy(self.reports[MODES[0]])
                item = next(r for r in reports if r['branch'] == 'design' and r['split'] == 'test')
                item[field] = value
                with self.assertRaises(ValueError):
                    build_summary(reports)

    def test_test_and_ood_labels_cannot_change_decoder_training_or_selection(self):
        changed = copy.deepcopy(self.data)
        held_out = set(self.manifest['test'] + self.manifest['ood'])
        for row in changed['samples']:
            if row['sample_id'] in held_out:
                row['performance'] = {name: 1e12 for name in row['performance']}
        context = quiet(probe.prepare_probe, changed, self.paths[MODES[0]])
        with tempfile.TemporaryDirectory() as folder:
            quiet(probe.run_pair, context, folder, seed=29, device='cpu')
            for branch in ('design', 'op'):
                before = torch.load(self.outputs[MODES[0]] / branch / 'decoder_best.pt', weights_only=True)
                after = torch.load(Path(folder, branch, 'decoder_best.pt'), weights_only=True)
                self.assertEqual(before['epoch'], after['epoch'])
                self.assertEqual(before['selection'], after['selection'])
                for key in before['decoder_state']:
                    torch.testing.assert_close(before['decoder_state'][key], after['decoder_state'][key], rtol=0, atol=0)

    def test_multiple_seeds_are_decoder_repeats_with_fixed_pretraining(self):
        from experiments.branch_probe.summarize import build_summary
        old = [r for r in self.reports[MODES[0]] if r['split'] == 'val']
        with tempfile.TemporaryDirectory() as folder:
            new = quiet(probe.run_pair, self.contexts[MODES[0]], folder, seed=43, device='cpu')
        reports = old + new
        self.assertEqual({r['pretrain_seed'] for r in reports}, {17})
        self.assertEqual({r['decoder_seed'] for r in reports}, {29, 43})
        self.assertEqual(len({r['source_checkpoint_sha256'] for r in reports}), 1)
        result = build_summary(reports)
        self.assertEqual({r['num_seeds'] for r in result['summary']}, {2})
        self.assertTrue(all(r['mae_std'] is not None for r in result['summary'] if r['scale'] == 'physical'))


class RunnerBoundaryTests(unittest.TestCase):
    def test_results_cannot_escape_experiment_directory(self):
        from experiments.branch_probe import run
        with tempfile.TemporaryDirectory() as folder, patch.object(run, 'EXPERIMENT_ROOT', Path(folder)):
            base = Path(folder, 'results')
            accepted = run.result_directory(str(base / 'new_run'))
            self.assertEqual(accepted, base / 'new_run')
            self.assertFalse(accepted.exists())
            for path in (Path(folder, 'outside'), base):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    run.result_directory(str(path))
            with tempfile.TemporaryDirectory() as external:
                base.mkdir()
                (base / 'escape').symlink_to(external, target_is_directory=True)
                with self.assertRaises(ValueError):
                    run.result_directory(str(base / 'escape' / 'new_run'))


if __name__ == '__main__':
    unittest.main()
