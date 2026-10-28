"""Tests of the factorial protocol, not claims of real-circuit predictive gains."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import torch

from device_jepa.data import build_manifest, synthetic_dataset
from device_jepa.model import ModelConfig
from device_jepa.states import apply_op_states
from device_jepa.training import pretrain, state_digest
from experiments.branch_probe.ablation import (fit_feature_scaler, fit_readout,
    make_readout, predict, run_ablation, summarize_readouts, transform_memory)
from experiments.branch_probe.diagnostics import standardize_features
from experiments.branch_probe.probe import prepare_probe


ROOT = Path(__file__).resolve().parents[2]


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class FactorialMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.cfg = ModelConfig(dim=8, heads=2, encoder_layers=2, cross_layers=1,
                               ffn_dim=16, decoder_layers=1, decoder_ffn_dim=16, dropout=0.)

    def test_train_only_scaling_preserves_slots_and_matches_prior_diagnostic(self):
        memory = torch.randn(20, 3, 8) * 1e-4 + torch.randn(1, 3, 8)
        original = memory.clone()
        scaler = fit_feature_scaler(memory[:16])
        scaled = transform_memory(memory, 'standardized', scaler)
        train, val, _ = standardize_features(memory[:16], memory[16:])
        torch.testing.assert_close(scaled[:16].flatten(1), train, atol=0, rtol=0)
        torch.testing.assert_close(scaled[16:].flatten(1), val, atol=0, rtol=0)
        self.assertIs(transform_memory(memory, 'raw', scaler), memory)
        torch.testing.assert_close(memory, original, atol=0, rtol=0)
        changed = memory.clone()
        changed[16:] += 100
        again = fit_feature_scaler(changed[:16])
        torch.testing.assert_close(scaler['mean'], again['mean'], atol=0, rtol=0)
        torch.testing.assert_close(scaler['scale'], again['scale'], atol=0, rtol=0)
        self.assertEqual(scaler['fitted_samples'], 16)

    def test_shared_training_loop_learns_signal_for_both_heads_and_restores_state(self):
        x, val = torch.randn(64, 2, 8), torch.randn(16, 2, 8)
        y, vy = x[:, :, 0].mean(1, keepdim=True), val[:, :, 0].mean(1, keepdim=True)
        mask, vm = torch.ones_like(y, dtype=torch.bool), torch.ones_like(vy, dtype=torch.bool)
        paired = []
        for kind in ('attention', 'mlp'):
            a, b, info, state = quiet(fit_readout, kind, self.cfg, x, y.double(), mask, val, vy.double(), vm,
                epochs=35, batch_size=16, hidden=16, lr=.01, seed=29)
            self.assertLess(float((b - vy).square().mean()), .15)
            self.assertEqual(info['early_stopping']['epochs_ran'], 35)
            self.assertEqual(info['optimizer_steps'], 35 * 4)
            best = min(info['history'], key=lambda h: h['val_normalized_mse'])
            self.assertEqual(best['epoch'], info['early_stopping']['best_epoch'])
            self.assertAlmostEqual(float((b - vy).square().mean()), best['val_normalized_mse'])
            restored = make_readout(kind, self.cfg, 2, 1, 16)
            restored.load_state_dict(state)
            self.assertEqual(state_digest(restored), info['selected_digest'])
            torch.testing.assert_close(predict(restored, kind, val, 16), b, atol=0, rtol=0)
            paired.append(info)
        self.assertEqual(paired[0]['sample_order_sha256'], paired[1]['sample_order_sha256'])

    def test_scaling_pairs_share_initialization_and_rng_and_are_deterministic(self):
        x, val = torch.randn(16, 2, 8), torch.randn(8, 2, 8)
        y, vy = torch.randn(16, 2).double(), torch.randn(8, 2).double()
        mask, vm = torch.ones_like(y, dtype=torch.bool), torch.ones_like(vy, dtype=torch.bool)
        cfg = copy.deepcopy(self.cfg)
        cfg.dropout = .1
        for kind in ('attention', 'mlp'):
            a = quiet(fit_readout, kind, cfg, x, y, mask, val, vy, vm, epochs=2, batch_size=8)
            b = quiet(fit_readout, kind, cfg, x, y, mask, val, vy, vm, epochs=2, batch_size=8)
            c = quiet(fit_readout, kind, cfg, x * 3, y, mask, val * 3, vy, vm, epochs=2, batch_size=8)
            torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
            for key in ('initial_digest', 'sample_order_sha256', 'optimizer_steps'):
                self.assertEqual(a[2][key], c[2][key])
            self.assertEqual(a[2]['selected_digest'], b[2]['selected_digest'])

    def test_early_stopping_is_optional_and_does_not_change_exact_best_rule(self):
        x, y = torch.randn(8, 2, 8), torch.randn(8, 1).double()
        mask = torch.ones_like(y, dtype=torch.bool)
        # Zero LR deliberately produces a constant objective to exercise patience.
        _, _, fixed, _ = quiet(fit_readout, 'mlp', self.cfg, x, y, mask, x, y, mask,
                              epochs=4, lr=0., patience=0)
        _, _, early, _ = quiet(fit_readout, 'mlp', self.cfg, x, y, mask, x, y, mask,
                              epochs=4, lr=0., patience=1)
        self.assertEqual(fixed['early_stopping']['epochs_ran'], 4)
        self.assertEqual(early['early_stopping']['epochs_ran'], 2)
        self.assertEqual(early['early_stopping']['best_epoch'], 1)

    def test_summary_uses_sample_std_and_matched_seed_differences(self):
        reports = []
        for seed, offset in ((17, 0.), (29, 2.)):
            for normalization, error in (('raw', 3. + offset), ('standardized', 1. + offset)):
                metric = dict(count=10, mae=error, rmse=error, r2=None, normalized_rmse=error)
                reports.append(dict(source='design', head='mlp', normalization=normalization, seed=seed,
                                    train=dict(metrics={'gain': metric}), val=dict(metrics={'gain': metric})))
        summary, deltas = summarize_readouts(reports)
        raw = next(r for r in summary if r['normalization'] == 'raw' and r['split'] == 'val')
        self.assertEqual(raw['mae_mean'], 4.)
        self.assertAlmostEqual(raw['mae_std'], 2 ** .5)
        self.assertIsNone(raw['r2_mean'])
        self.assertTrue(all(r['mae_standardized_minus_raw'] == -2. for r in deltas))
        one, _ = summarize_readouts(reports[:2])
        self.assertTrue(all(r['mae_std'] is None for r in one))
        with self.assertRaisesRegex(ValueError, 'Missing cell'):
            summarize_readouts(reports[:1])


class FactorialIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        options = json.loads((ROOT / 'configs/smoke.json').read_text())
        options['model'].update(dim=8, heads=2, ffn_dim=16, decoder_ffn_dim=16, decoder_layers=1)
        options['training'].update(pretrain_epochs=1, batch_size=8)
        options['dataset'].update(train_circuits=['synthetic_0'], ood_circuits=[])
        data = synthetic_dataset(32)
        data['samples'] = [r for r in data['samples'] if r['circuit_id'] == 'synthetic_0']
        for row in data['samples']:
            row['devices'] = row['devices'][:4]
        data = apply_op_states(data, options['op_state'])
        cls.manifest = build_manifest(data, config=options['dataset'])
        cls.data, cls.contexts, cls.paths = data, {}, {}
        for mode in ('jepa_zerosim', 'jepa_zerosim_ema'):
            path = quiet(pretrain, data, cls.manifest, options, cls.root / mode, mode=mode)
            cls.paths[mode] = Path(path)
            cls.contexts[mode] = quiet(prepare_probe, data, path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_both_checkpoints_complete_all_cells_with_equal_budgets_and_frozen_source(self):
        for mode, context in self.contexts.items():
            with self.subTest(mode=mode):
                before = hashlib.sha256(self.paths[mode].read_bytes()).hexdigest()
                out = self.root / (mode + '_factorial')
                reports = quiet(run_ablation, context, out, seeds=(17, 29), epochs=2, hidden=8,
                                batch_size=8, device='cpu')
                self.assertEqual(len(reports), 16)
                self.assertEqual(before, hashlib.sha256(self.paths[mode].read_bytes()).hexdigest())
                plan = json.loads((out / 'run.json').read_text())
                self.assertTrue(all(plan[k] for k in ('complete', 'encoder_frozen_verified',
                                                     'initialization_pairing_verified', 'fixed_budget_verified')))
                for split in ('train', 'val'):
                    self.assertLessEqual(set(plan['sample_ids'][split]),
                                         set(self.manifest['labeled_train' if split == 'train' else split]))
                self.assertFalse(set(plan['sample_ids']['train'] + plan['sample_ids']['val']) & set(self.manifest['test']))
                for seed in (17, 29):
                    selected = [r for r in reports if r['seed'] == seed]
                    self.assertEqual(len({r['training']['sample_order_sha256'] for r in selected}), 1)
                    self.assertEqual(len({r['training']['optimizer_steps'] for r in selected}), 1)
                    for head in ('attention', 'mlp'):
                        self.assertEqual(len({r['training']['initial_digest'] for r in selected if r['head'] == head}), 1)
                for filename in ('readouts.json', 'readouts.csv', 'summary.json', 'summary.csv',
                                 'paired_deltas.csv', 'train_mean.json', 'feature_scalers.pt'):
                    self.assertTrue((out / filename).is_file())
                self.assertEqual(len(list(out.glob('seed_*/*/*/*/head_best.pt'))), 16)
                for r in reports:
                    self.assertAlmostEqual(r['val']['normalized_mse'], r['training']['early_stopping']['best_value'])

    def test_inactive_extreme_validation_labels_are_excluded_before_conversion(self):
        context = copy.deepcopy(self.contexts['jepa_zerosim_ema'])
        for row in context['split_rows']['train']:
            row['performance']['power'] = None
        for row in context['split_rows']['val']:
            row['performance']['power'] = 1e100
        reports = quiet(run_ablation, context, self.root / 'inactive', epochs=1, hidden=8, device='cpu')
        self.assertTrue(all(r['val']['metrics']['power']['count'] == 0 for r in reports))

    def test_invalid_options_and_outputs_fail_before_writes(self):
        context = self.contexts['jepa_zerosim']
        for options in ({'seeds': ()}, {'seeds': (17, 17)}, {'sample_seed': -1}, {'epochs': 0},
                        {'weight_decay': -1}, {'std_floor': 0}, {'patience': -1}, {'max_train': 1},
                        {'circuit': 'missing'}, {'lr': float('nan')}):
            out = self.root / 'invalid'
            with self.subTest(options=options), self.assertRaises(ValueError):
                quiet(run_ablation, context, out, device='cpu', **options)
            self.assertFalse(out.exists())
        out = self.root / 'occupied'
        out.mkdir()
        (out / 'keep').write_text('original')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            quiet(run_ablation, context, out, device='cpu')
        self.assertEqual(list(out.iterdir()), [out / 'keep'])
        self.assertEqual((out / 'keep').read_text(), 'original')

    def test_subset_is_independent_of_decoder_seeds(self):
        plans = []
        for seeds in ((17,), (29,)):
            out = self.root / f'seeds_{seeds[0]}'
            quiet(run_ablation, self.contexts['jepa_zerosim'], out, seeds=seeds, sample_seed=43,
                  max_train=8, max_val=2, epochs=1, hidden=8, device='cpu')
            plans.append(json.loads((out / 'run.json').read_text()))
        self.assertEqual(plans[0]['sample_ids'], plans[1]['sample_ids'])
        self.assertEqual(plans[0]['target_scaler_digest'], plans[1]['target_scaler_digest'])

    def test_cli_executes_and_submission_preview_preserves_options(self):
        data = self.root / 'data with spaces.json'
        data.write_text(json.dumps(self.data))
        results = ROOT / 'experiments/branch_probe/results'
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ablation_cli_test_', dir=results) as parent:
            out = Path(parent) / 'run'
            args = ['--data', str(data), '--pretrained', str(self.paths['jepa_zerosim_ema']),
                    '--output', str(out), '--epochs', '1', '--hidden', '8', '--device', 'cpu']
            command = [sys.executable, str(ROOT / 'experiments/branch_probe/ablate.py'), *args]
            done = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertTrue(json.loads((out / 'run.json').read_text())['complete'])
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
            blocked = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('refusing to overwrite', blocked.stderr)
            self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
            preview = subprocess.run(['bash', str(ROOT / 'experiments/branch_probe/submit_ablation.sh'),
                                      '--dry-run', *args], capture_output=True, text=True)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            parts = shlex.split(preview.stdout)
            self.assertIn(str(ROOT / 'experiments/branch_probe/ablate.py'), parts)
            self.assertIn(str(data), parts)
            self.assertNotIn('--ablate', parts)
            self.assertIn('--cpus-per-task=8', parts)
            self.assertIn('--gres=gpu:1', parts)


if __name__ == '__main__':
    unittest.main()
