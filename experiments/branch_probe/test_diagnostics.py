"""Mechanism checks for collapse diagnostics; no GPU/real-data performance claim."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from device_jepa.data import build_manifest, synthetic_dataset
from device_jepa.states import apply_op_states
from device_jepa.training import pretrain
from experiments.branch_probe.diagnostics import (embedding_variation, fit_targets,
    mlp_predict, ridge_predict, run_diagnostics, select_rows, standardize_features)
from experiments.branch_probe.probe import prepare_probe


ROOT = Path(__file__).resolve().parents[2]


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class DiagnosticMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_device_identity_only_has_zero_within_design_variation(self):
        z = torch.randn(1, 8, 16).expand(32, -1, -1).clone()
        result = embedding_variation(z, [f'M{i}' for i in range(8)])
        self.assertEqual(result['within_device_std_rms'], 0.)
        self.assertGreater(result['pooled_token_std_rms'], .5)
        self.assertEqual(result['within_fraction_of_total_variance'], 0.)
        self.assertTrue(all(d['fraction_std_below_1e_6'] == 1 for d in result['per_device']))

    def test_variance_decomposition_detects_sizing_variation(self):
        offsets = torch.randn(64, 1, 12, dtype=torch.float64)
        devices = torch.randn(1, 5, 12, dtype=torch.float64) * 10
        z = offsets + devices
        r = embedding_variation(z, [str(i) for i in range(5)])
        self.assertAlmostEqual(r['within_device_std_rms'], float(offsets.var(0, unbiased=False).mean().sqrt()))
        self.assertAlmostEqual(r['pooled_token_std_rms'], float(z.flatten(0, 1).var(0, unbiased=False).mean().sqrt()))
        self.assertGreater(r['within_fraction_of_total_variance'], 0.)

    def test_scaling_uses_train_only_and_does_not_amplify_tiny_noise(self):
        train = torch.tensor([[1., 0.], [3., 1e-8]])
        val = torch.tensor([[101., 2e-8], [103., 3e-8]])
        a, b, info = standardize_features(train, val)
        torch.testing.assert_close(a[:, 0], torch.tensor([-1., 1.]))
        torch.testing.assert_close(b[:, 0], torch.tensor([99., 101.]))
        self.assertLess(float(a[:, 1].abs().max()), 1e-3)
        self.assertEqual(info['columns_below_floor'], 1)

    def test_ridge_recovers_linear_signal_and_handles_missing_labels(self):
        x, v = torch.randn(128, 6), torch.randn(32, 6)
        weights = torch.randn(6, 3)
        y, vy = x @ weights + 3, v @ weights + 3
        mask = torch.ones_like(y, dtype=torch.bool)
        mask[::3, 1] = False
        mask[:, 2] = False
        corrupted = y.clone()
        corrupted[~mask] = 1e25
        a, b = quiet(ridge_predict, x, corrupted, mask, v, alpha=1e-5)
        torch.testing.assert_close(b[:, :2], vy[:, :2].double(), atol=2e-5, rtol=2e-5)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.equal(b[:, 2], torch.zeros_like(b[:, 2])))

    def test_dual_ridge_matches_augmented_regularized_least_squares(self):
        x, v, y = torch.randn(8, 20, dtype=torch.float64), torch.randn(4, 20, dtype=torch.float64), torch.randn(8, 2, dtype=torch.float64)
        alpha = .3
        _, actual = quiet(ridge_predict, x, y, torch.ones_like(y, dtype=torch.bool), v, alpha)
        mx, my = x.mean(0), y.mean(0)
        a = torch.cat([x-mx, torch.eye(20, dtype=x.dtype)*alpha**.5])
        target = torch.cat([y-my, torch.zeros(20, 2, dtype=y.dtype)])
        expected = (v-mx) @ torch.linalg.lstsq(a, target).solution + my
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    def test_constant_features_produce_train_mean_prediction(self):
        x, v, y = torch.ones(32, 5), torch.ones(8, 5), torch.randn(32, 2)
        _, pred = quiet(ridge_predict, x, y, torch.ones_like(y, dtype=torch.bool), v)
        torch.testing.assert_close(pred, y.double().mean(0).expand_as(pred))

    def test_small_mlp_can_read_known_signal_and_repeats_exactly(self):
        x, val = torch.randn(128, 3), torch.randn(32, 3)
        y = (x[:, 0] + .5 * x[:, 1]).unsqueeze(-1)
        vy = (val[:, 0] + .5 * val[:, 1]).unsqueeze(-1)
        m, vm = torch.ones_like(y, dtype=torch.bool), torch.ones_like(vy, dtype=torch.bool)
        options = dict(seed=29, epochs=40, batch_size=32, hidden=16, lr=.02, patience=10)
        a = quiet(mlp_predict, x, y.double(), m, val, vy.double(), vm, **options)
        b = quiet(mlp_predict, x, y.double(), m, val, vy.double(), vm, **options)
        self.assertLess(float((a[1] - vy).square().mean()), .03)
        torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
        self.assertEqual(a[2], b[2])


class DiagnosticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        options = json.loads((ROOT / 'configs/smoke.json').read_text())
        options['model'].update(dim=16, heads=4, ffn_dim=32, decoder_ffn_dim=32)
        options['training'].update(pretrain_epochs=1, batch_size=8)
        options['dataset'].update(train_circuits=['synthetic_0'], ood_circuits=[])
        data = synthetic_dataset(32)
        data['samples'] = [r for r in data['samples'] if r['circuit_id'] == 'synthetic_0']
        # A genuine fixed device layout is essential for the flattening probe.
        for row in data['samples']:
            row['devices'] = row['devices'][:4]
        data = apply_op_states(data, options['op_state'])
        manifest = build_manifest(data, config=options['dataset'])
        cls.data, cls.manifest, cls.contexts, cls.paths = data, manifest, {}, {}
        for mode in ('jepa_zerosim', 'jepa_zerosim_ema'):
            path = quiet(pretrain, data, manifest, options, cls.root / mode, mode=mode)
            cls.paths[mode] = path
            cls.contexts[mode] = quiet(prepare_probe, data, path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_both_modes_complete_without_mutating_source(self):
        for mode, context in self.contexts.items():
            with self.subTest(mode=mode):
                path = Path(self.paths[mode])
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                out = self.root / (mode + '_diagnostic')
                diagnostic, reports = quiet(run_diagnostics, context, out, device='cpu', epochs=2, hidden=16, batch_size=8)
                self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertTrue(diagnostic['encoder_frozen_verified'])
                self.assertEqual(len(reports), 7)
                heads = [r for r in reports if r['head'] == 'mlp' and r['source'] != 'raw_op']
                self.assertEqual(heads[0]['training']['initial_digest'], heads[1]['training']['initial_digest'])
                for filename in ('run.json', 'diagnostics.json', 'readouts.json', 'readouts.csv', 'device_variation.csv'):
                    self.assertTrue((out / filename).is_file())
                plan = json.loads((out / 'run.json').read_text())
                self.assertTrue(plan['complete'])
                self.assertLessEqual(set(plan['sample_ids']['train']), set(self.manifest['labeled_train']))
                self.assertLessEqual(set(plan['sample_ids']['val']), set(self.manifest['val']))
                self.assertFalse(set(plan['sample_ids']['train'] + plan['sample_ids']['val']) & set(self.manifest['test']))

    def test_target_scalers_ignore_val_and_inactive_extreme_is_masked(self):
        context = copy.deepcopy(self.contexts['jepa_zerosim_ema'])
        for row in context['split_rows']['train']:
            row['performance']['power'] = None
        rows, _ = select_rows(context)
        prep = fit_targets(context, rows['train'], 'cpu')
        for row in context['split_rows']['val']:
            row['performance']['power'] = 1e100
        after = fit_targets(context, rows['train'], 'cpu')
        self.assertEqual(prep['performance'], after['performance'])
        _, reports = quiet(run_diagnostics, context, self.root/'inactive', device='cpu', epochs=1, hidden=8)
        self.assertTrue(all(r['val']['metrics']['power']['count'] == 0 for r in reports))

    def test_subsampling_and_slot_identity_do_not_depend_on_labels_or_device_order(self):
        context = self.contexts['jepa_zerosim']
        first, _ = select_rows(context, max_train=8, max_val=2)
        changed = copy.deepcopy(context)
        for split in ('train', 'val'):
            changed['split_rows'][split].reverse()
            for row in changed['split_rows'][split]:
                row['performance'] = {k: 0 for k in row['performance']}
                row['devices'].reverse()
        second, _ = select_rows(changed, max_train=8, max_val=2)
        for split in ('train', 'val'):
            self.assertEqual([r['sample_id'] for r in first[split]], [r['sample_id'] for r in second[split]])
        changed['split_rows']['val'][0]['devices'][0]['terminals']['d'] = 'changed_net'
        with self.assertRaisesRegex(ValueError, 'topology'):
            select_rows(changed)

    def test_bad_options_and_nonempty_output_fail_without_writes(self):
        context = self.contexts['jepa_zerosim']
        for config in ({'epochs':0}, {'std_floor':0}, {'ridge_alpha':-1}, {'circuit':'missing'}, {'max_train':1}):
            with self.subTest(config=config):
                out = self.root/'should_not_exist'
                with self.assertRaises(ValueError):
                    quiet(run_diagnostics, context, out, device='cpu', **config)
                self.assertFalse(out.exists())
        out = self.root/'occupied'
        out.mkdir()
        (out/'keep').write_text('original')
        with self.assertRaises(ValueError):
            quiet(run_diagnostics, context, out, device='cpu')
        self.assertEqual(list(out.iterdir()), [out/'keep'])
        self.assertEqual((out/'keep').read_text(), 'original')

    def test_stats_only_skips_head_training(self):
        out = self.root/'stats'
        _, reports = quiet(run_diagnostics, self.contexts['jepa_zerosim'], out, device='cpu', stats_only=True)
        self.assertEqual(reports, [])
        self.assertFalse((out/'readouts.json').exists())
        self.assertTrue(json.loads((out/'run.json').read_text())['complete'])

    def test_submission_preview_selects_diagnostic_runner_and_preserves_spaces(self):
        script = ROOT/'experiments/branch_probe/submit_diagnostics.sh'
        result = subprocess.run(['bash', str(script), '--dry-run', '--data', '/tmp/data with space.json',
                                 '--pretrained', '/tmp/source.pt', '--stats-only'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        import shlex
        command = shlex.split(result.stdout)
        self.assertIn(str(script.with_name('diagnose.py')), command)
        self.assertIn('/tmp/data with space.json', command)
        self.assertNotIn('--diagnose', command)
        self.assertIn('--stats-only', command)

    def test_cli_end_to_end_and_refuses_to_overwrite(self):
        data_path = self.root/'data.json'
        data_path.write_text(json.dumps(self.data))
        results = ROOT/'experiments/branch_probe/results'
        results.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='diagnostic_cli_test_', dir=results) as parent:
            output = Path(parent)/'run'
            command = [sys.executable, str(ROOT/'experiments/branch_probe/diagnose.py'),
                       '--data', str(data_path), '--pretrained', str(self.paths['jepa_zerosim_ema']),
                       '--output', str(output), '--epochs', '1', '--hidden', '8', '--device', 'cpu']
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads((output/'run.json').read_text())['complete'])
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            blocked = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('refusing to overwrite', blocked.stderr)
            self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
