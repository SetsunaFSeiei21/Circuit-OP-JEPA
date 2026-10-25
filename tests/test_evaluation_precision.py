"""Finite physical scales survive reporting, without float32 label round trips."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from device_jepa.baselines import evaluate_baseline, train_baseline
from device_jepa.comparison import regression_metrics
from device_jepa.data import build_manifest, synthetic_dataset
from device_jepa.states import apply_op_states
from device_jepa.tensors import denormalize, evaluation_targets
from device_jepa.training import _metrics, evaluate, load_checkpoint, pretrain, seed_everything, train_decoder


class PhysicalPrecisionTests(unittest.TestCase):
    def stats(self, kind='asinh', scale=1e-20, mean=0., std=1.):
        return {'kind': [kind], 'mean': [mean], 'std': [std], 'asinh_scale': [scale]}

    def test_scaled_sinh_stays_finite_when_float32_sinh_overflows(self):
        value = denormalize(torch.tensor([[100.]]), self.stats()).item()
        self.assertTrue(math.isclose(value, math.sinh(100)*1e-20, rel_tol=1e-14))

    def test_tiny_scale_is_not_cast_to_float32_zero(self):
        value = denormalize(torch.tensor([[1.]]), self.stats(scale=1e-250)).item()
        self.assertGreater(value, 0.)
        self.assertTrue(math.isclose(value, math.sinh(1)*1e-250, rel_tol=1e-14))

    def test_log_space_inverse_avoids_unscaled_float64_overflow(self):
        value = denormalize(torch.tensor([[1000.]]), self.stats(scale=1e-250)).item()
        expected = math.exp(1000 + math.log(1e-250) - math.log(2))
        self.assertTrue(math.isclose(value, expected, rel_tol=1e-14))

    def test_linear_offset_keeps_small_variation(self):
        value = denormalize(torch.tensor([[-1.], [1.]]), self.stats('linear', mean=1e9))
        self.assertEqual(value.dtype, torch.float64)
        self.assertEqual(value[:, 0].tolist(), [1e9-1, 1e9+1])

    def test_unrepresentable_inverse_raises_not_clips(self):
        with self.assertRaisesRegex(ValueError, 'finite physical range'):
            denormalize(torch.tensor([[1000.]]), self.stats(scale=1.))

    def test_original_truth_and_normalized_truth_are_double(self):
        prep = {'feature_spec': {'metric_names': ['gain']},
                'performance': self.stats('linear', mean=1e9)}
        rows = [{'performance': {'gain': 1e9+.125}}, {'performance': {'gain': None}}]
        raw, normalized = evaluation_targets(rows, prep)
        self.assertEqual(raw.dtype, torch.float64)
        self.assertEqual(raw.tolist(), [[1e9+.125], [0.]])
        self.assertEqual(normalized.tolist(), [[.125], [0.]])

    def test_metrics_match_raw_records_for_high_offset_low_variance(self):
        prep = {'feature_spec': {'metric_names': ['gain']},
                'performance': self.stats('linear', mean=1e9)}
        truth = [1e9-1, 1e9+1]
        prediction = [1e9-.5, 1e9+.5]
        metric = _metrics(torch.tensor(prediction, dtype=torch.float64)[:, None],
                          torch.tensor(truth, dtype=torch.float64)[:, None],
                          torch.ones(2, 1, dtype=torch.bool), prep)['gain']
        for key, value in regression_metrics(truth, prediction).items():
            self.assertEqual(metric[key], value)
        self.assertEqual(metric['r2'], .75)


class SafeRuntimeTests(unittest.TestCase):
    def test_unsafe_torch_version_rejected_before_deserialization(self):
        for version in ('2.2.0', '2.5.1', '2.6.0rc1', 'unknown'):
            with self.subTest(version=version), patch.object(torch, '__version__', version), \
                    patch.object(torch, 'load') as load:
                with self.assertRaisesRegex(RuntimeError, 'stable PyTorch >=2.6.0'):
                    load_checkpoint('not_opened.pt')
                load.assert_not_called()

    def test_patched_torch_version_keeps_weights_only(self):
        checkpoint = {'checkpoint_version': 2, 'preprocessing': {'preprocessing_version': 2}}
        for version in ('2.6.0', '2.14.1+cpu'):
            with self.subTest(version=version), patch.object(torch, '__version__', version), \
                    patch.object(torch, 'load', return_value=checkpoint) as load:
                self.assertEqual(load_checkpoint('example.pt'), checkpoint)
                load.assert_called_once_with('example.pt', map_location='cpu', weights_only=True)

    def test_cuda_deterministic_workspace_default_set_before_seeding(self):
        def seeded(_seed):
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        with patch.dict(os.environ, {}, clear=True), patch.object(torch, 'manual_seed', side_effect=seeded), \
                patch.object(torch.cuda, 'is_available', return_value=False):
            seed_everything(17)

    def test_cuda_explicit_supported_workspace_preserved(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}):
            seed_everything(17)
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':16:8')

    def test_cuda_bad_workspace_rejected_before_seeding(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':bad'}), patch.object(torch, 'manual_seed') as seed:
            with self.assertRaisesRegex(ValueError, 'Deterministic CUDA'):
                seed_everything(17)
            seed.assert_not_called()


class EvaluationPathsPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = json.loads((Path(__file__).parents[1] / 'configs' / 'smoke.json').read_text())
        cls.options['training'].update(pretrain_epochs=1, decoder_epochs=1, baseline_epochs=1)
        cls.options['model'].update(dim=8, heads=2, ffn_dim=16, decoder_ffn_dim=16, max_devices=64)
        cls.data = apply_op_states(synthetic_dataset(6), cls.options['op_state'])
        for i, row in enumerate(cls.data['samples']):
            row['performance']['gain'] = 1e9 + i * .125
        cls.manifest = build_manifest(cls.data, config=cls.options['dataset'])
        cls.temp = tempfile.TemporaryDirectory()
        cls.paths = {}
        pre = pretrain(cls.data, cls.manifest, cls.options, Path(cls.temp.name)/'jepa')
        cls.paths['jepa_zerosim'] = train_decoder(cls.data, pre, cls.options, Path(cls.temp.name)/'jepa')
        for mode in ('zerosim_pin', 'zerosim_device'):
            cls.paths[mode] = train_baseline(cls.data, cls.manifest, cls.options, Path(cls.temp.name)/mode, mode)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assert_mode_precision(self, mode):
        evaluator = evaluate if mode == 'jepa_zerosim' else evaluate_baseline
        report = evaluator(self.data, self.paths[mode], split='test')
        records = report['predictions']
        expected = regression_metrics([r['target']['gain'] for r in records],
                                      [r['prediction']['gain'] for r in records])
        self.assertIsNotNone(expected['r2'])
        for key, value in expected.items():
            self.assertEqual(report['metrics']['gain'][key], value, (mode, key))
        truth_by_id = {r['sample_id']: r['performance']['gain'] for r in self.data['samples']}
        for record in records:
            self.assertEqual(record['target']['gain'], truth_by_id[record['sample_id']])

    def test_jepa_evaluation_raw_precision(self):
        self.assert_mode_precision('jepa_zerosim')

    def test_pin_evaluation_raw_precision(self):
        self.assert_mode_precision('zerosim_pin')

    def test_device_evaluation_raw_precision(self):
        self.assert_mode_precision('zerosim_device')


if __name__ == '__main__':
    unittest.main()
