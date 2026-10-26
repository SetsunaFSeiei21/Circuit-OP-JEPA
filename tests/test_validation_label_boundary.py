"""Inactive validation labels cannot fail before the supervised loss masks them."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from device_jepa.baselines import (BASELINE_MODES, collate_pin_samples, evaluate_baseline,
                                   fit_pin_vocabulary, train_baseline)
from device_jepa.data import (build_manifest, dataset_fingerprint, fit_preprocessor,
                              synthetic_dataset, transform_value)
from device_jepa.states import apply_op_states
from device_jepa.tensors import collate_samples, evaluation_targets
from device_jepa.training import evaluate, load_checkpoint, pretrain, train_decoder


class ValidationLabelBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.config = json.loads((Path(__file__).parents[1] / 'configs/smoke.json').read_text())
        cls.config['model'].update(dim=8, heads=2, ffn_dim=16, decoder_ffn_dim=16,
                                   decoder_layers=1, dropout=0.)
        cls.config['training'].update(pretrain_epochs=1, decoder_epochs=2, torch_threads=1)
        cls.config['normalization']['performance'] = {
            'gain': {'kind': 'linear'}, 'power': {'kind': 'linear'}}
        cls.data = apply_op_states(synthetic_dataset(6, 17), cls.config['op_state'])
        manifest = build_manifest(cls.data, config=cls.config['dataset'])
        for row in cls.data['samples']:
            if row['sample_id'] in manifest['labeled_train']:
                row['performance']['gain'] = None
        cls.manifest = build_manifest(cls.data, config=cls.config['dataset'])
        cls.prep = fit_preprocessor(cls.data, cls.manifest, include_performance=True,
                                    config=cls.config['normalization'])
        cls.pin_vocab = fit_pin_vocabulary(cls.data, cls.manifest)
        cls.source = pretrain(cls.data, cls.manifest, cls.config, cls.root / 'source')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def validation_data(self, gain, power='original'):
        data = copy.deepcopy(self.data)
        for row in data['samples']:
            if row['sample_id'] in self.manifest['val']:
                row['performance']['gain'] = gain
                if power != 'original':
                    row['performance']['power'] = power
        return data

    def train(self, mode, data, output):
        if mode == 'jepa_zerosim':
            return train_decoder(data, self.source, self.config, output)
        manifest = copy.deepcopy(self.manifest)
        manifest['dataset_fingerprint'] = dataset_fingerprint(data)
        return train_baseline(data, manifest, self.config, output, mode)

    def collate(self, kind, samples, **kwargs):
        if kind == 'pin':
            return collate_pin_samples(samples, self.prep, self.pin_vocab, **kwargs)
        return collate_samples(samples, self.prep, include_op=False,
                               include_performance=kwargs.pop('include_performance', True), **kwargs)

    def test_extreme_untrained_labels_leave_all_five_training_runs_unchanged(self):
        for mode in ('jepa_zerosim', *BASELINE_MODES):
            runs = []
            for index, value in enumerate((None, 1e100, -1e100)):
                with self.subTest(mode=mode, value=value):
                    output = self.root / 'unchanged' / mode / str(index)
                    cp = load_checkpoint(self.train(mode, self.validation_data(value), output))
                    history_name = 'decoder' if mode == 'jepa_zerosim' else mode
                    history = json.loads((output / f'{history_name}_history.json').read_text())
                    self.assertEqual(cp['selection_metric_names'], ['power'])
                    self.assertEqual(cp['unobserved_training_metrics'], ['gain'])
                    runs.append((cp['epoch'], cp['encoder_digest'], cp['decoder_digest'], history))
            self.assertEqual(runs[0], runs[1], mode)
            self.assertEqual(runs[0], runs[2], mode)

    def test_extreme_supported_labels_still_fail_in_all_five_modes(self):
        for mode in ('jepa_zerosim', *BASELINE_MODES):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'standardized feature power'):
                self.train(mode, self.validation_data(1e100, power=1e100),
                           self.root / 'supported_overflow' / mode)

    def test_untrained_extreme_labels_do_not_hide_missing_supported_validation(self):
        for mode in ('jepa_zerosim', *BASELINE_MODES):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'trained-metric.*val'):
                self.train(mode, self.validation_data(1e100, power=None),
                           self.root / 'no_supported_labels' / mode)

    def test_device_and_pin_selection_skip_normalization_before_conversion(self):
        sample = copy.deepcopy(self.data['samples'][0])
        sample['performance'].update(gain=1e100, power=1e-5)
        def checked_transform(value, stats, index):
            if stats is self.prep['performance'] and index == 0:
                self.fail('Inactive metric reached normalization')
            return transform_value(value, stats, index)
        for kind in ('device', 'pin'):
            for selected in ([False, True], (False, True), torch.tensor([False, True])):
                with self.subTest(kind=kind, mask_type=type(selected).__name__):
                    with patch('device_jepa.tensors.transform_value', side_effect=checked_transform):
                        batch = self.collate(kind, [sample], performance_metric_mask=selected)
                    self.assertEqual(batch['performance_mask'].tolist(), [[False, True]])
                    self.assertEqual(batch['performance'][0, 0].item(), 0.)
                    self.assertTrue(torch.isfinite(batch['performance']).all())

    def test_default_tensorization_keeps_all_labels_and_rejects_overflow(self):
        sample = copy.deepcopy(self.data['samples'][0])
        sample['performance'].update(gain=60., power=1e-5)
        for kind in ('device', 'pin'):
            with self.subTest(kind=kind):
                batch = self.collate(kind, [sample])
                self.assertEqual(batch['performance_mask'].tolist(), [[True, True]])
                sample['performance']['gain'] = 1e100
                with self.assertRaisesRegex(ValueError, 'standardized feature gain'):
                    self.collate(kind, [sample])
                sample['performance']['gain'] = 60.
        sample['performance']['gain'] = 1e100
        raw, normalized = evaluation_targets([sample], self.prep)
        self.assertEqual(raw.dtype, torch.float64)
        self.assertEqual(raw[0, 0].item(), 1e100)
        self.assertTrue(torch.isfinite(normalized).all())

    def test_metric_mask_rejects_bad_shapes_types_and_unused_selection(self):
        invalid = ([], [True], [True, False, True], [0, 1], 'gain',
                   [[False, True]], torch.tensor([0, 1]), torch.tensor([[False, True]]))
        for kind in ('device', 'pin'):
            for selected in invalid:
                with self.subTest(kind=kind, mask=repr(selected)):
                    with self.assertRaisesRegex(ValueError, 'performance_metric_mask'):
                        self.collate(kind, self.data['samples'][:1], performance_metric_mask=selected)
            with self.assertRaisesRegex(ValueError, 'requires include_performance'):
                self.collate(kind, self.data['samples'][:1], include_performance=False,
                             performance_metric_mask=[False, True])

    def test_evaluation_retains_observed_untrained_metrics_in_all_five_modes(self):
        data = self.validation_data(60.)
        for mode in ('jepa_zerosim', *BASELINE_MODES):
            with self.subTest(mode=mode):
                path = self.train(mode, data, self.root / 'evaluation' / mode)
                evaluator = evaluate if mode == 'jepa_zerosim' else evaluate_baseline
                report = evaluator(data, path, split='val')
                self.assertEqual(report['unobserved_training_metrics'], ['gain'])
                records = report['predictions']
                self.assertTrue(records)
                self.assertTrue(all(row['target']['gain'] == 60. for row in records))


if __name__ == '__main__':
    unittest.main()
