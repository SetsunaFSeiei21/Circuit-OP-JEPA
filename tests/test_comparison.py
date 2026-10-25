import copy
import math
import unittest
from device_jepa.comparison import compare_reports, regression_metrics, MODES


def report(mode, seed=17, split='test'):
    return dict(split=split, synthetic=True, num_samples=3, metric_names=['gain'],
                provenance=dict(mode=mode, backend=mode, seed=seed, model_config={'dim': 32}, partition_fingerprint='partition',
                                input_op_fingerprint='input', input_transform_digest='transform', performance_fingerprint='perf', target_scaler_digest='scaler',
                                labeled_train_ids=['train1'], train_ids=['train1', 'train2'], timing={},
                                training_config=dict(decoder_epochs=1, decoder_lr=.001, batch_size=8, weight_decay=.0001, grad_clip=1)),
                predictions=[dict(sample_id=str(i), circuit_id='circuit', target={'gain': y},
                                  prediction={'gain': 2.0}, normalized_target={'gain': y},
                                  normalized_prediction={'gain': 2.0}) for i, y in enumerate([1.0, 3.0, None])])


class ComparisonTests(unittest.TestCase):
    def test_statistics_and_missing(self):
        result = regression_metrics([1, 3, None, math.nan], [2, 2, None, 8])
        self.assertEqual(result, dict(count=2, mae=1.0, rmse=1.0, r2=0.0))
        self.assertIsNone(regression_metrics([2, 2], [2, 3])['r2'])
        self.assertIsNone(regression_metrics([1], [1])['r2'])
        self.assertIsNone(regression_metrics([None], [3])['mae'])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            regression_metrics([1], [math.inf])

    def test_extreme_finite_errors_do_not_underflow_or_overflow_rmse(self):
        for scale in (1e-200, 1e200):
            metric = regression_metrics([0., scale], [scale, 2*scale])
            self.assertTrue(math.isclose(metric['rmse'], scale, rel_tol=1e-14))
            self.assertTrue(math.isclose(metric['r2'], -3., rel_tol=1e-14))

    def test_unrepresentable_report_value_raises_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'reporting range'):
            regression_metrics([-1e308], [1e308])
        with self.assertRaisesRegex(ValueError, 'R2 exceeds'):
            regression_metrics([0., 1e-300], [1e300, 1e300])

    def test_reject_architecture_mismatch_across_modes(self):
        for mode in MODES:
            reports = [report(m) for m in MODES]
            next(r for r in reports if r['provenance']['mode'] == mode)['provenance']['model_config']['dim'] = 64
            with self.assertRaisesRegex(ValueError, 'Model configuration'):
                compare_reports(reports)

    def test_allow_only_intended_pin_architecture_differences(self):
        reports = [report(m) for m in MODES]
        for item in reports:
            item['provenance']['model_config'].update(max_devices=32, terminal_bias=True)
        reports[0]['provenance']['model_config'].update(max_devices=128, terminal_bias=False)
        compare_reports(reports)
        reports[1]['provenance']['model_config']['terminal_bias'] = False
        with self.assertRaisesRegex(ValueError, 'between device and JEPA'):
            compare_reports(reports)

    def test_matching_seeds_and_separate_splits(self):
        reports = [report(mode, seed, split) for mode in MODES for seed in [17, 29] for split in ['test', 'ood']]
        result = compare_reports(reports)
        self.assertEqual(len(result['summary']), 24)
        self.assertTrue(all(r['num_seeds'] == 2 for r in result['summary']))
        self.assertTrue(all(r['mae_std'] == 0 for r in result['summary']))
        self.assertTrue(all(r['count_per_seed'] == 2 for r in result['summary']))

    def test_reject_mismatch(self):
        for key in ['partition_fingerprint', 'input_op_fingerprint', 'performance_fingerprint', 'target_scaler_digest', 'input_transform_digest', 'labeled_train_ids']:
            reports = [report(m) for m in MODES]
            reports[1]['provenance'][key] = 'different'
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                compare_reports(reports)

    def test_reject_supervised_budget_mismatch(self):
        for key in ('decoder_epochs', 'decoder_lr', 'batch_size', 'weight_decay', 'grad_clip'):
            reports = [report(m) for m in MODES]
            reports[1]['provenance']['training_config'][key] *= 2
            with self.assertRaisesRegex(ValueError, 'training budget'):
                compare_reports(reports)

    def test_reject_architecture_mismatch_within_mode(self):
        reports = [report(m, s) for m in MODES for s in [17, 29]]
        reports[1]['provenance']['model_config']['dim'] = 64
        with self.assertRaisesRegex(ValueError, 'Model configuration'):
            compare_reports(reports)

    def test_reject_missing_modes_and_seeds(self):
        with self.assertRaisesRegex(ValueError, 'matching seed'):
            compare_reports([report(MODES[0])])
        with self.assertRaisesRegex(ValueError, 'matching seed'):
            compare_reports([report(m) for m in MODES] + [report(MODES[0], 29)])
        self.assertEqual(len(compare_reports([report(MODES[0])], False)['summary']), 4)

    def test_reject_different_targets_ids_and_duplicate_reports(self):
        reports = [report(m) for m in MODES]
        for key, value in [('sample_id', 'wrong'), ('target', {'gain': 22})]:
            bad = copy.deepcopy(reports)
            bad[1]['predictions'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'IDs or targets'):
                compare_reports(bad)
        with self.assertRaisesRegex(ValueError, 'Duplicate mode'):
            compare_reports(reports + [reports[0]])

    def test_reject_duplicate_ids_and_missing_normalization(self):
        bad = report(MODES[0])
        bad['predictions'][1]['sample_id'] = '0'
        with self.assertRaisesRegex(ValueError, 'Duplicate evaluation'):
            compare_reports([bad], False)
        bad = report(MODES[0])
        del bad['predictions'][0]['normalized_target']
        with self.assertRaisesRegex(ValueError, 'Normalized prediction'):
            compare_reports([bad], False)


if __name__ == '__main__':
    unittest.main()
