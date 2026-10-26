import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from device_jepa.comparison import compare_reports, regression_metrics, main, MODES, LEGACY_MODES, OP_MODES


def report(mode, seed=17, split='test'):
    result = dict(split=split, synthetic=True, num_samples=3, metric_names=['gain'],
                provenance=dict(mode=mode, backend=mode, seed=seed, model_config={'dim': 32}, partition_fingerprint='partition',
                                requires_op_at_inference=mode in OP_MODES, op_input_transform_digest='op_transform',
                                input_op_fingerprint='input', input_transform_digest='transform', performance_fingerprint='perf', target_scaler_digest='scaler',
                                labeled_train_ids=['train1'], train_ids=['train1', 'train2'], timing={},
                                training_config=dict(pretrain_epochs=1, decoder_epochs=1, baseline_epochs=2,
                                                     decoder_lr=.001, batch_size=8, weight_decay=.0001, grad_clip=1)),
                predictions=[dict(sample_id=str(i), circuit_id='circuit', target={'gain': y},
                                  prediction={'gain': 2.0}, normalized_target={'gain': y},
                                  normalized_prediction={'gain': 2.0}) for i, y in enumerate([1.0, 3.0, None])])
    p = result['provenance']
    p.update(run_identity=f'{mode}/{seed}', decoder_seed=seed, selection_metric_names=['gain'])
    if mode == 'jepa_zerosim':
        protocol = {'protocol_version': 1, 'optimizer': 'AdamW', 'model_config': {'dim': 32},
                    'architecture': 'shared_structure', 'input_contract_version': 3,
                    'loss': {'invariance': 25, 'variance': 25, 'covariance': 1, 'variance_floor': 1},
                    'op_state': {}, 'feature_spec': {}, 'normalization': {},
                    'training': {'pretrain_epochs': 1, 'pretrain_lr': .001, 'batch_size': 8,
                                 'weight_decay': .0001, 'grad_clip': 1., 'torch_threads': 1}}
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        p.update(pretraining_protocol_digest=digest, pretrain_seed=seed,
                 pretrain_provenance={'protocol': protocol, 'protocol_digest': digest, 'seed': seed,
                                     'source_options': {'loss': copy.deepcopy(protocol['loss']),
                                         'training': {**protocol['training'], 'seed': seed},
                                         'model': copy.deepcopy(protocol['model_config']), 'op_state': {}},
                                     'input_op_fingerprint': 'input', 'partition_fingerprint': 'partition'})
    return result


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
        for item in reports:
            if item['provenance']['mode'] in ('zerosim_pin', 'zerosim_pin_op'):
                item['provenance']['model_config'].update(max_devices=128, terminal_bias=False)
        compare_reports(reports)
        reports[1]['provenance']['model_config']['terminal_bias'] = False
        with self.assertRaisesRegex(ValueError, 'between device and JEPA'):
            compare_reports(reports)

    def test_matching_seeds_and_separate_splits(self):
        reports = [report(mode, seed, split) for mode in MODES for seed in [17, 29] for split in ['test', 'ood']]
        result = compare_reports(reports)
        self.assertEqual(len(result['summary']), 40)
        self.assertTrue(all(r['num_seeds'] == 2 for r in result['summary']))
        self.assertTrue(all(r['mae_std'] == 0 for r in result['summary']))
        self.assertTrue(all(r['count_per_seed'] == 2 for r in result['summary']))

    def test_legacy_three_mode_reports_remain_supported_without_new_metadata(self):
        reports = [report(m) for m in LEGACY_MODES]
        for item in reports:
            del item['provenance']['requires_op_at_inference']
            del item['provenance']['op_input_transform_digest']
            for key in ('run_identity', 'pretraining_protocol_digest', 'pretrain_provenance',
                        'pretrain_seed', 'decoder_seed', 'selection_metric_names'):
                item['provenance'].pop(key, None)
        result = compare_reports(reports)
        self.assertEqual(result['expected_modes'], list(LEGACY_MODES))
        self.assertFalse(result['complete'])
        self.assertTrue(result['cohort_complete'])
        self.assertFalse(result['strict_provenance'])
        self.assertEqual(result['paired_per_seed'], [])
        self.assertTrue(all(not r['requires_op_at_inference'] for r in result['summary']))
        jepa_cost = next(r for r in result['run_costs'] if r['mode'] == 'jepa_zerosim')
        self.assertEqual(jepa_cost['op_training_use'], 'pretraining_target')
        self.assertEqual(jepa_cost['op_inference_use'], 'none')

    def test_explicit_cohort_and_five_mode_completeness(self):
        pair = ['zerosim_pin', 'zerosim_pin_op']
        with self.assertRaisesRegex(ValueError, 'matching seed'):
            compare_reports([report(m) for m in pair])
        result = compare_reports([report(m) for m in pair], expected_modes=pair)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['paired_per_seed']), 4)
        with self.assertRaisesRegex(ValueError, 'outside the declared'):
            compare_reports([report(m) for m in MODES], expected_modes=LEGACY_MODES)
        with self.assertRaisesRegex(ValueError, 'matching seed'):
            compare_reports([report(m) for m in LEGACY_MODES], expected_modes=MODES)
        for invalid in ([], ['unknown'], ['zerosim_pin', 'zerosim_pin']):
            with self.assertRaisesRegex(ValueError, 'Expected modes'):
                compare_reports([report('zerosim_pin')], expected_modes=invalid)

    def test_pairwise_deltas_are_op_minus_no_op_and_paired_before_summary(self):
        reports = [report(m, seed) for m in MODES for seed in (17, 29)]
        for item in reports:
            if item['provenance']['mode'] in OP_MODES:
                # Seed 17 predicts perfectly; seed 29 uses the no-OP predictions.
                if item['provenance']['seed'] == 17:
                    for row in item['predictions']:
                        row['prediction'] = dict(row['target'])
                        row['normalized_prediction'] = dict(row['normalized_target'])
        result = compare_reports(reports)
        self.assertEqual(len(result['paired_per_seed']), 16)
        self.assertEqual(len(result['paired_summary']), 8)
        for row in result['paired_summary']:
            self.assertEqual(row['num_paired_seeds'], 2)
            self.assertEqual(row['mae_delta_mean'], -.5)
            self.assertEqual(row['rmse_delta_mean'], -.5)
            self.assertEqual(row['r2_delta_mean'], .5)
            self.assertAlmostEqual(row['mae_delta_std'], math.sqrt(.5))

    def test_partial_pairing_uses_only_intersecting_seeds_and_keeps_undefined_r2(self):
        reports = [report('zerosim_pin', seed) for seed in (17, 29)] + [report('zerosim_pin_op', 29)]
        for item in reports:
            for row in item['predictions'][:2]:
                row['target']['gain'] = row['normalized_target']['gain'] = 1.
        result = compare_reports(reports, require_all_modes=False)
        self.assertFalse(result['complete'])
        self.assertEqual({r['seed'] for r in result['paired_per_seed']}, {29})
        self.assertTrue(all(r['r2_delta'] is None for r in result['paired_per_seed']))
        self.assertTrue(all(r['r2_delta_defined_seeds'] == 0 for r in result['paired_summary']))
        self.assertTrue(all(r['mae_delta_std'] is None for r in result['paired_summary']))

    def test_op_information_and_unknown_cost_are_not_silently_zero(self):
        result = compare_reports([report(m) for m in MODES])
        for row in result['evaluation_costs']:
            self.assertEqual(row['requires_op_at_inference'], row['mode'] in OP_MODES)
            self.assertIsNone(row['op_acquisition_cost_seconds'])
            self.assertEqual(row['op_acquisition_cost_status'], 'not_measured')
        for mode in MODES:
            bad = report(mode)
            bad['provenance']['requires_op_at_inference'] = mode not in OP_MODES
            with self.assertRaisesRegex(ValueError, 'requires_op_at_inference'):
                compare_reports([bad], False)
        bad = report('zerosim_pin_op')
        del bad['provenance']['requires_op_at_inference']
        with self.assertRaisesRegex(ValueError, 'requires_op_at_inference'):
            compare_reports([bad], False)

    def test_measured_acquisition_cost_needs_scope_and_does_not_replace_phase_timing(self):
        item = report('zerosim_pin_op')
        info = item['provenance']['op_information'] = {'acquisition_cost_seconds': 2., 'acquisition_cost_status': 'measured'}
        with self.assertRaisesRegex(ValueError, 'scope'):
            compare_reports([item], False)
        info['acquisition_cost_scope'] = 'all test-split DC solves, wall-clock seconds'
        result = compare_reports([item], False)
        self.assertEqual(result['evaluation_costs'][0]['op_acquisition_cost_seconds'], 2.)
        self.assertEqual(result['op_acquisition_costs'][0]['seconds'], 2.)
        self.assertEqual(result['run_costs'][0]['timing'], {})
        for seconds in (-1, math.nan, math.inf, True):
            info['acquisition_cost_seconds'] = seconds
            with self.assertRaisesRegex(ValueError, 'Measured OP acquisition cost'):
                compare_reports([item], False)

    def test_reject_inference_training_use_contradictions_and_op_scaler_mismatch(self):
        for key, value in (('training_use', 'none'), ('inference_use', 'none')):
            item = report('zerosim_pin_op')
            item['provenance']['op_information'] = {key: value}
            with self.assertRaisesRegex(ValueError, 'Incorrect OP'):
                compare_reports([item], False)
        reports = [report(m) for m in MODES]
        reports[-1]['provenance']['op_input_transform_digest'] = 'different'
        with self.assertRaisesRegex(ValueError, 'OP input scalers'):
            compare_reports(reports)
        del reports[-1]['provenance']['op_input_transform_digest']
        with self.assertRaisesRegex(ValueError, 'OP input scaler provenance'):
            compare_reports(reports)

    def test_reject_discrete_states_and_nontrain_op_scaler_membership(self):
        item = report('zerosim_pin_op')
        item['provenance']['op_information'] = {'state_features': ['saturation_pass']}
        with self.assertRaisesRegex(ValueError, 'discrete OP states'):
            compare_reports([item], False)
        for mode in (*OP_MODES, 'jepa_zerosim'):
            item = report(mode)
            item['provenance']['op_information'] = {'scaler_fitted_sample_ids': ['train1', 'train2', 'test1']}
            with self.assertRaisesRegex(ValueError, 'ID training samples'):
                compare_reports([item], False)

    def test_reject_different_jepa_op_target_protocols_across_seeds(self):
        for key, values in [('state_features', (['current_missing', 'current_below', 'current_pass'], [])),
                            ('feature_names', (['id', 'gm'], ['id'])),
                            ('eligibility', ('MOS', 'all devices')),
                            ('state_config', ({'mode': 'collector_checked'}, {'mode': 'preserve_only'})),
                            ('state_config', ({'thresholds': {'min_signal_current_a': 1e-9}},
                                              {'thresholds': {'min_signal_current_a': 1e-8}}))]:
            reports = [report('jepa_zerosim', seed) for seed in (17, 29)]
            for item, value in zip(reports, values):
                item['provenance']['op_information'] = {key: value}
            with self.assertRaisesRegex(ValueError, 'OP information protocol'):
                compare_reports(reports, expected_modes=['jepa_zerosim'])

    def test_op_coverage_is_preserved_without_treating_split_coverage_as_protocol(self):
        reports = [report(mode, split=split) for mode in MODES for split in ('test', 'ood')]
        for item in reports:
            if item['provenance']['mode'] in OP_MODES:
                item['provenance']['op_information'] = {
                    'source': 'dataset_dc_operating_point', 'feature_names': ['id', 'gm'],
                    'state_features': [], 'eligibility': 'MOS by default',
                    'evaluation_coverage': {'counting_unit': 'original devices, not repeated pin tokens',
                                            'samples_without_observed_op': 1 if item['split'] == 'ood' else 0}}
        result = compare_reports(reports)
        for cost in result['evaluation_costs']:
            if cost['mode'] in OP_MODES:
                self.assertEqual(cost['op_source'], 'dataset_dc_operating_point')
                self.assertEqual(cost['op_feature_names'], ['id', 'gm'])
                self.assertEqual(cost['op_state_features'], [])
                self.assertEqual(cost['op_evaluation_coverage']['samples_without_observed_op'],
                                 1 if cost['split'] == 'ood' else 0)
            else:
                self.assertIsNone(cost['op_evaluation_coverage'])

    def test_reject_different_architecture_within_op_pair(self):
        for base, op in [('zerosim_pin', 'zerosim_pin_op'), ('zerosim_device', 'zerosim_device_op')]:
            reports = [report(base), report(op)]
            reports[-1]['provenance']['model_config']['max_devices'] = 128
            with self.assertRaisesRegex(ValueError, 'OP/no-OP pair'):
                compare_reports(reports, expected_modes=[base, op])
        modes = ['zerosim_device_op', 'jepa_zerosim']
        reports = [report(mode) for mode in modes]
        reports[0]['provenance']['model_config']['terminal_bias'] = False
        with self.assertRaisesRegex(ValueError, 'device OP and JEPA'):
            compare_reports(reports, expected_modes=modes)

    def test_cli_exports_pair_tables_and_blank_unknown_cost_then_clears_stale_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for mode in MODES:
                path = root/f'{mode}.json'
                item = report(mode)
                if mode in OP_MODES:
                    item['provenance']['op_information'] = {'source': 'dataset_dc_operating_point',
                        'feature_names': ['id', 'gm'], 'state_features': [],
                        'evaluation_coverage': {'eligible_devices': 3, 'observed_values': 5}}
                path.write_text(json.dumps(item))
                paths.append(str(path))
            output = root/'comparison'
            main(['--reports', *paths, '--output', str(output)])
            result = json.loads((output/'comparison.json').read_text())
            self.assertEqual(result['schema_version'], 5)
            with (output/'evaluation_costs.csv').open() as handle:
                costs = list(csv.DictReader(handle))
            self.assertTrue(all(row['op_acquisition_cost_seconds'] == '' for row in costs))
            for row in costs:
                if row['mode'] in OP_MODES:
                    self.assertEqual(json.loads(row['op_feature_names']), ['id', 'gm'])
                    self.assertEqual(json.loads(row['op_state_features']), [])
                    self.assertEqual(json.loads(row['op_evaluation_coverage'])['observed_values'], 5)
            with (output/'paired_summary.csv').open() as handle:
                pairs = list(csv.DictReader(handle))
            self.assertEqual({row['node_granularity'] for row in pairs}, {'pin', 'device'})
            self.assertTrue(all(row['mae_delta_std'] == '' for row in pairs))
            main(['--reports', *paths[:3], '--output', str(output)])
            self.assertFalse((output/'paired_summary.csv').exists())
            self.assertFalse((output/'paired_per_seed.csv').exists())

    def test_reject_mismatch(self):
        for key in ['partition_fingerprint', 'input_op_fingerprint', 'performance_fingerprint', 'target_scaler_digest', 'input_transform_digest', 'labeled_train_ids']:
            reports = [report(m) for m in MODES]
            reports[1]['provenance'][key] = 'different'
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                compare_reports(reports)

    def test_reject_supervised_budget_mismatch(self):
        for key in ('baseline_epochs', 'decoder_lr', 'batch_size', 'weight_decay', 'grad_clip'):
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
