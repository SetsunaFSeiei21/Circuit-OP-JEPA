"""Regression tests for independently reproduced comparison failures."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from device_jepa.comparison import compare_reports, main, regression_metrics
from test_comparison import report


def exact_r2(target, prediction):
    y, p = [Fraction(v) for v in target], [Fraction(v) for v in prediction]
    center = sum(y) / len(y)
    return float(1 - sum((a-b)**2 for a, b in zip(y, p)) / sum((v-center)**2 for v in y))


def sign_protocol(item):
    source = item['provenance']['pretrain_provenance']
    digest = hashlib.sha256(json.dumps(source['protocol'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    source['protocol_digest'] = item['provenance']['pretraining_protocol_digest'] = digest


class StableR2AuditTests(unittest.TestCase):
    def test_adjacent_float_targets_match_exact_binary_reference(self):
        for first in (1., 1e9, 1e16, 1e-200, 1e200):
            y = [first, math.nextafter(first, math.inf)]
            p = [first, first]
            with self.subTest(first=first):
                self.assertAlmostEqual(regression_metrics(y, p)['r2'], exact_r2(y, p), places=14)

    def test_subnormal_spread_does_not_need_representable_physical_std(self):
        tiny = math.ulp(0.)
        y, p = [0., tiny], [0., 0.]
        self.assertAlmostEqual(regression_metrics(y, p)['r2'], -1., places=14)
        self.assertEqual(regression_metrics(y, y)['r2'], 1.)

    def test_large_opposite_sign_targets_and_extreme_scales(self):
        self.assertEqual(regression_metrics([-1e308, 1e308], [0., 0.])['r2'], 0.)
        for scale in (1e-200, 1e200):
            result = regression_metrics([0., scale], [scale, 2*scale])
            self.assertTrue(math.isclose(result['rmse'], scale, rel_tol=1e-14))
            self.assertAlmostEqual(result['r2'], -3., places=14)
        with self.assertRaisesRegex(ValueError, 'reporting range'):
            regression_metrics([-1e308], [1e308])
        with self.assertRaisesRegex(ValueError, 'R2 exceeds'):
            regression_metrics([0., 1e-300], [1e300, 1e300])


class CircuitScopeAuditTests(unittest.TestCase):
    def test_reserved_name_remains_a_distinct_circuit_and_does_not_duplicate_seeds(self):
        modes = ['zerosim_device', 'zerosim_device_op']
        for seeds in ([17], [17, 29]):
            reports = [report(mode, seed) for mode in modes for seed in seeds]
            for item in reports:
                for row in item['predictions']:
                    row['circuit_id'] = '__all__'
                extra = copy.deepcopy(item['predictions'][0])
                extra.update(sample_id='other', circuit_id='normal_circuit')
                extra['target'] = extra['normalized_target'] = {'gain': 9.}
                item['predictions'].append(extra)
                item['num_samples'] += 1
            result = compare_reports(reports, expected_modes=modes)
            self.assertTrue(result['complete'])
            self.assertTrue(all(row['num_seeds'] == len(seeds) for row in result['summary']))
            self.assertTrue(all(row['num_paired_seeds'] == len(seeds) for row in result['paired_summary']))
            pooled = [r for r in result['summary'] if r['scope'] == 'pooled']
            circuit = [r for r in result['summary'] if r['scope'] == 'circuit' and r['circuit'] == '__all__']
            self.assertTrue(all(r['count_per_seed'] == 3 for r in pooled))
            self.assertTrue(all(r['count_per_seed'] == 2 for r in circuit))
            self.assertTrue(all(r['rmse_mean'] == 1. for r in circuit))
            if len(seeds) == 1:
                self.assertTrue(all(r['rmse_std'] is None for r in result['summary']))


class ExecutedProvenanceAuditTests(unittest.TestCase):
    def test_different_actual_pretraining_protocol_rejected_after_same_stage2_options(self):
        for field, value in [('loss', {'invariance': 0.}),
                             ('training', {'pretrain_epochs': 2, 'pretrain_lr': .0005}),
                             ('model_config', {'dim': 64})]:
            reports = [report('jepa_zerosim', seed) for seed in (17, 29)]
            source = reports[-1]['provenance']['pretrain_provenance']
            source['protocol'][field].update(value)
            source['source_options']['model' if field == 'model_config' else field].update(value)
            sign_protocol(reports[-1])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Actual pretraining protocol'):
                compare_reports(reports, expected_modes=['jepa_zerosim'])

    def test_mismatched_source_digest_or_seed_rejected(self):
        item = report('jepa_zerosim')
        item['provenance']['pretrain_provenance']['protocol']['loss']['invariance'] = 0.
        with self.assertRaisesRegex(ValueError, 'digest does not match|loss differs from source'):
            compare_reports([item], expected_modes=['jepa_zerosim'])
        item = report('jepa_zerosim')
        item['provenance']['pretrain_seed'] = 29
        with self.assertRaisesRegex(ValueError, 'seed differs'):
            compare_reports([item], expected_modes=['jepa_zerosim'])

    def test_unknown_legacy_source_is_readable_but_never_strictly_complete(self):
        reports = [report('jepa_zerosim', split=split) for split in ('test', 'ood')]
        for item in reports:
            for key in ('run_identity', 'pretraining_protocol_digest', 'pretrain_provenance', 'pretrain_seed'):
                item['provenance'].pop(key, None)
        result = compare_reports(reports, expected_modes=['jepa_zerosim'])
        self.assertTrue(result['cohort_complete'])
        self.assertFalse(result['complete'])
        self.assertFalse(result['strict_provenance'])
        self.assertTrue(result['provenance_gaps'])
        self.assertEqual(len(result['run_costs']), 2)
        self.assertTrue(all(r['identity_status'] == 'legacy_unverified' for r in result['run_costs']))

    def test_incomplete_protocol_and_selection_metadata_are_not_strict(self):
        item = report('jepa_zerosim')
        del item['provenance']['pretrain_provenance']['protocol']['loss']
        del item['provenance']['selection_metric_names']
        sign_protocol(item)
        result = compare_reports([item], expected_modes=['jepa_zerosim'])
        self.assertFalse(result['complete'])
        self.assertTrue(any('incomplete executed' in gap for gap in result['provenance_gaps']))

    def test_different_selection_metrics_rejected(self):
        reports = [report('zerosim_device', seed) for seed in (17, 29)]
        reports[-1]['provenance']['selection_metric_names'] = []
        with self.assertRaisesRegex(ValueError, 'checkpoint-selection'):
            compare_reports(reports, expected_modes=['zerosim_device'])

    def test_empty_nested_protocol_is_not_strict_even_with_matching_digest(self):
        item = report('jepa_zerosim')
        source = item['provenance']['pretrain_provenance']
        source['protocol']['loss'] = {}
        source['protocol']['training'] = {}
        sign_protocol(item)
        result = compare_reports([item], expected_modes=['jepa_zerosim'])
        self.assertFalse(result['complete'])
        self.assertFalse(result['strict_provenance'])
        self.assertTrue(any('protocol loss fields' in gap for gap in result['provenance_gaps']))
        self.assertTrue(any('protocol training fields' in gap for gap in result['provenance_gaps']))

    def test_nested_protocol_requires_real_finite_numeric_fields(self):
        for section, key, value in [('loss', 'variance', True), ('loss', 'invariance', '25'),
                                    ('loss', 'variance', 0.), ('training', 'pretrain_epochs', 1.5),
                                    ('training', 'batch_size', 0), ('training', 'pretrain_lr', -1.)]:
            item = report('jepa_zerosim')
            item['provenance']['pretrain_provenance']['protocol'][section][key] = value
            sign_protocol(item)
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, 'Invalid executed'):
                compare_reports([item], expected_modes=['jepa_zerosim'])

    def test_explicit_null_nested_protocol_is_rejected(self):
        for section in ('loss', 'training', 'model_config', 'op_state', 'feature_spec', 'normalization',
                        'optimizer', 'architecture', 'protocol_version', 'input_contract_version'):
            item = report('jepa_zerosim')
            item['provenance']['pretrain_provenance']['protocol'][section] = None
            sign_protocol(item)
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'must be a'):
                compare_reports([item], expected_modes=['jepa_zerosim'])

    def test_unsupported_protocol_version_is_not_strict(self):
        item = report('jepa_zerosim')
        item['provenance']['pretrain_provenance']['protocol']['protocol_version'] = 2
        sign_protocol(item)
        result = compare_reports([item], expected_modes=['jepa_zerosim'])
        self.assertFalse(result['complete'])
        self.assertTrue(any('unsupported' in gap for gap in result['provenance_gaps']))

    def test_effective_loss_defaults_match_raw_source_options(self):
        item = report('jepa_zerosim')
        options = item['provenance']['pretrain_provenance']['source_options']
        del options['loss']['invariance']
        del options['loss']['variance_floor']
        del options['training']['torch_threads']
        result = compare_reports([item], expected_modes=['jepa_zerosim'])
        self.assertTrue(result['complete'])

    def test_conflicting_executed_source_options_rejected(self):
        for section, key, value in [('loss', 'invariance', 0.), ('training', 'pretrain_epochs', 2),
                                    ('training', 'seed', 29), ('model', 'dim', 64)]:
            item = report('jepa_zerosim')
            item['provenance']['pretrain_provenance']['source_options'][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, 'differs from source|seed differs'):
                compare_reports([item], expected_modes=['jepa_zerosim'])


class RunCostsAuditTests(unittest.TestCase):
    def reports(self):
        reports = [report('jepa_zerosim', split=split) for split in ('test', 'ood')]
        for item in reports:
            item['provenance']['timing'] = {'pretrain_seconds': 10., 'supervised_seconds': 2.}
            item['provenance']['training_cost'] = {'pretrain_optimizer_steps': 5, 'supervised_optimizer_steps': 2}
            item['provenance']['parameter_count'] = {'total': 100}
            item['provenance']['op_information'] = {
                'evaluation_coverage': {'observed_values': 2 if item['split'] == 'test' else 3}}
        return reports

    def test_one_run_has_one_training_cost_and_separate_split_coverage(self):
        result = compare_reports(self.reports(), expected_modes=['jepa_zerosim'])
        self.assertEqual(len(result['run_costs']), 1)
        cost = result['run_costs'][0]
        self.assertEqual(cost['timing']['pretrain_seconds'], 10.)
        self.assertEqual(cost['pretrain_optimizer_steps'], 5)
        self.assertEqual(cost['supervised_optimizer_steps'], 2)
        self.assertEqual(cost['evaluation_splits'], ['ood', 'test'])
        self.assertNotIn('op_evaluation_coverage', cost)
        self.assertEqual({r['split']: r['op_evaluation_coverage']['observed_values']
                          for r in result['evaluation_costs']}, {'test': 2, 'ood': 3})
        self.assertEqual(result['op_acquisition_costs'], [])
        self.assertTrue(all(r['op_acquisition_cost_seconds'] is None for r in result['evaluation_costs']))

    def test_same_run_conflicts_and_mixed_checkpoints_rejected(self):
        for field, value in [('timing', {'pretrain_seconds': 20.}),
                             ('training_cost', {'pretrain_optimizer_steps': 10}),
                             ('parameter_count', {'total': 200}), ('run_identity', 'another_checkpoint')]:
            reports = self.reports()
            reports[-1]['provenance'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'run identit'):
                compare_reports(reports, expected_modes=['jepa_zerosim'])

    def test_acquisition_costs_preserve_distinct_scopes_and_deduplicate_only_matching_scope(self):
        reports = self.reports()
        for item in reports:
            item['provenance']['op_information'].update(acquisition_cost_status='measured',
                acquisition_cost_seconds=4., acquisition_cost_scope='DC solves for ' + item['split'])
        result = compare_reports(reports, expected_modes=['jepa_zerosim'])
        self.assertEqual(len(result['run_costs']), 1)
        self.assertEqual(len(result['op_acquisition_costs']), 2)
        for item in reports:
            item['provenance']['op_information']['acquisition_cost_scope'] = 'shared DC preparation'
        result = compare_reports(reports, expected_modes=['jepa_zerosim'])
        self.assertEqual(len(result['op_acquisition_costs']), 1)
        self.assertEqual(result['op_acquisition_costs'][0]['seconds'], 4.)
        reports[-1]['provenance']['op_information']['acquisition_cost_seconds'] = 8.
        with self.assertRaisesRegex(ValueError, 'Conflicting OP acquisition'):
            compare_reports(reports, expected_modes=['jepa_zerosim'])

    def test_csv_exports_split_costs_separately_and_clears_stale_measured_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports, paths = self.reports(), []
            for i, item in enumerate(reports):
                item['provenance']['op_information'].update(acquisition_cost_status='measured',
                    acquisition_cost_seconds=4., acquisition_cost_scope='shared DC preparation')
                path = root / f'{i}.json'
                path.write_text(json.dumps(item))
                paths.append(str(path))
            out = root / 'out'
            args = ['--reports', *paths, '--output', str(out), '--modes', 'jepa_zerosim']
            main(args)
            self.assertTrue((out / 'op_acquisition_costs.csv').exists())
            self.assertTrue((out / 'evaluation_costs.csv').exists())
            for path, item in zip(paths, reports):
                for key in ('acquisition_cost_status', 'acquisition_cost_seconds', 'acquisition_cost_scope'):
                    del item['provenance']['op_information'][key]
                Path(path).write_text(json.dumps(item))
            main(args)
            self.assertFalse((out / 'op_acquisition_costs.csv').exists())


if __name__ == '__main__':
    unittest.main()
