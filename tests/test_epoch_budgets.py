"""Matched experiments use independent stage settings and executed total epochs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from device_jepa.budgets import comparison_epoch_budget, positive_epochs, smoke_epochs
from device_jepa.comparison import compare_reports
from device_jepa.methods import DEFAULT_MODELS as MODES
from test_comparison import report


ROOT = Path(__file__).resolve().parents[1]


def load_runner(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/run' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matched_reports(pretrain=30, decoder=50, baseline=80):
    reports = [report(mode) for mode in MODES]
    for item in reports:
        provenance = item['provenance']
        provenance['training_config'].update(pretrain_epochs=pretrain,
            decoder_epochs=decoder, baseline_epochs=baseline)
        if provenance['mode'] == 'jepa_zerosim':
            source = provenance['pretrain_provenance']
            source['protocol']['training']['pretrain_epochs'] = pretrain
            source['source_options']['training']['pretrain_epochs'] = pretrain
            digest = hashlib.sha256(json.dumps(source['protocol'], sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
            source['protocol_digest'] = provenance['pretraining_protocol_digest'] = digest
    return reports


class EpochBudgetTests(unittest.TestCase):
    def test_epoch_fields_require_explicit_positive_integers(self):
        for key in ('pretrain_epochs', 'decoder_epochs', 'baseline_epochs'):
            for value in (None, True, False, 0, -1, 1.0, '1', float('nan')):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        positive_epochs({key: value}, key)
            with self.assertRaisesRegex(ValueError, key):
                positive_epochs({}, key)
            self.assertEqual(positive_epochs({key: 3}, key), 3)

    def test_total_budget_is_equal_without_equal_supervised_budgets(self):
        options = {'training': {'pretrain_epochs': 30, 'decoder_epochs': 50,
                                'baseline_epochs': 80}}
        before = copy.deepcopy(options)
        budgets = comparison_epoch_budget(options, MODES)
        self.assertEqual(options, before)
        self.assertEqual({budget['total_epochs'] for budget in budgets.values()}, {80})
        self.assertEqual(budgets['jepa_zerosim']['supervised_epochs'], 50)
        self.assertEqual(budgets['zerosim_device']['supervised_epochs'], 80)
        options['training']['baseline_epochs'] = 50
        with self.assertRaisesRegex(ValueError, 'total training budget'):
            comparison_epoch_budget(options, MODES)
        self.assertEqual(options['training']['baseline_epochs'], 50)

    def test_smoke_uses_two_total_epochs_and_preserves_other_parameters(self):
        options = {'training': {'pretrain_epochs': 30, 'decoder_epochs': 50,
                                'baseline_epochs': 80, 'batch_size': 32},
                   'model': {'dim': 128}}
        smoke_epochs(options)
        self.assertEqual(options['training'], dict(pretrain_epochs=1, decoder_epochs=1,
            baseline_epochs=2, batch_size=32))
        self.assertEqual(options['model'], {'dim': 128})
        self.assertEqual({b['total_epochs'] for b in
                          comparison_epoch_budget(options, MODES).values()}, {2})

    def test_comparison_accepts_equal_totals_and_exposes_stage_counts(self):
        result = compare_reports(matched_reports())
        self.assertTrue(result['complete'])
        costs = {row['mode']: row for row in result['run_costs']}
        for mode in MODES:
            self.assertEqual(costs[mode]['total_epochs'], 80)
            self.assertEqual(costs[mode]['pretrain_epochs'], 30 if mode == 'jepa_zerosim' else 0)
            self.assertEqual(costs[mode]['supervised_epochs'], 50 if mode == 'jepa_zerosim' else 80)

    def test_equal_decoder_and_baseline_epochs_no_longer_certifies_equal_total(self):
        with self.assertRaisesRegex(ValueError, 'total training budget'):
            compare_reports(matched_reports(baseline=50))

    def test_actual_pretraining_budget_ignores_stage_two_wishes_and_best_epoch(self):
        reports = matched_reports()
        jepa = next(r['provenance'] for r in reports if r['provenance']['mode'] == 'jepa_zerosim')
        jepa['training_config']['pretrain_epochs'] = 999
        jepa['pretrain_provenance']['selected_epoch'] = 1
        jepa['epoch'] = 1
        result = compare_reports(reports)
        self.assertTrue(result['complete'])
        cost = next(row for row in result['run_costs'] if row['mode'] == 'jepa_zerosim')
        self.assertEqual(cost['total_epochs'], 80)
        for item in reports:
            item['provenance']['training_config']['baseline_epochs'] = 51
        jepa['training_config']['pretrain_epochs'] = 1
        with self.assertRaisesRegex(ValueError, 'total training budget'):
            compare_reports(reports)

    def test_missing_explicit_baseline_budget_is_legacy_incomplete(self):
        reports = matched_reports()
        baseline = next(r['provenance'] for r in reports if r['provenance']['mode'] == 'zerosim_device')
        del baseline['training_config']['baseline_epochs']
        result = compare_reports(reports)
        self.assertFalse(result['complete'])
        self.assertTrue(result['cohort_complete'])
        self.assertTrue(any('baseline_epochs' in gap for gap in result['provenance_gaps']))
        cost = next(row for row in result['run_costs'] if row['mode'] == 'zerosim_device')
        self.assertIsNone(cost['total_epochs'])

    def test_missing_executed_pretraining_budget_is_legacy_incomplete(self):
        reports = matched_reports()
        jepa = next(r['provenance'] for r in reports if r['provenance']['mode'] == 'jepa_zerosim')
        del jepa['pretrain_provenance']
        del jepa['pretraining_protocol_digest']
        result = compare_reports(reports)
        self.assertFalse(result['complete'])
        self.assertTrue(any('executed pretraining epoch budget' in gap
                            for gap in result['provenance_gaps']))
        cost = next(row for row in result['run_costs'] if row['mode'] == 'jepa_zerosim')
        self.assertIsNone(cost['total_epochs'])

    def test_recorded_cost_cannot_contradict_the_executed_budget(self):
        for mode in ('jepa_zerosim', 'zerosim_device'):
            for field in ('pretrain_epochs', 'supervised_epochs', 'total_epochs'):
                with self.subTest(mode=mode, field=field):
                    reports = matched_reports()
                    provenance = next(r['provenance'] for r in reports if r['provenance']['mode'] == mode)
                    provenance['training_cost'] = {field: 999}
                    with self.assertRaisesRegex(ValueError, 'differs from executed training budget'):
                        compare_reports(reports)

    def test_malformed_report_epoch_values_are_rejected(self):
        for mode, field in (('jepa_zerosim', 'decoder_epochs'),
                            ('zerosim_device', 'baseline_epochs')):
            for value in (True, 0, -1, 1.0, '80'):
                with self.subTest(mode=mode, field=field, value=value):
                    reports = matched_reports()
                    provenance = next(r['provenance'] for r in reports if r['provenance']['mode'] == mode)
                    provenance['training_config'][field] = value
                    with self.assertRaisesRegex(ValueError, 'training budget'):
                        compare_reports(reports)

    def test_canonical_runner_rejects_bad_budget_before_output_or_children(self):
        runner = load_runner('epoch_budget_canonical_runner', 'run_comparison.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((ROOT / 'configs/default.json').read_text())
            config['training'].update(pretrain_epochs=30, decoder_epochs=50, baseline_epochs=50)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            output = root / 'results'
            argv = ['run_comparison.py', '--data', str(root / 'data.json'),
                    '--config', str(config_path), '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'total training budget'):
                    runner.main()
                child.assert_not_called()
            self.assertFalse(output.exists())

    def test_raw_runner_rejects_bad_budget_before_import_or_output(self):
        runner = load_runner('epoch_budget_raw_runner', 'run_raw_comparison.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'raw'
            source.mkdir()
            config = json.loads((ROOT / 'configs/default.json').read_text())
            config['training'].update(pretrain_epochs=30, decoder_epochs=50, baseline_epochs=50)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            output = root / 'results'
            with patch.object(runner, 'source_identity', return_value=('Circuit', 60000)), \
                 patch.object(runner.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'total training budget'):
                    runner.main(['--input', str(source), '--config', str(config_path),
                                 '--output', str(output)])
                child.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
