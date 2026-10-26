"""Exercise real training termination, selected weights and matched reporting."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from device_jepa import baselines, training
from device_jepa.comparison import MODES, compare_reports
from device_jepa.data import build_manifest, synthetic_dataset
from device_jepa.early_stopping import (EarlyStopping, PRETRAIN_OBJECTIVE, SUPERVISED_OBJECTIVE,
                                       early_stopping_config, executed_epochs)
from device_jepa.states import apply_op_states
from test_epoch_budgets import matched_reports


POLICY = dict(enabled=True, patience=2, min_delta=0.0, min_epochs=1)


def stopped(training_config, maximum, objective, values=(3., 1., 2., 4.)):
    state = EarlyStopping(training_config, maximum, objective)
    for epoch, value in enumerate(values, 1):
        state.update(value, epoch)
    return state.summary()


def early_reports():
    reports = matched_reports()
    for item in reports:
        p = item['provenance']
        p['training_config']['early_stopping'] = dict(POLICY)
        maximum = 50 if p['mode'] == 'jepa_zerosim' else 80
        p['early_stopping'] = stopped(p['training_config'], maximum, SUPERVISED_OBJECTIVE)
        if p['mode'] == 'jepa_zerosim':
            source = p['pretrain_provenance']
            source['source_options']['training']['early_stopping'] = dict(POLICY)
            source['protocol']['early_stopping'] = dict(POLICY)
            source['execution'] = stopped(source['source_options']['training'], 30, PRETRAIN_OBJECTIVE)
            digest = training._json_digest(source['protocol'])
            source['protocol_digest'] = p['pretraining_protocol_digest'] = digest
        pretrain = 4 if p['mode'] == 'jepa_zerosim' else 0
        p['training_cost'] = dict(pretrain_epochs=pretrain, supervised_epochs=4, total_epochs=pretrain+4)
    return reports


class EarlyStoppingTests(unittest.TestCase):
    def test_config_legacy_defaults_and_strict_validation(self):
        self.assertFalse(early_stopping_config({})['enabled'])
        for raw in (None, True, [], {'unknown': 1}, {'enabled': 1}, {'patience': 0},
                    {'patience': True}, {'min_epochs': 0}, {'min_epochs': 1.5},
                    {'min_delta': True}, {'min_delta': -1}, {'min_delta': float('inf')},
                    {'min_delta': float('nan')}):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, 'early_stopping'):
                early_stopping_config({'early_stopping': raw})

    def test_exact_best_kept_even_when_min_delta_does_not_reset_patience(self):
        tr = {'early_stopping': {**POLICY, 'min_delta': 1.}}
        state = EarlyStopping(tr, 20, SUPERVISED_OBJECTIVE)
        for epoch, value in enumerate((10., 9.9, 9.8), 1):
            self.assertTrue(state.update(value, epoch))
        summary = state.summary()
        self.assertTrue(summary['stopped_early'])
        self.assertEqual(summary['best_epoch'], 3)
        self.assertEqual(summary['patience_best_epoch'], 1)
        self.assertEqual(executed_epochs(summary, tr, 20, SUPERVISED_OBJECTIVE), 3)

    def test_significant_improvement_resets_patience_and_minimum_epochs_is_honored(self):
        tr = {'early_stopping': {**POLICY, 'min_delta': 1., 'min_epochs': 6}}
        state = EarlyStopping(tr, 10, PRETRAIN_OBJECTIVE)
        for epoch, value in enumerate((10., 9.9, 8., 8.1, 8.2, 8.3), 1):
            state.update(value, epoch)
            self.assertEqual(state.should_stop, epoch == 6)
        self.assertEqual(state.best_epoch, 3)
        self.assertEqual(executed_epochs(state.summary(), tr, 10, PRETRAIN_OBJECTIVE), 6)

    def test_fixed_budget_short_stage_and_disabled_rules(self):
        for policy in ({**POLICY, 'min_epochs': 10}, {**POLICY, 'enabled': False}):
            tr = {'early_stopping': policy}
            state = EarlyStopping(tr, 3, SUPERVISED_OBJECTIVE)
            for epoch in range(1, 4):
                state.update(1., epoch)
                self.assertFalse(state.should_stop)
            self.assertEqual(state.best_epoch, 1)
            self.assertEqual(state.summary()['stop_reason'], 'max_epochs')
            self.assertEqual(executed_epochs(state.summary(), tr, 3, SUPERVISED_OBJECTIVE), 3)

    def test_nonfinite_and_incomplete_state_are_not_silently_finalized(self):
        for value in (float('nan'), float('inf'), True):
            with self.assertRaisesRegex(ValueError, 'finite'):
                EarlyStopping({}, 2, PRETRAIN_OBJECTIVE).update(value, 1)
        state = EarlyStopping({}, 2, PRETRAIN_OBJECTIVE)
        state.update(1., 1)
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            state.summary()
        with self.assertRaisesRegex(ValueError, 'sequential'):
            state.update(1., 1)

    def test_comparison_keeps_equal_maxima_with_different_actual_totals(self):
        result = compare_reports(early_reports())
        self.assertTrue(result['complete'])
        self.assertEqual(result['epoch_budget_policy'], 'equal_maximum_with_early_stopping')
        self.assertEqual(result['max_total_epochs'], 80)
        self.assertEqual({r['planned_total_epochs'] for r in result['run_costs']}, {80})
        self.assertEqual({r['total_epochs'] for r in result['run_costs']}, {4, 8})

    def test_execution_outcomes_can_differ_across_seeds_without_changing_protocol(self):
        first = early_reports()
        second = copy.deepcopy(first)
        for item in second:
            p = item['provenance']
            p.update(seed=29, decoder_seed=29, run_identity=p['run_identity'].replace('/17', '/29'))
            p['training_config']['seed'] = 29
            if p['mode'] == 'jepa_zerosim':
                p['pretrain_seed'] = 29
                source = p['pretrain_provenance']
                source['seed'] = source['source_options']['training']['seed'] = 29
                source['execution'] = stopped(source['source_options']['training'], 30, PRETRAIN_OBJECTIVE,
                                               (1., 2., 3.))
                p['training_cost'].update(pretrain_epochs=3, total_epochs=7)
        result = compare_reports(first + second)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['run_costs']), 10)

    def test_mixed_policy_and_forged_execution_are_rejected(self):
        for mutation in ('policy', 'missing', 'epoch_count', 'reason', 'cost', 'config', 'pretraining'):
            reports = early_reports()
            p = next(r['provenance'] for r in reports if r['provenance']['mode'] == 'zerosim_device')
            if mutation == 'policy':
                p['training_config']['early_stopping']['patience'] = 3
            elif mutation == 'missing':
                del p['early_stopping']
            elif mutation == 'epoch_count':
                p['early_stopping']['epochs_ran'] = 80
            elif mutation == 'reason':
                p['early_stopping']['stop_reason'] = 'max_epochs'
            elif mutation == 'cost':
                p['training_cost']['supervised_epochs'] = 80
            elif mutation == 'config':
                p['early_stopping']['config']['min_delta'] = .1
            else:
                p = next(r['provenance'] for r in reports if r['provenance']['mode'] == 'jepa_zerosim')
                p['pretrain_provenance']['execution']['config']['enabled'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_reports(reports)

    def test_invalid_policy_is_rejected_before_output_or_children(self):
        from test_epoch_budgets import load_runner
        runner = load_runner('early_stop_runner', 'run_raw_comparison.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((Path(__file__).parents[1] / 'configs/default.json').read_text())
            config['training']['early_stopping'] = {'patience': True}
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            with patch.object(runner, 'source_identity', return_value=('circuit', 10)), \
                 patch.object(runner.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'patience'):
                    runner.main(['--input', str(root), '--output', str(root / 'out'), '--config', str(path)])
                child.assert_not_called()
            self.assertFalse((root / 'out').exists())


class EarlyStoppingTrainingTests(unittest.TestCase):
    def exercise_training(self, device):
        config = json.loads((Path(__file__).parents[1] / 'configs/smoke.json').read_text())
        config['model'].update(dim=8, heads=2, ffn_dim=16, decoder_layers=1,
                               decoder_ffn_dim=16, dropout=0., max_devices=128)
        config['training'].update(pretrain_epochs=6, decoder_epochs=6, baseline_epochs=12,
                                  batch_size=128, early_stopping=dict(POLICY))
        data = apply_op_states(synthetic_dataset(8), config['op_state'])
        manifest = build_manifest(data, config=config['dataset'])

        def validation_objective(original, pretraining=False):
            values = iter((3., 1., 2., 4.))

            def objective(*args, **kwargs):
                result = original(*args, **kwargs)
                if torch.is_grad_enabled():
                    return result
                value = torch.tensor(next(values), device=device)
                return (value, result[1]) if pretraining else value
            return objective

        reports = []
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as log:
            root = Path(tmp)
            with patch.object(training, 'alignment_loss',
                    side_effect=validation_objective(training.alignment_loss, True)):
                # Preserve the function signature used to resolve effective loss kwargs.
                with patch.object(training, '_effective_alignment_loss', return_value=config['loss']):
                    source = training.pretrain(data, manifest, config, root / 'jepa', device)
            pre = training.load_checkpoint(source)
            self.assertEqual(pre['epoch'], 2)
            self.assertEqual(pre['training_cost']['pretrain_epochs'], 4)
            self.assertEqual(pre['training_cost']['pretrain_optimizer_steps'], 4)
            for mode in MODES:
                with self.subTest(mode=mode, device=device):
                    if mode == 'jepa_zerosim':
                        with patch.object(training, 'masked_mse',
                                side_effect=validation_objective(training.masked_mse)):
                            path = training.train_decoder(data, source, config, root / 'jepa', device)
                        history_path = root / 'jepa/decoder_history.json'
                        evaluator = training.evaluate
                    else:
                        with patch.object(baselines, 'masked_mse',
                                side_effect=validation_objective(baselines.masked_mse)):
                            path = baselines.train_baseline(data, manifest, config, root / mode, mode, device)
                        history_path = root / mode / f'{mode}_history.json'
                        evaluator = baselines.evaluate_baseline
                    cp = training.load_checkpoint(path)
                    history = json.loads(history_path.read_text())
                    self.assertEqual(len(history), 4)
                    self.assertEqual(cp['epoch'], 2)
                    self.assertEqual(cp['selection']['value'], 1.)
                    self.assertTrue(cp['early_stopping']['stopped_early'])
                    self.assertEqual(cp['training_cost']['supervised_epochs'], 4)
                    self.assertEqual(cp['training_cost']['supervised_optimizer_steps'], 4)
                    if mode == 'jepa_zerosim':
                        self.assertEqual(cp['encoder_digest'], pre['encoder_digest'])
                        self.assertTrue(cp['freeze_verified'])
                        self.assertEqual(cp['training_cost']['total_epochs'], 8)
                        self.assertEqual(cp['pretrain_provenance']['execution'], pre['early_stopping'])
                    for split in ('test', 'ood'):
                        reports.append(evaluator(data, path, split, device=device))
            self.assertIn('early stopping at 4/6', log.getvalue())
            self.assertIn('early stopping at 4/12', log.getvalue())
            result = compare_reports(reports)
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['run_costs']), 5)
            self.assertEqual({r['planned_total_epochs'] for r in result['run_costs']}, {12})
            self.assertEqual({r['total_epochs'] for r in result['run_costs']}, {4, 8})

    def test_cpu_all_five_modes_stop_restore_and_report_actual_cost(self):
        self.exercise_training('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_all_five_modes_stop_restore_and_report_actual_cost(self):
        self.exercise_training('cuda')


if __name__ == '__main__':
    unittest.main()
