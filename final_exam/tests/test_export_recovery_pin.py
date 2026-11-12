"""Recover completed metrics after export failure without loading training data."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from final_exam.summarize_pin import export_results_pin
from final_exam.tests.test_summary_pin import record, text_values, workbook, write_run

ROOT = Path(__file__).resolve().parents[2]


def snapshot(directory):
    return {path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


class ExportRecoveryPinTests(unittest.TestCase):
    def test_cli_recovers_without_site_packages_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(root / 'training_pin', [record(), record('gat_pin')],
                                  status='failed_export')
            before = snapshot(root)
            result = subprocess.run(
                [sys.executable, '-S', str(ROOT / 'final_exam/summarize_pin.py'),
                 str(root), '--recover-export'], capture_output=True, text=True, check=True)
            output = directory / 'summary_pin.xlsx'
            self.assertEqual(result.stdout.strip(), str(output))
            after = snapshot(root)
            del after[output.relative_to(root)]
            self.assertEqual(before, after)
            sheets = workbook(output)
            self.assertIn('STATUS: COMPLETE — export recovered; original job failed at export',
                          text_values(sheets['MAE']))
            self.assertTrue(any('EXPORT RECOVERED' in value for value in text_values(sheets['Configuration'])))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                export_results_pin(root, recover_export=True)

    def test_failed_export_is_opt_in_and_not_a_training_recovery_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for status in ('failed', 'running', 'ready_to_export', 'complete', 'failed_export'):
                with self.subTest(status=status):
                    directory = write_run(root / status, [record()], status=status)
                    before = snapshot(directory)
                    if status == 'failed_export':
                        with self.assertRaisesRegex(ValueError, 'Incomplete experiment'):
                            export_results_pin(directory)
                        for options in ({'allow_incomplete': True}, {'finalizing': True}):
                            with self.assertRaisesRegex(ValueError, 'cannot be combined'):
                                export_results_pin(directory, recover_export=True, **options)
                    else:
                        with self.assertRaisesRegex(ValueError, 'requires status=failed_export'):
                            export_results_pin(directory, recover_export=True)
                    self.assertEqual(snapshot(directory), before)

    def test_recovery_retains_all_completeness_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for missing in ('model', 'seed', 'config', 'stage', 'gate_comparison'):
                with self.subTest(missing=missing):
                    directory = root / missing
                    if missing == 'gate_comparison':
                        from final_exam.tests.test_gate_v2_summary_pin import versioned_run
                        versioned_run(directory, seeds=(17,))
                        metadata = json.loads((directory / 'run_pin.json').read_text())
                        metadata['status'] = 'failed_export'
                        (directory / 'run_pin.json').write_text(json.dumps(metadata))
                        next(directory.rglob('gate_comparison_pin.json')).unlink()
                    else:
                        write_run(directory, [record()], status='failed_export',
                                  models=['mlp_pin', 'gat_pin'] if missing == 'model' else None,
                                  seeds=[17, 29] if missing == 'seed' else None)
                        if missing == 'config':
                            (directory / 'seed_17_pin/mlp_pin/config_pin.json').unlink()
                        if missing == 'stage':
                            (directory / 'stages_pin.json').write_text('{"unique_stages": []}')
                    before = snapshot(directory)
                    with self.assertRaises(ValueError):
                        export_results_pin(directory, recover_export=True)
                    self.assertEqual(snapshot(directory), before)

    def test_workbook_failure_does_not_mark_run_complete_or_leave_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_run(Path(tmp), [record()], status='failed_export')
            before = snapshot(directory)
            with patch('final_exam.summarize_pin._write_workbook', side_effect=OSError('disk unavailable')):
                with self.assertRaisesRegex(OSError, 'disk unavailable'):
                    export_results_pin(directory, recover_export=True)
            self.assertEqual(snapshot(directory), before)
            self.assertFalse(list(directory.glob('.final-exam-report-pin-*')))

    def test_stale_registry_fails_on_runner_import_not_after_training(self):
        code = (
            'import final_exam.gate_version_registry_pin as registry\n'
            'del registry.V3_GATE_MODELS_PIN\n'
            'import final_exam.run_pin\n'
        )
        result = subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Pin report imports failed before training', result.stderr)
        self.assertIn('V3_GATE_MODELS_PIN', result.stderr)

    def test_runner_pins_reporter_before_registry_changes(self):
        code = (
            'import sys\n'
            'import final_exam.run_pin as runner\n'
            'import final_exam.gate_version_registry_pin as registry\n'
            'assert "final_exam.summarize_pin" in sys.modules\n'
            'reporter = runner.export_results_pin\n'
            'del registry.V3_GATE_MODELS_PIN\n'
            'assert runner.export_results_pin is reporter\n'
            'assert "torch" not in sys.modules\n'
        )
        subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT,
                       capture_output=True, text=True, check=True)

    def test_actual_v2_export_failure_recovers_without_any_training_or_loading(self):
        import torch
        from final_exam.run_pin import run_experiment_pin
        from final_exam.tests.test_run_pin import tiny_inputs
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, data_path, config_path = tiny_inputs(root)
            output = root / 'training_pin'
            models = ['device_op_cascade_pin', 'device_op_distill_gate_pin_v2']
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                    patch('final_exam.run_pin.export_results_pin', side_effect=ImportError('simulated export mismatch')):
                with self.assertRaisesRegex(ImportError, 'simulated export mismatch'):
                    run_experiment_pin(data_path, config_path, output, models, [17], 'cpu', smoke=True)
            self.assertEqual(json.loads((output / 'run_pin.json').read_text())['status'], 'failed_export')
            self.assertEqual(len(json.loads((output / 'results_pin.json').read_text())), len(models))
            for mode in models:
                self.assertTrue((output / 'seed_17_pin' / mode / 'checkpoint_pin.pt').is_file())
            before = snapshot(root)
            with patch('experiments.op_transfer.engine.fit_stage', side_effect=AssertionError('must not retrain')), \
                    patch('device_jepa.data.load_dataset', side_effect=AssertionError('must not load dataset')), \
                    patch('torch.load', side_effect=AssertionError('must not load checkpoints')):
                excel = export_results_pin(output, recover_export=True)
            self.assertIn('GateBeforeAfter', workbook(excel))
            after = snapshot(root)
            del after[excel.relative_to(root)]
            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
