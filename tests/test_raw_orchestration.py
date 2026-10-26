"""Raw circuit jobs keep import on the worker and isolate independent datasets."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from test_adapter import fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('raw_runner', ROOT / 'scripts/run/run_raw_comparison.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob('*') if p.is_file()}


class RawOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source folder not a circuit name'
        self.source.mkdir()
        fixture(self.source, circuit='ActualCircuit')
        self.output = self.root / 'new job' / 'circuit'
        self.args = ['--input', str(self.source), '--output', str(self.output)]

    def invoke(self, args=None, fail_import=False, complete=True):
        calls = []
        def child(command, **kwargs):
            calls.append(command)
            self.assertEqual(kwargs['cwd'], ROOT)
            self.assertTrue(kwargs['check'])
            if '-m' in command:
                if fail_import:
                    raise subprocess.CalledProcessError(1, command)
                Path(command[command.index('--output') + 1]).write_text('{"mock": true}')
            else:
                output = Path(command[command.index('--output') + 1])
                (output / 'comparison').mkdir(parents=True)
                (output / 'comparison/comparison.json').write_text(json.dumps({'complete': complete}))
            return subprocess.CompletedProcess(command, 0)
        with patch.object(runner.subprocess, 'run', side_effect=child), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main(args or self.args)
        return calls

    def test_import_precedes_training_and_uses_sqlite_identity_with_default_budgets(self):
        before = snapshot(self.source)
        calls = self.invoke()
        config = json.loads((self.output / 'config.json').read_text())
        self.assertEqual(config['dataset']['train_circuits'], ['ActualCircuit'])
        self.assertEqual(config['dataset']['ood_circuits'], [])
        self.assertEqual(config['training']['pretrain_epochs'], 30)
        self.assertEqual(config['training']['decoder_epochs'], 50)
        self.assertEqual(config['training']['baseline_epochs'], 80)
        self.assertEqual(calls[0][1:4], ['-m', 'device_jepa', 'import_analog_agent'])
        self.assertEqual(calls[0][calls[0].index('--input') + 1], str(self.source.resolve()))
        self.assertEqual(calls[0][calls[0].index('--max-table-rows') + 1], '1000000')
        self.assertIn(str(ROOT / 'scripts/run/run_comparison.py'), calls[1])
        self.assertEqual(calls[1][calls[1].index('--output') + 1], str(self.output / 'results'))
        self.assertEqual(calls[1][calls[1].index('--seeds') + 1:calls[1].index('--modes')], ['17', '29', '43'])
        self.assertEqual(calls[1][calls[1].index('--modes') + 1:], list(runner.MODES))
        self.assertEqual(snapshot(self.source), before)

    def test_smoke_and_explicit_row_limit_are_forwarded(self):
        calls = self.invoke(self.args + ['--smoke', '--max-table-rows', 'none', '--seeds', '29',
                                        '--modes', 'zerosim_device', 'zerosim_device_op'])
        config = json.loads((self.output / 'config.json').read_text())
        self.assertEqual([config['training'][k] for k in
                          ('pretrain_epochs', 'decoder_epochs', 'baseline_epochs')], [1, 1, 2])
        self.assertEqual(calls[0][-1], 'none')
        self.assertEqual(calls[1][-1], '--smoke')
        self.assertEqual(calls[1][calls[1].index('--seeds') + 1], '29')

    def test_existing_empty_output_is_allowed_and_second_run_cannot_modify_it(self):
        self.output.mkdir(parents=True)
        self.invoke()
        before = snapshot(self.output)
        with patch.object(runner, 'source_identity', side_effect=AssertionError('read before output guard')), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(self.args)
        self.assertEqual(snapshot(self.output), before)

    def test_old_data_or_output_file_is_rejected_before_writes(self):
        for file_output in (False, True):
            output = self.root / str(file_output)
            if file_output:
                output.write_text('keep')
            else:
                output.mkdir()
                (output / 'data.json').write_text('old data')
            before = snapshot(self.root)
            with patch.object(runner.subprocess, 'run') as child, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(['--input', str(self.source), '--output', str(output)])
            child.assert_not_called()
            self.assertEqual(snapshot(self.root), before)

    def test_duplicate_seeds_or_modes_fail_without_output(self):
        for extra in (['--seeds', '17', '17'], ['--modes', 'zerosim_pin', 'zerosim_pin']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(self.args + extra)
            self.assertFalse(self.output.exists())

    def test_source_views_and_missing_netlist_fail_without_output(self):
        (self.source / 'ActualCircuit.sp').unlink()
        with self.assertRaisesRegex(ValueError, 'netlist'):
            runner.main(self.args)
        self.assertFalse(self.output.exists())
        with sqlite3.connect(self.source / 'sampling_history.sqlite3') as db:
            db.execute('ALTER TABLE dataset_schema RENAME TO original_schema')
            db.execute('CREATE VIEW dataset_schema AS SELECT * FROM original_schema')
        with self.assertRaisesRegex(ValueError, 'ordinary SQLite table'):
            runner.main(self.args)
        self.assertFalse(self.output.exists())

    def test_import_failure_does_not_start_training(self):
        calls = []
        def fail(command, **kwargs):
            calls.append(command)
            raise subprocess.CalledProcessError(1, command)
        with patch.object(runner.subprocess, 'run', side_effect=fail), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            runner.main(self.args)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.output / 'results').exists())

    def test_incomplete_comparison_is_not_reported_as_success(self):
        with self.assertRaisesRegex(ValueError, 'strictly complete'):
            self.invoke(complete=False)

    def test_raw_slurm_preview_uses_worker_import_without_creating_artifacts(self):
        fake = self.root / 'sbatch'
        fake.write_text('#!/usr/bin/env bash\necho UNEXPECTED >&2\nexit 99\n')
        fake.chmod(0o755)
        env = {**os.environ, 'PATH': str(self.root) + os.pathsep + os.environ['PATH'],
               'DRY_RUN': '1', 'SLURM_GPUS': '1', 'SLURM_ACCOUNT': 'example',
               'SLURM_PARTITION': 'gpu', 'JEPA_CONDA_ENV': 'newbase'}
        result = subprocess.run(['bash', str(ROOT / 'scripts/run/submit_raw_comparison.sh'),
                                 *self.args, '--max-table-rows', 'none', '--device', 'cuda'],
                                env=env, capture_output=True, text=True, check=True)
        self.assertIn('scripts/run/slurm_run.sh', result.stdout)
        self.assertIn('scripts/run/run_raw_comparison.py', result.stdout)
        self.assertIn('source\\ folder\\ not\\ a\\ circuit\\ name', result.stdout)
        self.assertNotIn('UNEXPECTED', result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
