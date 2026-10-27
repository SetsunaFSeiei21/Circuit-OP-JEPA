"""Raw circuit jobs keep import on the worker and isolate independent datasets."""
import contextlib
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
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
from scripts.run import result_paths
from device_jepa.methods import DEFAULT_MODELS


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
        # These tests exercise orchestration, not model/CUDA availability.
        runtime = patch.object(runner, 'check_runtime')
        self.runtime = runtime.start()
        self.addCleanup(runtime.stop)

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
        self.assertEqual(calls[1][calls[1].index('--seeds') + 1:calls[1].index('--models')], ['17', '29', '43'])
        self.assertEqual(calls[1][calls[1].index('--models') + 1:], list(DEFAULT_MODELS))
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

    def test_default_output_uses_sqlite_circuit_and_date_under_shared_root(self):
        shared = self.root / 'original repo/runs/gpu_full'
        before = snapshot(self.source)
        with patch.dict(os.environ, {'JEPA_RESULTS_ROOT': str(shared)}), \
             patch.object(result_paths, 'datetime') as clock:
            clock.now.return_value = datetime(2026, 10, 26, 21, 30)
            calls = self.invoke(['--input', str(self.source)])
        output = shared / 'ActualCircuit_20261026_213000'
        self.assertEqual(calls[0][calls[0].index('--output') + 1], str(output / 'data.json'))
        self.assertEqual(calls[1][calls[1].index('--output') + 1], str(output / 'results'))
        config = json.loads((output / 'config.json').read_text())
        self.assertEqual(config['dataset']['train_circuits'], ['ActualCircuit'])
        self.assertEqual(snapshot(self.source), before)

    def test_explicit_output_overrides_results_root(self):
        shared = self.root / 'unused shared root'
        with patch.dict(os.environ, {'JEPA_RESULTS_ROOT': str(shared)}):
            self.invoke()
        self.assertTrue((self.output / 'config.json').is_file())
        self.assertFalse(shared.exists())

    def test_configured_models_and_cli_override_are_forwarded_and_saved(self):
        config = json.loads((ROOT / 'configs/default.json').read_text())
        config['experiment'] = {'models': ['zerosim_pin_op']}
        config_path = self.root / 'selection.json'
        config_path.write_text(json.dumps(config))
        for index, selected in enumerate((['zerosim_pin_op'], ['jepa_zerosim', 'zerosim_device'])):
            output = self.root / f'selection_{index}'
            args = ['--input', str(self.source), '--output', str(output), '--config', str(config_path)]
            if index:
                args += ['--models', *selected]
            calls = self.invoke(args)
            saved = json.loads((output / 'config.json').read_text())
            self.assertEqual(saved['experiment']['models'], selected)
            self.assertEqual(calls[1][calls[1].index('--models') + 1:], selected)
            self.assertEqual(saved['dataset']['ood_circuits'], [])

    def test_invalid_run_does_not_allocate_default_output(self):
        shared = self.root / 'runs/gpu_full'
        with patch.dict(os.environ, {'JEPA_RESULTS_ROOT': str(shared)}), \
             patch.object(runner, 'comparison_epoch_budget', side_effect=ValueError('budget')):
            with self.assertRaisesRegex(ValueError, 'budget'):
                runner.main(['--input', str(self.source)])
        self.assertFalse(shared.exists())

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

    def test_default_output_preview_does_not_read_source_or_create_directories(self):
        shared = self.root / 'shared results'
        env = {**os.environ, 'DRY_RUN': '1', 'JEPA_RESULTS_ROOT': str(shared)}
        result = subprocess.run(['bash', str(ROOT / 'scripts/run/submit_raw_comparison.sh'),
                                 '--input', str(self.root / 'not available on login node'),
                                 '--device', 'cuda'],
                                env=env, capture_output=True, text=True, check=True)
        self.assertIn('scripts/run/run_raw_comparison.py', result.stdout)
        self.assertFalse(shared.exists())

    def make_source(self, name):
        directory = self.root / ('directory for ' + name)
        directory.mkdir()
        fixture(directory, circuit=name)
        return directory

    def test_ood_imports_all_sources_and_preserves_explicit_roles(self):
        second = self.make_source('SecondID')
        holdout = self.make_source('HeldOut')
        before = snapshot(self.root)
        calls = self.invoke(['--input', str(self.source), str(second), str(holdout),
                             '--output', str(self.output), '--train-circuits', 'ActualCircuit', 'SecondID',
                             '--ood-circuits', 'HeldOut', '--models', 'all', '--device', 'cuda'])
        config = json.loads((self.output / 'config.json').read_text())
        self.assertEqual(config['dataset']['train_circuits'], ['ActualCircuit', 'SecondID'])
        self.assertEqual(config['dataset']['ood_circuits'], ['HeldOut'])
        self.assertEqual(len(config['experiment']['models']), 8)
        self.assertEqual(calls[0][calls[0].index('--input')+1:calls[0].index('--config')],
                         [str(self.source), str(second), str(holdout)])
        self.assertEqual(calls[1][calls[1].index('--device')+1], 'cuda')
        self.runtime.assert_called_once_with(config, 'cuda')
        after = snapshot(self.root)
        self.assertTrue(all(after[path] == content for path, content in before.items()))

    def test_ood_default_output_is_separate_from_single_circuit_results(self):
        holdout = self.make_source('HeldOut')
        shared = self.root/'runs/gpu_full'
        with patch.dict(os.environ, {'JEPA_RESULTS_ROOT': str(shared)}), \
             patch.object(result_paths, 'datetime') as clock:
            clock.now.return_value = datetime(2026, 10, 27, 20, 30)
            calls = self.invoke(['--input', str(self.source), str(holdout),
                                 '--train-circuits', 'ActualCircuit', '--ood-circuits', 'HeldOut'])
        expected = shared/'HeldOut_OOD_20261027_203000'
        self.assertEqual(calls[1][calls[1].index('--output')+1], str(expected/'results'))

    def test_ood_rejects_ambiguous_missing_overlapping_or_duplicate_roles_before_writes(self):
        holdout = self.make_source('HeldOut')
        base = ['--input', str(self.source), str(holdout), '--output', str(self.output)]
        invalid = [[], ['--train-circuits', 'ActualCircuit'], ['--ood-circuits', 'HeldOut'],
                   ['--train-circuits', 'ActualCircuit', 'ActualCircuit', '--ood-circuits', 'HeldOut'],
                   ['--train-circuits', 'ActualCircuit', '--ood-circuits', 'ActualCircuit'],
                   ['--train-circuits', 'FolderName', '--ood-circuits', 'HeldOut'],
                   ['--train-circuits', 'ActualCircuit', '--ood-circuits', 'HeldOut', 'Absent']]
        for options in invalid:
            with self.subTest(options=options), patch.object(runner.subprocess, 'run') as child:
                with self.assertRaises(ValueError):
                    runner.main(base + options)
                child.assert_not_called()
                self.assertFalse(self.output.exists())
        self.runtime.assert_not_called()

    def test_ood_rejects_duplicate_directories_and_source_identities(self):
        holdout = self.make_source('HeldOut')
        duplicate = self.make_source('ActualCircuit')
        for directories in ([self.source, self.source, holdout], [self.source, duplicate, holdout]):
            with self.subTest(directories=directories), self.assertRaisesRegex(ValueError, 'unique|distinct'):
                runner.source_plan(directories, ['ActualCircuit'], ['HeldOut'])
        self.assertFalse(self.output.exists())

    def test_imported_ood_circuit_is_excluded_from_all_id_splits(self):
        from device_jepa.adapter import adapt_analog_agent, combine_datasets
        from device_jepa.data import build_manifest
        second = self.make_source('SecondID')
        holdout = self.make_source('HeldOut')
        # Give the synthetic fixtures distinct topologies; copying an identical
        # design under a new circuit name is correctly rejected as leakage.
        for directory, name, old, new in ((second, 'SecondID', 'IBIAS OUT VSS', 'IBIAS IN VSS'),
                                          (holdout, 'HeldOut', 'RLOAD OUT VDD', 'RLOAD OUT VSS')):
            netlist = directory / (name + '.sp')
            netlist.write_text(netlist.read_text().replace(old, new))
        membership, count, label = runner.source_plan(
            [self.source, second, holdout], ['ActualCircuit', 'SecondID'], ['HeldOut'])
        self.assertEqual((count, label), (9, 'HeldOut_OOD'))
        dataset = combine_datasets([adapt_analog_agent(path) for path in (self.source, second, holdout)])
        config = {**membership, 'split_ratios': [.6, .2, .2], 'split_seed': 17, 'finetune_label_fraction': 1.0}
        manifest = build_manifest(dataset, config=config)
        by_id = {sample['sample_id']: sample['circuit_id'] for sample in dataset['samples']}
        heldout_ids = {sid for sid, circuit in by_id.items() if circuit == 'HeldOut'}
        self.assertEqual(set(manifest['ood']), heldout_ids)
        self.assertEqual(len(heldout_ids), 3)
        for split in ('train', 'val', 'test', 'labeled_train'):
            self.assertTrue(manifest[split])
            self.assertFalse(set(manifest[split]) & heldout_ids)

    def test_runtime_failure_prevents_import_and_output_creation(self):
        self.runtime.side_effect = ValueError('CUDA unavailable')
        with patch.object(runner.subprocess, 'run') as child, self.assertRaisesRegex(ValueError, 'CUDA'):
            runner.main(self.args)
        child.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_ood_launcher_preview_and_submit_forward_roles_models_and_gpu(self):
        fake = self.root/'sbatch'
        captured = self.root/'submitted'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$JEPA_TEST_CAPTURE"\nprintf "123456\\n"\n')
        fake.chmod(0o755)
        env = {**os.environ, 'PATH': str(self.root)+os.pathsep+os.environ['PATH'],
               'JEPA_TEST_CAPTURE': str(captured), 'JEPA_LOG_DIR': str(self.root/'logs'),
               'JEPA_RESULTS_ROOT': str(self.root/'results'), 'RAW_ROOT': str(self.root/'raw data'),
               'OOD_CIRCUIT': 'HeldOut', 'CONFIG': 'runs/configs/with spaces.json', 'DRY_RUN': '1'}
        command = ['bash', str(ROOT/'scripts/run/train_ood.sh')]
        result = subprocess.run(command+['--dry-run'], env=env, cwd=self.root,
                                capture_output=True, text=True, check=True)
        args = shlex.split(result.stdout)
        self.assertEqual(args[args.index('--train-circuits')+1:args.index('--ood-circuits')],
                         ['5t_ota', 'Opamp0', 'Opamp1'])
        self.assertEqual(args[args.index('--ood-circuits')+1:args.index('--config')], ['HeldOut'])
        self.assertEqual(len(args[args.index('--models')+1:args.index('--max-table-rows')]), 7)
        self.assertEqual(args[args.index('--input')+1:args.index('--train-circuits')],
                         [str(self.root/'raw data'/name) for name in ('5t_ota', 'Opamp0', 'Opamp1', 'HeldOut')])
        self.assertFalse(captured.exists())
        self.assertFalse((self.root/'logs').exists())
        self.assertFalse((self.root/'results').exists())
        for flag, code in (('--help', 0), ('--bad-option', 2)):
            result = subprocess.run(command+[flag], env=env, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, code)
            self.assertFalse(captured.exists())
        result = subprocess.run(command, env=env, cwd=self.root, capture_output=True, text=True, check=True)
        args = captured.read_text().rstrip('\0').split('\0')
        self.assertIn('--gres=gpu:1', args)
        self.assertIn('--cpus-per-task=8', args)
        self.assertEqual(args[args.index('--config')+1], 'runs/configs/with spaces.json')
        self.assertEqual(args[-2:], ['--device', 'cuda'])
        self.assertTrue(result.stdout.endswith('123456\n'))
        self.assertFalse((self.root/'results').exists())


if __name__ == '__main__':
    unittest.main()
