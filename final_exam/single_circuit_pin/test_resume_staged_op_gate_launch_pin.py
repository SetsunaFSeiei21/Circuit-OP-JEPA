"""Resume submission uses existing canonical data and leaves source runs intact."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from final_exam.common import resume_pin as resume


MODELS = ['device_op_distill_gate_pin', 'device_op_distill_gate_decoder_pin',
          'device_op_supervised_gate_pin']
SEEDS = [17, 29, 43]
CIRCUIT = 'two_stage_folded_opamp'
FINAL_ROOT = Path(__file__).resolve().parents[1]
REPO = FINAL_ROOT.parent


class ResumeStagedGateLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = self.root / 'final_exam/single_circuit_pin'

    def source_fixture(self, name='failed_pin', *, smoke=False):
        source = self.task / 'results_pin' / name
        training = source / 'training_pin'
        training.mkdir(parents=True)
        config = json.loads((FINAL_ROOT / 'configs/staged_op_gate_pin.json').read_text())
        metadata = dict(status='failed', granularity='pin', circuit=CIRCUIT,
                        models=MODELS, seeds=SEEDS, device='cuda', smoke=smoke,
                        dataset_fingerprint='f' * 64, slurm_job_id=None)
        manifest = dict(train=[1, 2], val=[3], test=[4], ood=[], labeled_train=[1, 2],
                        dataset_fingerprint='f' * 64)
        for name, value in (('config_pin.json', config), ('run_pin.json', metadata),
                            ('manifest_pin.json', manifest)):
            (training / name).write_text(json.dumps(value) + '\n')
        (source / 'config_pin.json').write_text(json.dumps(config) + '\n')
        (source / 'data_pin.json').write_text('{"samples": []}\n')
        (source / 'run_pin.json').write_text(json.dumps(metadata) + '\n')
        return source, training

    def inspect(self, source):
        output = io.StringIO()
        with patch.object(resume, 'PROJECT_ROOT', self.root), \
                patch.object(resume, 'TASK_ROOT', self.task), contextlib.redirect_stdout(output):
            resume.main(['--resume-from', str(source), '--inspect'])
        return json.loads(output.getvalue())

    def snapshot(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob('*') if path.is_file()}

    def call_main(self, source, *extra, fake=None):
        with patch.object(resume, 'PROJECT_ROOT', self.root), \
                patch.object(resume, 'TASK_ROOT', self.task), \
                patch.object(resume, 'run_logged', side_effect=fake) as runner, \
                contextlib.redirect_stdout(io.StringIO()):
            result = resume.main(['--resume-from', str(source), '--run-id', 'resumed_pin',
                                  '--device', 'cpu', *extra])
        return result, runner

    @staticmethod
    def fake_complete(command, log, lock):
        command = list(map(str, command))
        output = Path(command[command.index('--output') + 1])
        output.mkdir(parents=True)
        (output / 'run_pin.json').write_text('{"status":"complete"}\n')
        (output / 'summary_pin.xlsx').write_bytes(b'fixture workbook')

    def test_inspect_accepts_run_or_training_directory_and_chained_data(self):
        source, training = self.source_fixture()
        before = self.snapshot(source)
        first = self.inspect(source)
        self.assertEqual(first, self.inspect(training))
        self.assertEqual(first['models'], MODELS)
        self.assertEqual(first['seeds'], SEEDS)
        self.assertEqual(first['data_path'], str(source / 'data_pin.json'))
        self.assertEqual(first['source_training_output'], str(training))
        self.assertEqual(self.snapshot(source), before)

        chained, _ = self.source_fixture('already_resumed_pin')
        (chained / 'data_pin.json').unlink()
        (chained / 'data_source_pin.json').write_text(json.dumps({
            'data_path': str(source / 'data_pin.json'),
            'dataset_fingerprint': 'f' * 64,
            'source_training_output': str(training),
        }))
        self.assertEqual(self.inspect(chained)['data_path'], first['data_path'])

    def test_invalid_source_fails_without_creating_new_run(self):
        for invalid in ('foreign_model', 'missing_data'):
            with self.subTest(invalid=invalid):
                source, training = self.source_fixture(invalid + '_pin')
                if invalid == 'foreign_model':
                    record = json.loads((training / 'run_pin.json').read_text())
                    record['models'] = ['zerosim_supervised_pin']
                    (training / 'run_pin.json').write_text(json.dumps(record))
                else:
                    (source / 'data_pin.json').unlink()
                before = self.snapshot(self.task)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.call_main(source)
                self.assertEqual(self.snapshot(self.task), before)

    def test_resume_reuses_data_selection_manifest_and_leaves_source_unchanged(self):
        source, training = self.source_fixture(smoke=True)
        before = self.snapshot(source)
        commands = []

        def complete(command, log, lock):
            commands.append(list(map(str, command)))
            self.fake_complete(command, log, lock)

        output, runner = self.call_main(source, fake=complete)
        self.assertEqual(runner.call_count, 1)
        command = commands[0]
        self.assertNotIn('import_analog_agent', command)
        self.assertEqual(command[command.index('--data') + 1], str(source / 'data_pin.json'))
        self.assertEqual(command[command.index('--manifest') + 1], str(training / 'manifest_pin.json'))
        self.assertEqual(command[command.index('--resume-from') + 1], str(training))
        self.assertEqual(command[command.index('--models') + 1:command.index('--seeds')], MODELS)
        self.assertEqual(command[command.index('--seeds') + 1:command.index('--device')], list(map(str, SEEDS)))
        self.assertIn('--smoke', command)
        output = Path(output)
        self.assertEqual(output, self.task / 'results_pin/resumed_pin')
        self.assertFalse((output / 'data_pin.json').exists())
        self.assertEqual(json.loads((output / 'data_source_pin.json').read_text())['data_path'],
                         str(source / 'data_pin.json'))
        self.assertEqual(json.loads((output / 'run_pin.json').read_text())['status'], 'complete')
        self.assertTrue((self.task / 'logs_pin/resumed_pin/launch_pin.log').is_file())
        self.assertEqual(self.snapshot(source), before)

    def test_failure_records_original_exit_code_without_touching_source(self):
        source, _ = self.source_fixture()
        before = self.snapshot(source)
        error = subprocess.CalledProcessError(7, ['training'])
        with self.assertRaises(subprocess.CalledProcessError):
            self.call_main(source, fake=error)
        record = json.loads((self.task / 'results_pin/resumed_pin/run_pin.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['exit_code'], 7)
        self.assertEqual(self.snapshot(source), before)
        self.assertIn('CalledProcessError',
                      (self.task / 'logs_pin/resumed_pin/launch_pin.log').read_text())

    def test_changed_submission_snapshot_and_occupied_output_fail_before_writes(self):
        source, training = self.source_fixture()
        snapshot = self.root / 'submitted_resume_source_pin.json'
        snapshot.write_text(json.dumps(self.inspect(source)))
        config_path = training / 'config_pin.json'
        config = json.loads(config_path.read_text())
        config['op_transfer']['lr'] *= 2
        config_path.write_text(json.dumps(config))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.call_main(source, '--submitted-source', str(snapshot))
        self.assertFalse((self.task / 'results_pin/resumed_pin').exists())
        self.assertFalse((self.task / 'logs_pin/resumed_pin').exists())

        occupied = self.task / 'results_pin/resumed_pin'
        occupied.mkdir()
        sentinel = occupied / 'keep.txt'
        sentinel.write_text('do not overwrite')
        before = self.snapshot(occupied)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.call_main(source)
        self.assertEqual(self.snapshot(occupied), before)

    def bash_fixture(self, sbatch_body='echo 98765'):
        source, training = self.source_fixture()
        script = self.task / 'resume_staged_op_gate_pin.sh'
        shutil.copyfile(Path(__file__).with_name(script.name), script)
        common = self.root / 'final_exam/common'
        common.mkdir()
        for name in ('resume_pin.py', 'launch_pin.py'):
            shutil.copyfile(FINAL_ROOT / 'common' / name, common / name)
        # The shell must submit the existing root-aware Slurm wrapper, never
        # calculate the repository from the eventual /var/spool script path.
        wrapper = self.root / 'scripts/run/slurm_run.sh'
        wrapper.parent.mkdir(parents=True)
        shutil.copyfile(REPO / 'scripts/run/slurm_run.sh', wrapper)
        binary = self.root / 'bin/sbatch'
        binary.parent.mkdir()
        binary.write_text('#!/usr/bin/env bash\n' + sbatch_body + '\n')
        binary.chmod(0o755)
        environment = os.environ.copy()
        environment['PATH'] = str(binary.parent) + os.pathsep + environment['PATH']
        environment.update(JEPA_PYTHON=sys.executable, SLURM_ACCOUNT='b_phzhwu',
                           SLURM_PARTITION='ex01A800', SLURM_CPUS='8', SLURM_GPUS='1',
                           SLURM_MEM='80G')
        environment.pop('RESUME_FROM', None)
        return source, training, script, environment

    def test_bash_dry_run_is_read_only_and_submits_absolute_paths(self):
        source, training, script, environment = self.bash_fixture('exit 99')
        before = self.snapshot(self.root)
        names_before = sorted(str(p) for p in self.root.rglob('*'))
        result = subprocess.run(['bash', str(script), str(source), '--dry-run'],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(self.root), before)
        self.assertEqual(sorted(str(p) for p in self.root.rglob('*')), names_before)
        command = shlex.split(result.stdout)
        self.assertEqual(command[0], 'sbatch')
        self.assertIn('--gres=gpu:1', command)
        self.assertIn('--cpus-per-task=8', command)
        self.assertIn(str(self.root / 'scripts/run/slurm_run.sh'), command)
        self.assertIn(str(self.root / 'final_exam/common/resume_pin.py'), command)
        self.assertEqual(command[command.index('--resume-from') + 1], str(training))
        run_id = command[command.index('--run-id') + 1]
        self.assertIn('_resume_staged_op_gate_', run_id)
        self.assertTrue(run_id.endswith('_pin'))

    def test_bash_submission_exports_root_and_saves_source_snapshot(self):
        source, training, script, environment = self.bash_fixture(
            'test "$JEPA_PROJECT_ROOT" = "$PWD" || exit 21\n'
            'test "$CIRCUIT_JEPA_ROOT" = "$PWD" || exit 22\necho "98765;cluster"')
        result = subprocess.run(['bash', str(script), str(training)],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = list(self.task.glob('hpc_logs_pin/*/submission_pin.txt'))
        self.assertEqual(len(records), 1)
        saved = records[0].parent / 'submitted_resume_source_pin.json'
        self.assertEqual(json.loads(saved.read_text()), self.inspect(source))
        self.assertIn('job_id=98765;cluster', records[0].read_text())
        self.assertFalse((self.task / 'logs_pin').exists())
        self.assertEqual(list((self.task / 'results_pin').iterdir()), [source])

    def test_bash_submission_failure_preserves_exit_status(self):
        source, _, script, environment = self.bash_fixture('echo rejected >&2\nexit 9')
        result = subprocess.run(['bash', str(script), str(source)],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 9)
        self.assertIn('rejected', result.stderr)
        self.assertNotIn('Submitted job', result.stdout)
        self.assertEqual(list((self.task / 'results_pin').iterdir()), [source])

    def test_bash_blocks_active_source_job_before_allocating_new_directories(self):
        source, training, script, environment = self.bash_fixture('exit 99')
        metadata_path = training / 'run_pin.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['slurm_job_id'] = '12345'
        metadata_path.write_text(json.dumps(metadata))
        squeue = self.root / 'bin/squeue'
        squeue.write_text('#!/usr/bin/env bash\necho RUNNING\n')
        squeue.chmod(0o755)
        before = self.snapshot(self.root)
        result = subprocess.run(['bash', str(script), str(source)],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('12345', result.stderr)
        self.assertIn('still active', result.stderr)
        self.assertEqual(self.snapshot(self.root), before)
        self.assertFalse((self.task / 'hpc_logs_pin').exists())


if __name__ == '__main__':
    unittest.main()
