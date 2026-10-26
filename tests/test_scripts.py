import os
import shlex
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def test_shell_syntax(self):
        for path in (ROOT/'scripts').rglob('*.sh'):
            subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)

    def test_slurm_dry_run_never_calls_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/'sbatch'
            fake.write_text('#!/usr/bin/env bash\necho UNEXPECTED >&2\nexit 99\n')
            fake.chmod(0o755)
            env = {**os.environ, 'PATH': tmp+os.pathsep+os.environ['PATH'], 'DRY_RUN': '1',
                   'SLURM_GPUS': '0', 'SLURM_ACCOUNT': '', 'SLURM_PARTITION': '', 'SLURM_MAIL_USER': ''}
            for script in ['run/submit_comparison.sh', 'smoke_test/submit_comparison.sh']:
                result = subprocess.run(['bash', str(ROOT/'scripts'/script), '--data', '/path with spaces/data.json',
                                         '--config', 'configs/smoke.json', '--output', '/tmp/not-created'],
                                        env=env, text=True, capture_output=True, check=True)
                self.assertIn('sbatch --parsable', result.stdout)
                self.assertNotIn('--gres=', result.stdout)
                self.assertNotIn('--mail-user=', result.stdout)
                self.assertNotIn('--account=', result.stdout)
                self.assertNotIn('UNEXPECTED', result.stderr)
                self.assertIn('/path\\ with\\ spaces/data.json', result.stdout)

    def test_git_helper_refuses_enclosing_repo(self):
        # The helper only accepts a dedicated checkout and reviewed staged files.
        # Static guard also ensures a test cannot inadvertently invoke a real push.
        text = (ROOT/'scripts/git/push.sh').read_text()
        self.assertIn('PUSH_CONFIRM:-0', text)
        self.assertIn('PROJECT_ROOT" == "$REPO_ROOT', text)
        for command in ['git add -A', 'git reset', 'git push --force', 'git pull']:
            self.assertFalse(any(line.startswith(command) for line in text.splitlines()))

    def test_model_catalog_never_submits_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'sbatch'
            fake.write_text('#!/usr/bin/env bash\necho UNEXPECTED >&2\nexit 99\n')
            fake.chmod(0o755)
            env = {**os.environ, 'PATH': tmp + os.pathsep + os.environ['PATH'],
                   'DRY_RUN': '0', 'JEPA_LOG_DIR': str(root / 'logs'),
                   'JEPA_RESULTS_ROOT': str(root / 'results')}
            for script in ('run/submit_comparison.sh', 'run/submit_raw_comparison.sh',
                           'smoke_test/submit_comparison.sh', 'smoke_test/run_smoke.sh', 'run_smoke.sh'):
                result = subprocess.run(['bash', str(ROOT / 'scripts' / script), '--list-models'],
                                        env=env, text=True, capture_output=True, check=True)
                self.assertIn('jepa_zerosim', result.stdout)
                self.assertIn('zerosim_device_op', result.stdout)
                self.assertNotIn('sbatch', result.stdout)
                self.assertNotIn('UNEXPECTED', result.stderr)
            self.assertEqual({p.name for p in root.iterdir()}, {'sbatch'})

    def test_model_selection_survives_slurm_wrappers(self):
        env = {**os.environ, 'DRY_RUN': '1'}
        for script in ('run/submit_comparison.sh', 'run/submit_raw_comparison.sh',
                       'smoke_test/submit_comparison.sh'):
            result = subprocess.run(['bash', str(ROOT / 'scripts' / script),
                                     '--models', 'jepa_zerosim', 'zerosim_device',
                                     '--output', '/not-created'],
                                    env=env, text=True, capture_output=True, check=True)
            args = shlex.split(result.stdout)
            start = args.index('--models')
            self.assertEqual(args[start + 1:start + 3], ['jepa_zerosim', 'zerosim_device'])
            self.assertNotIn('zerosim_pin', args)
            self.assertNotIn('zerosim_device_op', args)

    def test_shared_log_directory_overrides_worktree_and_job_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root = root / 'original repo' / 'hpc_logs' / 'gpu_full'
            captured = root / 'sbatch-args'
            fake = root / 'sbatch'
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$JEPA_TEST_SBATCH_ARGS"\nprintf "123456\\n"\n')
            fake.chmod(0o755)
            env = {**os.environ, 'PATH': tmp+os.pathsep+os.environ['PATH'],
                   'JEPA_LOG_DIR': str(log_root), 'JEPA_JOB_KIND': 'opamp1_full',
                   'JEPA_TEST_SBATCH_ARGS': str(captured), 'DRY_RUN': '1'}
            for name in ('submit_comparison.sh', 'submit_raw_comparison.sh'):
                command = ['bash', str(ROOT/'scripts/run'/name), '--output', str(root/'results')]
                before = set(log_root.rglob('*')) if log_root.exists() else None
                preview = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
                args = shlex.split(preview.stdout)
                output = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--output=')))
                self.assertEqual(output.parent.parent, log_root)
                if before is None:
                    self.assertFalse(log_root.exists())
                else:
                    self.assertEqual(set(log_root.rglob('*')), before)
                self.assertFalse(captured.exists())
                result = subprocess.run(command, env={**env, 'DRY_RUN': '0'},
                                        text=True, capture_output=True, check=True)
                args = captured.read_bytes().decode().rstrip('\0').split('\0')
                output = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--output=')))
                error = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--error=')))
                self.assertEqual(output.parent.parent, log_root)
                self.assertEqual(output.parent, error.parent)
                self.assertTrue(output.parent.is_dir())
                self.assertIn('--job-name=circuit-jepa-opamp1_full', args)
                self.assertIn('--chdir=' + str(ROOT), args)
                self.assertTrue(result.stdout.endswith('123456\n'))
                captured.unlink()

    def test_relative_log_override_is_rejected_before_submit_or_mkdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/'sbatch'
            fake.write_text('#!/usr/bin/env bash\necho UNEXPECTED >&2\nexit 99\n')
            fake.chmod(0o755)
            env = {**os.environ, 'PATH': tmp+os.pathsep+os.environ['PATH'],
                   'JEPA_LOG_DIR': 'relative/gpu_full', 'DRY_RUN': '0'}
            result = subprocess.run(['bash', str(ROOT/'scripts/run/submit_comparison.sh'),
                                     '--output', str(Path(tmp)/'results')],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('JEPA_LOG_DIR must be an absolute directory', result.stderr)
            self.assertNotIn('UNEXPECTED', result.stderr)


if __name__ == '__main__':
    unittest.main()
