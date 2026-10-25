import os
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


if __name__ == '__main__':
    unittest.main()
