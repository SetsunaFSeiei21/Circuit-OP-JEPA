"""Git metadata must never be a prerequisite for either training runner."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from final_exam import run, run_pin


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (run, run_pin)
SHA = 'a' * 40


class RevisionTests(unittest.TestCase):
    def test_real_child_without_git_can_write_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = dict(os.environ, PATH='', PYTHONPATH=str(ROOT))
            destination = Path(tmp) / 'metadata.json'
            command = (
                'import json, pathlib, sys; from final_exam import run, run_pin; '
                'records = {"device": run._revision(), "pin": run_pin._revision()}; '
                'pathlib.Path(sys.argv[1]).write_text(json.dumps(records))'
            )
            result = subprocess.run([sys.executable, '-c', command, str(destination)],
                                    cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(destination.read_text()),
                             {'device': 'unavailable', 'pin': 'unavailable'})
            self.assertIn('code_commit', result.stderr)

    def test_valid_revision_is_preserved(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__), \
                    patch('subprocess.run', return_value=subprocess.CompletedProcess(
                        ['git'], 0, SHA + '\n', '')), contextlib.redirect_stderr(io.StringIO()) as log:
                self.assertEqual(runner._revision(), SHA)
                self.assertEqual(log.getvalue(), '')

    def test_unavailable_git_is_nonfatal(self):
        for runner in RUNNERS:
            for error in (FileNotFoundError('git'), PermissionError('git')):
                with self.subTest(runner=runner.__name__, error=type(error).__name__), \
                        patch('subprocess.run', side_effect=error), \
                        contextlib.redirect_stderr(io.StringIO()) as log:
                    self.assertEqual(runner._revision(), 'unavailable')
                    self.assertIn('code_commit', log.getvalue())

    def test_revision_lookup_has_a_timeout_and_timeout_is_nonfatal(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__), \
                    patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['git'], 5)) as child, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner._revision(), 'unavailable')
                self.assertGreater(child.call_args.kwargs['timeout'], 0)
                self.assertLessEqual(child.call_args.kwargs['timeout'], 5)

    def test_failed_git_stdout_is_not_a_commit(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__), \
                    patch('subprocess.run', return_value=subprocess.CompletedProcess(
                        ['git'], 128, 'HEAD\n', 'fatal: not a git repository')), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner._revision(), 'unavailable')

    def test_empty_successful_output_is_unavailable(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__), \
                    patch('subprocess.run', return_value=subprocess.CompletedProcess(
                        ['git'], 0, '  \n', '')), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner._revision(), 'unavailable')

    def test_interrupt_is_not_swallowed(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__), \
                    patch('subprocess.run', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runner._revision()

    @unittest.skipUnless(shutil.which('git'), 'Git is required for the positive control')
    def test_real_checkout_records_the_actual_revision(self):
        expected = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                self.assertEqual(runner._revision(), expected)

    @unittest.skipUnless(shutil.which('git'), 'Git is required for the non-repository control')
    def test_source_directory_without_git_metadata_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            for runner in RUNNERS:
                with self.subTest(runner=runner.__name__), patch.object(runner, 'ROOT', Path(tmp)), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(runner._revision(), 'unavailable')


if __name__ == '__main__':
    unittest.main()
