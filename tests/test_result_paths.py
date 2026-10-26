"""Default output locations share the main checkout and never reuse a run."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.run import result_paths


class ResultPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        env = patch.dict(os.environ, {'JEPA_RESULTS_ROOT': ''})
        env.start()
        self.addCleanup(env.stop)

    def test_main_checkout_and_linked_worktree_share_result_root(self):
        primary = self.root / 'original repo'
        linked = self.root / 'Opamp1 worktree'
        subprocess.run(['git', 'init', '-q', str(primary)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(primary), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'initial',
                        '--allow-empty'], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(primary), 'worktree', 'add', '-q', '--detach',
                        str(linked)], check=True, capture_output=True)
        for checkout in (primary, linked):
            self.assertEqual(result_paths.results_root(checkout), primary / 'runs/gpu_full')
        # An exported copy inside an enclosing repository must use its own root.
        exported = linked / 'exported source'
        exported.mkdir()
        self.assertEqual(result_paths.results_root(exported), exported / 'runs/gpu_full')
        self.assertFalse((primary / 'runs').exists())

    def test_result_root_override_must_be_absolute(self):
        with patch.dict(os.environ, {'JEPA_RESULTS_ROOT': str(self.root / 'shared results')}):
            self.assertEqual(result_paths.results_root(self.root), self.root / 'shared results')
        with patch.dict(os.environ, {'JEPA_RESULTS_ROOT': 'relative/gpu_full'}):
            with self.assertRaisesRegex(ValueError, 'absolute'):
                result_paths.create_default_output(self.root, 'Opamp1')
        self.assertEqual(list(self.root.iterdir()), [])

    def test_concurrent_runs_are_distinct_and_keep_existing_artifacts(self):
        output_root = self.root / 'runs/gpu_full'
        occupied = output_root / 'Opamp1_20261026_213000'
        occupied.mkdir(parents=True)
        sentinel = occupied / 'data.json'
        sentinel.write_text('old data')
        with patch.dict(os.environ, {'JEPA_RESULTS_ROOT': str(output_root)}), \
             patch.object(result_paths, 'datetime') as clock:
            clock.now.return_value = datetime(2026, 10, 26, 21, 30)
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs = list(pool.map(
                    lambda _: result_paths.create_default_output(self.root, 'Opamp1'), range(4)))
        self.assertEqual({p.name for p in outputs},
                         {f'Opamp1_20261026_213000_{i}' for i in range(2, 6)})
        self.assertTrue(all(p.is_dir() and p.parent == output_root for p in outputs))
        self.assertEqual(sentinel.read_text(), 'old data')

    def test_circuit_label_cannot_escape_results_root(self):
        output = result_paths.create_default_output(self.root, '../../odd/circuit')
        self.assertEqual(output.parent, self.root / 'runs/gpu_full')
        self.assertNotIn('/', output.name)


if __name__ == '__main__':
    unittest.main()
