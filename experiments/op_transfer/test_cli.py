"""CLI boundaries are testable without importing torch or touching cluster jobs."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

from experiments.op_transfer import run



class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.capture = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.capture))
        self.enterContext(contextlib.redirect_stderr(self.capture))

    def test_help_and_list_models_without_data(self):
        with self.assertRaises(SystemExit) as exc:
            run.main(['--help'])
        assert exc.exception.code == 0
        run.main(['--list-models'])
        output = self.capture.getvalue()
        assert 'predicted_op_joint' in output
        assert 'dual_design_control' in output
        assert 'student_predicted_op' in output
        assert 'student_predicted_op_distill' in output
        assert 'design_op_oracle' in output


    def test_output_validation_has_no_writes(self):
        with patch.object(run, 'EXPERIMENT_ROOT', self.tmp_path):
            path = run.result_directory()
            assert path.parent == self.tmp_path / 'results'
            assert not path.exists()
            with self.assertRaisesRegex(ValueError, 'subdirectory'):
                run.result_directory(self.tmp_path / 'outside')
            with self.assertRaisesRegex(ValueError, 'subdirectory'):
                run.result_directory(self.tmp_path / 'results')
            path.mkdir(parents=True)
            sentinel = path / 'data.json'
            sentinel.write_text('unchanged')
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                run.result_directory(path)
            assert sentinel.read_text() == 'unchanged'


    def test_output_symlink_escape_rejected(self):
        root = self.tmp_path / 'experiment'
        root.mkdir()
        outside = self.tmp_path / 'outside'
        outside.mkdir()
        (root / 'results').symlink_to(outside, target_is_directory=True)
        with patch.object(run, 'EXPERIMENT_ROOT', root):
            with self.assertRaisesRegex(ValueError, 'redirect'):
                run.result_directory()


    def test_invalid_selection_rejected_before_writes(self):
        for extra in (['--models', 'design_supervised', 'design_supervised'],
                      ['--seeds', '17', '17'], ['--seeds', '-1'],
                      ['--max-train', '0'], ['--models', 'nonexistent_model']):
            self._invalid_selection(extra)


    def _invalid_selection(self, extra):
        with patch.object(run, 'EXPERIMENT_ROOT', self.tmp_path):
            with self.assertRaises(SystemExit) as exc:
                run.main(['--data', str(self.tmp_path / 'missing.json'), *extra])
        assert exc.exception.code == 2
        assert not (self.tmp_path / 'results').exists()


    def test_main_forwards_only_selected_models(self):
        data = self.tmp_path / 'data.json'
        config = self.tmp_path / 'config.json'
        manifest = self.tmp_path / 'manifest.json'
        for path in (data, config, manifest):
            path.write_text('{}')
        calls = []
        engine = types.ModuleType('experiments.op_transfer.engine')
        engine.run_experiment = lambda **kwargs: calls.append(kwargs)
        with patch.object(run, 'EXPERIMENT_ROOT', self.tmp_path), patch.dict(sys.modules, {
                'experiments.op_transfer.engine': engine}):
            run.main(['--data', str(data), '--config', str(config), '--manifest', str(manifest),
                      '--models', 'op_distill', 'predicted_op', '--seeds', '17', '29',
                      '--device', 'cpu', '--max-train', '8', '--max-val', '4', '--max-test', '3'])
        assert len(calls) == 1
        assert calls[0]['models'] == ['op_distill', 'predicted_op']
        assert calls[0]['seeds'] == [17, 29]
        assert calls[0]['manifest_path'] == manifest
        assert calls[0]['max_train'] == 8
        assert calls[0]['max_val'] == 4
        assert calls[0]['max_test'] == 3
        assert not (self.tmp_path / 'results').exists()


    def test_submit_preview_preserves_spaces_and_has_no_output(self):
        source = Path(run.__file__).with_name('submit.sh')
        root = self.tmp_path / 'project with spaces'
        script = root / 'experiments' / 'op_transfer' / 'submit.sh'
        script.parent.mkdir(parents=True)
        script.write_text(source.read_text())
        env = {**os.environ, 'DRY_RUN': '1', 'SLURM_CPUS': '8', 'SLURM_GPUS': '1'}
        result = subprocess.run(['bash', str(script), '--data', '/input with spaces/data.json',
                                 '--config', '/input/config.json', '--models', 'predicted_op'],
                                env=env, text=True, capture_output=True, check=True)
        assert 'circuit-jepa-op-transfer' in result.stdout
        assert 'scripts/run/slurm_run.sh' in result.stdout
        assert '--cpus-per-task=8' in result.stdout
        assert '--gres=gpu:1' in result.stdout
        assert 'input\\ with\\ spaces/data.json' in result.stdout
        assert not (script.parent / 'logs').exists()
        assert not (script.parent / 'results').exists()


    def test_submit_log_symlink_rejected(self):
        source = Path(run.__file__).with_name('submit.sh')
        script = self.tmp_path / 'experiments' / 'op_transfer' / 'submit.sh'
        script.parent.mkdir(parents=True)
        script.write_text(source.read_text())
        (script.parent / 'logs').symlink_to(self.tmp_path, target_is_directory=True)
        result = subprocess.run(['bash', str(script), '--dry-run', '--data', '/some/data.json'],
                                text=True, capture_output=True)
        assert result.returncode == 2
        assert 'must not be a symlink' in result.stderr


if __name__ == '__main__':
    unittest.main()
