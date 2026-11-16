"""Explicit micrometer scale repair preserves sources or rolls back on failure."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from test_adapter import fixture
from scripts.run.fix_raw_geometry_scale import fix_geometry_scale, main


class RawGeometryScaleFixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, directory='a', circuit='Tiny', option=''):
        root = self.root / directory
        root.mkdir()
        fixture(root, circuit=circuit)
        path = root / (circuit + '.sp')
        path.write_text(path.read_text().replace('.option scale=1u\n', option))
        return root, path

    def test_missing_scale_backup_exact_bytes_db_untouched_and_idempotent(self):
        root, path = self.source(circuit='DifferentName')
        original = path.read_bytes().replace(b'\n', b'\r\n')
        path.write_bytes(original)
        database = root / 'sampling_history.sqlite3'
        digest = hashlib.sha256(database.read_bytes()).hexdigest()
        report, = fix_geometry_scale([root], max_samples=2)
        self.assertEqual(report['status'], 'fixed')
        self.assertFalse(report['scan_complete'])
        self.assertEqual(report['geometry_scale_to_m'], 1e-6)
        self.assertEqual(report['mismatches'], {'W_m': 0, 'L_m': 0, 'm': 0})
        self.assertEqual(Path(report['backup']).read_bytes(), original)
        self.assertTrue(path.read_bytes().endswith(original))
        self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), digest)
        modified = path.read_bytes()
        again, = fix_geometry_scale([root])
        self.assertEqual(again['status'], 'already_consistent')
        self.assertIsNone(again['backup'])
        self.assertEqual(path.read_bytes(), modified)
        self.assertEqual(len(list(root.glob('*.bak_*'))), 1)

    def test_existing_scale_only_value_changes_and_comments_survive(self):
        for i, option in enumerate(('.option scale=1 ; keep this\n',
                                     '  .OPTIONS scale = 1.0 $ comment\n')):
            with self.subTest(option=option):
                root, path = self.source(str(i), option=option)
                original = path.read_bytes()
                fix_geometry_scale([root])
                expected = original.replace(b'scale=1 ;', b'scale=1u ;').replace(
                    b'scale = 1.0 $', b'scale = 1u $')
                self.assertEqual(path.read_bytes(), expected)

    def test_true_geometry_mismatch_refused_without_backup(self):
        root, path = self.source()
        with sqlite3.connect(root / 'sampling_history.sqlite3') as db:
            db.execute("UPDATE dc_device_operating_points SET w_um=999 WHERE sample_id=1 AND device_name='XN'")
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'ratios must be 1e-6'):
            fix_geometry_scale([root])
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(root.glob('*.bak_*')))

    def test_postcheck_detects_outlier_beyond_examples_and_rolls_back_all(self):
        first, first_path = self.source('first')
        second, second_path = self.source('second')
        # The preflight report shows at most eight mismatch examples; this later
        # outlier must still be rejected by the complete postcheck of the scope.
        with sqlite3.connect(second / 'sampling_history.sqlite3') as db:
            db.execute("UPDATE dc_device_operating_points SET w_um=999 WHERE sample_id=3 AND device_name='XN'")
        originals = [path.read_bytes() for path in (first_path, second_path)]
        with self.assertRaisesRegex(ValueError, 'postcheck failed'):
            fix_geometry_scale([first, second])
        for root, path, original in zip((first, second), (first_path, second_path), originals):
            self.assertEqual(path.read_bytes(), original)
            backup, = root.glob('*.bak_*')
            self.assertEqual(backup.read_bytes(), original)

    def test_all_sources_preflight_before_any_write(self):
        first, first_path = self.source('first')
        second, _ = self.source('second', option='.option scale=1m\n')
        original = first_path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'only scale=1'):
            fix_geometry_scale([first, second])
        self.assertEqual(first_path.read_bytes(), original)
        self.assertFalse(list(first.glob('*.bak_*')))

    def test_ambiguous_or_unsupported_options_refused_without_changes(self):
        options = ('.option scale=1\n.option scale=1\n',
                   '.option scale=\n+ 1\n',
                   '.option scale=1 + 0\n',
                   '.options reltol=.001\n')
        for i, option in enumerate(options):
            with self.subTest(option=option):
                root, path = self.source(str(i), option=option)
                original = path.read_bytes()
                with self.assertRaises(ValueError):
                    fix_geometry_scale([root])
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(list(root.glob('*.bak_*')))

    def test_cli_reports_scope_and_rejects_duplicate_inputs(self):
        root, _ = self.source()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--input', str(root), '--max-samples', '1']), 0)
        report, = json.loads(output.getvalue())
        self.assertEqual(report['samples_scanned'], 1)
        self.assertFalse(report['scan_complete'])
        with self.assertRaisesRegex(ValueError, 'unique'):
            fix_geometry_scale([root, root])


if __name__ == '__main__':
    unittest.main()
