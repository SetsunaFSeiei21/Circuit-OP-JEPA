"""Geometry preflight reports real mismatches without changing source files."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from test_adapter import fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('geometry_check', ROOT / 'scripts/run/check_raw_geometry.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class RawGeometryCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)

    def test_full_check_is_consistent_and_read_only_with_missing_geometry_reported(self):
        before = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in self.root.iterdir() if p.is_file()}
        report = checker.inspect_geometry(self.root)
        self.assertEqual(report['status'], 'consistent_observed_geometry')
        self.assertEqual(report['geometry_scale_to_m'], 1e-6)
        self.assertEqual(report['samples_scanned'], 3)
        self.assertTrue(report['scan_complete'])
        self.assertEqual(report['comparisons'], {'W_m': 5, 'L_m': 5, 'm': 5})
        self.assertEqual(report['missing_geometry'], {'W_m': 1, 'L_m': 1, 'm': 1})
        after = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in self.root.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_scale_reports_the_million_factor_without_guessing_a_fix(self):
        p = self.root / 'Tiny.sp'
        p.write_text(p.read_text().replace('.option scale=1u\n', ''))
        before = p.read_bytes()
        report = checker.inspect_geometry(self.root)
        self.assertEqual(report['status'], 'mismatch')
        self.assertEqual(report['mismatches'], {'W_m': 5, 'L_m': 5, 'm': 0})
        self.assertAlmostEqual(report['examples'][0]['op_over_input_ratio'], 1e-6)
        self.assertEqual(p.read_bytes(), before)

    def test_real_sizing_and_multiplier_changes_are_still_rejected(self):
        with sqlite3.connect(self.root / 'sampling_history.sqlite3') as db:
            db.execute("UPDATE dc_device_operating_points SET w_um=999,multiplicity=999 WHERE sample_id=1 AND device_name='XN'")
        report = checker.inspect_geometry(self.root)
        self.assertEqual(report['status'], 'mismatch')
        self.assertEqual(report['mismatches'], {'W_m': 1, 'L_m': 0, 'm': 1})

    def test_limited_scan_is_not_claimed_as_full_validation(self):
        report = checker.inspect_geometry(self.root, max_samples=1)
        self.assertEqual(report['source_samples'], 3)
        self.assertEqual(report['samples_scanned'], 1)
        self.assertFalse(report['scan_complete'])
        self.assertEqual(report['comparisons'], {'W_m': 2, 'L_m': 2, 'm': 2})

    def test_fully_missing_dimensions_are_unverifiable(self):
        with sqlite3.connect(self.root / 'sampling_history.sqlite3') as db:
            db.execute('UPDATE dc_device_operating_points SET w_um=NULL,l_um=NULL,multiplicity=NULL')
        report = checker.inspect_geometry(self.root)
        self.assertEqual(report['status'], 'unverifiable')
        self.assertFalse(any(report['comparisons'].values()))
        self.assertEqual(report['missing_geometry'], {'W_m': 6, 'L_m': 6, 'm': 6})

    def test_orphans_and_device_type_conflicts_raise_explicit_errors(self):
        with sqlite3.connect(self.root / 'sampling_history.sqlite3') as db:
            db.execute("UPDATE dc_device_operating_points SET device_type='PMOS' WHERE sample_id=1 AND device_name='XN'")
        with self.assertRaisesRegex(ValueError, 'model/type mismatch'):
            checker.inspect_geometry(self.root)
        with sqlite3.connect(self.root / 'sampling_history.sqlite3') as db:
            db.execute("UPDATE dc_device_operating_points SET device_type='NMOS',sample_id=99 WHERE sample_id=1 AND device_name='XN'")
        with self.assertRaisesRegex(ValueError, 'Orphan OP'):
            checker.inspect_geometry(self.root)

    def test_cli_checks_every_circuit_and_returns_nonzero_for_failure(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = checker.main(['--input', str(self.root), str(self.root / 'missing')])
        reports = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual([r['status'] for r in reports], ['consistent_observed_geometry', 'error'])

    def test_cli_examples_are_bounded_and_invalid_limits_rejected(self):
        with sqlite3.connect(self.root / 'sampling_history.sqlite3') as db:
            db.execute('UPDATE dc_device_operating_points SET w_um=999')
        report = checker.inspect_geometry(self.root, max_examples=2)
        self.assertEqual(len(report['examples']), 2)
        self.assertEqual(report['mismatches']['W_m'], 5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            checker.main(['--input', str(self.root), '--max-samples', '0'])


if __name__ == '__main__':
    unittest.main()
