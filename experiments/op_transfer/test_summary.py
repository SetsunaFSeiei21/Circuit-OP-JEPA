"""Validate aggregation and OOXML export using only the standard library."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from experiments.op_transfer.summarize import export_results, summary_rows


def record(seed=17, mode='predicted_op', split='test', r2=.5, count=8):
    return dict(seed=seed, mode=mode, report=dict(split=split,
        requires_measured_op=mode in ('op_oracle', 'design_op_oracle'), metrics={
            'UGF_Hz': dict(count=count, mae=2e6, rmse=3e6, r2=r2)}))


class SummaryTests(unittest.TestCase):
    def test_sample_std_and_split_separation(self):
        other = record(seed=29, r2=.9)
        other['report']['metrics']['UGF_Hz']['mae'] = 4e6
        rows = summary_rows([record(), other, record(split='ood', r2=-1)])
        test = next(row for row in rows if row['split'] == 'test')
        self.assertEqual(test['mae_mean'], 3e6)
        self.assertAlmostEqual(test['mae_std'], 2**.5 * 1e6)
        self.assertAlmostEqual(test['r2_mean'], .7)
        self.assertEqual(test['num_seeds'], 2)
        self.assertEqual(test['scope'], 'pooled')
        self.assertEqual(test['circuit'], 'ALL_SELECTED_CIRCUITS')
        ood = next(row for row in rows if row['split'] == 'ood')
        self.assertIsNone(ood['r2_std'])
        self.assertEqual(ood['r2_mean'], -1)

    def test_undefined_scores_and_op_contract(self):
        rows = summary_rows([record(r2=None), record(seed=29, r2=.7), record(mode='op_oracle')])
        predicted = next(row for row in rows if row['mode'] == 'predicted_op')
        self.assertEqual(predicted['r2_defined_seeds'], 1)
        self.assertIsNone(predicted['r2_std'])
        self.assertFalse(predicted['requires_op_at_inference'])
        oracle = next(row for row in rows if row['mode'] == 'op_oracle')
        self.assertTrue(oracle['requires_op_at_inference'])
        self.assertFalse(oracle['uses_design_parameters'])

    def test_incompatible_reports_rejected(self):
        for reports in ([record(), record()], [record(), record(seed=29, count=9)],
                        [record(r2=float('nan'))], [record(count=0)]):
            with self.subTest(reports=reports), self.assertRaises(ValueError):
                summary_rows(reports)
        bad = record()
        bad['report']['requires_measured_op'] = True
        with self.assertRaisesRegex(ValueError, 'input contract'):
            summary_rows([bad])

    def test_distilled_cascade_and_design_op_teacher_input_contracts(self):
        rows = summary_rows([record(mode='student_predicted_op'), record(mode='design_op_oracle')])
        student = next(row for row in rows if row['mode'] == 'student_predicted_op')
        teacher = next(row for row in rows if row['mode'] == 'design_op_oracle')
        self.assertFalse(student['requires_op_at_inference'])
        self.assertTrue(student['uses_design_parameters'])
        self.assertTrue(teacher['requires_op_at_inference'])
        self.assertTrue(teacher['uses_design_parameters'])
        for mode in ('student_predicted_op', 'design_op_oracle'):
            invalid = record(mode=mode)
            invalid['report']['requires_measured_op'] = not invalid['report']['requires_measured_op']
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'input contract'):
                summary_rows([invalid])

    def test_export_includes_new_student_and_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'results.json').write_text(json.dumps([
                record(mode='student_predicted_op'), record(mode='design_op_oracle')]))
            path = export_results(root)
            with ZipFile(path) as archive:
                detail = archive.read('xl/worksheets/sheet2.xml').decode()
                self.assertIn('student_predicted_op', detail)
                self.assertIn('design_op_oracle', detail)

    def test_xlsx_export_retains_full_precision_and_missing_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'results.json').write_text(json.dumps([record(r2=.512345678901)]))
            (root/'run.json').write_text(json.dumps({'status':'complete'}))
            path = export_results(root)
            self.assertEqual(path, root/'summary.xlsx')
            with ZipFile(path) as archive:
                for filename in archive.namelist():
                    if filename.endswith('.xml'):
                        ET.fromstring(archive.read(filename))
                display = archive.read('xl/worksheets/sheet1.xml').decode()
                detail = archive.read('xl/worksheets/sheet2.xml').decode()
                self.assertIn('0.512345678901', detail)
                self.assertIn('one seed has no std', display)
                self.assertIn('scope=pooled', display)
                self.assertIn('MHz', display)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                export_results(root)
            self.assertEqual(path.read_bytes(), before)

    def test_cli_has_no_torch_dependency(self):
        script = Path(__file__).with_name('summarize.py')
        process = subprocess.run([sys.executable, '-S', str(script), '--help'],
                                 capture_output=True, text=True, check=True)
        self.assertIn('mean', process.stdout)


if __name__ == '__main__':
    unittest.main()
