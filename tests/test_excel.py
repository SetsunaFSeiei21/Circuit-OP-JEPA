import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from device_jepa.excel import export_summary, load_summary, main


NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
ROOT = Path(__file__).resolve().parents[1]


def summary_row(mode='zerosim_device', **changes):
    row = dict(mode=mode, backend='test_backend', split='test', scope='circuit',
               circuit='two_stage_opamp_otaf', metric='UGF_Hz', scale='physical',
               num_seeds=3, count_per_seed=6000, requires_op_at_inference=False,
               uses_design_parameters=True, op_training_use='none',
               mae_defined_seeds=3, mae_mean=2_000_000., mae_std=100_000.,
               rmse_defined_seeds=3, rmse_mean=4_000_000., rmse_std=200_000.,
               r2_defined_seeds=3, r2_mean=.9, r2_std=.01)
    row.update(changes)
    return row


def worksheet(path, number=1):
    with ZipFile(path) as archive:
        return ET.fromstring(archive.read(f'xl/worksheets/sheet{number}.xml'))


def cells(path, number=1):
    result = []
    for cell in worksheet(path, number).findall('.//s:c', NS):
        value = cell.find('s:is/s:t', NS) if cell.get('t') == 'inlineStr' else cell.find('s:v', NS)
        result.append((cell, value.text if value is not None else None))
    return result


class ExcelTests(unittest.TestCase):
    def test_units_linked_formulas_numeric_sources_and_best_means(self):
        rows = [summary_row(), summary_row('future_model', mae_mean=0., mae_std=0.,
                                          rmse_mean=3_000_000., r2_mean=.95)]
        with tempfile.TemporaryDirectory() as tmp:
            path = export_summary(rows, Path(tmp)/'report.xlsx', models=['zerosim_device', 'future_model'])
            by_ref = {cell.get('r'): (cell, value) for cell, value in cells(path)}
            self.assertEqual(by_ref['C14'][1], '2.0000 ± 0.1000')
            self.assertEqual(by_ref['D14'][1], '0.0000 ± 0.0000')
            self.assertEqual(by_ref['C16'][1], '0.9000 ± 0.0100')
            self.assertEqual(by_ref['D16'][0].get('s'), '9')
            self.assertEqual(by_ref['D14'][0].get('s'), '9')
            self.assertIn('ISNUMBER', by_ref['D14'][0].find('s:f', NS).text)
            self.assertIn('/1000000', by_ref['C14'][0].find('s:f', NS).text)
            self.assertNotIn('/1000000', by_ref['C16'][0].find('s:f', NS).text)
            self.assertIn('2000000.0', [value for _, value in cells(path, 2)])
            with ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                for name in archive.namelist():
                    ET.fromstring(archive.read(name))

    def test_single_seed_preserves_mean_and_undefined_r2(self):
        row = summary_row(num_seeds=1, mae_defined_seeds=1, mae_mean=0., mae_std=None,
                          rmse_defined_seeds=1, rmse_std=None, r2_defined_seeds=0, r2_mean=None, r2_std=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_summary([row], Path(tmp)/'report.xlsx')
            values = [value for _, value in cells(path)]
            self.assertIn('0.0000 ± n/a', values)
            self.assertIn('n/a', values)
            self.assertNotIn('0.0000 ± 0.0000', values)
            self.assertFalse(any(cell.get('s') == '9' for cell, _ in cells(path)))

    def test_selection_keeps_splits_circuits_scopes_and_backends_separate(self):
        rows = [summary_row(circuit=circuit, split=split, scope=scope, scale=scale)
                for circuit in ('Circuit B', '__all__') for split in ('test', 'ood')
                for scope in ('circuit', 'pooled') for scale in ('physical', 'normalized')]
        rows.append(summary_row(backend='another_backend'))
        rows.append(summary_row('zerosim_device_op_only', uses_design_parameters=False,
                                requires_op_at_inference=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = export_summary(rows, Path(tmp)/'all.xlsx')
            values = [value for _, value in cells(path)]
            self.assertEqual(sum('| split=' in str(v) for v in values), 5)
            self.assertIn('__all__ | split=test | scope=circuit | scale=physical', values)
            self.assertIn('zerosim_device\nanother_backend', values)
            self.assertIn('zerosim_device_op_only', values)
            self.assertEqual(len(worksheet(path, 2).findall('s:sheetData/s:row', NS)), len(rows)+7)
            chosen = export_summary(rows, Path(tmp)/'selected.xlsx', split='ood', scope='pooled',
                                    scale='normalized', models=['zerosim_device'])
            selected_values = [value for _, value in cells(chosen)]
            self.assertEqual(sum('| split=' in str(v) for v in selected_values), 2)
            self.assertIn('MAE (normalized)', selected_values)
            self.assertNotIn('MAE (MHz)', selected_values)

    def test_rounding_does_not_create_false_ties_and_small_values_remain_visible(self):
        rows = [summary_row(metric='gain', mae_mean=2.00004, mae_std=.01),
                summary_row('other', metric='gain', mae_mean=2.00003, mae_std=.01)]
        with tempfile.TemporaryDirectory() as tmp:
            path = export_summary(rows, Path(tmp)/'report.xlsx', models=['zerosim_device', 'other'])
            by_ref = {cell.get('r'): (cell, value) for cell, value in cells(path)}
            self.assertEqual(by_ref['C14'][1], by_ref['D14'][1])
            self.assertNotEqual(by_ref['C14'][0].get('s'), '9')
            self.assertEqual(by_ref['D14'][0].get('s'), '9')
            rows[0].update(mae_mean=1e-12, mae_std=1e-13)
            small = export_summary(rows[:1], Path(tmp)/'small.xlsx')
            self.assertIn('1.0000E-12 ± 1.0000E-13', [v for _, v in cells(small)])

    def test_unequal_counts_suppress_winner_highlight(self):
        rows = [summary_row(), summary_row('other', count_per_seed=5999)]
        with tempfile.TemporaryDirectory() as tmp:
            path = export_summary(rows, Path(tmp)/'report.xlsx')
            self.assertFalse(any(cell.get('s') == '9' for cell, _ in cells(path)))
            self.assertIn('varies', [v for _, v in cells(path)])

    def test_bad_inputs_leave_existing_report_unchanged(self):
        base = summary_row()
        invalid = [[base, base], [summary_row(mae_mean=float('nan'))],
                   [summary_row(rmse_std=-1)], [summary_row(r2_defined_seeds=0)],
                   [summary_row(mae_std=None)], [summary_row(num_seeds=0)]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'report.xlsx'
            path.write_bytes(b'previous report')
            for rows in invalid:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    export_summary(rows, path)
                self.assertEqual(path.read_bytes(), b'previous report')
            for kwargs in ({'split': 'ood'}, {'models': ['missing']}, {'decimals': 0}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    export_summary([base], path, **kwargs)
                self.assertEqual(path.read_bytes(), b'previous report')

    def test_csv_json_and_directory_entrypoints_without_site_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run with spaces'
            comparison = root/'results/comparison'
            comparison.mkdir(parents=True)
            row = summary_row()
            csv_path = comparison/'summary.csv'
            with csv_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            original = csv_path.read_bytes()
            loaded, metadata, path = load_summary(root)
            self.assertEqual(loaded, [row])
            self.assertEqual(metadata, {})
            self.assertEqual(path, csv_path)
            result = subprocess.run([sys.executable, '-S', str(ROOT/'scripts/run/export_results_excel.py'),
                                     '--input', str(root)], cwd=tmp, capture_output=True, text=True, check=True)
            output = comparison/'model_comparison.xlsx'
            self.assertTrue(output.is_file())
            self.assertIn(str(output), result.stdout)
            self.assertIn('Comparison status: UNKNOWN (CSV does not verify run compatibility)',
                          [v for _, v in cells(output)])
            self.assertEqual(csv_path.read_bytes(), original)
            (comparison/'comparison.json').write_text(json.dumps({'summary': [row], 'complete': False}))
            loaded, metadata, path = load_summary(root/'results')
            self.assertFalse(metadata['complete'])
            self.assertEqual(path.name, 'comparison.json')
            main(['--input', str(comparison)])
            self.assertIn('Comparison status: INCOMPLETE (as declared by comparison.json)',
                          [v for _, v in cells(output)])

    def test_source_text_is_not_interpreted_as_excel_formula(self):
        row = summary_row(circuit='=HYPERLINK("https://example.com")', metric='gain & <value>')
        with tempfile.TemporaryDirectory() as tmp:
            path = export_summary([row], Path(tmp)/'report.xlsx')
            original = [cell for cell, value in cells(path, 2) if value == row['circuit']]
            self.assertEqual(len(original), 1)
            self.assertEqual(original[0].get('t'), 'inlineStr')
            self.assertIsNone(original[0].find('s:f', NS))

    def test_comparison_command_automatically_exports_workbook(self):
        from device_jepa.comparison import main as compare
        from tests.test_comparison import report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'test_report.json'
            source.write_text(json.dumps(report('zerosim_device')))
            output = root/'comparison'
            compare(['--reports', str(source), '--output', str(output), '--models', 'zerosim_device'])
            self.assertTrue((output/'summary.csv').is_file())
            self.assertTrue((output/'model_comparison.xlsx').is_file())
            self.assertIn('1.0000 ± n/a', [v for _, v in cells(output/'model_comparison.xlsx')])


if __name__ == '__main__':
    unittest.main()
