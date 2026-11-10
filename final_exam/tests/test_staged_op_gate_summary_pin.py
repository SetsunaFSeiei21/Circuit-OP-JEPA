"""Staged-gate reports retain deployable inputs, shared costs and base comparisons."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from final_exam.summarize_pin import (
    STAGED_GATE_INPUTS_PIN, export_results_pin, summary_rows_pin,
)
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook, write_run


def staged_run(directory, seeds=(17, 29)):
    records = []
    for mode in STAGED_GATE_INPUTS_PIN:
        for seed in seeds:
            item = record(mode, seed, mae=2e6 if seed == 17 else 4e6)
            family = 'supervised' if 'supervised' in mode else 'distilled'
            item['stage_ids'] = [f'seed_{seed}_pin/device_op_predictor_pin',
                                 f'seed_{seed}_pin/device_op_{family}_base_pin',
                                 f'seed_{seed}_pin/{mode}']
            records.append(item)
    write_run(directory, records)
    for item in records:
        folder = directory / f'seed_{item["seed"]}_pin' / item['mode']
        before = copy.deepcopy(item['report'])
        before['metrics']['UGF_Hz'].update(mae=5e6, rmse=7e6, r2=.3)
        family = 'supervised' if 'supervised' in item['mode'] else 'distilled'
        comparison = dict(base_checkpoint=f'../shared_pin/device_op_{family}_base_pin/checkpoint_pin.pt',
                          base_test=before, post_gate_test=item['report'])
        (folder / 'gate_comparison_pin.json').write_text(json.dumps(comparison))
    return records


class StagedGateSummaryPinTests(unittest.TestCase):
    def test_three_models_keep_inputs_std_shared_cost_and_base_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = staged_run(root)
            rows = summary_rows_pin(records)
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertFalse(row['requires_op_at_inference'])
                self.assertTrue(row['uses_design_parameters'])
                self.assertEqual(row['mae_mean'], 3e6)
                self.assertAlmostEqual(row['mae_std'], 2**.5 * 1e6)
            sheets = workbook(export_results_pin(root))
            inputs = '\n'.join(text_values(sheets['ModelInputs']))
            for mode in STAGED_GATE_INPUTS_PIN:
                self.assertIn(mode, inputs)
            self.assertIn('gate and decoder', inputs)
            self.assertIn('No teacher or latent distillation dependency', inputs)
            self.assertIn('GateBeforeAfter', sheets)
            comparison = sheets['GateBeforeAfter']
            self.assertEqual(len(comparison.findall('x:sheetData/x:row', NS)), 11)
            # Six distinct model/seed rows; frozen bases are references, not added model runs.
            first = comparison.find('x:sheetData/x:row[@r="6"]', NS)
            cells = {cell.attrib['r']: cell for cell in first.findall('x:c', NS)}
            self.assertEqual(cells['G6'].find('x:v', NS).text, '5000000.0')
            self.assertEqual(cells['H6'].find('x:v', NS).text, '2000000.0')
            self.assertEqual(cells['I6'].find('x:v', NS).text, '-3000000.0')
            self.assertAlmostEqual(float(cells['O6'].find('x:v', NS).text), .2)
            costs = text_values(sheets['TrainingCost'])
            self.assertEqual(costs.count('seed_17_pin/device_op_distilled_base_pin'), 1)
            self.assertEqual(costs.count('seed_17_pin/device_op_predictor_pin'), 1)
            self.assertIn('additional training', '\n'.join(text_values(sheets['MAE'])))

    def test_missing_comparison_fails_before_writing_complete_workbook(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = staged_run(root, (17,))
            path = root / 'seed_17_pin' / records[0]['mode'] / 'gate_comparison_pin.json'
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing staged gate comparison'):
                export_results_pin(root)
            self.assertFalse((root / 'summary_pin.xlsx').exists())
            sheets = workbook(export_results_pin(root, allow_incomplete=True))
            self.assertIn('MISSING COMPARISON', text_values(sheets['GateBeforeAfter']))
            self.assertTrue(any('INCOMPLETE' in value for value in text_values(sheets['MAE'])))
            self.assertFalse(any(cell.attrib.get('s') == '9'
                                 for cell in sheets['MAE'].findall('.//x:c', NS)))

    def test_mismatched_final_or_counts_are_rejected(self):
        for kind in ('final', 'count', 'split', 'real_op'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                records = staged_run(root, (17,))
                path = root / 'seed_17_pin' / records[0]['mode'] / 'gate_comparison_pin.json'
                comparison = json.loads(path.read_text())
                if kind == 'final':
                    comparison['post_gate_test']['metrics']['UGF_Hz']['mae'] = 1
                elif kind == 'count':
                    comparison['base_test']['metrics']['UGF_Hz']['count'] = 9
                elif kind == 'split':
                    comparison['base_test']['split'] = 'val'
                else:
                    comparison['base_test']['requires_measured_op'] = True
                path.write_text(json.dumps(comparison))
                with self.assertRaises(ValueError):
                    export_results_pin(root)
                self.assertFalse((root / 'summary_pin.xlsx').exists())


if __name__ == '__main__':
    unittest.main()
