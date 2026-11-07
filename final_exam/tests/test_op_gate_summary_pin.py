"""OP gate reports preserve the A/B and measured-OP diagnostic contracts."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments.op_transfer.summarize import summary_rows as legacy_summary
from final_exam.summarize_pin import (
    GATE_INPUTS_PIN, ORACLES, SHUFFLE_CONTROLS_PIN, export_results_pin, summary_rows_pin,
)
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook, write_run


def gate_record(mode, seed=17, *, mae=2e6):
    value = record(mode, seed, mae=mae)
    value['report']['requires_measured_op'] = mode in ORACLES
    return value


class GateSummaryPinTests(unittest.TestCase):
    def test_all_eight_modes_keep_identity_and_oracle_contract(self):
        records = [gate_record(mode, seed, mae=2e6 if seed == 17 else 4e6)
                   for mode in GATE_INPUTS_PIN for seed in (17, 29)]
        snapshot = copy.deepcopy(records)
        rows = summary_rows_pin(records)
        self.assertEqual(records, snapshot)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row['requires_op_at_inference'], row['mode'] in ORACLES)
            self.assertTrue(row['uses_design_parameters'])
            self.assertEqual(row['mae_mean'], 3e6)
            self.assertAlmostEqual(row['mae_std'], 2**.5 * 1e6)
            self.assertEqual(row['seed_ids'], [17, 29])

    def test_misreported_oracle_and_design_input_fail_closed(self):
        for mode in GATE_INPUTS_PIN:
            value = gate_record(mode)
            value['report']['requires_measured_op'] = not value['report']['requires_measured_op']
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'input contract'):
                summary_rows_pin([value])
        # New oracle declarations are opt-in at the pin exporter; legacy contract stays strict.
        value = gate_record('device_op_gate_oracle_pin')
        value['mode'] = value['mode'][:-4]
        with self.assertRaisesRegex(ValueError, 'input contract'):
            legacy_summary([value])

    def test_xlsx_explains_feature_gate_oracle_and_shuffle_inputs(self):
        records = [gate_record(mode, mae=1 if mode in SHUFFLE_CONTROLS_PIN else 2e6)
                   for mode in GATE_INPUTS_PIN]
        with tempfile.TemporaryDirectory() as tmp:
            path = export_results_pin(write_run(Path(tmp), records))
            sheets = workbook(path)
            self.assertIn('ModelInputs', sheets)
            descriptions = '\n'.join(text_values(sheets['ModelInputs']))
            self.assertIn('A: predicted OP features', descriptions)
            self.assertIn('B: OP only controls', descriptions)
            self.assertIn('deterministic donors from training', descriptions)
            self.assertIn('missing fields use predicted OP', descriptions)
            headers = sheets['MAE'].find('x:sheetData/x:row[@r="7"]', NS)
            header_cells = headers.findall('x:c', NS)
            by_column = {cell.attrib['r'][:-1]: ''.join(cell.itertext()) for cell in header_cells}
            for name in SHUFFLE_CONTROLS_PIN:
                column = next(column for column, text in by_column.items() if name in text)
                self.assertIn('SHUFFLE CONTROL', by_column[column])
                self.assertNotEqual(sheets['MAE'].find(f'.//x:c[@r="{column}8"]', NS).attrib.get('s'), '9')
            for name in ORACLES & GATE_INPUTS_PIN.keys():
                label = next(text for text in by_column.values() if name in text)
                self.assertIn('TRUE OP REFERENCE', label)
            self.assertIn('TrainingCost', sheets)
            self.assertIn('Configuration', sheets)


if __name__ == '__main__':
    unittest.main()
