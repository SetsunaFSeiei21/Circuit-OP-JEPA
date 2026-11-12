"""Gate+decoder experiments retain their own identities and paired base costs."""
import json
from pathlib import Path
import tempfile
import unittest

from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_gate_v1_summary_pin import v1_run
from final_exam.tests.test_gate_v2_summary_pin import versioned_run
from final_exam.tests.test_gate_v3_summary_pin import v3_contract, v3_run
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook


class GateDecoderVersionsSummaryPinTests(unittest.TestCase):
    fixtures = (('v1', v1_run), ('v2', versioned_run), ('v3', v3_run))

    def test_all_six_variants_keep_decoder_label_scope_seed_statistics_and_shared_costs(self):
        for version, build in self.fixtures:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records = build(root)
                costs = json.loads((root / 'stages_pin.json').read_text())
                sheets = workbook(export_results_pin(root))
                rows = {row['mode']: row for row in summary_rows_pin(records)}
                for fusion in (False, True):
                    mode = f'device_op_distill_gate_decoder{"_fusion" if fusion else ""}_pin_{version}'
                    scope = 'fusion_self_attention' if fusion else 'all_encoder_self_attention'
                    self.assertEqual(rows[mode]['seed_ids'], [17, 29])
                    self.assertEqual(rows[mode]['mae_mean'], 3e6)
                    self.assertAlmostEqual(rows[mode]['mae_std'], 2**.5 * 1e6)
                    self.assertFalse(rows[mode]['requires_op_at_inference'])
                    input_rows = [text_values(row) for row in sheets['ModelInputs'].findall('x:sheetData/x:row', NS)
                                  if mode in text_values(row)]
                    self.assertEqual(len(input_rows), 1)
                    self.assertIn('gate and decoder', input_rows[0])
                    self.assertIn('distilled student', input_rows[0])
                    self.assertIn(scope, input_rows[0])
                    self.assertIn(version, input_rows[0])
                    comparisons = [text_values(row) for row in sheets['GateBeforeAfter'].findall('x:sheetData/x:row', NS)
                                   if mode in text_values(row)]
                    self.assertEqual(len(comparisons), 2)
                    for values in comparisons:
                        self.assertIn('gate and decoder', values)
                        self.assertIn(scope, values)
                        self.assertIn(version, values)
                self.assertEqual(text_values(sheets['GateStudy']).count('device_op_distilled_base_pin'), 1)
                self.assertEqual(text_values(sheets['GateStudy']).count('device_op_cascade_pin'), 1)
                self.assertEqual(len(sheets['TrainingCost'].findall('x:sheetData/x:row', NS)),
                                 len(costs['unique_stages']) + 5)
                self.assertEqual(costs, json.loads((root / 'stages_pin.json').read_text()))

    def test_report_cannot_mislabel_decoder_finetuning_as_gate_only(self):
        for version, _ in self.fixtures:
            for fusion in (False, True):
                mode = f'device_op_distill_gate_decoder{"_fusion" if fusion else ""}_pin_{version}'
                contract = v3_contract(mode) if version == 'v3' else dict(
                    gate_version=version,
                    attention_scope='fusion_self_attention' if fusion else 'all_encoder_self_attention')
                contract.update(base_training='distilled', final_trainable='gate_and_decoder')
                row = record(mode)
                row['report']['op_gate_pin'] = contract
                self.assertEqual(summary_rows_pin([row])[0]['mode'], mode)
                for key, wrong in (('base_training', 'supervised'), ('final_trainable', 'gate_only')):
                    with self.subTest(mode=mode, key=key):
                        row['report']['op_gate_pin'] = {**contract, key: wrong}
                        with self.assertRaisesRegex(ValueError, key):
                            summary_rows_pin([row])

    def test_comparison_rejects_wrong_phase_or_base_family_before_export(self):
        for version, build in self.fixtures:
            for key, wrong in (('final_phase', 'gate_only'), ('base_family', 'supervised')):
                with self.subTest(version=version, key=key), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    build(root)
                    mode = f'device_op_distill_gate_decoder_fusion_pin_{version}'
                    path = root / 'seed_17_pin' / mode / 'gate_comparison_pin.json'
                    comparison = json.loads(path.read_text())
                    comparison[key] = wrong
                    path.write_text(json.dumps(comparison))
                    with self.assertRaisesRegex(ValueError, key):
                        export_results_pin(root)
                    self.assertFalse((root / 'summary_pin.xlsx').exists())


if __name__ == '__main__':
    unittest.main()
