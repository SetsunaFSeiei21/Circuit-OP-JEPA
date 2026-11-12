"""Versioned gates preserve names and expose shared bases without duplicate runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from final_exam.gate_version_registry_pin import V1_GATE_MODELS_PIN, V1_STAGED_GATE_MODELS_PIN
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook, write_run


def v1_run(directory, seeds=(17, 29)):
    models = ['device_op_cascade_pin', *V1_GATE_MODELS_PIN]
    records = [record(mode, seed, mae=2e6 if seed == 17 else 4e6)
               for mode in models for seed in seeds]
    for item in records:
        if item['mode'] == 'device_op_cascade_pin':
            item['report']['metrics']['UGF_Hz'].update(
                mae=6e6 if item['seed'] == 17 else 8e6, rmse=9e6, r2=.3)
            item['report']['ungated_base_checkpoint_pin'] = (
                'shared_pin/device_op_supervised_base_pin/checkpoint_pin.pt')
    write_run(directory, records)
    for item in records:
        if item['mode'] not in V1_STAGED_GATE_MODELS_PIN:
            continue
        spec = staged_gate_spec_pin(item['mode'])
        family = spec['base_training']
        before = copy.deepcopy(item['report'])
        if family == 'supervised':
            before['metrics'] = copy.deepcopy(next(r['report']['metrics'] for r in records
                if r['mode'] == 'device_op_cascade_pin' and r['seed'] == item['seed']))
        else:
            before['metrics']['UGF_Hz'].update(
                mae=5e6 if item['seed'] == 17 else 7e6, rmse=9e6, r2=.3)
        value = dict(base_checkpoint=f'../shared_pin/device_op_{family}_base_pin/checkpoint_pin.pt',
                     base_test=before, post_gate_test=item['report'],
                     attention_scope=spec['attention_scope'], gate_version='v1',
                     base_family=family, final_phase=spec['final_phase'])
        path = directory / f'seed_{item["seed"]}_pin' / item['mode'] / 'gate_comparison_pin.json'
        path.write_text(json.dumps(value))
    return records


class GateV1SummaryPinTests(unittest.TestCase):
    def test_exact_versioned_names_do_not_collide_with_legacy(self):
        names = ['device_op_gate_pin', *V1_GATE_MODELS_PIN]
        records = [record(name, seed) for name in names for seed in (17, 29)]
        original = copy.deepcopy(records)
        rows = summary_rows_pin(records)
        self.assertEqual(records, original)
        self.assertEqual({row['mode'] for row in rows}, set(names))
        self.assertTrue(all(row['seed_ids'] == [17, 29] for row in rows))
        self.assertFalse(any(row['mode'].endswith('_v1_pin') for row in rows))
        with self.assertRaises(ValueError):
            summary_rows_pin([record('unregistered_pin_v1')])

    def test_workbook_keeps_all_models_and_one_shared_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = v1_run(root)
            sheets = workbook(export_results_pin(root))
            inputs = '\n'.join(text_values(sheets['ModelInputs']))
            self.assertIn('Gate version', inputs)
            self.assertIn('v1', inputs)
            self.assertIn('fusion_self_attention', inputs)
            self.assertIn('encoder, decoder and gate', inputs)
            study = sheets['GateStudy']
            self.assertEqual(len(study.findall('x:sheetData/x:row', NS)), 5 + len(V1_GATE_MODELS_PIN) + 2)
            study_text = text_values(study)
            self.assertEqual(study_text.count('device_op_distilled_base_pin'), 1)
            self.assertNotIn('device_op_supervised_base_pin', study_text)
            self.assertEqual(study_text.count('device_op_cascade_pin'), 1)
            for row in study.findall('x:sheetData/x:row', NS):
                if 'device_op_distilled_base_pin' not in text_values(row):
                    continue
                cells = {''.join(c for c in cell.attrib['r'] if c.isalpha()): cell
                         for cell in row.findall('x:c', NS)}
                self.assertEqual(float(cells['M'].find('x:v', NS).text), 6e6)
                self.assertAlmostEqual(float(cells['N'].find('x:v', NS).text), 2**.5 * 1e6)
            self.assertEqual(len(summary_rows_pin(records)), len(V1_GATE_MODELS_PIN) + 1)
            self.assertNotIn('device_op_distilled_base_pin', text_values(sheets['MAE']))
            self.assertIn('Gate version', text_values(sheets['GateBeforeAfter']))

    def test_shared_base_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1_run(root, (17,))
            path = root / 'seed_17_pin/device_op_distill_gate_fusion_pin_v1/gate_comparison_pin.json'
            value = json.loads(path.read_text())
            value['base_test']['metrics']['UGF_Hz']['mae'] += 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Conflicting reports'):
                export_results_pin(root)
            self.assertFalse((root / 'summary_pin.xlsx').exists())

    def test_alias_cannot_hide_a_different_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1_run(root, (17,))
            path = root / 'results_pin.json'
            records = json.loads(path.read_text())
            records[0]['report']['metrics']['UGF_Hz']['mae'] += 1
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'disagrees with its shared ungated'):
                export_results_pin(root)

    def test_version_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1_run(root, (17,))
            path = root / 'seed_17_pin/device_op_distill_gate_pin_v1/gate_comparison_pin.json'
            value = json.loads(path.read_text())
            value['gate_version'] = 'v0'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'version differs'):
                export_results_pin(root)


if __name__ == '__main__':
    unittest.main()
