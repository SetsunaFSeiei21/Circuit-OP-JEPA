"""v2 reports retain normalization identity and deduplicate ungated references."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from final_exam.gate_version_registry_pin import (
    V1_GATE_MODELS_PIN, V2_GATE_MODELS_PIN, gate_version_pin,
)
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook, write_run


def versioned_run(directory, seeds=(17, 29), mixed=False):
    models = ['device_op_cascade_pin', *((V1_GATE_MODELS_PIN + V2_GATE_MODELS_PIN) if mixed else V2_GATE_MODELS_PIN)]
    records = [record(mode, seed, mae=2e6 if seed == 17 else 4e6)
               for mode in models for seed in seeds]
    for item in records:
        mode = item['mode']
        if mode == 'device_op_cascade_pin':
            item['report']['metrics']['UGF_Hz'].update(
                mae=6e6 if item['seed'] == 17 else 8e6, rmse=9e6, r2=.3)
            item['report']['ungated_base_checkpoint_pin'] = (
                'shared_pin/device_op_supervised_base_pin/checkpoint_pin.pt')
            continue
        scope = 'fusion_self_attention' if '_fusion_' in mode else 'all_encoder_self_attention'
        contract = dict(gate_version=gate_version_pin(mode), attention_scope=scope)
        if mode in V2_GATE_MODELS_PIN:
            contract.update(gate_relative_normalization='per_sample_per_field_all_valid_mos',
                            gate_input_dim=90, gate_hidden_dim=128, norm_eps=1e-5)
        item['report']['op_gate_pin'] = contract
    write_run(directory, records)
    for item in records:
        mode = item['mode']
        if not ('_distill_' in mode or '_supervised_' in mode):
            continue
        spec = staged_gate_spec_pin(mode)
        family = spec['base_training']
        before = copy.deepcopy(item['report'])
        del before['op_gate_pin']
        if family == 'supervised':
            before['metrics'] = copy.deepcopy(next(r['report']['metrics'] for r in records
                if r['mode'] == 'device_op_cascade_pin' and r['seed'] == item['seed']))
        else:
            before['metrics']['UGF_Hz'].update(
                mae=5e6 if item['seed'] == 17 else 7e6, rmse=9e6, r2=.3)
        comparison = dict(base_checkpoint=f'../shared_pin/device_op_{family}_base_pin/checkpoint_pin.pt',
            base_test=before, post_gate_test=item['report'], attention_scope=spec['attention_scope'],
            gate_version=gate_version_pin(mode), base_family=family, final_phase=spec['final_phase'])
        path = directory / f'seed_{item["seed"]}_pin' / mode / 'gate_comparison_pin.json'
        path.write_text(json.dumps(comparison))
    return records


class GateV2SummaryPinTests(unittest.TestCase):
    def test_names_and_aggregate_seeds_remain_version_specific(self):
        modes = ['device_op_gate_pin', *V1_GATE_MODELS_PIN, *V2_GATE_MODELS_PIN]
        records = [record(mode, seed) for mode in modes for seed in (17, 29)]
        before = copy.deepcopy(records)
        rows = summary_rows_pin(records)
        self.assertEqual(before, records)
        self.assertEqual({row['mode'] for row in rows}, set(modes))
        self.assertTrue(all(row['seed_ids'] == [17, 29] for row in rows))
        for invalid in ('unregistered_pin_v2', 'device_op_gate_pin_v4', 'device_op_gate_v2_pin_v2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                summary_rows_pin([record(invalid)])

    def test_v2_workbook_shows_normalization_and_all_models_without_extra_stage_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = versioned_run(root)
            stage_count = len(json.loads((root / 'stages_pin.json').read_text())['unique_stages'])
            sheets = workbook(export_results_pin(root))
            self.assertEqual(len(summary_rows_pin(records)), len(V2_GATE_MODELS_PIN) + 1)
            study = sheets['GateStudy']
            self.assertEqual(len(study.findall('x:sheetData/x:row', NS)), 5 + len(V2_GATE_MODELS_PIN) + 2)
            self.assertEqual(text_values(study).count('device_op_distilled_base_pin'), 1)
            self.assertEqual(text_values(study).count('device_op_cascade_pin'), 1)
            self.assertNotIn('device_op_supervised_base_pin', text_values(study))
            self.assertNotIn('device_op_distilled_base_pin', text_values(sheets['MAE']))
            self.assertEqual(len(sheets['TrainingCost'].findall('x:sheetData/x:row', NS)), 5 + stage_count)
            inputs = sheets['ModelInputs']
            self.assertIn('Gate relative normalization', text_values(inputs))
            self.assertIn('v2', text_values(sheets['GateBeforeAfter']))
            v2_rows = 0
            for row in inputs.findall('x:sheetData/x:row', NS):
                text = text_values(row)
                if not any(mode in text for mode in V2_GATE_MODELS_PIN):
                    continue
                v2_rows += 1
                self.assertIn('per_sample_per_field_all_valid_mos', text)
                cells = {''.join(c for c in cell.attrib['r'] if c.isalpha()): cell
                         for cell in row.findall('x:c', NS)}
                self.assertEqual(float(cells['M'].find('x:v', NS).text), 90)
                self.assertEqual(float(cells['N'].find('x:v', NS).text), 128)
                self.assertEqual(float(cells['O'].find('x:v', NS).text), 1e-5)
            self.assertEqual(v2_rows, len(V2_GATE_MODELS_PIN))

    def test_mixed_v1_v2_studies_keep_names_and_share_each_base_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versioned_run(root, mixed=True)
            sheets = workbook(export_results_pin(root))
            study = sheets['GateStudy']
            self.assertEqual(len(study.findall('x:sheetData/x:row', NS)), 5 + len(V1_GATE_MODELS_PIN) + len(V2_GATE_MODELS_PIN) + 2)
            text = text_values(study)
            self.assertEqual(text.count('device_op_distilled_base_pin'), 1)
            self.assertEqual(text.count('device_op_cascade_pin'), 1)
            self.assertNotIn('device_op_supervised_base_pin', text)
            for mode in V1_GATE_MODELS_PIN + V2_GATE_MODELS_PIN:
                self.assertEqual(text.count(mode), 1)
            self.assertIn('v1', text)
            self.assertIn('v2', text)

    def test_v2_report_contract_rejects_version_scope_or_normalization_mismatch(self):
        for update in (dict(gate_version='v1'), dict(attention_scope='all_encoder_self_attention'),
                       dict(gate_relative_normalization='per_device'), dict(gate_input_dim=49),
                       dict(gate_hidden_dim=0), dict(norm_eps=0), dict(norm_eps=float('nan'))):
            row = record('device_op_gate_fusion_pin_v2')
            row['report']['op_gate_pin'] = dict(gate_version='v2', attention_scope='fusion_self_attention',
                gate_relative_normalization='per_sample_per_field_all_valid_mos', gate_input_dim=90,
                gate_hidden_dim=128, norm_eps=1e-5)
            row['report']['op_gate_pin'].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                summary_rows_pin([row])

    def test_metadata_cannot_silently_change_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = versioned_run(root)
            row = next(r for r in records if r['mode'] == 'device_op_gate_pin_v2' and r['seed'] == 29)
            row['report']['op_gate_pin']['norm_eps'] = 1e-4
            (root / 'results_pin.json').write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'differs between seeds/splits'):
                export_results_pin(root)
            self.assertFalse((root / 'summary_pin.xlsx').exists())

    def test_v2_comparison_version_and_shared_base_conflicts_are_rejected(self):
        for target, change, error in (
            ('gate_version', 'v1', 'version differs'),
            ('base_mae', 1., 'Conflicting reports'),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                versioned_run(root, seeds=(17,))
                path = root / 'seed_17_pin/device_op_distill_gate_fusion_pin_v2/gate_comparison_pin.json'
                saved = json.loads(path.read_text())
                if target == 'base_mae':
                    saved['base_test']['metrics']['UGF_Hz']['mae'] += change
                else:
                    saved[target] = change
                path.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, error):
                    export_results_pin(root)
                self.assertFalse((root / 'summary_pin.xlsx').exists())


if __name__ == '__main__':
    unittest.main()
