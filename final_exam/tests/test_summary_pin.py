"""Verify report aggregation, planned completeness and Excel presentation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from final_exam.summarize_pin import export_results_pin as export_results, summary_rows_pin as summary_rows


NS = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def record(mode='mlp_pin', seed=17, *, mae=2e6, r2=.5, split='test', circuit='Opamp1'):
    return dict(mode=mode, seed=seed, circuit=circuit,
                stage_ids=[f'seed_{seed}_pin/shared_pin', f'seed_{seed}_pin/{mode}'],
                report=dict(split=split, requires_measured_op=mode in ('op_oracle_pin', 'design_op_oracle_pin'),
                            metrics={'UGF_Hz': dict(count=8, mae=mae, rmse=mae * 1.5,
                                                   r2=r2, normalized_rmse=.1)}))


def write_run(directory, records, *, models=None, seeds=None, status='complete'):
    directory.mkdir(parents=True, exist_ok=True)
    config = {'model': {'dim': 32}, 'dataset': {'train_circuits': ['Opamp1']}}
    run = dict(status=status, models=models or list(dict.fromkeys(item['mode'] for item in records)),
               seeds=seeds or sorted({item['seed'] for item in records}), evaluation_splits=['test'], circuit='Opamp1')
    stages = []
    for stage_id in dict.fromkeys(stage_id for item in records for stage_id in item['stage_ids']):
        stages.append(dict(stage_id=stage_id, planned_epochs=2, epochs=1, seconds=2.5,
                           optimizer_steps=4, examples_seen=32, parameters=12, trainable_parameters=8))
    for filename, value in [('results_pin.json', records), ('config_pin.json', config), ('run_pin.json', run),
                            ('stages_pin.json', {'unique_stages': stages})]:
        (directory / filename).write_text(json.dumps(value))
    for item in records:
        destination = directory / f'seed_{item["seed"]}_pin' / item['mode']
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'config_pin.json').write_text(json.dumps(config))
    return directory


def workbook(path):
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read('xl/workbook.xml'))
        names = [entry.attrib['name'] for entry in root.findall('x:sheets/x:sheet', NS)]
        for name in archive.namelist():
            if name.endswith('.xml'):
                ET.fromstring(archive.read(name))
        return {name: ET.fromstring(archive.read(f'xl/worksheets/sheet{index}.xml'))
                for index, name in enumerate(names, 1)}


def text_values(root):
    return [item.text for item in root.findall('.//x:t', NS)]


class SummaryPinTests(unittest.TestCase):
    def test_pin_names_are_retained_without_mutating_source_records(self):
        records = [record('op_oracle_pin'), record('design_op_oracle_pin'), record('predicted_op_pin')]
        original = copy.deepcopy(records)
        rows = summary_rows(records)
        self.assertEqual(records, original)
        by_model = {row['mode']: row for row in rows}
        self.assertEqual(set(by_model), {'op_oracle_pin', 'design_op_oracle_pin', 'predicted_op_pin'})
        self.assertFalse(by_model['op_oracle_pin']['uses_design_parameters'])
        self.assertTrue(by_model['design_op_oracle_pin']['uses_design_parameters'])
        self.assertTrue(by_model['op_oracle_pin']['requires_op_at_inference'])
        self.assertFalse(by_model['predicted_op_pin']['requires_op_at_inference'])
        self.assertTrue(all(row['backend'] == 'final_exam_pin' for row in rows))

    def test_device_artifacts_and_unsuffixed_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'end in _pin'):
            summary_rows([record('mlp')])
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_run(Path(tmp), [record()])
            with self.assertRaisesRegex(ValueError, '_pin.xlsx'):
                export_results(directory, directory/'summary.xlsx')
            self.assertFalse((directory/'summary.xlsx').exists())
            run_path = directory/'run_pin.json'
            run = json.loads(run_path.read_text())
            run['granularity'] = 'device'
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'device-level'):
                export_results(directory)
            self.assertFalse((directory/'summary_pin.xlsx').exists())
            run['granularity'] = 'pin'
            run['models'].append('gat')
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'end in _pin'):
                export_results(directory, allow_incomplete=True)
            self.assertFalse((directory/'summary_pin.xlsx').exists())
            (directory/'results.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'results_pin.json'):
                export_results(directory/'results.json')

    def test_sample_std_circuits_and_undefined_values(self):
        records = [record(), record(seed=29, mae=4e6, r2=None), record(circuit='Opamp0', mae=1e6)]
        rows = summary_rows(records)
        first = next(item for item in rows if item['circuit'] == 'Opamp1')
        self.assertEqual(first['mae_mean'], 3e6)
        self.assertAlmostEqual(first['mae_std'], 2**.5 * 1e6)
        self.assertEqual(first['r2_defined_seeds'], 1)
        self.assertIsNone(first['r2_std'])
        self.assertEqual(first['seed_ids'], [17, 29])
        self.assertEqual(first['scope'], 'circuit')

    def test_circuit_fallback_does_not_invent_disaggregated_results(self):
        item = record()
        del item['circuit']
        rows = summary_rows([item], {'dataset': {'train_circuits': ['Opamp1']}})
        self.assertEqual(rows[0]['circuit'], 'Opamp1')
        rows = summary_rows([item], {'dataset': {'train_circuits': ['Opamp1', 'Opamp0']}})
        self.assertEqual(rows[0]['circuit'], 'ALL_SELECTED_CIRCUITS')
        self.assertEqual(rows[0]['scope'], 'pooled')

    def test_original_validation_remains_active(self):
        for records in ([record(), record()], [record(mae=float('nan'))],
                        [record(), record(seed=29, r2=float('inf'))]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                summary_rows(records)
        invalid = record()
        invalid['report']['requires_measured_op'] = True
        with self.assertRaisesRegex(ValueError, 'input contract'):
            summary_rows([invalid])

    def test_workbook_units_precise_values_blank_std_and_group_highlights(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_run(Path(tmp), [record(), record('gat_pin', mae=1e6, r2=.712345678901),
                                               record('design_op_oracle_pin', mae=.5e6, r2=.9)])
            output = export_results(directory)
            sheets = workbook(output)
            self.assertEqual(list(sheets), ['MAE', 'RMSE', 'R2', 'PerSeed', 'Statistics', 'TrainingCost', 'Configuration'])
            self.assertIn('MHz', text_values(sheets['MAE']))
            stats_values = [node.text for node in sheets['Statistics'].findall('.//x:v', NS)]
            self.assertIn('0.712345678901', stats_values)
            self.assertIn('1000000.0', stats_values)
            # Columns E/F are deployable mlp/gat. The oracle's better score must
            # not prevent gat from being highlighted, or itself win this group.
            for name in ('MAE', 'RMSE', 'R2'):
                winners = [cell.attrib['r'] for cell in sheets[name].findall('.//x:c', NS)
                           if cell.attrib.get('s') == '9']
                self.assertEqual(winners, ['F8'])
            mae_cell = sheets['MAE'].find('.//x:c[@r="F8"]', NS)
            self.assertEqual(mae_cell.find('x:v', NS).text, '1.0000')
            self.assertIn("'Statistics'!", mae_cell.find('x:f', NS).text)
            self.assertIn('TRUE OP REFERENCE', '\n'.join(text_values(sheets['MAE'])))
            # Only one shared stage exists despite three dependent models.
            costs = text_values(sheets['TrainingCost'])
            self.assertEqual(costs.count('seed_17_pin/shared_pin'), 1)
            self.assertTrue(any('TOTAL unique stage seconds = 10.0' in value for value in costs))
            content = output.read_bytes()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                export_results(directory)
            self.assertEqual(output.read_bytes(), content)

    def test_missing_planned_models_and_seeds_are_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_run(Path(tmp), [record()], models=['mlp_pin', 'gat_pin'], seeds=[17, 29])
            with self.assertRaisesRegex(ValueError, 'missing model/seed/split'):
                export_results(directory)
            self.assertFalse((directory / 'summary_pin.xlsx').exists())
            output = export_results(directory, allow_incomplete=True)
            sheets = workbook(output)
            self.assertTrue(any('INCOMPLETE' in value for value in text_values(sheets['MAE'])))
            self.assertFalse(any(cell.attrib.get('s') == '9' for cell in sheets['MAE'].findall('.//x:c', NS)))

    def test_failed_runs_require_explicit_provisional_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_run(Path(tmp), [record(), record('gat_pin')], status='failed')
            with self.assertRaisesRegex(ValueError, 'Incomplete experiment'):
                export_results(directory)
            with self.assertRaisesRegex(ValueError, 'Incomplete experiment'):
                export_results(directory, finalizing=True)
            export_results(directory, allow_incomplete=True)

    def test_finalization_and_raw_directory_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_run(root / 'training_pin', [record()], status='ready_to_export')
            with self.assertRaisesRegex(ValueError, 'Incomplete experiment'):
                export_results(root)
            output = export_results(root, finalizing=True)
            self.assertEqual(output, directory / 'summary_pin.xlsx')

    def test_missing_config_or_dependency_cost_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_run(Path(tmp), [record()])
            config_path = directory / 'seed_17_pin/mlp_pin/config_pin.json'
            config = config_path.read_bytes()
            config_path.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing per-model'):
                export_results(directory)
            config_path.write_bytes(config)
            stage_path = directory / 'stages_pin.json'
            stages = json.loads(stage_path.read_text())
            duplicate = copy.deepcopy(stages)
            duplicate['unique_stages'].append(duplicate['unique_stages'][0])
            stage_path.write_text(json.dumps(duplicate))
            with self.assertRaisesRegex(ValueError, 'Duplicate stage'):
                export_results(directory)
            stages['unique_stages'].pop()
            stage_path.write_text(json.dumps(stages))
            with self.assertRaisesRegex(ValueError, 'Missing costs'):
                export_results(directory)

    def test_schema_mismatch_and_invalid_cost_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [record(), record('gat_pin')]
            records[1]['report']['metrics']['extra'] = dict(count=8, mae=1, rmse=2, r2=.5)
            directory = write_run(Path(tmp), records)
            with self.assertRaisesRegex(ValueError, 'Metric schema'):
                export_results(directory)
            directory = write_run(Path(tmp), [record()])
            stages = json.loads((directory / 'stages_pin.json').read_text())
            stages['unique_stages'][0]['seconds'] = -1
            (directory / 'stages_pin.json').write_text(json.dumps(stages))
            with self.assertRaisesRegex(ValueError, 'invalid seconds'):
                export_results(directory)

    def test_cli_works_without_site_packages(self):
        script = Path(__file__).resolve().parents[1] / 'summarize_pin.py'
        result = subprocess.run([sys.executable, '-S', str(script), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--allow-incomplete', result.stdout)

    def test_tree_cost_reports_rounds_with_blank_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_run(Path(tmp), [record(), record('xgboost_pin')])
            stage_path = directory / 'stages_pin.json'
            payload = json.loads(stage_path.read_text())
            tree = next(stage for stage in payload['unique_stages'] if stage['stage_id'].endswith('/xgboost_pin'))
            tree.update(epochs=None, planned_epochs=None, optimizer_steps=0,
                        parameters=None, trainable_parameters=None,
                        budget_unit='boosting_rounds_per_metric', planned_boosting_rounds=60,
                        boosting_rounds={'UGF_Hz': 23})
            stage_path.write_text(json.dumps(payload))
            output = export_results(directory)
            sheet = workbook(output)['TrainingCost']
            values = text_values(sheet)
            self.assertIn('boosting_rounds_per_metric', values)
            self.assertIn('{"UGF_Hz": 23}', values)
            self.assertIn('Planned boosting rounds per metric', values)
            # The tree is the third unique stage and occupies row eight.
            for column in ('C', 'D'):
                cell = sheet.find(f'.//x:c[@r="{column}8"]', NS)
                self.assertIsNotNone(cell)
                self.assertIsNone(cell.find('x:v', NS))
            self.assertEqual(sheet.find('.//x:c[@r="F8"]/x:v', NS).text, '60')
            for patch in ({'epochs': 23}, {'planned_boosting_rounds': 0},
                          {'boosting_rounds': {'UGF_Hz': 61}}, {'boosting_rounds': {}},
                          {'boosting_rounds': {'UGF_Hz': True}}, {'budget_unit': 'unknown'}):
                invalid = copy.deepcopy(payload)
                invalid['unique_stages'][-1].update(patch)
                stage_path.write_text(json.dumps(invalid))
                with self.subTest(patch=patch), self.assertRaises(ValueError):
                    export_results(directory, directory / 'invalid_pin.xlsx')
                self.assertFalse((directory / 'invalid_pin.xlsx').exists())


if __name__ == '__main__':
    unittest.main()
