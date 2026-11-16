"""Best/worst whole-seed selection, source validation, matched gates and costs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

from final_exam.seed_selection_pin import select_seeds_pin
from final_exam.summarize_pin import export_results_pin
from final_exam.tests.test_gate_v1_summary_pin import v1_run
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook, write_run


ROOT = Path(__file__).resolve().parents[2]


def records():
    values = {'mlp_pin': [.2, .9, .4], 'gat_pin': [.1, .8, .6], 'zerosim_supervised_pin': [.3, .5, .7]}
    return [record(model, seed, mae=(i + 1) * 1e6, r2=value)
            for model, scores in values.items()
            for i, (seed, value) in enumerate(zip((17, 29, 43), scores))]


def select(items, **kwargs):
    options = dict(k=2, models_better=['mlp_pin'], models_worse=['gat_pin'])
    options.update(kwargs)
    return select_seeds_pin(items, circuit_of=lambda r: r.get('circuit', 'Opamp1'), **options)


def cells(row):
    """Read exact typed/cache values from the portable OOXML writer."""
    result = []
    for cell in row.findall('x:c', NS):
        value = cell.find('x:v', NS)
        if cell.attrib.get('t') == 'inlineStr':
            result.append(''.join(text_values(cell)))
        elif value is None:
            result.append(None)
        elif cell.attrib.get('t') == 'b':
            result.append(value.text == '1')
        else:
            result.append(float(value.text))
    return result


def table(sheet, header_row):
    rows = sheet.findall('x:sheetData/x:row', NS)
    header = cells(rows[header_row - 1])
    return [dict(zip(header, cells(r))) for r in rows[header_row:] if len(cells(r)) == len(header)]


class SeedSelectionPinTests(unittest.TestCase):
    def test_whole_seed_directions_unlisted_models_and_negative_scores(self):
        original = records()
        snapshot = copy.deepcopy(original)
        selected, audit = select(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(audit['selected_seeds'], {'mlp_pin': [29, 43], 'gat_pin': [17, 43],
                                                  'zerosim_supervised_pin': [17, 29, 43]})
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(audit['rankings']), 9)
        for r in original:
            r['report']['metrics']['UGF_Hz']['r2'] -= 100
        self.assertEqual(select(original)[1]['selected_seeds'], audit['selected_seeds'])

    def test_macro_r2_selects_one_seed_for_every_metric_not_per_metric(self):
        a, b = record(seed=17, r2=.9), record(seed=29, r2=.6)
        a['report']['metrics']['DC_Gain_dB'] = dict(count=8, mae=1, rmse=1, r2=.1)
        b['report']['metrics']['DC_Gain_dB'] = dict(count=8, mae=2, rmse=2, r2=.8)
        selected, audit = select([a, b], k=1, models_worse=[])
        self.assertEqual(selected, [b])
        self.assertAlmostEqual(next(r['test_macro_r2'] for r in audit['rankings'] if r['seed'] == 29), .7)
        self.assertEqual(selected[0]['report']['metrics']['DC_Gain_dB']['mae'], 2)

    def test_ties_use_lower_seed_id_in_both_directions(self):
        items = [record(seed=43), record(seed=17), record(seed=29)]
        for options in ({'models_better': ['mlp_pin'], 'models_worse': []},
                        {'models_better': [], 'models_worse': ['mlp_pin']}):
            self.assertEqual(select(items, k=1, **options)[1]['selected_seeds']['mlp_pin'], [17])

    def test_invalid_groups_and_k_fail_without_truncating_candidates(self):
        for overrides in (dict(k=0), dict(k=True), dict(k=4), dict(models_better=['missing_pin']),
                          dict(models_worse=['mlp_pin']), dict(models_better=['mlp_pin', 'mlp_pin']),
                          dict(models_better=[], models_worse=[])):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                select(records(), **overrides)
        self.assertEqual(len(select(records(), k=3)[0]), 9)

    def test_undefined_r2_rules_are_explicit(self):
        items = records()
        for r in items:
            r['report']['metrics']['constant'] = dict(count=8, mae=1, rmse=1, r2=None)
        audit = select(items)[1]
        self.assertEqual(audit['excluded_undefined_metrics']['Opamp1'], ['constant'])
        items[0]['report']['metrics']['constant']['r2'] = .1
        with self.assertRaisesRegex(ValueError, 'every candidate'):
            select(items)
        for r in items:
            r['report']['metrics'] = {'constant': dict(count=8, mae=1, rmse=1, r2=None)}
        with self.assertRaisesRegex(ValueError, 'No defined R2'):
            select(items)

    def test_test_selection_carries_to_ood_without_ranking_ood(self):
        items = records()
        ood = copy.deepcopy(items)
        for r in ood:
            r['report']['split'] = 'ood'
            r['report']['metrics']['UGF_Hz']['r2'] = -r['report']['metrics']['UGF_Hz']['r2']
            r['circuit'] = 'Unseen'
        selected, audit = select(items + ood)
        for model, seeds in audit['selected_seeds'].items():
            self.assertEqual(sorted(r['seed'] for r in selected if r['mode'] == model
                                    and r['report']['split'] == 'ood'), seeds)
        with self.assertRaisesRegex(ValueError, 'requires test'):
            select(ood)

    def test_multi_circuit_rank_has_equal_circuit_weights_and_complete_coverage(self):
        items = [record(seed=17, r2=.9), record(seed=29, r2=.6)]
        items += [record(seed=17, r2=.1, circuit='Opamp0'), record(seed=29, r2=.8, circuit='Opamp0')]
        self.assertEqual(select(items, k=1, models_worse=[])[1]['selected_seeds']['mlp_pin'], [29])
        with self.assertRaisesRegex(ValueError, 'Missing test circuits'):
            select(items[:-1], k=1, models_worse=[])

    def test_export_preserves_source_all_seed_statistics_and_full_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_run(Path(tmp) / 'training_pin', records())
            export_results_pin(root)
            snapshot = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob('*') if p.is_file()}
            output = export_results_pin(root.parent, seed_selection=dict(
                k=2, models_better=['mlp_pin'], models_worse=['gat_pin']))
            self.assertEqual(output.name, 'summary_seed_selection_k2_pin.xlsx')
            sheets = workbook(output)
            self.assertTrue({'MAE', 'RMSE', 'R2', 'PerSeed', 'Statistics', 'TrainingCost',
                             'Configuration', 'SeedSelection', 'AllSeedStatistics',
                             'AllSeedPerSeed', 'SelectedTrainingCost'} <= sheets.keys())
            for name in ('MAE', 'RMSE', 'R2', 'PerSeed', 'Statistics'):
                self.assertIn('TEST-SELECTED SEEDS', text_values(sheets[name])[0])
            stats = {r['mode']: r for r in table(sheets['Statistics'], 4)}
            self.assertEqual(stats['mlp_pin']['seed_ids'], '[29, 43]')
            self.assertEqual(stats['gat_pin']['seed_ids'], '[17, 43]')
            self.assertAlmostEqual(stats['mlp_pin']['r2_mean'], .65)
            self.assertAlmostEqual(stats['mlp_pin']['r2_std'], statistics.stdev([.9, .4]))
            all_stats = {r['mode']: r for r in table(sheets['AllSeedStatistics'], 4)}
            self.assertAlmostEqual(all_stats['mlp_pin']['r2_mean'], .5)
            self.assertEqual(len(table(sheets['AllSeedPerSeed'], 4)), 9)
            self.assertEqual(len(table(sheets['PerSeed'], 4)), 7)
            # Three shared stages plus nine model stages; retained view has
            # three shared stages plus seven model stages, not duplicated.
            self.assertEqual(len(table(sheets['TrainingCost'], 5)), 12)
            self.assertEqual(len(table(sheets['SelectedTrainingCost'], 5)), 10)
            self.assertFalse(any(c.attrib.get('s') == '9' for c in sheets['R2'].findall('.//x:c', NS)))
            for path, digest in snapshot.items():
                self.assertEqual(hashlib.sha256((root/path).read_bytes()).hexdigest(), digest)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                export_results_pin(root, seed_selection=dict(k=2, models_better=['mlp_pin'], models_worse=[]))

    def test_single_seed_has_blank_std_and_invalid_source_never_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_run(Path(tmp), records())
            options = dict(k=1, models_better=['mlp_pin'], models_worse=['gat_pin'])
            sheets = workbook(export_results_pin(root, seed_selection=options))
            stats = {r['mode']: r for r in table(sheets['Statistics'], 4)}
            self.assertIsNone(stats['mlp_pin']['r2_std'])
            with self.assertRaisesRegex(ValueError, 'original summary'):
                export_results_pin(root, root/'summary_pin.xlsx', seed_selection=options)
            # Even a discarded seed must have an effective config.
            (root/'seed_17_pin/mlp_pin/config_pin.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing per-model'):
                export_results_pin(root, root/'new_pin.xlsx', seed_selection=options)
            self.assertFalse((root/'new_pin.xlsx').exists())

    def test_failed_export_requires_explicit_recovery_and_complete_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_run(Path(tmp), records(), status='failed_export')
            options = dict(k=1, models_better=['mlp_pin'], models_worse=[])
            with self.assertRaisesRegex(ValueError, 'Incomplete experiment'):
                export_results_pin(root, seed_selection=options)
            export_results_pin(root, seed_selection=options, recover_export=True)
            data = json.loads((root/'results_pin.json').read_text())
            (root/'results_pin.json').write_text(json.dumps(data[:-1]))
            with self.assertRaisesRegex(ValueError, 'missing model/seed/split'):
                export_results_pin(root, root/'incomplete_pin.xlsx', seed_selection=options, recover_export=True)
            self.assertFalse((root/'incomplete_pin.xlsx').exists())

    def test_gate_base_rows_match_each_gate_models_selected_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1_run(root)
            best, worst = 'device_op_distill_gate_pin_v1', 'device_op_distill_gate_fusion_pin_v1'
            data = json.loads((root/'results_pin.json').read_text())
            for r in data:
                if r['mode'] not in (best, worst):
                    continue
                r['report']['metrics']['UGF_Hz']['r2'] = .4 if r['seed'] == 17 else .8
                path = root/f'seed_{r["seed"]}_pin'/r['mode']/'gate_comparison_pin.json'
                comparison = json.loads(path.read_text())
                comparison['post_gate_test'] = r['report']
                path.write_text(json.dumps(comparison))
            (root/'results_pin.json').write_text(json.dumps(data))
            sheets = workbook(export_results_pin(root, seed_selection=dict(
                k=1, models_better=[best], models_worse=[worst])))
            study = table(sheets['GateStudy'], 5)
            for mode, expected in ((best, '[29]'), (worst, '[17]')):
                paired = [r for r in study if r['Paired gate model'] == mode]
                self.assertEqual(len(paired), 2)
                self.assertEqual({r['Seeds'] for r in paired}, {expected})
                self.assertEqual({r['Role'] for r in paired}, {'selected model', 'ungated base reference'})

    def test_cli_and_shell_work_without_torch_and_handle_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='seed selection ') as tmp:
            root = write_run(Path(tmp)/'training_pin', records())
            command = [sys.executable, '-S', str(ROOT/'final_exam/summarize_selected_seeds_pin.py'),
                       str(root.parent), '--k', '1', '--models-better', 'mlp_pin', '--models-worse', 'gat_pin']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('test macro R2', result.stdout)
            script = (ROOT/'final_exam/single_circuit_pin/summarize_selected_seeds_pin.sh').read_text()
            script = script.replace('width_decoder_pin\n    width_decoder_op_pin', 'mlp_pin')
            script = script.replace('device_op_cascade_pin', 'gat_pin')
            script = script.replace('PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"', f'PROJECT_ROOT="{ROOT}"')
            launch = Path(tmp)/'launch.sh'
            launch.write_text(script)
            output = root/'shell_selected_pin.xlsx'
            result = subprocess.run(['bash', str(launch)], text=True, capture_output=True,
                env=dict(os.environ, SOURCE_RUN=str(root.parent), K='2', OUTPUT=str(output), JEPA_PYTHON=sys.executable))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())


if __name__ == '__main__':
    unittest.main()
