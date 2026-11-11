"""Recover staged pin experiments without retraining completed dependencies."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.run_pin import STAGED_GATE_MODELS_PIN, load_checkpoint_pin, run_experiment_pin
from final_exam.tests.test_run_pin import tiny_inputs


class StagedResumePinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.config, self.data_path, self.cfg_path = tiny_inputs(self.root)
        torch.set_num_threads(1)

    @contextlib.contextmanager
    def quiet(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield

    def run_models(self, name, *, source=None, data=None, config=None, manifest=None,
                   models=STAGED_GATE_MODELS_PIN, seeds=(17,)):
        output = self.root / name
        with self.quiet():
            result = run_experiment_pin(data or self.data_path, config or self.cfg_path,
                output, list(models), list(seeds), 'cpu', smoke=True,
                manifest_path=manifest, resume_from=source)
        return output, result

    def failed_source(self, name='failed_pin', *, seeds=(17,), resume_from=None):
        output = self.root / name
        original = engine.fit_stage

        def fail_at_gate(ctx, model, stage_id, *args, **kwargs):
            if stage_id == STAGED_GATE_MODELS_PIN[0]:
                raise RuntimeError('simulated first gate backward failure')
            return original(ctx, model, stage_id, *args, **kwargs)

        with patch.object(engine, 'fit_stage', side_effect=fail_at_gate):
            with self.assertRaisesRegex(RuntimeError, 'simulated first gate'):
                self.run_models(name, seeds=seeds, source=resume_from)
        self.assertEqual(json.loads((output / 'run_pin.json').read_text())['status'], 'failed')
        return output

    @staticmethod
    def snapshot(directory):
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob('*') if path.is_file()}

    @staticmethod
    def make_legacy(source):
        # Actual old complete snapshots contain no optimizer, RNG or stage
        # identity. Preserve their original complete/cost/selected-weight data.
        fields = {'state_dict', 'cost', 'config', 'preprocessing', 'stage_metadata', 'complete'}
        for path in source.rglob('last_pin.pt'):
            path.unlink()
        for path in source.rglob('best_pin.pt'):
            value = torch.load(path, map_location='cpu', weights_only=True)
            if value.get('complete'):
                value = {name: item for name, item in value.items() if name in fields}
                value['cost'].pop('resume', None)
                torch.save(value, path)

    def test_legacy_gate_failure_reuses_four_completed_stages_and_preserves_source(self):
        source = self.failed_source()
        self.make_legacy(source)
        before = self.snapshot(source)
        current_stage = [None]
        steps = {}
        original_fit, original_step = engine.fit_stage, torch.optim.AdamW.step

        def track_stage(ctx, model, stage_id, *args, **kwargs):
            current_stage[0] = stage_id
            steps.setdefault(stage_id, 0)
            try:
                return original_fit(ctx, model, stage_id, *args, **kwargs)
            finally:
                current_stage[0] = None

        def track_step(optimizer, *args, **kwargs):
            self.assertIsNotNone(current_stage[0])
            steps[current_stage[0]] += 1
            return original_step(optimizer, *args, **kwargs)

        with patch.object(engine, 'fit_stage', side_effect=track_stage), \
                patch.object(torch.optim.AdamW, 'step', new=track_step), \
                patch('final_exam.data_pin.fit_preprocessor', side_effect=AssertionError('refitted normalization')), \
                patch.object(engine, 'fit_scaler', side_effect=AssertionError('refitted teacher scaler')):
            output, result = self.run_models('recovered_pin', source=source)

        completed = ('device_op_predictor_pin', 'design_op_oracle_pin',
                     'device_op_distill_encoder_pin', 'device_op_distill_readout_pin')
        for stage in completed:
            self.assertEqual(steps[stage], 0, stage)
        for stage in (*STAGED_GATE_MODELS_PIN, 'device_op_supervised_readout_pin'):
            self.assertGreater(steps[stage], 0, stage)
        self.assertEqual(len(result['results']), 3)
        self.assertEqual(len(result['stages']), len({item['stage_id'] for item in result['stages']}))
        self.assertTrue((output / 'summary_pin.xlsx').is_file())
        self.assertEqual(json.loads((output / 'run_pin.json').read_text())['status'], 'complete')
        self.assertEqual(self.snapshot(source), before)
        for name in ('device_op_predictor_pin', 'design_op_oracle_pin', 'device_op_distilled_base_pin'):
            models = [load_checkpoint_pin(root / 'seed_17_pin' / 'shared_pin' / name /
                                          'checkpoint_pin.pt')[0] for root in (source, output)]
            self.assertEqual(state_digest(models[0]), state_digest(models[1]), name)
        self.assertEqual(json.loads((source / 'preprocessing_pin.json').read_text()),
                         json.loads((output / 'preprocessing_pin.json').read_text()))

    def test_completed_experiment_replays_without_optimizer_steps(self):
        source, original = self.run_models('completed_pin')
        before = self.snapshot(source)
        with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retrained completed stage')):
            output, resumed = self.run_models('replayed_pin', source=source)
        self.assertEqual(self.snapshot(source), before)
        self.assertEqual(original['results'], resumed['results'])
        self.assertTrue((output / 'summary_pin.xlsx').is_file())
        # Inspect the actual exported workbook, not the worksheet builder. A
        # recovery must report zero new optimization while retaining the cost
        # of producing the selected weights in the source experiment.
        ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        relations_ns = 'http://schemas.openxmlformats.org/package/2006/relationships'
        document_relation = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
        with zipfile.ZipFile(output / 'summary_pin.xlsx') as workbook:
            sheets = ET.fromstring(workbook.read('xl/workbook.xml')).findall('s:sheets/s:sheet', ns)
            sheet = next((item for item in sheets if item.get('name') == 'Resume'), None)
            self.assertIsNotNone(sheet)
            relations = ET.fromstring(workbook.read('xl/_rels/workbook.xml.rels'))
            target = next(item.get('Target') for item in relations.findall(f'{{{relations_ns}}}Relationship')
                          if item.get('Id') == sheet.get(document_relation))
            path = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = ET.fromstring(workbook.read(path)).findall('s:sheetData/s:row', ns)
        stage_rows = [row for row in rows if int(row.get('r')) >= 5]
        self.assertEqual(len(stage_rows), len(resumed['stages']))
        retained_steps = 0
        for row in stage_rows:
            cells = {''.join(char for char in cell.get('r') if char.isalpha()): cell
                     for cell in row.findall('s:c', ns)}
            self.assertEqual(int(cells['F'].find('s:v', ns).text), 0)
            retained_steps += int(cells['I'].find('s:v', ns).text)
        self.assertEqual(retained_steps, sum(stage['optimizer_steps'] for stage in original['stages']))
        self.assertGreater(retained_steps, 0)
        cache_id = 'seed_17_pin/frozen_device_op_cache_pin'
        old_cache = next(stage for stage in original['stages'] if stage['stage_id'] == cache_id)
        new_cache = next(stage for stage in resumed['stages'] if stage['stage_id'] == cache_id)
        audit = new_cache['resume']
        self.assertEqual(audit['policy'], 'rebuilt_cache')
        self.assertEqual(new_cache['optimizer_steps'], 0)
        self.assertEqual(audit['new_optimizer_steps'], 0)
        self.assertEqual(new_cache['seconds'], old_cache['seconds'] + audit['new_seconds'])
        self.assertEqual(new_cache['examples_seen'],
                         old_cache['examples_seen'] + audit['new_examples_seen'])
        self.assertEqual(audit['new_examples_seen'], old_cache['examples_seen'])
        for name in STAGED_GATE_MODELS_PIN:
            models = [load_checkpoint_pin(root / 'seed_17_pin' / name / 'checkpoint_pin.pt')[0]
                      for root in (source, output)]
            self.assertEqual(state_digest(models[0]), state_digest(models[1]))

    def test_seeds_not_reached_in_failed_run_train_normally_after_recovery(self):
        source = self.failed_source(seeds=(17, 29))
        self.assertFalse((source / 'seed_29_pin').exists())
        before = self.snapshot(source)
        current = [None]
        steps = {}
        original_fit, original_step = engine.fit_stage, torch.optim.AdamW.step

        def track_stage(ctx, model, stage_id, *args, **kwargs):
            key = (ctx.output.name, stage_id)
            current[0] = key
            steps.setdefault(key, 0)
            try:
                return original_fit(ctx, model, stage_id, *args, **kwargs)
            finally:
                current[0] = None

        def track_step(optimizer, *args, **kwargs):
            self.assertIsNotNone(current[0])
            steps[current[0]] += 1
            return original_step(optimizer, *args, **kwargs)

        with patch.object(engine, 'fit_stage', side_effect=track_stage), \
                patch.object(torch.optim.AdamW, 'step', new=track_step):
            output, result = self.run_models('later_seed_pin', source=source, seeds=(17, 29))
        self.assertEqual(steps[('seed_17_pin', 'device_op_predictor_pin')], 0)
        self.assertGreater(steps[('seed_29_pin', 'device_op_predictor_pin')], 0)
        self.assertGreater(steps[('seed_29_pin', 'design_op_oracle_pin')], 0)
        self.assertEqual(len(result['results']), 6)
        self.assertEqual(self.snapshot(source), before)
        self.assertTrue((output / 'summary_pin.xlsx').is_file())

    def test_recovery_can_itself_fail_and_be_recovered_again(self):
        first = self.failed_source('original_failure_pin')
        self.make_legacy(first)
        first_snapshot = self.snapshot(first)
        second = self.failed_source('resumed_failure_pin', resume_from=first)
        self.assertEqual(self.snapshot(first), first_snapshot)
        second_snapshot = self.snapshot(second)
        output, result = self.run_models('chained_recovery_pin', source=second)
        self.assertEqual(len(result['results']), 3)
        self.assertEqual(json.loads((output / 'run_pin.json').read_text())['status'], 'complete')
        self.assertTrue((output / 'summary_pin.xlsx').is_file())
        self.assertEqual(self.snapshot(first), first_snapshot)
        self.assertEqual(self.snapshot(second), second_snapshot)
        for name in ('device_op_predictor_pin', 'design_op_oracle_pin', 'device_op_distilled_base_pin'):
            models = [load_checkpoint_pin(root / 'seed_17_pin' / 'shared_pin' / name /
                                          'checkpoint_pin.pt')[0] for root in (first, second, output)]
            self.assertEqual(state_digest(models[0]), state_digest(models[1]), name)
            self.assertEqual(state_digest(models[1]), state_digest(models[2]), name)

    def test_changed_config_data_manifest_or_seed_fails_before_destination_writes(self):
        source = self.failed_source()
        changed_config = copy.deepcopy(self.config)
        changed_config['op_transfer']['lr'] *= 2
        cfg = self.root / 'changed_config.json'
        cfg.write_text(json.dumps(changed_config))
        changed_data = copy.deepcopy(self.data)
        row = changed_data['samples'][0]
        metric = next(name for name, value in row['performance'].items()
                      if isinstance(value, (int, float)))
        row['performance'][metric] += 1.
        data = self.root / 'changed_data.json'
        data.write_text(json.dumps(changed_data))
        manifest = json.loads((source / 'manifest_pin.json').read_text())
        manifest['val'][0], manifest['test'][0] = manifest['test'][0], manifest['val'][0]
        manifest_path = self.root / 'changed_manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        before = self.snapshot(source)
        cases = [('config', {'config': cfg}), ('data', {'data': data}),
                 ('manifest', {'manifest': manifest_path}), ('seed', {'seeds': (29,)}),
                 ('model', {'models': ('zerosim_supervised_pin',)})]
        for name, arguments in cases:
            with self.subTest(name=name):
                destination = self.root / f'invalid_{name}_pin'
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.run_models(destination.name, source=source, **arguments)
                self.assertFalse(destination.exists())
        self.assertEqual(self.snapshot(source), before)

    def test_missing_saved_normalization_fails_before_destination_writes(self):
        source = self.failed_source()
        (source / 'preprocessing_pin.json').unlink()
        before = self.snapshot(source)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_models('missing_prep_pin', source=source)
        self.assertFalse((self.root / 'missing_prep_pin').exists())
        self.assertEqual(self.snapshot(source), before)

    def test_recovery_never_writes_inside_source_tree(self):
        source = self.failed_source()
        before = self.snapshot(source)
        destination = source / 'new_nested_pin'
        with self.quiet(), self.assertRaises(ValueError):
            run_experiment_pin(self.data_path, self.cfg_path, destination,
                list(STAGED_GATE_MODELS_PIN), [17], 'cpu', smoke=True, resume_from=source)
        self.assertFalse(destination.exists())
        self.assertEqual(self.snapshot(source), before)


if __name__ == '__main__':
    unittest.main()
