"""Read-only gate inspection uses real checkpoint forwards and original cohorts."""
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from final_exam.gate_probe_pin import capture_forward, inspect_source, run_probe, select_ids
from final_exam.gate_version_registry_pin import V2_GATE_MODELS_PIN
from final_exam.run_pin import load_checkpoint_pin, run_experiment_pin
from final_exam.tests.test_run_pin import tiny_inputs


def snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


class GateProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / 'source_pin'
        cls.source.mkdir()
        data, config, _, cfg = tiny_inputs(cls.source)
        # Real sources include geometry OP fields that are excluded from the
        # train-fitted OP feature schema. The probe must use raw_feature_spec.
        data['feature_spec']['op_names'].append('w_um')
        for row in data['samples']:
            for dev in row['devices']:
                dev['op']['w_um'] = 2.
        cls.data_path = cls.source / 'data_pin.json'
        cls.data_path.write_text(json.dumps(data))
        cls.data = data
        cls.models = ['device_op_cascade_pin', *V2_GATE_MODELS_PIN,
                      'device_op_gate_pin_v1', 'device_op_gate_pin',
                      'device_op_gate_oracle_pin', 'device_op_gate_shuffle_pin']
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_experiment_pin(cls.data_path, cfg, cls.source / 'training_pin',
                               cls.models, [17], 'cpu', smoke=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def probe(self, name, models=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_probe(self.source, circuit='synthetic_0',
                models=models or ['device_op_supervised_gate_fusion_pin_v2'],
                seed=17, device='cpu', output=self.root / name,
                log_dir=self.root / (name + '_logs'), **kwargs)

    @staticmethod
    def csv_rows(path):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))

    def test_v2_reports_capture_all_heads_and_protected_pairs_without_mutation(self):
        before = snapshot(self.source)
        with (patch('device_jepa.data.fit_preprocessor', side_effect=AssertionError('refit')),
              patch('torch.optim.AdamW', side_effect=AssertionError('training'))):
            out = self.probe('all_v2_pin', self.models[:7], num_samples=3, viewer_samples=1, batch_size=2)
        record = json.loads((out / 'probe_pin.json').read_text())
        self.assertEqual(record['status'], 'complete')
        selected = record['selected_sample_ids']
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        manifest = json.loads((self.source / 'training_pin/manifest_pin.json').read_text())
        self.assertLessEqual(set(selected), set(manifest['test']))
        scores = self.csv_rows(out / 'gate_scores_pin.csv')
        self.assertEqual({r['model'] for r in scores}, set(V2_GATE_MODELS_PIN))
        self.assertEqual({r['sample_id'] for r in scores}, set(selected))
        self.assertEqual({int(r['head']) for r in scores}, {1, 2})
        self.assertTrue(any(r['protection_reason'] == 'same_net' for r in scores))
        self.assertTrue(any(r['applied'] == 'True' for r in scores))
        for row in scores:
            score, bias = float(row['gate_score']), float(row['attention_bias'])
            self.assertTrue(0 <= score <= 1)
            self.assertLessEqual(bias, 0.)
            self.assertAlmostEqual(math.exp(bias), float(row['effective_multiplier']), places=12)
            if row['applied'] == 'False':
                self.assertEqual(bias, 0.)
                self.assertIn(row['protection_reason'], ('self', 'same_net', 'no_supported_op'))
            else:
                self.assertAlmostEqual(bias, math.log(1e-4 + (1 - 1e-4) * score), places=6)
        grouped = self.csv_rows(out / 'gate_summary_pin.csv')
        self.assertTrue(all(int(row['samples']) == 3 for row in grouped))
        for row in grouped[:5]:
            matching = [float(r['gate_score']) for r in scores if r['applied'] == 'True'
                        and all(r[k] == row[k] for k in ('model', 'device_id', 'query_role', 'key_role', 'head'))]
            self.assertAlmostEqual(float(row['gate_mean']), sum(matching) / len(matching), places=12)
        self.assertEqual(record['checkpoints']['device_op_gate_pin_v2']['self_attention_layers_1based'], [1, 2])
        self.assertEqual(record['checkpoints']['device_op_gate_fusion_pin_v2']['self_attention_layers_1based'], [2])
        self.assertFalse(record['checkpoints']['device_op_cascade_pin']['gate_active'])
        html = (out / 'gate_viewer_pin.html').read_text()
        payload = json.loads(re.search(r'const payload=(.*);\nconst byId', html).group(1))
        self.assertEqual(payload['viewer_samples'], 1)
        self.assertTrue(all(len(samples) == 1 for samples in payload['data'].values()))
        with zipfile.ZipFile(out / 'gate_scores_pin.xlsx') as book:
            xml = book.read('xl/workbook.xml').decode()
            self.assertIn('ByDeviceHead', xml)
            self.assertIn('ScorePreview', xml)
        self.assertEqual(snapshot(self.source), before)

    def test_legacy_versions_and_oracle_shuffle_use_the_actual_source(self):
        out = self.probe('legacy_pin', self.models[7:], num_samples=2)
        rec = json.loads((out / 'probe_pin.json').read_text())
        self.assertEqual(rec['checkpoints']['device_op_gate_pin_v1']['gate_version'], 'v1')
        self.assertEqual(rec['checkpoints']['device_op_gate_pin']['gate_version'], 'v0')
        self.assertTrue(rec['checkpoints']['device_op_gate_oracle_pin']['requires_measured_op'])
        self.assertFalse(rec['checkpoints']['device_op_gate_pin']['requires_measured_op'])
        self.assertEqual(rec['checkpoints']['device_op_gate_shuffle_pin']['gate_source'], 'shuffle')
        self.assertEqual({row['model'] for row in self.csv_rows(out / 'gate_scores_pin.csv')}, set(self.models[7:]))

    def test_capture_does_not_change_predictions_or_leave_hooks(self):
        from final_exam.data_pin import PinResidentDataset
        path = self.source / 'training_pin/seed_17_pin/device_op_gate_pin_v2/checkpoint_pin.pt'
        model, meta = load_checkpoint_pin(path)
        cache = PinResidentDataset(self.data['samples'][:2], meta['preprocessing'], 'cpu',
                                   include_op=False, include_performance=False, max_tokens=32)
        _, batch = next(cache.batches(2))
        with torch.no_grad():
            before = model(batch)
            capture = capture_forward(model, batch)
            after = model(batch)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        self.assertEqual(capture['logits'].shape[-3:], (4, 4, 2))
        self.assertFalse(model.gate._forward_hooks)
        self.assertFalse(model.gate.score_head._forward_hooks)
        poisoned = dict(batch, owner_op=torch.full_like(capture['values'], float('nan')),
                        owner_op_mask=~capture['supported'])
        other = capture_forward(model, poisoned)
        torch.testing.assert_close(capture['gate_score'], other['gate_score'], atol=0, rtol=0)

    def test_invalid_requests_do_not_overwrite_inputs_or_existing_outputs(self):
        before = snapshot(self.source)
        for circuit, models, seed in (('wrong', [self.models[1]], 17),
                ('synthetic_0', ['not_a_model'], 17), ('synthetic_0', [self.models[1]], 29)):
            with self.assertRaises(ValueError):
                inspect_source(self.source, circuit, models, seed)
        existing = self.root / 'existing_pin'
        existing.mkdir(); (existing / 'keep.txt').write_text('original')
        with self.assertRaisesRegex(ValueError, 'new output'):
            self.probe('existing_pin')
        with self.assertRaisesRegex(ValueError, 'separate'):
            self.probe('source_pin/nested_pin')
        self.assertEqual((existing / 'keep.txt').read_text(), 'original')
        self.assertEqual(snapshot(self.source), before)

    def test_all_requested_when_limit_exceeds_saved_cohort_and_resume_pointer(self):
        pointer_root = self.root / 'pointer_source_pin'
        pointer_root.mkdir()
        # Resolve through a genuine resumed-run data_source pointer (not by copying raw data).
        import shutil
        shutil.copytree(self.source / 'training_pin', pointer_root / 'training_pin')
        rec = json.loads((pointer_root / 'training_pin/run_pin.json').read_text())
        (pointer_root / 'data_source_pin.json').write_text(json.dumps(dict(
            data_path=str(self.data_path), dataset_fingerprint=rec['dataset_fingerprint'])))
        with contextlib.redirect_stdout(io.StringIO()):
            out = run_probe(pointer_root, circuit='synthetic_0', models=[self.models[1]],
                num_samples=999, device='cpu', output=self.root / 'all_samples_pin',
                log_dir=self.root / 'all_samples_logs_pin')
        record = json.loads((out / 'probe_pin.json').read_text())
        self.assertEqual(len(record['selected_sample_ids']), record['available_samples'])

    def test_deterministic_selection_and_invalid_counts(self):
        ids = ['c', 'a', 'b', 'd']
        self.assertEqual(select_ids(ids, 3, 17), select_ids(list(reversed(ids)), 3, 17))
        self.assertEqual(set(select_ids(ids, 99, 17)), set(ids))
        for count, seed in ((0, 17), (2, -1), (True, 17)):
            with self.assertRaises(ValueError):
                select_ids(ids, count, seed)

    def test_changed_source_data_is_rejected_before_inference(self):
        import shutil
        changed = self.root / 'changed_source_pin'
        shutil.copytree(self.source, changed)
        data = json.loads((changed / 'data_pin.json').read_text())
        data['samples'][0]['devices'][0]['op']['w_um'] = 3.
        (changed / 'data_pin.json').write_text(json.dumps(data))
        output = self.root / 'changed_probe_pin'
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'final_exam.gate_probe_pin.capture_forward', side_effect=AssertionError('inference')):
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                run_probe(changed, circuit='synthetic_0', models=[self.models[1]],
                    device='cpu', output=output, log_dir=self.root / 'changed_logs_pin')
        self.assertEqual(json.loads((output / 'probe_pin.json').read_text())['status'], 'failed')
        self.assertFalse((output / 'gate_scores_pin.csv').exists())


if __name__ == '__main__':
    unittest.main()
