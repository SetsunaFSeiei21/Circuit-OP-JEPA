"""v3 diagnostics replay the real frozen M-MLP gate and preserve relative OP."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import torch

from final_exam.data_pin import PinResidentDataset
from final_exam.gate_probe_pin import capture_forward, gate_rows, run_probe
from final_exam.run_pin import load_checkpoint_pin, run_experiment_pin
from final_exam.tests.test_gate_probe_pin import snapshot
from final_exam.tests.test_run_pin import tiny_inputs
from final_exam.tests.test_summary_pin import text_values, workbook


class GateV3ProbePinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / 'source_pin'
        cls.source.mkdir()
        cls.data, cfg, _, cfg_path = tiny_inputs(cls.source)
        cls.data_path = cls.source / 'data_pin.json'
        cls.data_path.write_text(json.dumps(cls.data))
        cfg['op_gate_pin'] = dict(n_experts=3, mlp_dropout=.2, norm_eps=2e-5)
        cfg_path.write_text(json.dumps(cfg))
        cls.modes = ['device_op_supervised_gate_fusion_pin_v3']
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_experiment_pin(cls.data_path, cfg_path, cls.source / 'training_pin',
                               cls.modes, [17], 'cpu', smoke=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_checkpoint_probe_preserves_weights_and_exposes_architecture(self):
        before = snapshot(self.source)
        out = self.root / 'probe_pin'
        with (contextlib.redirect_stdout(io.StringIO()),
              patch('device_jepa.data.fit_preprocessor', side_effect=AssertionError('refit')),
              patch('torch.optim.AdamW', side_effect=AssertionError('training'))):
            run_probe(self.source, circuit='synthetic_0', models=self.modes, device='cpu',
                      num_samples=2, output=out, log_dir=self.root / 'probe_logs_pin')
        self.assertEqual(snapshot(self.source), before)
        record = json.loads((out / 'probe_pin.json').read_text())
        meta = record['checkpoints'][self.modes[0]]
        self.assertEqual(meta['gate_version'], 'v3')
        self.assertEqual(meta['gate_n_experts'], 3)
        self.assertEqual(meta['gate_mlp_dropout'], .2)
        self.assertEqual(meta['gate_architecture']['kind'], 'mixed_mlp')
        self.assertEqual(meta['gate_normalization']['norm_eps'], 2e-5)
        self.assertEqual(meta['self_attention_layers_1based'], [2])
        with (out / 'gate_op_inputs_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
            ops = list(csv.DictReader(handle))
        self.assertTrue(ops)
        self.assertTrue(all(row['relative_gate_op'] != '' for row in ops))
        self.assertTrue(any(abs(float(row['relative_gate_op'])) > 1e-5 for row in ops))
        with (out / 'gate_scores_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
            scores = list(csv.DictReader(handle))
        self.assertTrue(scores)
        for row in scores:
            self.assertEqual(row['gate_version'], 'v3')
            bias = float(row['attention_bias'])
            if row['applied'] == 'True':
                self.assertAlmostEqual(bias, math.log(1e-4 + .9999 * float(row['gate_score'])), places=6)
            else:
                self.assertEqual(bias, 0.)
        sheets = workbook(out / 'gate_scores_pin.xlsx')
        self.assertIn('Experts per layer', text_values(sheets['Models']))
        self.assertIn('mixed_mlp', text_values(sheets['Models']))
        payload = json.loads(re.search(r'const payload=(.*);\nconst byId',
            (out / 'gate_viewer_pin.html').read_text()).group(1))
        self.assertEqual(payload['models'][self.modes[0]]['gate_architecture'], meta['gate_architecture'])

    def test_capture_matches_direct_network_and_physical_device_mapping(self):
        mode = self.modes[0]
        model, metadata = load_checkpoint_pin(self.source / f'training_pin/seed_17_pin/{mode}/checkpoint_pin.pt')
        rows = self.data['samples'][:2]
        cache = PinResidentDataset(rows, metadata['preprocessing'], 'cpu',
                                   include_op=False, include_performance=False, max_tokens=32)
        _, batch = next(cache.batches(2))
        captured = capture_forward(model, batch)
        values, supported = captured['values'], captured['supported']
        relative = model.gate.relative_op(batch, values, supported)
        torch.testing.assert_close(captured['relative'], relative, rtol=0, atol=0)
        features = torch.cat((torch.where(supported, values, 0.), relative), -1)
        B, D, F = features.shape
        eye = torch.eye(4)
        inputs = torch.cat((features[:, :, None, None].expand(B, D, 4, 4, F),
            eye[None, None, :, None].expand(B, D, 4, 4, 4),
            eye[None, None, None, :].expand(B, D, 4, 4, 4)), -1)
        with torch.no_grad():
            logits = model.gate.score_head(inputs)
        torch.testing.assert_close(logits, captured['logits'], rtol=0, atol=0)
        self.assertFalse(model.gate.score_head._forward_hooks)
        self.assertFalse(model.gate._forward_hooks)
        result_rows = list(gate_rows(model, batch, rows, captured, mode, 17))
        for row in result_rows:
            sample = next(s for s in rows if s['sample_id'] == row['sample_id'])
            device = next(d for d in sample['devices'] if d['device_id'] == row['device_id'])
            self.assertEqual(row['query_net'], device['terminals'][row['query_role'].lower()])
            self.assertEqual(row['key_net'], device['terminals'][row['key_role'].lower()])
            if row['query_net'] == row['key_net']:
                self.assertFalse(row['applied'])


if __name__ == '__main__':
    unittest.main()
