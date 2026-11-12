"""Signed shared/independent gates retain actual layer identity in every report."""
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


class GateV45ProbePinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / 'source_pin'
        cls.source.mkdir()
        cls.data, cfg, _, cfg_path = tiny_inputs(cls.source)
        cls.data_path = cls.source / 'data_pin.json'
        cls.data_path.write_text(json.dumps(cls.data))
        cfg['model'].update(encoder_layers=6, cross_layers=3)
        cfg['op_gate_pin'] = dict(strength=1., n_experts=3, mlp_dropout=.2, norm_eps=2e-5)
        cfg_path.write_text(json.dumps(cfg))
        cls.modes = ['device_op_supervised_gate_fusion_pin_v4', 'device_op_supervised_gate_fusion_pin_v5']
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_experiment_pin(cls.data_path, cfg_path, cls.source / 'training_pin',
                               cls.modes, [17], 'cpu', smoke=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load_model(self, version):
        mode = f'device_op_supervised_gate_fusion_pin_{version}'
        model, metadata = load_checkpoint_pin(self.source / f'training_pin/seed_17_pin/{mode}/checkpoint_pin.pt')
        rows = self.data['samples'][:2]
        cache = PinResidentDataset(rows, metadata['preprocessing'], 'cpu',
                                   include_op=False, include_performance=False, max_tokens=32)
        _, batch = next(cache.batches(2))
        return mode, model, rows, batch

    def test_real_probe_writes_distinct_layer_rows_and_signed_metadata_without_refitting(self):
        before = snapshot(self.source)
        out = self.root / 'probe_pin'
        with (contextlib.redirect_stdout(io.StringIO()),
              patch('device_jepa.data.fit_preprocessor', side_effect=AssertionError('refit')),
              patch('torch.optim.AdamW', side_effect=AssertionError('training'))):
            run_probe(self.source, circuit='synthetic_0', models=self.modes, device='cpu',
                      num_samples=2, output=out, log_dir=self.root / 'probe_logs_pin')
        self.assertEqual(snapshot(self.source), before)
        record = json.loads((out / 'probe_pin.json').read_text())
        self.assertFalse(record['shared_layers'])
        for mode,meta in record['checkpoints'].items():
            self.assertEqual(meta['self_attention_layers_1based'], [5,6])
            self.assertEqual(meta['gated_pe_layers'], [2,3])
            self.assertEqual(meta['gate_score_transform'], 'tanh')
            self.assertEqual(meta['shared_gate_across_layers'], mode.endswith('_v4'))
            self.assertEqual(meta['gate_n_experts'], 3)
        with (out / 'gate_scores_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
            scores = list(csv.DictReader(handle))
        self.assertEqual({row['pe_layer'] for row in scores}, {'2','3'})
        self.assertEqual({row['encoder_layer'] for row in scores}, {'5','6'})
        for row in scores:
            self.assertEqual(row['gate_score_transform'], 'tanh')
            score,bias = float(row['gate_score']),float(row['attention_bias'])
            self.assertAlmostEqual(score, math.tanh(float(row['gate_logit'])), places=6)
            self.assertAlmostEqual(bias, score if row['applied']=='True' else 0., places=6)
            self.assertAlmostEqual(float(row['effective_multiplier']),math.exp(bias),places=12)
        with (out / 'gate_summary_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
            groups = list(csv.DictReader(handle))
        self.assertTrue(groups)
        self.assertTrue(all(row['samples']=='2' for row in groups))
        sheets = workbook(out / 'gate_scores_pin.xlsx')
        self.assertIn('Shared across layers',text_values(sheets['Models']))
        self.assertIn('Fraction enhanced',text_values(sheets['Models']))
        html = (out / 'gate_viewer_pin.html').read_text()
        payload=json.loads(re.search(r'const payload=(.*);\nconst byId',html).group(1))
        self.assertIn('id="layer"',html)
        self.assertIn("p[8]===Number(layer)",html)
        for model in payload['data'].values():
            for sample in model.values():
                for device in sample.values():
                    self.assertEqual({p[8] for p in device['pairs']},{2,3})

    def test_v5_capture_tracks_independent_scores_and_cleanup(self):
        mode, model, rows, batch = self.load_model('v5')
        heads=list(model.gate.iter_layer_score_heads())
        self.assertEqual([x[0] for x in heads],[2,3])
        # Controlled in-memory intervention proves the probe does not duplicate
        # one gate's output across layers or assume all signed gates suppress.
        with torch.no_grad():
            for sign,(_,head) in zip((1.,-1.),heads):
                head[-1].weight.zero_();head[-1].bias.fill_(sign)
        model.eval()
        with torch.no_grad():before=model(batch)
        captured=capture_forward(model,batch)
        with torch.no_grad():after=model(batch)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertEqual([x['pe_layer'] for x in captured['layer_captures']],[2,3])
        result=list(gate_rows(model,batch,rows,captured,mode,17))
        for row in result:
            if row['applied']:
                self.assertGreater(row['attention_bias'] if row['pe_layer']==2 else -row['attention_bias'],0)
        self.assertFalse(model.gate._forward_hooks)
        for _,head in heads:self.assertFalse(head._forward_hooks)

    def test_v4_same_computed_table_is_reported_for_each_actual_layer(self):
        mode,model,rows,batch=self.load_model('v4')
        capture=capture_forward(model,batch)
        first,second=capture['layer_captures']
        for key in ('logits','gate_score','bias'):
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        self.assertNotEqual(first['encoder_layer'],second['encoder_layer'])
        # OP arrays are held once at top level: report input rows must not be
        # multiplied by the number of PE layers.
        self.assertEqual(capture['values'].shape,first['values'].shape)


if __name__ == '__main__':
    unittest.main()
