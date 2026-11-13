"""Probe the actual appended polarity coordinates without refitting or training."""
import contextlib
import csv
import io
import json
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


class GateV67ProbePinTests(unittest.TestCase):
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
        cls.modes = ['device_op_supervised_gate_fusion_pin_v6', 'device_op_supervised_gate_fusion_pin_v7']
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_experiment_pin(cls.data_path, cfg_path, cls.source / 'training_pin',
                               cls.modes, [17], 'cpu', smoke=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_probe_exports_actual_type_inputs_once_per_owner_and_correct_sharing(self):
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
        for mode, meta in record['checkpoints'].items():
            self.assertEqual(meta['self_attention_layers_1based'], [5, 6])
            self.assertEqual(meta['gated_pe_layers'], [2, 3])
            self.assertEqual(meta['gate_score_transform'], 'tanh')
            self.assertEqual(meta['shared_gate_across_layers'], mode.endswith('_v6'))
            self.assertEqual(meta['device_type_encoding']['categories'], ['nmos', 'pmos'])
        with (out / 'gate_device_types_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
            types = list(csv.DictReader(handle))
        self.assertEqual(len(types), 2 * 2 * 3)  # models x samples x physical MOS owners
        self.assertEqual({row['device_type'] for row in types}, {'nmos', 'pmos'})
        for row in types:
            expected = (1., 0.) if row['device_type'] == 'nmos' else (0., 1.)
            self.assertEqual((float(row['nmos_one_hot']), float(row['pmos_one_hot'])), expected)
        with (out / 'gate_op_inputs_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), len(types) * 3)
        with (out / 'gate_scores_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
            scores = list(csv.DictReader(handle))
        self.assertEqual({row['pe_layer'] for row in scores}, {'2', '3'})
        self.assertEqual({row['gate_score_transform'] for row in scores}, {'tanh'})
        sheets = workbook(out / 'gate_scores_pin.xlsx')
        self.assertIn('Device type encoding', text_values(sheets['Models']))
        html = (out / 'gate_viewer_pin.html').read_text()
        payload = json.loads(re.search(r'const payload=(.*);\nconst byId', html).group(1))
        self.assertIn('deviceTypeInput', html)
        for model in payload['data'].values():
            for sample in model.values():
                for device in sample.values():
                    expected = [1., 0.] if device['device_type'] == 'nmos' else [0., 1.]
                    self.assertEqual(device['device_type_one_hot'], expected)
                    self.assertEqual({p[8] for p in device['pairs']}, {2, 3})

    def test_capture_observes_actual_appended_inputs_and_each_network(self):
        for version in ('v6', 'v7'):
            mode = f'device_op_supervised_gate_fusion_pin_{version}'
            model, metadata = load_checkpoint_pin(self.source / f'training_pin/seed_17_pin/{mode}/checkpoint_pin.pt')
            rows = self.data['samples'][:2]
            cache = PinResidentDataset(rows, metadata['preprocessing'], 'cpu',
                                       include_op=False, include_performance=False, max_tokens=32)
            _, batch = next(cache.batches(2))
            captured = capture_forward(model, batch)
            self.assertEqual(captured['device_type_one_hot'].shape, (2, 4, 2))
            expected = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 0.]])
            torch.testing.assert_close(captured['device_type_one_hot'], expected.expand(2, -1, -1))
            self.assertEqual([c['pe_layer'] for c in captured['layer_captures']], [2, 3])
            self.assertFalse(model.gate._forward_hooks)
            for _, head in model.gate.iter_layer_score_heads():
                self.assertFalse(head._forward_hooks)


if __name__ == '__main__':
    unittest.main()
