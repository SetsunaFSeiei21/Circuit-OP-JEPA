"""Numerical attention replay, physical pin identities, and read-only reports."""
import contextlib
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from final_exam.attention_probe_pin import (
    UNGATED_MODELS_PIN, attention_rows, capture_pe_forward, downstream_encoder,
    inspect_source, reconstruct_attention, run_probe,
)
from final_exam.tests.test_gate_probe_pin import snapshot
from final_exam.tests.test_run_pin import tiny_inputs
from final_exam.tests.test_summary_pin import text_values, workbook
from final_exam.run_pin import load_checkpoint_pin, run_experiment_pin
from final_exam.attention_full_circuit_pin import full_circuit_records, FullCircuitReport

ROOT = Path(__file__).resolve().parents[2]


class AttentionMathTests(unittest.TestCase):
    def check_replay(self, device='cpu'):
        torch.manual_seed(23)
        module = torch.nn.MultiheadAttention(12, 3, batch_first=True, dropout=.3).to(device).eval()
        x = torch.randn(2, 7, 12, device=device)
        padding = torch.zeros(2, 7, dtype=torch.bool, device=device)
        padding[0, -1] = True
        boolean = torch.zeros(7, 7, dtype=torch.bool, device=device)
        boolean[2, 4] = True
        additive = torch.randn(6, 7, 7, device=device) * .2
        additive[:, 1, 3] = float('-inf')
        for mask in (None, boolean, additive):
            with self.subTest(mask='none' if mask is None else str(mask.dtype)), torch.no_grad():
                # Match padding mask type so native MHA does not warn about
                # deprecated mismatched mask dtypes.
                pad = padding if mask is None or mask.dtype == torch.bool else torch.zeros_like(padding, dtype=x.dtype).masked_fill(padding, float('-inf'))
                kwargs = dict(attn_mask=mask, key_padding_mask=pad)
                item = reconstruct_attention(module, (x, x, x), kwargs)
                output, weights = module(x, x, x, need_weights=True, average_attn_weights=False, **kwargs)
                torch.testing.assert_close(item['attention_weight'], weights, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(item['replay_output'], output, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
                self.assertTrue(bool((weights[0, :, :, -1] == 0).all()))
                if mask is additive:
                    self.assertTrue(bool((weights[:, :, 1, 3] == 0).all()))

    def test_replay_matches_native_per_head_weights_and_masks(self):
        self.check_replay()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_replay_matches_native_per_head_weights(self):
        self.check_replay('cuda')

    def test_interleaved_pin_mapping_and_full_sequence_mass(self):
        # Pin positions are deliberately not ordered by device or D/G/S/B.
        # 0=external, 1=R.p, 2=M.s, 3=M.d, 4=R.n, 5=M.b, 6=M.g, 7=padding.
        terms = ['g', 'b', 'd', 's', 'p', 'n']
        batch = dict(owner_index=torch.tensor([[-1, 1, 0, 0, 1, 0, 0, -1]]),
            pin_terminal_index=torch.tensor([[-1, 4, 3, 2, 5, 1, 0, -1]]),
            pin_net_index=torch.tensor([[9, 8, 0, 2, 0, 0, 3, -1]]),
            device_mask=torch.tensor([[True] * 7 + [False]]))
        sample = dict(sample_id='s1', circuit_id='C', devices=[
            dict(device_id='Z_R', device_type='resistor', terminals={'p': 'vp', 'n': 'gnd'}),
            dict(device_id='A_M', device_type='nmos', terminals={'d': 'vd', 'g': 'vg', 's': 'gnd', 'b': 'gnd'})])
        logits = torch.arange(2 * 9 * 9, dtype=torch.float64).reshape(1, 2, 9, 9) / 300
        logits[0, 1] = logits[0, 1].flip(-1)
        logits[:, :, :, -1] = float('-inf')
        weights = logits.softmax(-1)
        capture = [dict(pe_layer=1, encoder_layer=4, mask_scope='global',
            qk_scaled=torch.zeros_like(logits), additive_bias=logits,
            attention_logit=logits, attention_weight=weights)]
        rows = list(attention_rows(batch, [sample], capture, mode='device_op_cascade_pin', seed=17, terminal_names=terms))
        scores = [row for kind, row in rows if kind == 'score']
        masses = [row for kind, row in rows if kind == 'mass']
        self.assertEqual(len(scores), 3 * 16)
        self.assertEqual(len(masses), 3 * 4)
        self.assertEqual({row['device_id'] for row in scores}, {'A_M'})
        ds = next(row for row in scores if row['head'] == 1 and row['query_role'] == 'd' and row['key_role'] == 's')
        self.assertEqual((ds['query_pin_index'], ds['key_pin_index']), (3, 2))
        self.assertEqual(ds['attention_weight'], float(weights[0, 0, 4, 3]))
        mean = next(row for row in scores if row['head'] == 'mean' and row['query_role'] == 'd' and row['key_role'] == 's')
        self.assertAlmostEqual(mean['attention_weight'], float(weights[0, :, 4, 3].mean()))
        for row in masses:
            self.assertAlmostEqual(row['total_mass'], 1.)
            self.assertLess(row['intra_device_mass'], 1.)
            self.assertGreater(row['graph_token_mass'], 0.)
            self.assertEqual(row['padding_mass'], 0.)
            self.assertAlmostEqual(sum(row[key] for key in (
                'intra_device_mass', 'other_device_mass', 'external_port_mass', 'graph_token_mass', 'padding_mass')), 1.)

        # Full matrices must contain CROSS-device entries, resistor terminals,
        # external ports and the graph token, with no padding or row re-fit.
        sample['ports'] = {'VOUT': 'output'}
        batch['device_ids'] = [['port:VOUT', 'pin:Z_R:p', 'pin:A_M:s', 'pin:A_M:d',
                               'pin:Z_R:n', 'pin:A_M:b', 'pin:A_M:g']]
        records = list(full_circuit_records(batch, [sample], capture,
            mode='device_op_cascade_pin', seed=17, terminal_names=terms))
        meta, arrays = records[0]
        self.assertEqual(meta['shape'], [1, 3, 8, 8])
        self.assertEqual(meta['heads'], [1, 2, 'mean'])
        self.assertEqual([t['label'] for t in meta['tokens']],
                         ['GRAPH', 'PORT:VOUT', 'Z_R.P', 'A_M.S', 'A_M.D', 'Z_R.N', 'A_M.B', 'A_M.G'])
        np.testing.assert_allclose(arrays['attention_weight'][0, :2], weights[0, :, :8, :8].numpy(), rtol=1e-6)
        np.testing.assert_allclose(arrays['attention_weight'][0, 2], weights[0, :, :8, :8].mean(0).numpy(), rtol=1e-6)
        np.testing.assert_allclose(arrays['attention_weight'].sum(-1), 1., atol=1e-6)
        self.assertGreater(arrays['attention_weight'][0, 0, 4, 2], 0.)  # M.d -> R.p
        self.assertAlmostEqual(float(arrays['attention_weight'][0, 0, 4, 2]), float(weights[0, 0, 4, 2]), places=7)
        with tempfile.TemporaryDirectory() as tmp:
            report = FullCircuitReport(tmp, ['s1'])
            report.add(records)
            report.finish()
            with np.load(Path(tmp) / 'full_attention_pin/000001_pin.npz', allow_pickle=False) as archive:
                self.assertEqual(json.loads(str(archive['metadata_json'])), meta)
                for field, values in arrays.items():
                    np.testing.assert_array_equal(archive[field], values)
            import base64
            compact = np.frombuffer(base64.b64decode(report.viewer[0]['weights_f32le_b64']), dtype='<f4')
            np.testing.assert_array_equal(compact.reshape(meta['shape']), arrays['attention_weight'])
        batch['device_ids'][0][3] = 'pin:Z_R:d'
        with self.assertRaisesRegex(ValueError, 'physical device/terminal disagree'):
            list(full_circuit_records(batch, [sample], capture,
                mode='device_op_cascade_pin', seed=17, terminal_names=terms))


class AttentionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / 'source_pin'
        cls.source.mkdir()
        data, config, _, cfg = tiny_inputs(cls.source)
        config['model'].update(encoder_layers=4, cross_layers=2)
        cfg.write_text(json.dumps(config))
        cls.data = data
        cls.data_path = cls.source / 'data_pin.json'
        cls.data_path.write_text(json.dumps(data))
        cls.models = list(UNGATED_MODELS_PIN) + ['device_op_gate_pin_v2']
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_experiment_pin(cls.data_path, cfg, cls.source / 'training_pin', cls.models, [17], 'cpu', smoke=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def probe(self, name, models=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_probe(self.source, circuit='synthetic_0', models=models or ['device_op_cascade_pin'],
                device='cpu', output=self.root / name, log_dir=self.root / (name + '_logs'), **kwargs)

    def test_all_ungated_variants_export_pe_heads_matrices_without_mutation(self):
        before = snapshot(self.source)
        with patch('device_jepa.data.fit_preprocessor', side_effect=AssertionError('refit')), \
                patch('torch.optim.AdamW', side_effect=AssertionError('retrain')):
            output = self.probe('all_pin', list(UNGATED_MODELS_PIN), num_samples=3, viewer_samples=1, batch_size=2)
        self.assertEqual(before, snapshot(self.source))
        meta = json.loads((output / 'probe_pin.json').read_text())
        self.assertEqual(meta['status'], 'complete')
        self.assertEqual(meta['viewer_samples'], 1)
        for info in meta['checkpoints'].values():
            self.assertEqual(info['pe_layers'], 2)
            self.assertEqual(info['encoder_layers_1based'], [3, 4])
            self.assertFalse(info['gate_active'])
        with (output / 'attention_scores_pin.csv').open(encoding='utf-8-sig') as stream:
            scores = list(csv.DictReader(stream))
        self.assertEqual({r['model'] for r in scores}, set(UNGATED_MODELS_PIN))
        self.assertEqual({r['pe_layer'] for r in scores}, {'1', '2'})
        self.assertEqual({r['encoder_layer'] for r in scores}, {'3', '4'})
        self.assertEqual({r['head'] for r in scores}, {'1', '2', 'mean'})
        self.assertEqual({r['device_type'] for r in scores}, {'nmos', 'pmos'})
        self.assertTrue(all(0 <= float(r['attention_weight']) <= 1 for r in scores))
        matrices = {}
        for row in scores:
            key = tuple(row[k] for k in ('model', 'sample_id', 'pe_layer', 'device_id', 'head'))
            matrices[key] = matrices.get(key, 0) + 1
        self.assertTrue(all(count == 16 for count in matrices.values()))
        sheets = workbook(output / 'attention_scores_pin.xlsx')
        self.assertIn('SampleMatrices', sheets)
        self.assertIn('MeanMatrices', sheets)
        self.assertTrue(any('PE 2' in (value or '') for value in text_values(sheets['SampleMatrices'])))
        html = (output / 'attention_viewer_pin.html').read_text()
        payload = json.loads(re.search(r'const payload=(.*);\nconst byId', html).group(1))
        self.assertEqual(len({r['sample_id'] for r in payload['records']}), 1)
        self.assertEqual({str(r['head']) for r in payload['records']}, {'1', '2', 'mean'})
        self.assertTrue(all(len(r['attention_weight']) == 4 for r in payload['records']))
        self.assertTrue(all(all(v is not None for v in r['masses']) for r in payload['records']))
        self.assertEqual(len(payload['full_circuit']), len(UNGATED_MODELS_PIN))
        full_index = json.loads((output / 'full_attention_pin/manifest_pin.json').read_text())
        self.assertEqual(len(full_index['entries']), len(UNGATED_MODELS_PIN) * len(meta['selected_sample_ids']))
        # Full archives reuse precisely the same probabilities as the original
        # intra-MOS CSV, while retaining entries between different devices.
        entry = full_index['entries'][0]
        with np.load(output / 'full_attention_pin' / entry['file'], allow_pickle=False) as archive:
            saved = json.loads(str(archive['metadata_json']))
            positions = {token['sequence_index']: i for i, token in enumerate(saved['tokens'])}
            for row in scores:
                if row['model'] != entry['model'] or row['sample_id'] != entry['sample_id']:
                    continue
                layer = next(i for i, item in enumerate(saved['layers']) if str(item['pe_layer']) == row['pe_layer'])
                head = [str(h) for h in saved['heads']].index(row['head'])
                q, k = positions[int(row['query_sequence_index'])], positions[int(row['key_sequence_index'])]
                self.assertAlmostEqual(float(archive['attention_weight'][layer, head, q, k]),
                                       float(row['attention_weight']), places=6)

    def test_capture_preserves_predictions_and_cleans_hooks_even_on_failure(self):
        from final_exam.data_pin import PinResidentDataset
        model, meta = load_checkpoint_pin(self.source / 'training_pin/seed_17_pin/device_op_cascade_pin/checkpoint_pin.pt')
        cache = PinResidentDataset(self.data['samples'][:2], meta['preprocessing'], 'cpu',
                                   include_op=False, include_performance=False, max_tokens=32)
        _, batch = next(cache.batches(2))
        encoder, _ = downstream_encoder(model)
        with torch.inference_mode():
            before = model(batch)
            captures = capture_pe_forward(model, batch)
            after = model(batch)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        self.assertEqual([c['pe_layer'] for c in captures], [1, 2])
        self.assertEqual([c['mask_scope'] for c in captures], ['local', 'global'])
        # First PE Self-Attention precedes any feature cross-attention: the
        # same topology produces the same scores despite different designs.
        torch.testing.assert_close(captures[0]['attention_weight'][0],
                                   captures[0]['attention_weight'][1], atol=0, rtol=0)
        poison = dict(batch, owner_op=torch.full((2, 4, 3), float('nan')),
                      op=torch.full((2, batch['device_mask'].shape[1], 3), float('nan')))
        poisoned_captures = capture_pe_forward(model, poison)
        for clean, poisoned in zip(captures, poisoned_captures):
            torch.testing.assert_close(clean['attention_weight'], poisoned['attention_weight'], atol=0, rtol=0)
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        with patch.object(encoder.fusion_blocks[1].self_attn, 'forward', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                capture_pe_forward(model, batch)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    def test_rejects_gated_models_and_source_overwrites(self):
        before = snapshot(self.source)
        with self.assertRaisesRegex(ValueError, 'natively ungated'):
            inspect_source(self.source, 'synthetic_0', ['device_op_gate_pin_v2'], 17)
        gated, _ = load_checkpoint_pin(self.source / 'training_pin/seed_17_pin/device_op_gate_pin_v2/checkpoint_pin.pt')
        with self.assertRaisesRegex(ValueError, 'contains a gate'):
            downstream_encoder(gated)
        with self.assertRaisesRegex(ValueError, 'separate from the source'):
            run_probe(self.source, circuit='synthetic_0', models=['device_op_cascade_pin'], device='cpu',
                      output=self.source / 'do_not_create_pin', log_dir=self.root / 'bad_logs')
        self.assertEqual(snapshot(self.source), before)

    def test_cli_metadata_preflight_has_no_torch_dependency(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'final_exam/attention_probe_pin.py'),
            '--source-run', str(self.source), '--circuit', 'synthetic_0', '--inspect-source'],
            capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['models'], ['device_op_cascade_pin'])

    def test_changed_source_is_rejected_and_failed_probe_is_recorded(self):
        original = self.data_path.read_bytes()
        changed = json.loads(original)
        row = next(row for row in changed['samples'] if row['performance']['gain'] is not None)
        row['performance']['gain'] += 1.
        try:
            self.data_path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.probe('changed_pin', num_samples=1)
            self.assertEqual(json.loads((self.root / 'changed_pin/probe_pin.json').read_text())['status'], 'failed')
        finally:
            self.data_path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
