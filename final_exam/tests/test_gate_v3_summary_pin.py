"""M-MLP reports preserve per-version metrics, routing identity and expert costs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from final_exam.gate_version_registry_pin import V2_GATE_MODELS_PIN, V3_GATE_MODELS_PIN
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook, write_run


def v3_contract(mode, experts=16, dropout=.1, hidden=128, fields=41, heads=8):
    dims = [fields * 2 + 8, hidden, hidden, heads]
    architecture = dict(kind='mixed_mlp', layer_dims=dims, n_experts=experts, dropout=dropout,
        activation='gelu', block_order='mo_linear_dropout_gelu', routing='input_softmax_dense',
        output='per_head_logits', shared_across_layers=True,
        initialization='xavier_hidden_small_random_output_logit_bias',
        source_repository='https://github.com/Yutong-Feng/FlowNet',
        source_commit='57ad6128c2a1980d0aa73325a844659262271797')
    return dict(gate_version='v3', attention_scope='fusion_self_attention' if '_fusion_' in mode
        else 'all_encoder_self_attention', gate_relative_normalization='per_sample_per_field_all_valid_mos',
        gate_input_dim=dims[0], gate_hidden_dim=hidden, norm_eps=1e-5, gate_network='mixed_mlp',
        gate_n_experts=experts, gate_mlp_dropout=dropout,
        gate_network_parameters=sum(experts * (a*b + b + a + 1) for a, b in zip(dims, dims[1:])),
        gate_architecture=architecture)


def v3_run(directory, mixed=False):
    models = ['device_op_cascade_pin', *V3_GATE_MODELS_PIN]
    if mixed:
        models += list(V2_GATE_MODELS_PIN)
    records = [record(mode, seed, mae=2e6 if seed == 17 else 4e6)
               for mode in models for seed in (17, 29)]
    for item in records:
        mode = item['mode']
        if mode == 'device_op_cascade_pin':
            item['report']['ungated_base_checkpoint_pin'] = 'shared_pin/device_op_supervised_base_pin/checkpoint_pin.pt'
        else:
            contract = v3_contract(mode)
            if mode in V2_GATE_MODELS_PIN:
                contract = {k: v for k, v in contract.items() if k in ('gate_version', 'attention_scope',
                    'gate_relative_normalization', 'gate_input_dim', 'gate_hidden_dim', 'norm_eps')}
                contract['gate_version'] = 'v2'
            item['report']['op_gate_pin'] = contract
    write_run(directory, records)
    for item in records:
        mode = item['mode']
        if '_distill_' not in mode and '_supervised_' not in mode:
            continue
        spec = staged_gate_spec_pin(mode)
        family = spec['base_training']
        before = copy.deepcopy(item['report'])
        del before['op_gate_pin']
        if family == 'supervised':
            before['metrics'] = copy.deepcopy(next(r['report']['metrics'] for r in records
                if r['mode'] == 'device_op_cascade_pin' and r['seed'] == item['seed']))
        else:
            before['metrics']['UGF_Hz']['mae'] += 1e6
        comparison = dict(base_checkpoint=f'../shared_pin/device_op_{family}_base_pin/checkpoint_pin.pt',
            base_test=before, post_gate_test=item['report'], attention_scope=spec['attention_scope'],
            gate_version=item['report']['op_gate_pin']['gate_version'])
        path = directory / f'seed_{item["seed"]}_pin' / mode / 'gate_comparison_pin.json'
        path.write_text(json.dumps(comparison))
    return records


class GateV3SummaryPinTests(unittest.TestCase):
    def test_all_six_models_report_expert_budget_and_shared_base_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = v3_run(root)
            original = copy.deepcopy(records)
            rows = summary_rows_pin(records)
            self.assertEqual(records, original)
            self.assertEqual({r['mode'] for r in rows}, {'device_op_cascade_pin', *V3_GATE_MODELS_PIN})
            sheets = workbook(export_results_pin(root))
            self.assertEqual(len(sheets['GateStudy'].findall('x:sheetData/x:row', NS)), 5 + 8)
            self.assertEqual(text_values(sheets['GateStudy']).count('device_op_distilled_base_pin'), 1)
            self.assertIn('Gate experts per layer', text_values(sheets['ModelInputs']))
            self.assertIn('Gate network parameters', text_values(sheets['ModelInputs']))
            names = text_values(sheets['ModelInputs'])
            self.assertEqual(names.count('mixed_mlp'), 6)
            self.assertIn('v3', text_values(sheets['GateBeforeAfter']))
            for row in sheets['ModelInputs'].findall('x:sheetData/x:row', NS):
                if not any(mode in text_values(row) for mode in V3_GATE_MODELS_PIN):
                    continue
                values = {''.join(c for c in cell.attrib['r'] if c.isalpha()): cell
                          for cell in row.findall('x:c', NS)}
                self.assertEqual(float(values['Q'].find('x:v', NS).text), 16)
                self.assertEqual(float(values['R'].find('x:v', NS).text), .1)
                self.assertEqual(float(values['S'].find('x:v', NS).text), 472656)
                self.assertIn('input_softmax_dense', text_values(values['T'])[0])

    def test_mixed_versions_do_not_merge_models_or_duplicate_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v3_run(root, mixed=True)
            sheets = workbook(export_results_pin(root))
            study = sheets['GateStudy']
            self.assertEqual(len(study.findall('x:sheetData/x:row', NS)), 5 + 14)
            for mode in (*V2_GATE_MODELS_PIN, *V3_GATE_MODELS_PIN):
                self.assertEqual(text_values(study).count(mode), 1)
            self.assertEqual(text_values(study).count('device_op_distilled_base_pin'), 1)

    def test_bad_architecture_and_expert_counts_are_rejected(self):
        mode = 'device_op_gate_fusion_pin_v3'
        changes = [dict(gate_version='v2'), dict(attention_scope='all_encoder_self_attention'),
            dict(gate_relative_normalization='batch'), dict(gate_input_dim=49),
            dict(gate_hidden_dim=0), dict(norm_eps=0), dict(gate_network='mlp'),
            dict(gate_n_experts=True), dict(gate_n_experts=0), dict(gate_n_experts=8),
            dict(gate_mlp_dropout=1), dict(gate_mlp_dropout=float('nan')),
            dict(gate_network_parameters=12), dict(gate_architecture=None)]
        for change in changes:
            row = record(mode)
            row['report']['op_gate_pin'] = {**v3_contract(mode), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary_rows_pin([row])
        for key, value in [('routing', 'top_k'), ('layer_dims', [90, 128, 8]),
                           ('shared_across_layers', False), ('source_commit', '')]:
            row = record(mode)
            row['report']['op_gate_pin'] = v3_contract(mode)
            row['report']['op_gate_pin']['gate_architecture'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summary_rows_pin([row])
        with self.assertRaisesRegex(ValueError, 'saved M-MLP'):
            summary_rows_pin([record(mode)])

    def test_different_expert_budgets_cannot_be_aggregated_across_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = v3_run(root)
            row = next(r for r in records if r['mode'] == V3_GATE_MODELS_PIN[0] and r['seed'] == 29)
            row['report']['op_gate_pin'] = v3_contract(row['mode'], experts=4)
            (root / 'results_pin.json').write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'differs between seeds/splits'):
                export_results_pin(root)
            self.assertFalse((root / 'summary_pin.xlsx').exists())


if __name__ == '__main__':
    unittest.main()
