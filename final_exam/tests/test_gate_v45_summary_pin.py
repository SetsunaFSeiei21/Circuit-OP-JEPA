"""V4/V5 Excel contracts distinguish signed bias, skipped PE1 and layer sharing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from final_exam.gate_version_registry_pin import V4_GATE_MODELS_PIN, V5_GATE_MODELS_PIN
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_gate_v3_summary_pin import v3_contract
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook, write_run


def signed_contract(mode, *, fusion_layers=3, strength=1.):
    version = mode.rsplit('_', 1)[-1]
    result = v3_contract(mode)
    result.update(gate_version=version, attention_scope='fusion_after_first_self_attention')
    count = 1 if version == 'v4' else fusion_layers - 1
    result['gate_network_parameters'] *= count
    architecture = result['gate_architecture']
    architecture.update(shared_across_layers=version == 'v4',
        gated_pe_layers=list(range(2, fusion_layers + 1)), independent_gate_count=count,
        bias_transform='strength*tanh(logits)', bias_strength=strength,
        initialization='xavier_hidden_small_random_output_zero_bias')
    if version == 'v5':
        architecture['layer_initialization'] = 'identical_copies_independent_storage'
    if '_distill_' in mode or '_supervised_' in mode:
        spec = staged_gate_spec_pin(mode)
        result.update(base_training=spec['base_training'],
            final_trainable='gate_and_decoder' if spec['final_phase'] == 'gate_decoder' else 'gate_only')
    return result


def signed_run(directory):
    models = ['device_op_cascade_pin', *V4_GATE_MODELS_PIN, *V5_GATE_MODELS_PIN]
    records = [record(mode, seed, mae=2e6 if seed == 17 else 4e6)
               for mode in models for seed in (17, 29)]
    for item in records:
        mode = item['mode']
        if mode == 'device_op_cascade_pin':
            item['report']['ungated_base_checkpoint_pin'] = 'shared_pin/device_op_supervised_base_pin/checkpoint_pin.pt'
        else:
            item['report']['op_gate_pin'] = signed_contract(mode)
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
            gate_version=item['report']['op_gate_pin']['gate_version'],
            base_family=family, final_phase=spec['final_phase'])
        (directory / f'seed_{item["seed"]}_pin' / mode / 'gate_comparison_pin.json').write_text(json.dumps(comparison))
    return records


class SignedGateSummaryTests(unittest.TestCase):
    def test_eight_models_keep_signed_shared_or_independent_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = signed_run(root)
            original = copy.deepcopy(records)
            rows = summary_rows_pin(records)
            self.assertEqual(records, original)
            self.assertEqual(len(rows), 9)
            sheets = workbook(export_results_pin(root))
            self.assertEqual(text_values(sheets['GateStudy']).count('device_op_distilled_base_pin'), 1)
            inputs = text_values(sheets['ModelInputs'])
            for heading in ('Gate parameter sharing', 'Gated PE layers', 'Gate score transform',
                            'Gate bias range', 'Independent gate networks', 'Gate initialization'):
                self.assertIn(heading, inputs)
            self.assertEqual(inputs.count('PE2, PE3'), 8)
            self.assertEqual(inputs.count('shared'), 4)
            self.assertEqual(inputs.count('independent_per_layer'), 4)
            self.assertEqual(inputs.count('strength*tanh(logits)'), 8)
            self.assertIn('v4', text_values(sheets['GateBeforeAfter']))
            self.assertIn('v5', text_values(sheets['GateBeforeAfter']))
            for row in sheets['ModelInputs'].findall('x:sheetData/x:row', NS):
                names = text_values(row)
                modes = [m for m in (*V4_GATE_MODELS_PIN, *V5_GATE_MODELS_PIN) if m in names]
                if not modes:
                    continue
                values = {''.join(c for c in cell.attrib['r'] if c.isalpha()): cell
                          for cell in row.findall('x:c', NS)}
                expected = 472656 if modes[0].endswith('_v4') else 945312
                self.assertEqual(float(values['S'].find('x:v', NS).text), expected)
                self.assertIn('fusion_after_first_self_attention', names)

    def test_invalid_signed_scope_transform_sharing_and_budget_rejected(self):
        for version in ('v4', 'v5'):
            mode = f'device_op_gate_fusion_pin_{version}'
            base = signed_contract(mode)
            for change in ({'gate_version': 'v3'}, {'attention_scope': 'fusion_self_attention'},
                           {'gate_network_parameters': 472656 + 1}):
                item = record(mode)
                item['report']['op_gate_pin'] = {**copy.deepcopy(base), **change}
                with self.subTest(version=version, change=change), self.assertRaises(ValueError):
                    summary_rows_pin([item])
            for field, value in [('gated_pe_layers', [1, 2, 3]), ('gated_pe_layers', []),
                    ('gated_pe_layers', [2, 4]), ('gated_pe_layers', [2., 3.]),
                    ('independent_gate_count', 0), ('independent_gate_count', True),
                    ('bias_transform', 'strength*log(sigmoid(logits))'),
                    ('bias_strength', -1), ('bias_strength', float('nan')),
                    ('shared_across_layers', version == 'v5'),
                    ('initialization', 'xavier_hidden_small_random_output_logit_bias')]:
                item = record(mode)
                item['report']['op_gate_pin'] = copy.deepcopy(base)
                item['report']['op_gate_pin']['gate_architecture'][field] = value
                with self.subTest(version=version, field=field, value=value), self.assertRaises(ValueError):
                    summary_rows_pin([item])
            for field in ('gate_version', 'attention_scope'):
                item = record(mode)
                item['report']['op_gate_pin'] = copy.deepcopy(base)
                del item['report']['op_gate_pin'][field]
                with self.subTest(version=version, missing=field), self.assertRaises(ValueError):
                    summary_rows_pin([item])
            with self.assertRaisesRegex(ValueError, 'saved M-MLP'):
                summary_rows_pin([record(mode)])

    def test_v5_parameters_scale_with_actual_number_of_gated_layers(self):
        mode = 'device_op_gate_fusion_pin_v5'
        item = record(mode)
        item['report']['op_gate_pin'] = signed_contract(mode, fusion_layers=5)
        self.assertEqual(item['report']['op_gate_pin']['gate_network_parameters'], 4 * 472656)
        self.assertEqual(len(summary_rows_pin([item])), 1)
        del item['report']['op_gate_pin']['gate_architecture']['layer_initialization']
        with self.assertRaisesRegex(ValueError, 'independent layer parameters'):
            summary_rows_pin([item])

    def test_cross_seed_strength_or_gated_layers_cannot_be_silently_aggregated(self):
        mode = 'device_op_gate_fusion_pin_v5'
        for change in ({'strength': 2.}, {'fusion_layers': 4}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records = [record(mode, seed) for seed in (17, 29)]
                for item in records:
                    item['report']['op_gate_pin'] = signed_contract(mode, **(change if item['seed'] == 29 else {}))
                write_run(root, records)
                with self.assertRaisesRegex(ValueError, 'differs between seeds/splits'):
                    export_results_pin(root)
                self.assertFalse((root / 'summary_pin.xlsx').exists())

    def test_comparison_contract_cannot_disagree_with_saved_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signed_run(root)
            mode = 'device_op_distill_gate_fusion_pin_v5'
            path = root / 'seed_17_pin' / mode / 'gate_comparison_pin.json'
            data = json.loads(path.read_text())
            data['post_gate_test']['op_gate_pin']['gate_architecture']['bias_strength'] = 2.
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'comparison contract differs'):
                export_results_pin(root)
            self.assertFalse((root / 'summary_pin.xlsx').exists())


if __name__ == '__main__':
    unittest.main()
