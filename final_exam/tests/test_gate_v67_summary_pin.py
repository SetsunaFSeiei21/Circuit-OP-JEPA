"""Type-conditioned reports retain polarity encoding, input width and layer costs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from final_exam.gate_version_registry_pin import V6_GATE_MODELS_PIN, V7_GATE_MODELS_PIN
from final_exam.staged_gate_registry_pin import staged_gate_spec_pin
from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_gate_v45_summary_pin import signed_contract
from final_exam.tests.test_summary_pin import NS, record, text_values, workbook, write_run


TYPE_ENCODING = dict(kind='one_hot', categories=['nmos', 'pmos'], width=2,
                     placement='after_terminal_roles', normalization='none')


def typed_contract(mode, *, fusion_layers=3):
    version = mode.rsplit('_', 1)[-1]
    parent = 'v4' if version == 'v6' else 'v5'
    result = signed_contract(mode[:-2] + parent, fusion_layers=fusion_layers)
    result['gate_version'] = version
    result['gate_input_dim'] += 2
    architecture = result['gate_architecture']
    architecture['layer_dims'][0] += 2
    architecture['device_type_encoding'] = copy.deepcopy(TYPE_ENCODING)
    dims = architecture['layer_dims']
    result['gate_network_parameters'] = architecture['independent_gate_count'] * sum(
        result['gate_n_experts'] * (a*b + b + a + 1) for a, b in zip(dims, dims[1:]))
    return result


def typed_run(directory):
    models = ['device_op_cascade_pin', *V6_GATE_MODELS_PIN, *V7_GATE_MODELS_PIN]
    records = [record(mode, seed, mae=2e6 if seed == 17 else 4e6)
               for mode in models for seed in (17, 29)]
    for item in records:
        mode = item['mode']
        if mode == 'device_op_cascade_pin':
            item['report']['ungated_base_checkpoint_pin'] = 'shared_pin/device_op_supervised_base_pin/checkpoint_pin.pt'
        else:
            item['report']['op_gate_pin'] = typed_contract(mode)
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
        comparison = dict(base_checkpoint=f'../shared_pin/device_op_{family}_base_pin/checkpoint_pin.pt',
            base_test=before, post_gate_test=item['report'], attention_scope=spec['attention_scope'],
            gate_version=item['report']['op_gate_pin']['gate_version'],
            base_family=family, final_phase=spec['final_phase'])
        (directory / f'seed_{item["seed"]}_pin' / mode / 'gate_comparison_pin.json').write_text(json.dumps(comparison))
    return records


class TypedGateSummaryTests(unittest.TestCase):
    def test_all_eight_models_export_actual_input_width_and_parameter_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = typed_run(root)
            self.assertEqual(len(summary_rows_pin(records)), 9)
            sheets = workbook(export_results_pin(root))
            inputs = text_values(sheets['ModelInputs'])
            self.assertIn('Gate device-type encoding', inputs)
            self.assertIn('Gate device-type category order', inputs)
            self.assertEqual(inputs.count('NMOS [1, 0]; PMOS [0, 1]'), 8)
            self.assertEqual(inputs.count('PE2, PE3'), 8)
            self.assertEqual(inputs.count('shared'), 4)
            self.assertEqual(inputs.count('independent_per_layer'), 4)
            self.assertEqual(text_values(sheets['GateStudy']).count('device_op_distilled_base_pin'), 1)
            for row in sheets['ModelInputs'].findall('x:sheetData/x:row', NS):
                modes = [m for m in (*V6_GATE_MODELS_PIN, *V7_GATE_MODELS_PIN) if m in text_values(row)]
                if not modes:
                    continue
                values = {''.join(c for c in cell.attrib['r'] if c.isalpha()): cell
                          for cell in row.findall('x:c', NS)}
                self.assertEqual(float(values['M'].find('x:v', NS).text), 92)
                expected = 476784 if modes[0].endswith('_v6') else 953568
                self.assertEqual(float(values['S'].find('x:v', NS).text), expected)

    def test_missing_reordered_or_normalized_type_encoding_is_rejected(self):
        for version in ('v6', 'v7'):
            mode = f'device_op_gate_fusion_pin_{version}'
            for encoding in (None, {**TYPE_ENCODING, 'categories': ['pmos', 'nmos']},
                             {**TYPE_ENCODING, 'width': 3}, {**TYPE_ENCODING, 'normalization': 'zscore'},
                             {**TYPE_ENCODING, 'placement': 'before_op'}):
                item = record(mode)
                item['report']['op_gate_pin'] = typed_contract(mode)
                item['report']['op_gate_pin']['gate_architecture']['device_type_encoding'] = encoding
                with self.subTest(version=version, encoding=encoding), self.assertRaisesRegex(ValueError, 'NMOS/PMOS'):
                    summary_rows_pin([item])

    def test_v7_parameter_count_scales_and_cannot_masquerade_as_v5(self):
        mode = 'device_op_gate_fusion_pin_v7'
        item = record(mode)
        contract = typed_contract(mode, fusion_layers=5)
        item['report']['op_gate_pin'] = contract
        self.assertEqual(contract['gate_network_parameters'], 4 * 476784)
        self.assertEqual(len(summary_rows_pin([item])), 1)
        contract['gate_version'] = 'v5'
        with self.assertRaisesRegex(ValueError, 'differs from the model name'):
            summary_rows_pin([item])


if __name__ == '__main__':
    unittest.main()
