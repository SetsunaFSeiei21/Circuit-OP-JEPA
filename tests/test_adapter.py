"""Small synthetic fixtures only; no supplied source dataset is committed."""
import copy
import hashlib
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from device_jepa.adapter import (
    AdapterError, adapt_analog_agent, combine_datasets, evaluate_spice_expression,
    inspect_analog_agent, parse_spice_netlist,
)
from device_jepa.data import validate_dataset


OP_NAMES = [
    'id_a', 'gm_s', 'gds_s', 'gmbs_s', 'vth_v', 'vdsat_v',
    'vd_v', 'vg_v', 'vs_v', 'vb_v', 'vgs_v', 'vds_v', 'vbs_v',
    'model_vgs_v', 'model_vds_v', 'model_vbs_v', 'vgs_normalized_v',
    'vds_normalized_v', 'vbs_normalized_v', 'reverse_body_bias_v',
    'gmid_v_inv', 'intrinsic_gain_v_v', 'saturation_margin_v',
    'w_um', 'l_um', 'multiplicity', 'cgg_f', 'cgs_f', 'cgd_f', 'cgb_f',
    'cdg_f', 'cdd_f', 'cds_f', 'cdb_f', 'csg_f', 'csd_f', 'css_f',
    'csb_f', 'cbg_f', 'cbd_f', 'cbs_f', 'cbb_f', 'capbd_f', 'capbs_f',
]
NMOS = 'sky130_fd_pr__nfet_01v8_lvt'
PMOS = 'sky130_fd_pr__pfet_01v8_lvt'
NETLIST = f'''* Synthetic fixture, not measured source data.
.include "Tiny_params.sp"
.include "not-fetched-model-library.sp"
.option scale=1u
.subckt DUT IN OUT VDD VSS
IBIAS OUT VSS DC {{IBIAS_A}}
XN OUT IN VSS VSS {NMOS} W={{width}} L={{length}} m={{2 * mult}}
XP OUT IN VDD VDD {PMOS} W={{width}} L={{length}} m=1
VBIAS IN VSS DC 0.7
RLOAD OUT VDD 10k
CLOAD OUT VSS 1p
.ends DUT
'''


def fixture(root, circuit='Tiny'):
    """Create a minimal Analog-Agent-shaped SQLite source for unit tests."""
    (root / f'{circuit}.sp').write_text(NETLIST.replace('Tiny_params', circuit + '_params'))
    (root / f'{circuit}_params.sp').write_text('.param width=99 length=99 mult=99 IBIAS_A=99\n.param doubled={2*width}\n')
    (root / 'circuit_metadata.json').write_text(json.dumps({'circuit_name': circuit}))
    connection = sqlite3.connect(root / 'sampling_history.sqlite3')
    connection.executescript('''
        CREATE TABLE dataset_schema (id INTEGER, circuit_name TEXT, circuit_type TEXT,
          parameter_names_json TEXT, metric_names_json TEXT);
        CREATE TABLE samples (sample_id INTEGER, design_key TEXT, design_values_json TEXT,
          metric_values_json TEXT, success INTEGER, run_id TEXT, observation_json TEXT);
        CREATE TABLE dc_operating_points (sample_id INTEGER, status TEXT, converged INTEGER,
          feasible INTEGER, condition_json TEXT, bias_topology TEXT, reasons_json TEXT,
          missing_fields_json TEXT, node_voltages_json TEXT, follower_error_v REAL, vout_v REAL);
        CREATE TABLE sampling_runs (run_id TEXT, config_json TEXT);
    ''')
    connection.execute('CREATE TABLE dc_device_operating_points (sample_id INTEGER, device_name TEXT, device_type TEXT, model_name TEXT, primitive_path TEXT, ' + ','.join(name + ' REAL' for name in OP_NAMES) + ')')
    design_names = ['width', 'length', 'mult', 'IBIAS_A']
    metric_names = ['Gain_dB', 'UGF_Hz', 'Power_uW']
    connection.execute('INSERT INTO dataset_schema VALUES (?,?,?,?,?)', (1, circuit, 'synthetic_unit_test', json.dumps(design_names), json.dumps(metric_names)))
    observation = {'checked_devices': ['XN'], 'reasons': ['XN:saturation']}
    for sample_id in (1, 2, 3):
        connection.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)', (
            sample_id, f'design-{sample_id}', json.dumps([2.0 * sample_id, 0.5, 3.0, 1e-6]),
            '[10, NaN, 3]' if sample_id == 1 else '[20, 1e6, 4]' if sample_id == 2 else None,
            0 if sample_id != 2 else 1, 'run-one', json.dumps(observation)))
        connection.execute('INSERT INTO dc_operating_points VALUES (?,?,?,?,?,?,?,?,?,?,?)', (
            sample_id, 'complete', 1, 0, json.dumps({'CORNER': 'tt', 'TEMP': 27, 'PDK_PATH': '/private/not/read'}),
            'unity_gain_follower', '["XN:saturation"]', '{}', '{"out":0.7}', 0.1, 0.7))
        for name, kind, model, multiplicity in [('XN', 'NMOS', NMOS, 6), ('XP', 'PMOS', PMOS, 1)]:
            if sample_id == 3 and name == 'XP':
                continue
            values = {key: (index + 1) * 0.01 for index, key in enumerate(OP_NAMES)}
            values.update(w_um=2.0 * sample_id, l_um=0.5, multiplicity=multiplicity,
                          cgs_f=-1e-15, vgs_v=-0.7 if name == 'XP' else 0.7,
                          vds_v=-0.6 if name == 'XP' else 0.6)
            if sample_id == 3:
                values['gm_s'] = None
            row = [sample_id, name, kind, model, f'm.xdut.{name.lower()}.model'] + [values[k] for k in OP_NAMES]
            connection.execute('INSERT INTO dc_device_operating_points VALUES (' + ','.join('?' for _ in row) + ')', row)
    connection.execute('INSERT INTO sampling_runs VALUES (?,?)', ('run-one', json.dumps({
        'sampling': {'constraints': {'saturation_margin_min_v': 0.0, 'min_signal_current_a': 1e-12}},
        'implementation': {'operating_point.py': 'synthetic-hash'}, 'api_secret': 'not-exported'})))
    connection.commit()
    connection.close()


class ExpressionTests(unittest.TestCase):
    def test_arithmetic_case_and_suffixes(self):
        self.assertEqual(evaluate_spice_expression('{2*DESVAR_M2}', {'desvar_m2': 47}), 94)
        self.assertAlmostEqual(evaluate_spice_expression('-(W + 2) / 2 + 2**3', {'w': 4}), 5)
        for source, value in [('1u', 1e-6), ('2N', 2e-9), ('3p', 3e-12), ('4m', 0.004),
                              ('5k', 5000), ('6MEG', 6e6), ('1e-3', 0.001), ('.2u', 2e-7)]:
            with self.subTest(source=source):
                self.assertAlmostEqual(evaluate_spice_expression(source), value)
        self.assertEqual(evaluate_spice_expression('1M'), 0.001)

    def test_rejects_executable_or_unbounded_expressions(self):
        for source in ["__import__('os').system('x')", 'open(1)', '[1][0]', 'x.y',
                       '2**1000', '2**(2**16)', '1/0', '1e999', '(-1)**0.5',
                       '{a}', '1xy', 'True', '1//2', '2%1', '{1}{2}', '1 if 2 else 3']:
            with self.subTest(source=source), self.assertRaises(AdapterError):
                evaluate_spice_expression(source)
        with self.assertRaises(AdapterError):
            evaluate_spice_expression('1+' * 200 + '1')
        with self.assertRaises(AdapterError):
            evaluate_spice_expression('1' * 4097)

    def test_rejects_duplicate_parameters_and_nonfinite_names(self):
        with self.assertRaises(AdapterError):
            evaluate_spice_expression('a', {'a': 1, 'A': 2})
        with self.assertRaises(AdapterError):
            evaluate_spice_expression('a', {'a': float('nan')})


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)

    def change_sql(self, sql, values=()):
        with sqlite3.connect(self.root / 'sampling_history.sqlite3') as c:
            c.execute(sql, values)

    def change_netlist(self, old, new):
        path = self.root / 'Tiny.sp'
        path.write_text(path.read_text().replace(old, new))

    def test_complete_import_retains_failed_samples_and_missing_metrics(self):
        dataset = adapt_analog_agent(self.root)
        validate_dataset(dataset)
        self.assertEqual(len(dataset['samples']), 3)
        self.assertEqual(len(dataset['samples'][0]['devices']), 6)
        self.assertEqual(dataset['feature_spec']['op_names'], OP_NAMES)
        self.assertEqual(len(OP_NAMES), 44)
        self.assertIsNone(dataset['samples'][0]['performance']['UGF_Hz'])
        self.assertEqual(dataset['samples'][2]['performance'], dict.fromkeys(['Gain_dB', 'UGF_Hz', 'Power_uW']))
        self.assertEqual(dataset['samples'][0]['design_id'], 'design-1')
        self.assertEqual(dataset['samples'][0]['sample_id'], 'Tiny:1')
        self.assertEqual(dataset['samples'][0]['raw_design_parameters']['width'], 2)
        self.assertEqual(dataset['provenance']['source_counts']['device_op_rows'], 5)
        json.dumps(dataset, allow_nan=False)

    def test_uses_sample_parameters_not_illustrative_defaults(self):
        dataset = adapt_analog_agent(self.root)
        first, second = dataset['samples'][:2]
        nmos = first['devices'][1]
        self.assertEqual(nmos['parameters'], {'W_m': 2e-6, 'L_m': 0.5e-6, 'm': 6})
        self.assertEqual(second['devices'][1]['parameters']['W_m'], 4e-6)
        self.assertEqual(first['devices'][0]['parameters'], {'dc_current_A': 1e-6})
        self.assertEqual(first['devices'][3]['parameters'], {'dc_voltage_V': 0.7})
        self.assertEqual(first['devices'][4]['parameters'], {'R_ohm': 10000})
        self.assertEqual(first['devices'][5]['parameters'], {'C_F': 1e-12})
        self.assertEqual(nmos['parameter_expressions']['m'], '{2 * mult}')

    def test_raw_signed_op_and_non_mos_targets(self):
        sample = adapt_analog_agent(self.root)['samples'][0]
        pmos = sample['devices'][2]
        self.assertEqual(pmos['op']['vgs_v'], -0.7)
        self.assertEqual(pmos['op']['cgs_f'], -1e-15)
        self.assertEqual(pmos['op']['w_um'], 2)
        self.assertTrue(pmos['op_target'])
        for device in [sample['devices'][i] for i in (0, 3, 4, 5)]:
            self.assertFalse(device['op_target'])
            self.assertEqual(device['op'], {})
            self.assertEqual(device['op_status'], 'not_supported')
        self.assertEqual(pmos['terminals'], {'d': 'OUT', 'g': 'IN', 's': 'VDD', 'b': 'VDD'})
        self.assertEqual(sample['devices'][0]['terminals'], {'p': 'OUT', 'n': 'VSS'})
        self.assertEqual(sample['ports'], {'IN': 'IN', 'OUT': 'OUT', 'VDD': 'VDD', 'VSS': 'VSS'})

    def test_missing_op_rows_remain_maskable_nulls(self):
        sample = adapt_analog_agent(self.root)['samples'][2]
        self.assertEqual(sample['devices'][2]['op_status'], 'missing')
        self.assertTrue(all(v is None for v in sample['devices'][2]['op'].values()))
        self.assertEqual(sample['devices'][1]['op_status'], 'partial')
        self.assertEqual(sample['op_status'], 'partial')

    def test_readonly_hash_provenance_no_file_mutation(self):
        db = self.root / 'sampling_history.sqlite3'
        before = hashlib.sha256(db.read_bytes()).hexdigest()
        files_before = set(self.root.iterdir())
        dataset = adapt_analog_agent(self.root)
        self.assertEqual(before, hashlib.sha256(db.read_bytes()).hexdigest())
        self.assertEqual(files_before, set(self.root.iterdir()))
        self.assertEqual(dataset['provenance']['source_hashes_sha256'][db.name], before)
        self.assertEqual(dataset['provenance']['sqlite_open_mode'], 'read-only')
        self.assertNotIn('api_secret', json.dumps(dataset))
        self.assertNotIn('/private/not/read', json.dumps(dataset))
        self.assertFalse(dataset['provenance']['includes'][1]['parameter_file_read'])

    def test_state_metadata_is_not_inferred_from_absent_reasons(self):
        dataset = adapt_analog_agent(self.root)
        observation = dataset['samples'][0]['source_observation']
        self.assertEqual(observation['reasons'], ['XN:saturation'])
        self.assertEqual(observation['checked_devices'], ['XN'])
        self.assertTrue(observation['target_only'])
        self.assertNotIn('op_state', dataset['samples'][0]['devices'][2])
        self.assertEqual(dataset['provenance']['runs'][0]['constraints']['min_signal_current_a'], 1e-12)

    def test_join_uses_sample_id_and_full_device_name_not_order(self):
        self.change_sql("UPDATE dc_device_operating_points SET device_name=lower(device_name)")
        dataset = adapt_analog_agent(self.root)
        self.assertEqual(dataset['samples'][1]['devices'][1]['op']['w_um'], 4)
        self.assertEqual(dataset['samples'][0]['devices'][2]['op_source']['device_name'], 'xp')

    def test_unknown_device_model_is_rejected(self):
        self.change_netlist(NMOS, 'unrecognized_model')
        with self.assertRaisesRegex(AdapterError, 'unknown MOS model'):
            adapt_analog_agent(self.root)

    def test_unknown_subcircuit_or_primitive_is_rejected(self):
        for extra in ['XOTHER IN OUT magic_subckt', 'D1 IN OUT diode', 'B1 IN OUT v=1']:
            self.change_netlist('.ends DUT', extra + '\n.ends DUT')
            with self.subTest(extra=extra), self.assertRaises(AdapterError):
                adapt_analog_agent(self.root)
            self.change_netlist(extra + '\n', '')

    def test_model_mismatch_or_orphan_op_rejected(self):
        self.change_sql("UPDATE dc_device_operating_points SET model_name='different' WHERE sample_id=1 AND device_name='XN'")
        with self.assertRaisesRegex(AdapterError, 'model/type mismatch'):
            adapt_analog_agent(self.root)
        self.change_sql("UPDATE dc_device_operating_points SET model_name=? WHERE sample_id=1 AND device_name='XN'", (NMOS,))
        self.change_sql("UPDATE dc_device_operating_points SET sample_id=999 WHERE sample_id=1 AND device_name='XN'")
        with self.assertRaisesRegex(AdapterError, 'Orphan OP'):
            adapt_analog_agent(self.root)

    def test_duplicate_and_unknown_op_device_rejected(self):
        self.change_sql("UPDATE dc_device_operating_points SET device_name='XN' WHERE sample_id=1 AND device_name='XP'")
        with self.assertRaises(AdapterError):
            adapt_analog_agent(self.root)
        self.change_sql("UPDATE dc_device_operating_points SET device_name='NO_SUCH_DEVICE' WHERE sample_id=1")
        with self.assertRaisesRegex(AdapterError, 'absent from DUT'):
            adapt_analog_agent(self.root)

    def test_wrong_vector_order_length_is_not_guessed(self):
        self.change_sql("UPDATE samples SET design_values_json='[1,2]' WHERE sample_id=1")
        with self.assertRaisesRegex(AdapterError, 'expected 4 values'):
            adapt_analog_agent(self.root)

    def test_geometry_mismatch_strict_or_reported(self):
        self.change_sql("UPDATE dc_device_operating_points SET w_um=999 WHERE sample_id=1 AND device_name='XN'")
        with self.assertRaisesRegex(AdapterError, 'disagrees'):
            adapt_analog_agent(self.root)
        dataset = adapt_analog_agent(self.root, strict=False)
        self.assertEqual(len(dataset['provenance']['warnings']), 1)
        self.assertEqual(dataset['samples'][0]['devices'][1]['op']['w_um'], 999)
        self.assertEqual(dataset['samples'][0]['devices'][1]['parameters']['W_m'], 2e-6)

    def test_unknown_directives_and_transient_sources_rejected(self):
        self.change_netlist('DC {IBIAS_A}', 'PULSE(0 1 1n)')
        with self.assertRaises(AdapterError):
            adapt_analog_agent(self.root)
        self.change_netlist('PULSE(0 1 1n)', 'DC {IBIAS_A}')
        self.change_netlist('.ends DUT', '.control\nshell dangerous\n.endc\n.ends DUT')
        with self.assertRaisesRegex(AdapterError, 'unsupported directive'):
            adapt_analog_agent(self.root)

    def test_parameters_file_cannot_execute_or_include(self):
        (self.root / 'Tiny_params.sp').write_text('.include "outside.sp"\n')
        with self.assertRaisesRegex(AdapterError, 'only .param'):
            adapt_analog_agent(self.root)

    def test_cyclic_parameter_definitions_rejected(self):
        (self.root / 'Tiny_params.sp').write_text('.param a={b} b={a}\n')
        with self.assertRaisesRegex(AdapterError, 'Cyclic or unresolved'):
            adapt_analog_agent(self.root)

    def test_path_and_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            external = Path(other) / 'external.sp'
            external.write_text(NETLIST)
            with self.assertRaisesRegex(AdapterError, 'escapes'):
                parse_spice_netlist(external, root=self.root)
            with self.assertRaisesRegex(AdapterError, 'inside'):
                parse_spice_netlist(self.root / 'Tiny.sp', parameter_file=external)
            (self.root / 'Tiny_params.sp').unlink()
            (self.root / 'Tiny_params.sp').symlink_to(external)
            with self.assertRaisesRegex(AdapterError, 'inside'):
                adapt_analog_agent(self.root)

    def test_views_cannot_substitute_for_source_tables(self):
        self.change_sql('DROP TABLE dc_device_operating_points')
        self.change_sql('CREATE VIEW dc_device_operating_points AS SELECT 1 AS sample_id')
        with self.assertRaisesRegex(AdapterError, 'ordinary SQLite tables'):
            adapt_analog_agent(self.root)

    def test_duplicate_json_keys_rejected(self):
        self.change_sql('UPDATE samples SET observation_json=? WHERE sample_id=1', ('{"reasons":[],"reasons":["bad"]}',))
        with self.assertRaisesRegex(AdapterError, 'duplicate JSON'):
            adapt_analog_agent(self.root)

    def test_m_primitive_and_continuation(self):
        self.change_netlist('XN OUT IN VSS VSS ' + NMOS, 'MN OUT IN VSS VSS custom_n')
        self.change_netlist('.subckt DUT', '.model custom_n NMOS (level=1)\n.subckt DUT')
        self.change_netlist('L={length} m={2 * mult}', '\n+ L={length} m={2 * mult}')
        self.change_sql("UPDATE dc_device_operating_points SET device_name='MN',model_name='custom_n' WHERE device_name='XN'")
        dataset = adapt_analog_agent(self.root)
        self.assertEqual(dataset['samples'][0]['devices'][1]['device_id'], 'MN')
        self.assertEqual(dataset['samples'][0]['devices'][1]['parameters']['m'], 6)

    def test_nonfinite_raw_values_become_json_null(self):
        self.change_sql('UPDATE dc_device_operating_points SET gm_s=? WHERE sample_id=1', (math.inf,))
        dataset = adapt_analog_agent(self.root)
        self.assertIsNone(dataset['samples'][0]['devices'][1]['op']['gm_s'])
        json.dumps(dataset, allow_nan=False)

    def test_inspection(self):
        summary = inspect_analog_agent(self.root)
        self.assertEqual(summary['circuit_ids'], ['Tiny'])
        self.assertEqual(summary['samples'], 3)
        self.assertEqual(summary['missing_metric_values'], 4)

    def test_combine_stable_unions_and_duplicate_identity_check(self):
        one = adapt_analog_agent(self.root)
        with tempfile.TemporaryDirectory() as other:
            fixture(Path(other), 'Tiny2')
            two = adapt_analog_agent(other)
        combined = combine_datasets([one, two])
        self.assertEqual(len(combined['samples']), 6)
        self.assertEqual(combined['feature_spec']['op_names'], OP_NAMES)
        with self.assertRaisesRegex(ValueError, 'duplicate sample_id'):
            combine_datasets([one, one])
        two['feature_spec']['op_units']['id_a'] = 'mA'
        with self.assertRaisesRegex(AdapterError, 'Incompatible op units'):
            combine_datasets([one, two])


    def test_combined_source_ids_are_stable_and_inputs_are_not_mutated(self):
        one = adapt_analog_agent(self.root)
        before = copy.deepcopy(one)
        with tempfile.TemporaryDirectory() as other:
            fixture(Path(other), 'Tiny2')
            two = adapt_analog_agent(other)
        # Missing run metadata in A must remain scoped to A, not borrowed from B.
        one['provenance']['runs'] = []
        combined = combine_datasets([one, two])
        sources = {leaf['source_id']: leaf for leaf in combined['provenance']['sources']}
        self.assertEqual(len(sources), 2)
        for sample in combined['samples']:
            source = sources[sample['source_id']]
            self.assertIn(sample['circuit_id'], source['circuit_ids'])
            self.assertEqual(source['circuit_ids'], [sample['circuit_id']])
            self.assertEqual(bool(source['runs']), sample['circuit_id'] == 'Tiny2')
        reversed_combined = combine_datasets([two, one])
        self.assertEqual({s['sample_id']: s['source_id'] for s in combined['samples']},
                         {s['sample_id']: s['source_id'] for s in reversed_combined['samples']})
        self.assertNotIn('source_id', one['samples'][0])
        self.assertNotIn('source_id', one['provenance'])
        self.assertEqual(before, adapt_analog_agent(self.root))

    def test_nested_combination_preserves_scopes_and_deduplicates_leaves(self):
        one = adapt_analog_agent(self.root)
        combined = combine_datasets([one])
        self.assertEqual(combined, combine_datasets([combined]))
        left, right = copy.deepcopy(combined), copy.deepcopy(combined)
        left['samples'] = left['samples'][:1]
        right['samples'] = right['samples'][1:]
        recombined = combine_datasets([left, right])
        self.assertEqual(recombined, combined)
        self.assertEqual(len(recombined['provenance']['sources']), 1)

    def test_legacy_unscoped_combined_sources_are_not_guessed(self):
        one = adapt_analog_agent(self.root)
        legacy = copy.deepcopy(one)
        legacy['metadata']['adapter'] = 'analog_agent_combined_v1'
        legacy['provenance'] = {'sources': [legacy['provenance']]}
        with self.assertRaisesRegex(AdapterError, 'source_id'):
            combine_datasets([legacy])
        legacy['provenance'] = {}
        with self.assertRaisesRegex(AdapterError, 'missing scoped'):
            combine_datasets([legacy])

    def test_combined_scope_rejects_unknown_ids_and_wrong_circuits(self):
        combined = combine_datasets([adapt_analog_agent(self.root)])
        combined['samples'][0]['source_id'] = 'not-a-source'
        with self.assertRaisesRegex(AdapterError, 'no matching scoped'):
            combine_datasets([combined])
        combined = combine_datasets([adapt_analog_agent(self.root)])
        combined['provenance']['sources'][0]['circuit_ids'] = ['NotTiny']
        with self.assertRaisesRegex(AdapterError, 'outside its source scope'):
            combine_datasets([combined])

    def test_combined_source_collision_rejected(self):
        one = combine_datasets([adapt_analog_agent(self.root)])
        two = copy.deepcopy(one)
        for sample in two['samples']:
            sample['sample_id'] += ':other'
        two['provenance']['sources'][0]['runs'] = []
        with self.assertRaisesRegex(AdapterError, 'Conflicting provenance'):
            combine_datasets([one, two])


if __name__ == '__main__':
    unittest.main()
