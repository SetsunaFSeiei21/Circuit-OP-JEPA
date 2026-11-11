"""Raw sampling occurs before payload decoding, in one read-only snapshot."""
import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from device_jepa.adapter import AdapterError, _read_database, adapt_analog_agent, combine_datasets
from device_jepa.sampling import select_sample_ids
from tests.test_adapter import fixture


class RawSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.database = self.root / 'sampling_history.sqlite3'

    def change_sql(self, query, values=()):
        with sqlite3.connect(self.database) as db:
            db.execute(query, values)

    def test_selected_ids_are_recorded_and_raw_source_is_unchanged(self):
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        dataset = adapt_analog_agent(self.root, max_samples_per_circuit=2, sampling_seed=29)
        expected = select_sample_ids(['Tiny:1', 'Tiny:2', 'Tiny:3'], 'Tiny', 2, 29)
        self.assertEqual(sorted(sample['sample_id'] for sample in dataset['samples']), expected)
        self.assertEqual(dataset['sampling'], {
            'schema_version': 1, 'algorithm': 'sha256_rank_v1',
            'max_samples_per_circuit': 2, 'sampling_seed': 29,
            'circuits': {'Tiny': {'source_count': 3, 'selected_count': 2, 'selected_ids': expected}}})
        self.assertEqual(dataset['provenance']['source_counts']['samples'], 3)
        self.assertEqual(dataset['provenance']['source_counts']['selected_samples'], 2)
        self.assertEqual(dataset['provenance']['source_counts']['device_op_rows'], 5)
        self.assertEqual(before, hashlib.sha256(self.database.read_bytes()).hexdigest())
        self.assertEqual(dataset, adapt_analog_agent(self.root, max_samples_per_circuit=2, sampling_seed=29))
        json.dumps(dataset, allow_nan=False)

    def test_equal_or_larger_cap_keeps_every_sample(self):
        complete = adapt_analog_agent(self.root)
        self.assertNotIn('sampling', complete)
        for cap in (3, 60_000):
            with self.subTest(cap=cap):
                selected = adapt_analog_agent(self.root, max_samples_per_circuit=cap)
                self.assertEqual(selected['samples'], complete['samples'])
                self.assertEqual(selected['sampling']['circuits']['Tiny']['selected_count'], 3)
        self.assertEqual(complete, adapt_analog_agent(self.root, max_samples_per_circuit=None, sampling_seed=99))

    def test_filtering_happens_before_sample_and_op_payload_validation(self):
        expected = select_sample_ids(['Tiny:1', 'Tiny:2', 'Tiny:3'], 'Tiny', 1, 17)
        selected_id = int(expected[0].split(':')[-1])
        self.change_sql("UPDATE samples SET design_values_json='invalid JSON' WHERE sample_id != ?", (selected_id,))
        self.change_sql("UPDATE dc_device_operating_points SET model_name='invalid model' WHERE sample_id != ?", (selected_id,))
        selected = adapt_analog_agent(self.root, max_samples_per_circuit=1)
        self.assertEqual([row['sample_id'] for row in selected['samples']], expected)
        with self.assertRaises(AdapterError):
            adapt_analog_agent(self.root)
        self.change_sql("UPDATE samples SET design_values_json='invalid JSON' WHERE sample_id = ?", (selected_id,))
        with self.assertRaisesRegex(AdapterError, 'invalid JSON'):
            adapt_analog_agent(self.root, max_samples_per_circuit=1)

    def test_filtered_reader_returns_only_selected_related_rows(self):
        selected = select_sample_ids(['Tiny:1', 'Tiny:2', 'Tiny:3'], 'Tiny', 2, 17)
        raw_ids = {int(value.split(':')[-1]) for value in selected}
        with _read_database(self.database, max_samples_per_circuit=2) as source:
            for table in ('samples', 'dc_device_operating_points', 'dc_operating_points'):
                self.assertNotIsInstance(source[table], list)
                rows = list(source[table])
                self.assertEqual({row['sample_id'] for row in rows}, raw_ids)
            self.assertEqual(source['table_counts']['samples'], 3)

    def test_sampling_does_not_bypass_table_size_guard(self):
        with self.assertRaisesRegex(AdapterError, 'max_table_rows=4'):
            adapt_analog_agent(self.root, max_samples_per_circuit=1, max_table_rows=4)

    def test_sixty_thousand_source_ids_select_twenty_thousand_without_sql_parameter_limit(self):
        database = self.root / 'large.sqlite3'
        with sqlite3.connect(database) as db:
            db.executescript('''
CREATE TABLE dataset_schema (id INTEGER, circuit_name TEXT);
INSERT INTO dataset_schema VALUES (1, 'Large');
CREATE TABLE samples (sample_id INTEGER PRIMARY KEY);
WITH RECURSIVE ids(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<60000)
INSERT INTO samples SELECT n FROM ids;
CREATE TABLE dc_device_operating_points (
 sample_id INTEGER, device_name TEXT, device_type TEXT, model_name TEXT,
 primitive_path TEXT, id_a REAL);
INSERT INTO dc_device_operating_points
 SELECT sample_id, 'MN', 'NMOS', 'nmos', 'synthetic', 1e-6 FROM samples;
''')
        with _read_database(database, max_samples_per_circuit=20_000) as source:
            samples = [row['sample_id'] for row in source['samples']]
            op_ids = [row['sample_id'] for row in source['dc_device_operating_points']]
            self.assertEqual(source['table_counts']['samples'], 60_000)
            self.assertEqual(len(samples), 20_000)
            self.assertEqual(op_ids, samples)
            self.assertEqual(len(set(samples)), len(samples))
            self.assertGreater(max(samples), 40_000)

    def test_sampling_does_not_hide_duplicate_source_ids_or_orphan_op(self):
        self.change_sql('INSERT INTO samples SELECT * FROM samples WHERE sample_id=1')
        with self.assertRaisesRegex(AdapterError, 'unique sample_id'):
            adapt_analog_agent(self.root, max_samples_per_circuit=1)
        self.change_sql('DELETE FROM samples WHERE rowid=(SELECT MAX(rowid) FROM samples)')
        self.change_sql('UPDATE dc_device_operating_points SET sample_id=99 WHERE sample_id=1')
        with self.assertRaisesRegex(AdapterError, 'Orphan OP sample_id'):
            adapt_analog_agent(self.root, max_samples_per_circuit=1)

    def test_invalid_options_are_rejected(self):
        for cap in (0, -1, True, 2.5, '2'):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                adapt_analog_agent(self.root, max_samples_per_circuit=cap)
        for seed in (True, 1.5, '17'):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                adapt_analog_agent(self.root, max_samples_per_circuit=1, sampling_seed=seed)

    def test_combination_preserves_every_circuit_selection(self):
        one = adapt_analog_agent(self.root, max_samples_per_circuit=2)
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp), circuit='Another')
            two = adapt_analog_agent(tmp, max_samples_per_circuit=2)
        combined = combine_datasets([one, two])
        self.assertEqual(len(combined['samples']), 4)
        self.assertEqual(set(combined['sampling']['circuits']), {'Tiny', 'Another'})
        for item in (one, two):
            for circuit, record in item['sampling']['circuits'].items():
                self.assertEqual(combined['sampling']['circuits'][circuit], record)
        with self.assertRaisesRegex(AdapterError, 'sampled and unsampled'):
            combine_datasets([one, adapt_analog_agent(self.root)])
        modified = copy.deepcopy(two)
        modified['sampling']['sampling_seed'] = 43
        with self.assertRaisesRegex(AdapterError, 'different sampling options'):
            combine_datasets([one, modified])


if __name__ == '__main__':
    unittest.main()
