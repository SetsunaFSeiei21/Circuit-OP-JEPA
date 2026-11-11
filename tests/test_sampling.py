import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from device_jepa.data import (build_manifest, dataset_fingerprint, load_dataset,
                              synthetic_dataset, verify_manifest)
from device_jepa.sampling import (apply_dataset_sampling, check_sampling_config,
                                  select_sample_ids, validate_sampling_options)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_dataset(30, 17)
        self.options = dict(max_samples_per_circuit=20, sampling_seed=17,
            train_circuits=['synthetic_0', 'synthetic_1', 'synthetic_2', 'synthetic_3'],
            ood_circuits=['synthetic_4'], split_ratios=[.8, .1, .1], split_seed=17,
            finetune_label_fraction=1.)

    def test_cap_applies_per_circuit_before_id_split_and_ood_holdout(self):
        sampled = apply_dataset_sampling(self.data, self.options)
        self.assertEqual(len(sampled['samples']), 100)
        manifest = build_manifest(sampled, config=self.options)
        self.assertEqual([len(manifest[k]) for k in ('train', 'val', 'test', 'ood')], [64, 8, 8, 20])
        self.assertEqual(manifest['sampling'], sampled['sampling'])
        for record in manifest['sampling']['circuits'].values():
            self.assertEqual(record['source_count'], 30)
            self.assertEqual(record['selected_count'], 20)
            self.assertEqual(len(set(record['selected_ids'])), 20)
        self.assertEqual(len(self.data['samples']), 150)
        self.assertNotIn('sampling', self.data)

    def test_order_circuit_membership_rng_labels_do_not_change_selection(self):
        first = apply_dataset_sampling(self.data, self.options)
        shuffled = copy.deepcopy(self.data)
        random.Random(123).shuffle(shuffled['samples'])
        for row in shuffled['samples']:
            row['performance'] = {key: None for key in row['performance']}
        random.seed(829)
        second = apply_dataset_sampling(shuffled, self.options)
        self.assertEqual(first['sampling'], second['sampling'])
        single = dict(self.data, samples=[r for r in self.data['samples'] if r['circuit_id'] == 'synthetic_0'])
        third = apply_dataset_sampling(single, self.options)
        self.assertEqual(third['sampling']['circuits']['synthetic_0'], first['sampling']['circuits']['synthetic_0'])

    def test_seed_changes_subset_cap_increase_is_nested(self):
        ids = [f'x:{i}' for i in range(100)]
        small = select_sample_ids(ids, 'x', 20, 17)
        self.assertNotEqual(small, select_sample_ids(ids, 'x', 20, 29))
        self.assertLessEqual(set(small), set(select_sample_ids(ids, 'x', 40, 17)))

    def test_unlimited_preserves_legacy_identity_and_large_caps_take_all(self):
        before = dataset_fingerprint(self.data)
        for config in (None, {}, dict(max_samples_per_circuit=None, sampling_seed=123)):
            self.assertIs(apply_dataset_sampling(self.data, config), self.data)
            self.assertEqual(dataset_fingerprint(self.data), before)
        for cap in (30, 31, 1000):
            result = apply_dataset_sampling(self.data, dict(self.options, max_samples_per_circuit=cap))
            self.assertEqual(result['samples'], self.data['samples'])

    def test_roundtrip_and_manifest_reconstruct_same_selection(self):
        result = apply_dataset_sampling(self.data, self.options)
        manifest = build_manifest(result, config=self.options)
        with tempfile.TemporaryDirectory() as folder:
            full, selected = Path(folder)/'full.json', Path(folder)/'selected.json'
            full.write_text(json.dumps(self.data))
            selected.write_text(json.dumps(result))
            original = full.read_bytes()
            with patch('device_jepa.sampling.select_sample_ids', side_effect=AssertionError('redrew samples')):
                self.assertEqual(load_dataset(full, sampling_config=manifest['sampling']), result)
            self.assertEqual(load_dataset(selected, sampling_config=self.options), result)
            self.assertEqual(load_dataset(selected), result)
            self.assertEqual(full.read_bytes(), original)
        self.assertIs(apply_dataset_sampling(result, self.options), result)
        for updates in ({'max_samples_per_circuit': 19}, {'sampling_seed': 43}, {'max_samples_per_circuit': None}):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'differs from saved sampling'):
                apply_dataset_sampling(result, dict(self.options, **updates))

    def test_saved_ids_cannot_be_changed_or_omitted(self):
        result = apply_dataset_sampling(self.data, self.options)
        manifest = build_manifest(result, config=self.options)
        bad = copy.deepcopy(result)
        bad['sampling']['circuits']['synthetic_0']['selected_ids'][0] = 'unseen'
        with self.assertRaisesRegex(ValueError, 'sampling IDs/counts'):
            apply_dataset_sampling(bad, self.options)
        bad = copy.deepcopy(result)
        del bad['sampling']['sampling_seed']
        with self.assertRaisesRegex(ValueError, 'missing required fields'):
            apply_dataset_sampling(bad, self.options)
        bad_manifest = copy.deepcopy(manifest)
        del bad_manifest['sampling']
        with self.assertRaisesRegex(ValueError, 'manifest sampling'):
            verify_manifest(result, bad_manifest)
        with self.assertRaisesRegex(ValueError, 'Apply per-circuit sampling'):
            build_manifest(self.data, config=self.options)

    def test_grouped_split_still_keeps_duplicate_design_together(self):
        for row in self.data['samples']:
            row['group_id'] = row['circuit_id'] + ':' + str(int(row['sample_id'].rsplit('_', 1)[-1]) // 2)
        sampled = apply_dataset_sampling(self.data, self.options)
        manifest = build_manifest(sampled, config=self.options)
        verify_manifest(sampled, manifest)
        assignments = {sid: split for split in ('train', 'val', 'test', 'ood') for sid in manifest[split]}
        groups = {}
        for row in sampled['samples']:
            groups.setdefault(row['group_id'], set()).add(assignments[row['sample_id']])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_invalid_options_fail_including_bools(self):
        for value in (0, -1, True, False, '20000', 20.5):
            with self.subTest(cap=value), self.assertRaises(ValueError):
                validate_sampling_options(value, 17)
        for value in (-1, True, '17', 1.5, None):
            with self.subTest(seed=value), self.assertRaises(ValueError):
                validate_sampling_options(20, value)
        check_sampling_config({}, None)
        duplicate = copy.deepcopy(self.data)
        duplicate['samples'].append(duplicate['samples'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate sample_id'):
            apply_dataset_sampling(duplicate, self.options)


if __name__ == '__main__':
    unittest.main()
