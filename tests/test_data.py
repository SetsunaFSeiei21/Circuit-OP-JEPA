"""Standard-library tests for schema, grouped splits and train-only scaling."""

import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from device_jepa.data import (
    TERMINAL_NAMES,
    build_manifest,
    dataset_fingerprint,
    design_fingerprint,
    fit_preprocessor,
    load_dataset,
    samples_for_split,
    synthetic_dataset,
    validate_dataset,
    verify_manifest,
)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = synthetic_dataset(n_per_circuit=8)

    def test_synthetic_schema_and_repeatability(self):
        self.assertEqual(self.dataset, synthetic_dataset(n_per_circuit=8))
        self.assertNotEqual(self.dataset, synthetic_dataset(n_per_circuit=8, seed=18))
        self.assertTrue(self.dataset["metadata"]["synthetic"])
        self.assertEqual(len(self.dataset["samples"]), 40)
        self.assertEqual(len({row["circuit_id"] for row in self.dataset["samples"]}), 5)
        self.assertEqual({len(row["devices"]) for row in self.dataset["samples"]}, {4, 5, 6})
        self.assertEqual(self.dataset["feature_spec"]["terminal_names"], TERMINAL_NAMES)
        self.assertTrue(any(row["performance"]["gain"] is None for row in self.dataset["samples"]))
        self.assertTrue(any(device["op"]["vds"] is None for row in self.dataset["samples"] for device in row["devices"]))

    def test_load_round_trip_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            path.write_text(json.dumps(self.dataset), encoding="utf-8")
            loaded = load_dataset(path)
        self.assertEqual(loaded, self.dataset)
        self.assertIn("ports", loaded["samples"][0])
        self.assertIn("op_status", loaded["samples"][0])
        self.assertIn("b", loaded["samples"][0]["devices"][0]["terminals"])

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                load_dataset(path)

    def test_bad_schema_rejected(self):
        for value in (None, [], {}, {"schema_version": True}, {"schema_version": 2}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_dataset(value)

    def test_nonfinite_and_boolean_measurements_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "1.0"):
            bad = copy.deepcopy(self.dataset)
            bad["samples"][0]["devices"][0]["op"]["id"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "null"):
                validate_dataset(bad)

    def test_nonfinite_context_rejected(self):
        self.dataset["samples"][0]["context"]["nested"] = {"array": [float("nan")]}
        with self.assertRaisesRegex(ValueError, "null"):
            validate_dataset(self.dataset)

    def test_unknown_features_rejected(self):
        self.dataset["samples"][0]["devices"][0]["op"]["surprise"] = 1.0
        with self.assertRaisesRegex(ValueError, "undeclared features"):
            validate_dataset(self.dataset)

    def test_missing_observations_can_be_null_or_omitted(self):
        self.dataset["samples"][0]["devices"][0]["parameters"].pop("W")
        self.dataset["samples"][0]["devices"][0]["op"]["id"] = None
        self.assertIs(validate_dataset(self.dataset), self.dataset)

    def test_duplicate_ids_rejected(self):
        duplicate = copy.deepcopy(self.dataset["samples"][0])
        self.dataset["samples"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            validate_dataset(self.dataset)
        self.dataset["samples"].pop()
        devices = self.dataset["samples"][0]["devices"]
        devices.append(copy.deepcopy(devices[0]))
        with self.assertRaisesRegex(ValueError, "duplicate device_id"):
            validate_dataset(self.dataset)

    def test_feature_order_and_names_must_be_valid(self):
        for key, value in (("parameter_names", ["W", "W"]), ("op_names", []),
                           ("metric_names", [1]), ("terminal_names", list(reversed(TERMINAL_NAMES)))):
            bad = copy.deepcopy(self.dataset)
            bad["feature_spec"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_dataset(bad)

    def test_terminals_ports_and_status_validation(self):
        mutations = (
            lambda row: row["devices"][0]["terminals"].update({"drain": "x"}),
            lambda row: row["devices"][0]["terminals"].update({"d": ""}),
            lambda row: row.update({"ports": {"out": 10}}),
            lambda row: row.update({"op_status": "unconverged"}),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.dataset)
            mutate(bad["samples"][0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_dataset(bad)

    def test_varying_context_is_rejected_without_context_model_features(self):
        self.dataset["samples"][0]["context"]["temperature_c"] = 95.0
        with self.assertRaisesRegex(ValueError, "context must be fixed"):
            validate_dataset(self.dataset)

    def test_synthetic_minimum_and_seed_validation(self):
        for count in (0, 2, 3.1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                synthetic_dataset(count)
        with self.assertRaises(ValueError):
            synthetic_dataset(seed=True)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.dataset = synthetic_dataset(n_per_circuit=12)
        self.manifest = build_manifest(self.dataset, "synthetic_4")

    def assignment(self, manifest):
        return {sid: split for split in ("train", "val", "test", "ood") for sid in manifest[split]}

    def add_copy(self, row, sample_id, design_id=None, **extra):
        duplicate = copy.deepcopy(row)
        duplicate["sample_id"] = sample_id
        if design_id is not None:
            duplicate["design_id"] = design_id
        duplicate.update(extra)
        self.dataset["samples"].append(duplicate)
        return duplicate

    def test_exact_four_id_one_ood_and_full_partition(self):
        verify_manifest(self.dataset, self.manifest)
        assignment = self.assignment(self.manifest)
        self.assertEqual(len(assignment), len(self.dataset["samples"]))
        self.assertEqual(len(self.manifest["id_circuits"]), 4)
        for split in ("train", "val", "test"):
            rows = samples_for_split(self.dataset, self.manifest, split)
            self.assertEqual({row["circuit_id"] for row in rows}, set(self.manifest["id_circuits"]))
        self.assertEqual({row["circuit_id"] for row in samples_for_split(self.dataset, self.manifest, "ood")}, {"synthetic_4"})

    def test_reordering_is_deterministic(self):
        shuffled = copy.deepcopy(self.dataset)
        rng = random.Random(43)
        rng.shuffle(shuffled["samples"])
        for row in shuffled["samples"]:
            rng.shuffle(row["devices"])
        self.assertEqual(build_manifest(shuffled, "synthetic_4"), self.manifest)
        self.assertEqual(dataset_fingerprint(shuffled), dataset_fingerprint(self.dataset))

    def test_seed_changes_assignments(self):
        other = build_manifest(self.dataset, "synthetic_4", seed=22)
        self.assertNotEqual(other["train"], self.manifest["train"])
        self.assertEqual(other["ood"], self.manifest["ood"])

    def test_same_design_conditions_and_transitive_groups_stay_together(self):
        first, second = self.dataset["samples"][:2]
        first["group_id"] = second["group_id"] = "repeat_family"
        duplicate = self.add_copy(first, "repeat_measurement")
        duplicate["devices"][0]["parameters"]["W"] *= 1.1
        third = self.add_copy(duplicate, "alternate_design_id", "different_name")
        # A duplicate-fingerprint link and a group_id link connect all records.
        third.pop("group_id")
        assignments = self.assignment(build_manifest(self.dataset, "synthetic_4"))
        self.assertEqual(len({assignments[row["sample_id"]] for row in (first, second, duplicate, third)}), 1)

    def test_duplicate_fingerprint_ignores_labels_and_identifiers(self):
        original = self.dataset["samples"][0]
        duplicate = self.add_copy(original, "copy_of_design", "another_design")
        duplicate["performance"] = {"gain": 123.0, "power": 456.0}
        duplicate["op_status"] = "failed"
        duplicate["devices"].reverse()
        for i, device in enumerate(duplicate["devices"]):
            device["device_id"] = f"renamed_{i}"
            device["op"] = {"id": None, "gm": None, "vds": None}
        spec = self.dataset["feature_spec"]
        self.assertEqual(design_fingerprint(original, spec), design_fingerprint(duplicate, spec))
        assignments = self.assignment(build_manifest(self.dataset, "synthetic_4"))
        self.assertEqual(assignments[original["sample_id"]], assignments[duplicate["sample_id"]])

    def test_ports_and_context_are_part_of_fingerprint(self):
        original = self.dataset["samples"][0]
        spec = self.dataset["feature_spec"]
        for key, value in (("ports", {"output": "n3"}), ("context", {"temperature_c": -30})):
            changed = copy.deepcopy(original)
            changed[key] = value
            self.assertNotEqual(design_fingerprint(original, spec), design_fingerprint(changed, spec))

    def test_cross_circuit_duplicates_rejected_even_if_both_id(self):
        original = self.dataset["samples"][0]
        for circuit in ("synthetic_1", "synthetic_4"):
            duplicate = self.add_copy(original, f"duplicate_{circuit}", circuit_id=circuit)
            with self.subTest(circuit=circuit), self.assertRaisesRegex(ValueError, "cross circuit"):
                build_manifest(self.dataset, "synthetic_4")
            self.dataset["samples"].remove(duplicate)

    def test_exactly_five_circuits_required(self):
        self.dataset["samples"] = [row for row in self.dataset["samples"] if row["circuit_id"] != "synthetic_3"]
        with self.assertRaisesRegex(ValueError, "exactly five"):
            build_manifest(self.dataset, "synthetic_4")

    def test_minimum_independent_groups_and_fraction_edges(self):
        minimal = synthetic_dataset(3)
        manifest = build_manifest(minimal, "synthetic_4", val_fraction=0.8, test_fraction=0.1)
        self.assertEqual([len(manifest[s]) for s in ("train", "val", "test", "ood")], [4, 4, 4, 3])
        for row in minimal["samples"]:
            if row["circuit_id"] == "synthetic_0":
                row["design_id"] = "one_group"
        with self.assertRaisesRegex(ValueError, "at least 3 independent"):
            build_manifest(minimal, "synthetic_4")

    def test_bad_fractions_seed_and_circuit(self):
        for kwargs in ({"val_fraction": 0}, {"test_fraction": -1}, {"val_fraction": float("nan")},
                       {"test_fraction": True}, {"val_fraction": 0.6, "test_fraction": 0.4}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_manifest(self.dataset, "synthetic_4", **kwargs)
        with self.assertRaises(ValueError):
            build_manifest(self.dataset, "missing_circuit")

    def test_stale_fingerprint_rejected(self):
        self.dataset["samples"][0]["performance"]["power"] *= 2
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            verify_manifest(self.dataset, self.manifest)

    def test_unknown_duplicate_omitted_and_wrong_circuit_rejected(self):
        mutations = (
            lambda m: m["train"].append("unknown_sample"),
            lambda m: m["train"].append(m["train"][0]),
            lambda m: m["train"].pop(),
            lambda m: m["train"].append(m["ood"].pop()),
            lambda m: m.update({"id_circuits": list(reversed(m["id_circuits"]))}),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.manifest)
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                verify_manifest(self.dataset, bad)

    def test_group_split_tampering_rejected(self):
        original = self.dataset["samples"][0]
        self.add_copy(original, "repeat_heldout")
        manifest = build_manifest(self.dataset, "synthetic_4")
        assignments = self.assignment(manifest)
        source = assignments["repeat_heldout"]
        target = "val" if source != "val" else "train"
        manifest[source].remove("repeat_heldout")
        manifest[target].append("repeat_heldout")
        with self.assertRaisesRegex(ValueError, "design group"):
            verify_manifest(self.dataset, manifest)

    def test_split_accessor_uses_manifest_order(self):
        self.manifest["train"].reverse()
        rows = samples_for_split(self.dataset, self.manifest, "train")
        self.assertEqual([row["sample_id"] for row in rows], self.manifest["train"])
        with self.assertRaises(ValueError):
            samples_for_split(self.dataset, self.manifest, "validation")


class PreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.dataset = synthetic_dataset(n_per_circuit=12)
        self.manifest = build_manifest(self.dataset, "synthetic_4")

    def refresh_integrity(self):
        self.manifest["dataset_fingerprint"] = dataset_fingerprint(self.dataset)

    def test_train_only_vocabulary_and_statistics(self):
        baseline = fit_preprocessor(self.dataset, self.manifest, include_performance=True)
        train_ids = set(self.manifest["train"])
        for row_index, row in enumerate(self.dataset["samples"]):
            if row["sample_id"] in train_ids:
                continue
            row["performance"] = {"gain": 1e12, "power": 1e12}
            for device in row["devices"]:
                device["device_type"] = "HELD_OUT_ONLY_TYPE"
                device["parameters"] = {"W": 1e10 + row_index, "L": 1e10, "R": 1e10}
                device["op"] = {"id": 1e10, "gm": 1e10, "vds": 1e10}
        self.refresh_integrity()
        actual = fit_preprocessor(self.dataset, self.manifest, include_performance=True)
        self.assertEqual(actual, baseline)
        self.assertNotIn("HELD_OUT_ONLY_TYPE", actual["device_vocab"])
        self.assertEqual(actual["device_vocab"]["<PAD>"], 0)
        self.assertEqual(actual["device_vocab"]["<UNK>"], 1)

    def test_stage_one_has_no_performance_statistics_or_label_dependence(self):
        baseline = fit_preprocessor(self.dataset, self.manifest)
        self.assertNotIn("performance", baseline)
        for row in self.dataset["samples"]:
            row["performance"] = {"gain": 1e100, "power": None}
        self.refresh_integrity()
        self.assertEqual(fit_preprocessor(self.dataset, self.manifest), baseline)
        self.assertIn("performance", fit_preprocessor(self.dataset, self.manifest, include_performance=True))

    def test_exact_population_statistics_and_missing_values(self):
        train_ids = set(self.manifest["train"])
        observed = []
        counter = 0
        for row in self.dataset["samples"]:
            if row["sample_id"] not in train_ids:
                continue
            for device in row["devices"]:
                value = None if counter % 3 == 0 else counter * 1e-12
                device["op"]["id"] = value
                if value is not None:
                    observed.append(value)
                counter += 1
        self.refresh_integrity()
        pp = fit_preprocessor(self.dataset, self.manifest)
        expected_mean = math.fsum(observed) / len(observed)
        expected_std = math.sqrt(math.fsum((value - expected_mean) ** 2 for value in observed) / len(observed))
        self.assertAlmostEqual(pp["op"]["mean"][0] / expected_mean, 1.0, places=14)
        self.assertAlmostEqual(pp["op"]["std"][0] / expected_std, 1.0, places=14)
        self.assertEqual(pp["op"]["count"][0], len(observed))
        self.assertLess(pp["op"]["std"][0], 1e-9)

    def test_absent_and_constant_features(self):
        for row in self.dataset["samples"]:
            for device in row["devices"]:
                device["op"]["id"] = None
                device["op"]["gm"] = 3e-15
        self.refresh_integrity()
        pp = fit_preprocessor(self.dataset, self.manifest)
        self.assertEqual(pp["op"]["mean"][0], 0.0)
        self.assertEqual(pp["op"]["std"][0], 1.0)
        self.assertEqual(pp["op"]["count"][0], 0)
        self.assertEqual(pp["op"]["std"][1], 1.0)
        self.assertAlmostEqual(pp["op"]["mean"][1] / 3e-15, 1.0)

    def test_tiny_units_do_not_underflow_variance(self):
        for row in self.dataset["samples"]:
            for i, device in enumerate(row["devices"]):
                device["op"]["id"] = (i + 1) * 1e-250
        self.refresh_integrity()
        pp = fit_preprocessor(self.dataset, self.manifest)
        self.assertGreater(pp["op"]["std"][0], 0)
        self.assertLess(pp["op"]["std"][0], 1e-248)

    def test_preprocessor_serializes_without_nan_and_does_not_alias_spec(self):
        pp = fit_preprocessor(self.dataset, self.manifest)
        json.dumps(pp, allow_nan=False)
        self.assertEqual(pp["fitted_sample_ids"], self.manifest["train"])
        pp["feature_spec"]["parameter_names"].append("not_in_dataset")
        self.assertNotIn("not_in_dataset", self.dataset["feature_spec"]["parameter_names"])


if __name__ == "__main__":
    unittest.main()
