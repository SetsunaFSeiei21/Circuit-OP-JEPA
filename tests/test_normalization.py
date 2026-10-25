"""v2 grouped partitions, train-only physics transforms and MOS eligibility."""
import copy
import json
import math
import unittest

from device_jepa.data import (
    build_manifest, dataset_fingerprint, fit_preprocessor, input_op_fingerprint,
    inverse_transform_value, is_op_target, partition_fingerprint, samples_for_split,
    synthetic_dataset, transform_value, validate_dataset, verify_manifest,
)


class ExplicitSplitTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_dataset(20)
        self.config = {"train_circuits": ["synthetic_0", "synthetic_1", "synthetic_2"],
                       "ood_circuits": ["synthetic_3", "synthetic_4"],
                       "split_ratios": [0.8, 0.1, 0.1], "split_seed": 51,
                       "finetune_label_fraction": 0.25}

    def test_explicit_multiple_ood_and_per_circuit_ratios(self):
        manifest = build_manifest(self.data, config=self.config)
        self.assertIsNone(manifest["ood_circuit"])
        self.assertEqual(manifest["ood_circuits"], ["synthetic_3", "synthetic_4"])
        for circuit in self.config["train_circuits"]:
            for split, count in (("train", 16), ("val", 2), ("test", 2), ("labeled_train", 4)):
                self.assertEqual(sum(row["circuit_id"] == circuit for row in samples_for_split(self.data, manifest, split)), count)
        self.assertEqual(len(manifest["ood"]), 40)
        self.assertTrue(set(manifest["labeled_train"]) < set(manifest["train"]))
        self.assertEqual(manifest, build_manifest(self.data, config=self.config))

    def test_single_circuit_requires_explicit_empty_ood(self):
        self.data["samples"] = [row for row in self.data["samples"] if row["circuit_id"] == "synthetic_0"]
        with self.assertRaises(ValueError):
            build_manifest(self.data)
        with self.assertRaises(ValueError):
            build_manifest(self.data, "synthetic_0")
        manifest = build_manifest(self.data, config={"train_circuits": ["synthetic_0"], "ood_circuits": []})
        self.assertEqual(manifest["ood"], [])
        self.assertEqual(manifest["ood_circuits"], [])
        self.assertEqual(manifest["labeled_train"], manifest["train"])
        verify_manifest(self.data, manifest)

    def test_stable_design_key_grouping_and_grouped_labels(self):
        first = self.data["samples"][0]
        duplicate = copy.deepcopy(first)
        first["design_key"] = duplicate["design_key"] = "stable_source_design"
        duplicate["sample_id"] = "new_measurement"
        duplicate["design_id"] = "different_export_name"
        duplicate["devices"][0]["parameters"]["W"] *= 1.1
        self.data["samples"].append(duplicate)
        manifest = build_manifest(self.data, config=self.config)
        for split in ("train", "val", "test", "labeled_train"):
            self.assertEqual(first["sample_id"] in manifest[split], duplicate["sample_id"] in manifest[split])
        for row in self.data["samples"]:
            row["design_key"] = row.get("design_key", row.pop("design_id", "unused"))
            row.pop("design_id", None)
        validate_dataset(self.data)
        build_manifest(self.data, config=self.config)

    def test_additions_fail_existing_manifest_without_resplit(self):
        manifest = build_manifest(self.data, config=self.config)
        old = copy.deepcopy(manifest)
        extra = copy.deepcopy(self.data["samples"][0])
        extra["sample_id"] = "late_measurement"
        self.data["samples"].append(extra)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            verify_manifest(self.data, manifest)
        self.assertEqual(manifest, old)
        new = build_manifest(self.data, config=self.config)
        self.assertNotEqual(new["dataset_fingerprint"], old["dataset_fingerprint"])

    def test_label_subset_tampering_rejected_and_in_provenance(self):
        manifest = build_manifest(self.data, config=self.config)
        for sid in (manifest["val"][0], manifest["ood"][0], manifest["labeled_train"][0]):
            changed = copy.deepcopy(manifest)
            changed["labeled_train"].append(sid)
            with self.assertRaisesRegex(ValueError, "labeled_train"):
                verify_manifest(self.data, changed)
        changed = copy.deepcopy(manifest)
        changed["labeled_train"].pop()
        self.assertNotEqual(partition_fingerprint(manifest), partition_fingerprint(changed))
        with self.assertRaisesRegex(ValueError, "labeled_train"):
            verify_manifest(self.data, changed)

    def test_whole_group_split_tampering_rejected(self):
        manifest = build_manifest(self.data, config=self.config)
        manifest["val"][0], manifest["test"][0] = manifest["test"][0], manifest["val"][0]
        with self.assertRaisesRegex(ValueError, "deterministic grouped split"):
            verify_manifest(self.data, manifest)

    def test_invalid_explicit_config(self):
        changes = [
            {"train_circuits": ["synthetic_0"]},
            {"ood_circuits": ["synthetic_0", "synthetic_3", "synthetic_4"]},
            {"ood_circuits": None}, {"train_circuits": []},
            {"split_ratios": [0.8, 0.2, 0.2]}, {"split_ratios": [True, 0, 0]},
            {"split_ratios": [0, 0.5, 0.5]}, {"split_seed": True},
            {"finetune_label_fraction": 0}, {"finetune_label_fraction": 1.01},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_manifest(self.data, config={**self.config, **change})

    def test_explicit_zero_holdout_ratios(self):
        data = copy.deepcopy(self.data)
        data["samples"] = data["samples"][:1]
        manifest = build_manifest(data, config={"train_circuits": ["synthetic_0"], "ood_circuits": [],
                                                "split_ratios": [1, 0, 0]})
        self.assertEqual(manifest["val"], [])
        self.assertEqual(manifest["test"], [])
        self.assertEqual(len(manifest["labeled_train"]), 1)


class PhysicsNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_dataset(12)
        self.config = {"train_circuits": [f"synthetic_{i}" for i in range(4)],
                       "ood_circuits": ["synthetic_4"], "finetune_label_fraction": 0.3}
        self.manifest = build_manifest(self.data, config=self.config)

    def refresh(self):
        self.manifest["dataset_fingerprint"] = dataset_fingerprint(self.data)

    def test_signed_zero_missing_and_round_trip(self):
        values = [-1e-250, 0.0, 1e-250, None]
        train_ids = set(self.manifest["train"])
        observed = []
        count = 0
        for row in self.data["samples"]:
            if row["sample_id"] in train_ids:
                for device in row["devices"]:
                    device["op"]["id"] = values[count % len(values)]
                    if is_op_target(device) and values[count % len(values)] is not None:
                        observed.append(values[count % len(values)])
                    count += 1
        self.refresh()
        pp = fit_preprocessor(self.data, self.manifest)
        stats = pp["op"]
        self.assertEqual(stats["count"][0], len(observed))
        self.assertEqual(stats["asinh_scale"][0], 1e-250)
        self.assertEqual(stats["kind"][0], "asinh")
        normalized = [transform_value(x, stats, 0) for x in values[:3]]
        self.assertLess(normalized[0], normalized[1])
        self.assertLess(normalized[1], normalized[2])
        for value, norm in zip(values[:3], normalized):
            self.assertAlmostEqual(inverse_transform_value(norm, stats, 0) / 1e-250, value / 1e-250, places=13)
        with self.assertRaises(ValueError):
            transform_value(None, stats, 0)

    def test_equal_circuit_weighting_not_oversampled_topology(self):
        # Keep all data training to isolate weighting from split proportions.
        self.data["samples"] = [row for row in self.data["samples"] if row["circuit_id"] in ("synthetic_0", "synthetic_1")]
        for row in self.data["samples"]:
            value = 0.0 if row["circuit_id"] == "synthetic_0" else 10.0
            for device in row["devices"]:
                device["op"]["vds"] = value
            row["performance"]["gain"] = value
        # Duplicate one circuit tenfold with unique sample IDs, grouped designs.
        original = list(self.data["samples"])
        for k in range(9):
            for row in original:
                if row["circuit_id"] == "synthetic_1":
                    extra = copy.deepcopy(row)
                    extra["sample_id"] += f"_repeat{k}"
                    self.data["samples"].append(extra)
        manifest = build_manifest(self.data, config={"train_circuits": ["synthetic_0", "synthetic_1"],
                                                    "ood_circuits": [], "split_ratios": [1, 0, 0]})
        pp = fit_preprocessor(self.data, manifest, include_performance=True)
        self.assertAlmostEqual(pp["op"]["mean"][2], 5.0)
        self.assertAlmostEqual(pp["op"]["std"][2], 5.0)
        self.assertAlmostEqual(pp["performance"]["mean"][0], 5.0)
        observation = fit_preprocessor(self.data, manifest, config={"weighting": "observation"})
        self.assertGreater(observation["op"]["mean"][2], 8.5)

    def test_performance_only_labeled_train_and_stage_one_all_train(self):
        baseline = fit_preprocessor(self.data, self.manifest, True)
        unlabeled_ids = set(self.manifest["train"]) - set(self.manifest["labeled_train"])
        self.assertTrue(unlabeled_ids)
        before_inputs = input_op_fingerprint(self.data)
        for row in self.data["samples"]:
            if row["sample_id"] not in self.manifest["labeled_train"]:
                row["performance"] = {"gain": 1e90, "power": -1e90}
        self.assertEqual(input_op_fingerprint(self.data), before_inputs)
        self.refresh()
        actual = fit_preprocessor(self.data, self.manifest, True)
        self.assertEqual(actual, baseline)
        self.assertEqual(actual["performance_fitted_sample_ids"], self.manifest["labeled_train"])
        self.assertEqual(actual["fit_ids"]["op"], self.manifest["train"])
        self.assertNotIn("performance", fit_preprocessor(self.data, self.manifest))

    def test_ood_mutation_cannot_change_transform_or_vocabulary(self):
        before = fit_preprocessor(self.data, self.manifest, True)
        for row in self.data["samples"]:
            if row["sample_id"] in self.manifest["ood"]:
                row["performance"] = {"gain": -1e200, "power": 1e200}
                for device in row["devices"]:
                    device["device_type"] = "OOD_UNKNOWN_TYPE"
                    device["op"] = {name: -1e200 for name in device["op"]}
        self.refresh()
        self.assertEqual(before, fit_preprocessor(self.data, self.manifest, True))

    def test_non_mos_graph_nodes_excluded_unless_explicit_target(self):
        baseline = fit_preprocessor(self.data, self.manifest)
        for row in self.data["samples"]:
            for device in row["devices"]:
                if device["device_type"] == "resistor":
                    device["op"] = {"id": 1e200, "gm": 1e200, "vds": 1e200}
        self.refresh()
        changed = fit_preprocessor(self.data, self.manifest)
        self.assertEqual(changed["op"], baseline["op"])
        self.assertEqual(changed["parameters"], baseline["parameters"])
        self.assertIn("resistor", changed["device_vocab"])
        device = self.data["samples"][0]["devices"][0]
        self.assertTrue(is_op_target({**device, "device_type": "NMOS"}))
        self.assertFalse(is_op_target({**device, "op_target": False}))
        self.assertTrue(is_op_target({**device, "device_type": "resistor", "op_target": True}))

    def test_raw_geometry_preserved_selected_electrical_fields(self):
        added = ["w_um", "l_um", "multiplicity", "id_a", "cgd_f", "intrinsic_gain_v_v", "vd_v"]
        self.data["feature_spec"]["op_names"].extend(added)
        for row in self.data["samples"]:
            for device in row["devices"]:
                device["op"].update({name: 1e-15 if name.endswith("_f") else -2.0 for name in added})
        original = copy.deepcopy(self.data)
        self.refresh()
        pp = fit_preprocessor(self.data, self.manifest)
        self.assertEqual(self.data, original)
        self.assertEqual(pp["raw_feature_spec"], self.data["feature_spec"])
        for name in ("w_um", "l_um", "multiplicity"):
            self.assertNotIn(name, pp["feature_spec"]["op_names"])
        kinds = dict(zip(pp["feature_spec"]["op_names"], pp["op"]["kind"]))
        self.assertEqual(kinds["id_a"], "asinh")
        self.assertEqual(kinds["cgd_f"], "asinh")
        self.assertEqual(kinds["intrinsic_gain_v_v"], "asinh")
        self.assertEqual(kinds["vd_v"], "linear")
        pp = fit_preprocessor(self.data, self.manifest, config={"op_fields": ["id_a", "vd_v"]})
        self.assertEqual(pp["feature_spec"]["op_names"], ["id_a", "vd_v"])

    def test_fixed_and_fitted_per_collection_transform_config(self):
        config = {"normalization": {"parameters": {"W": {"kind": "asinh", "scale": 1e-6}},
                                     "op": {"id": {"kind": "asinh", "scale": 1e-9}},
                                     "performance": {"power": {"kind": "asinh", "scale": "fit"}}}}
        pp = fit_preprocessor(self.data, self.manifest, True, config)
        self.assertEqual(pp["parameters"]["asinh_scale"][0], 1e-6)
        self.assertEqual(pp["op"]["asinh_scale"][0], 1e-9)
        self.assertEqual(pp["performance"]["kind"][1], "asinh")
        self.assertGreater(pp["performance"]["asinh_scale"][1], 0)
        self.assertEqual(pp["preprocessing_version"], 2)
        self.assertEqual(pp, fit_preprocessor(self.data, self.manifest, True, pp["normalization_config"]))
        json.dumps(pp, allow_nan=False)
        for scale in (0, -1, True, math.inf, "learn"):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                fit_preprocessor(self.data, self.manifest, config={"op": {"id": {"scale": scale}}})

    def test_extreme_asinh_and_linear_finite_round_trips(self):
        stats = {"kind": ["asinh"], "mean": [0.0], "std": [1.0], "asinh_scale": [1e-300]}
        for value in (-1e300, -1e-300, 0.0, 1e-300, 1e300):
            transformed = transform_value(value, stats, 0)
            self.assertTrue(math.isfinite(transformed))
            recovered = inverse_transform_value(transformed, stats, 0)
            if value:
                self.assertAlmostEqual(recovered / value, 1.0, places=12)
            else:
                self.assertEqual(recovered, 0.0)
        linear = {"kind": ["linear"], "mean": [-1e308], "std": [1e308], "asinh_scale": [1.0]}
        self.assertEqual(transform_value(1e308, linear, 0), 2.0)
        self.assertEqual(inverse_transform_value(2.0, linear, 0), 1e308)


if __name__ == "__main__":
    unittest.main()
