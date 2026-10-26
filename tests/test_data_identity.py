"""Regression tests for model/port identity and signed-zero-safe grouping."""

import copy
import math
import unittest

from device_jepa.data import (
    DESIGN_FINGERPRINT_VERSION, INPUT_CONTRACT_VERSION, TERMINAL_NAMES,
    build_manifest, canonical_model_name, canonical_port_name,
    dataset_fingerprint, design_fingerprint, fit_preprocessor,
    input_op_fingerprint, partition_fingerprint, synthetic_dataset,
    validate_dataset, verify_manifest,
)


def source_dataset():
    dataset = {
        "schema_version": 1,
        "feature_spec": {"parameter_names": ["dc_voltage_V"], "op_names": ["id"],
                         "metric_names": ["gain"], "terminal_names": list(TERMINAL_NAMES)},
        "samples": [],
    }
    for index, value in enumerate([0.0, -0.0, .2, .4, .6, .8]):
        dataset["samples"].append({
            "sample_id": f"s{index}", "design_id": f"export_{index}", "circuit_id": "C",
            "ports": {"VOUT": "out"}, "performance": {"gain": 1.0},
            "devices": [{"device_id": "V1", "device_type": "independent_voltage_source",
                         "parameters": {"dc_voltage_V": value}, "op": {},
                         "terminals": {"p": "out", "n": "0"}, "op_target": False}],
        })
    return dataset


class DesignIdentityTests(unittest.TestCase):
    def test_signed_zero_identity_is_canonical_and_raw_sign_is_preserved(self):
        data = source_dataset()
        a, b = data["samples"][:2]
        b["devices"][0]["op"]["id"] = -0.0
        before = copy.deepcopy(data)
        self.assertEqual(design_fingerprint(a, data["feature_spec"]),
                         design_fingerprint(b, data["feature_spec"]))
        self.assertEqual(math.copysign(1, b["devices"][0]["parameters"]["dc_voltage_V"]), -1)
        self.assertEqual(math.copysign(1, b["devices"][0]["op"]["id"]), -1)
        self.assertEqual(data, before)
        changed = copy.deepcopy(b)
        changed["devices"][0]["parameters"]["dc_voltage_V"] = None
        self.assertNotEqual(design_fingerprint(a, data["feature_spec"]),
                            design_fingerprint(changed, data["feature_spec"]))
        plus, minus = copy.deepcopy(a), copy.deepcopy(a)
        plus["devices"][0]["parameters"]["dc_voltage_V"] = 1e-30
        minus["devices"][0]["parameters"]["dc_voltage_V"] = -1e-30
        self.assertNotEqual(design_fingerprint(plus, data["feature_spec"]),
                            design_fingerprint(minus, data["feature_spec"]))

    def test_signed_zero_duplicates_never_cross_splits(self):
        data = source_dataset()
        for seed in range(20):
            with self.subTest(seed=seed):
                manifest = build_manifest(data, config={"train_circuits": ["C"], "ood_circuits": [],
                    "split_ratios": [.5, 1 / 6, 1 / 3], "split_seed": seed})
                assignment = {sid: split for split in ("train", "val", "test") for sid in manifest[split]}
                self.assertEqual(assignment["s0"], assignment["s1"])
                verify_manifest(data, manifest)

    def test_cross_circuit_signed_zero_copy_is_rejected(self):
        data = source_dataset()
        data["samples"][1]["circuit_id"] = "OOD"
        with self.assertRaisesRegex(ValueError, "exact duplicate designs cross circuit"):
            build_manifest(data, config={"train_circuits": ["C"], "ood_circuits": ["OOD"]})

    def test_model_identity_changes_design_but_case_does_not(self):
        data = synthetic_dataset(3)
        original = copy.deepcopy(data["samples"][0])
        original["devices"][0]["model_name"] = "sky130_fd_pr__nfet_01v8"
        case_variant = copy.deepcopy(original)
        case_variant["devices"][0]["model_name"] = "SKY130_FD_PR__NFET_01V8"
        lvt_variant = copy.deepcopy(original)
        lvt_variant["devices"][0]["model_name"] += "_lvt"
        self.assertEqual(design_fingerprint(original, data["feature_spec"]),
                         design_fingerprint(case_variant, data["feature_spec"]))
        self.assertNotEqual(design_fingerprint(original, data["feature_spec"]),
                            design_fingerprint(lvt_variant, data["feature_spec"]))
        self.assertEqual(case_variant["devices"][0]["model_name"], "SKY130_FD_PR__NFET_01V8")

    def test_other_model_same_geometry_is_not_false_cross_circuit_duplicate(self):
        data = synthetic_dataset(3)
        originals = [sample for sample in data["samples"] if sample["circuit_id"] == "synthetic_0"]
        for sample in originals:
            sample["devices"][0]["model_name"] = "standard_model"
        variants = copy.deepcopy(originals)
        for sample in variants:
            sample["sample_id"] = "other_" + sample["sample_id"]
            sample["circuit_id"] = "other_model_circuit"
            sample["devices"][0]["model_name"] = "lvt_model"
        data["samples"] = originals + variants
        manifest = build_manifest(data, config={"train_circuits": ["synthetic_0"],
                                               "ood_circuits": ["other_model_circuit"]})
        self.assertEqual(len(manifest["ood"]), 3)
        # A case-only rename is the same model and must not evade duplicate detection.
        for sample in variants:
            sample["devices"][0]["model_name"] = "STANDARD_MODEL"
        with self.assertRaisesRegex(ValueError, "exact duplicate designs cross circuit"):
            build_manifest(data, config={"train_circuits": ["synthetic_0"],
                                         "ood_circuits": ["other_model_circuit"]})

    def test_missing_and_null_model_identity_agree_without_becoming_unknown(self):
        data = synthetic_dataset(3)
        absent = copy.deepcopy(data["samples"][0])
        null = copy.deepcopy(absent)
        null["devices"][0]["model_name"] = None
        named = copy.deepcopy(absent)
        named["devices"][0]["model_name"] = "new_model"
        self.assertIsNone(canonical_model_name(absent["devices"][0]))
        self.assertEqual(design_fingerprint(absent, data["feature_spec"]),
                         design_fingerprint(null, data["feature_spec"]))
        self.assertNotEqual(design_fingerprint(absent, data["feature_spec"]),
                            design_fingerprint(named, data["feature_spec"]))

    def test_invalid_model_names_fail_schema_validation(self):
        data = synthetic_dataset(3)
        for value in (0, True, [], "", "  ", "<PAD>", "<unk>", "<NoNe>"):
            with self.subTest(value=value):
                bad = copy.deepcopy(data)
                bad["samples"][0]["devices"][0]["model_name"] = value
                with self.assertRaisesRegex(ValueError, "model_name"):
                    validate_dataset(bad)

    def test_port_roles_are_case_insensitive_and_ambiguous_duplicates_rejected(self):
        self.assertEqual(canonical_port_name("VINP"), "vinp")
        data = source_dataset()
        first = copy.deepcopy(data["samples"][0])
        second = copy.deepcopy(first)
        second["ports"] = {"vout": "out"}
        self.assertEqual(design_fingerprint(first, data["feature_spec"]),
                         design_fingerprint(second, data["feature_spec"]))
        data["samples"][0]["ports"]["vout"] = "different_net"
        with self.assertRaisesRegex(ValueError, "duplicate case-insensitive port"):
            validate_dataset(data)

    def test_manifest_explicitly_records_and_requires_new_grouping_semantics(self):
        data = source_dataset()
        manifest = build_manifest(data, config={"train_circuits": ["C"], "ood_circuits": []})
        self.assertEqual(manifest["design_fingerprint_version"], DESIGN_FINGERPRINT_VERSION)
        for version in (None, 1, 3, True, 2.0):
            with self.subTest(version=version):
                legacy = copy.deepcopy(manifest)
                if version is None:
                    legacy.pop("design_fingerprint_version")
                else:
                    legacy["design_fingerprint_version"] = version
                with self.assertRaisesRegex(ValueError, "design_fingerprint_version"):
                    verify_manifest(data, legacy)
        legacy = copy.deepcopy(manifest)
        legacy.pop("design_fingerprint_version")
        self.assertNotEqual(partition_fingerprint(manifest), partition_fingerprint(legacy))

    def test_grouping_version_does_not_break_allowed_performance_addition(self):
        data = source_dataset()
        manifest = build_manifest(data, config={"train_circuits": ["C"], "ood_circuits": []})
        before = input_op_fingerprint(data)
        partition = partition_fingerprint(manifest)
        data["samples"][0]["performance"]["gain"] = 4.0
        self.assertEqual(input_op_fingerprint(data), before)
        manifest["dataset_fingerprint"] = dataset_fingerprint(data)
        verify_manifest(data, manifest)
        self.assertEqual(partition_fingerprint(manifest), partition)


class TrainOnlyIdentityVocabularyTests(unittest.TestCase):
    def test_new_vocabularies_fit_only_id_train_and_have_distinct_sentinels(self):
        data = synthetic_dataset(10)
        for sample in data["samples"]:
            sample["devices"][0]["model_name"] = "TRAIN_MODEL"
            sample["ports"]["VDD"] = "supply"
        manifest = build_manifest(data, "synthetic_4")
        train_ids = set(manifest["train"])
        for sample in data["samples"]:
            if sample["sample_id"] not in train_ids:
                sample["devices"][0]["model_name"] = "HELD_OUT_MODEL"
                sample["ports"]["ONLY_HELD_OUT_PORT"] = "special"
        manifest["dataset_fingerprint"] = dataset_fingerprint(data)
        pre = fit_preprocessor(data, manifest)
        self.assertEqual(pre["input_contract_version"], INPUT_CONTRACT_VERSION)
        self.assertEqual(pre["model_vocab"], {"<PAD>": 0, "<UNK>": 1, "<NONE>": 2, "train_model": 3})
        self.assertNotIn("held_out_model", pre["model_vocab"])
        self.assertNotIn("only_held_out_port", pre["port_role_vocab"])
        self.assertIn("vdd", pre["port_role_vocab"])
        self.assertEqual(pre["port_role_vocab"]["<PAD>"], 0)
        self.assertEqual(pre["port_role_vocab"]["<UNK>"], 1)
        self.assertEqual(pre["fit_ids"]["model_vocabulary"], sorted(manifest["train"]))
        self.assertEqual(pre["fit_ids"]["port_role_vocabulary"], sorted(manifest["train"]))
        unknown_id = pre["model_vocab"].get(canonical_model_name({"model_name": "held_out_model"}), pre["model_vocab"]["<UNK>"])
        self.assertEqual(unknown_id, 1)
        self.assertNotEqual(unknown_id, pre["model_vocab"]["<NONE>"])
        self.assertNotEqual(unknown_id, pre["model_vocab"]["<PAD>"])

    def test_case_variants_share_one_model_vocab_entry(self):
        data = synthetic_dataset(10)
        for index, sample in enumerate(data["samples"]):
            sample["devices"][0]["model_name"] = "MODEL_A" if index % 2 else "model_a"
        manifest = build_manifest(data, "synthetic_4")
        pre = fit_preprocessor(data, manifest)
        self.assertEqual(pre["model_vocab"], {"<PAD>": 0, "<UNK>": 1, "<NONE>": 2, "model_a": 3})


if __name__ == "__main__":
    unittest.main()
