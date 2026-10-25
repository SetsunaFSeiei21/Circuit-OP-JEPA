"""Independent OP-check categories, evidence scope, and immutable source data."""
import copy
import json
import unittest

from device_jepa.data import validate_dataset
from device_jepa.states import STATE_FEATURES, apply_op_states, encode_device_checks, validate_op_states


THRESHOLDS = {"min_signal_current_a": 1e-12, "saturation_margin_min_v": 0.0}


def fixture():
    names = ["id_a", "saturation_margin_v"] + [f"raw_{i}" for i in range(42)]
    op = {key: (-1) ** index * (index + 1) * 0.01 for index, key in enumerate(names)}
    op.update(id_a=-2e-12, saturation_margin_v=0.1)
    devices = [
        {"device_id": "XP", "device_type": "PMOS", "op_target": True,
         "parameters": {"W": 2}, "op": op, "terminals": {"d": "out", "g": "in", "s": "vdd", "b": "vdd"}},
        {"device_id": "XN", "device_type": "NMOS", "op_target": True,
         "parameters": {"W": 1}, "op": copy.deepcopy(op), "terminals": {"d": "out", "g": "in", "s": "gnd", "b": "gnd"}},
        {"device_id": "IREF", "device_type": "independent_current_source", "op_target": False,
         "parameters": {"W": None}, "op": {}, "terminals": {"p": "out", "n": "gnd"}},
    ]
    data = {
        "schema_version": 1,
        "feature_spec": {"parameter_names": ["W"], "op_names": names, "metric_names": ["gain"],
                         "terminal_names": ["d", "g", "s", "b", "p", "n"]},
        "samples": [{"sample_id": "unit:1", "source_sample_id": 1, "circuit_id": "unit", "design_id": "design:1",
                     "devices": devices, "performance": {"gain": None},
                     "source_sampling": {"run_id": "run1"},
                     "source_observation": {"status": "complete", "feasible": 0, "reasons": []}}],
        "metadata": {"synthetic": True},
        "provenance": {"runs": [{"run_id": "run1", "constraints": dict(THRESHOLDS)}]},
    }
    return validate_dataset(data)


class CheckEncodingTests(unittest.TestCase):
    def test_independent_simultaneous_failures(self):
        result = encode_device_checks({"id_a": 0, "saturation_margin_v": -0.01}, True, THRESHOLDS)
        self.assertEqual(result, dict(zip(STATE_FEATURES, [0, 1, 0, 0, 1, 0])))
        self.assertEqual(sum(result.values()), 2)

    def test_equal_threshold_passes_and_signed_pmos_current(self):
        for current in (1e-12, -1e-12, -2e-12):
            with self.subTest(current=current):
                result = encode_device_checks({"id_a": current, "saturation_margin_v": 0}, True, THRESHOLDS)
                self.assertEqual(result["current_pass"], 1)
                self.assertEqual(result["saturation_pass"], 1)

    def test_zero_and_missing_are_different(self):
        missing = encode_device_checks({"id_a": None, "saturation_margin_v": None}, True, THRESHOLDS)
        zero = encode_device_checks({"id_a": 0, "saturation_margin_v": 0}, True, THRESHOLDS)
        self.assertEqual(missing, dict(zip(STATE_FEATURES, [1, 0, 0, 1, 0, 0])))
        self.assertEqual(zero, dict(zip(STATE_FEATURES, [0, 1, 0, 0, 0, 1])))

    def test_missing_checked_group_known_without_threshold(self):
        result = encode_device_checks({}, {"current": True}, {})
        self.assertEqual([result[key] for key in STATE_FEATURES[:3]], [1, 0, 0])
        self.assertTrue(all(result[key] is None for key in STATE_FEATURES[3:]))

    def test_nonmissing_checked_without_threshold_remains_unknown(self):
        result = encode_device_checks({"id_a": 0, "saturation_margin_v": -10}, True)
        self.assertTrue(all(value is None for value in result.values()))

    def test_unchecked_is_masked_not_missing_or_passing(self):
        result = encode_device_checks({}, False, THRESHOLDS)
        self.assertTrue(all(value is None for value in result.values()))

    def test_saturation_keeps_sign(self):
        result = encode_device_checks({"saturation_margin_v": -0.1}, {"saturation": True}, THRESHOLDS)
        self.assertEqual(result["saturation_below"], 1)

    def test_each_group_needs_its_own_threshold(self):
        result = encode_device_checks({"id_a": 1, "saturation_margin_v": 1}, True,
                                      {"saturation_margin_min_v": 0})
        self.assertTrue(all(result[key] is None for key in STATE_FEATURES[:3]))
        self.assertEqual(result["saturation_pass"], 1)

    def test_invalid_checked_and_threshold_inputs_rejected(self):
        for checked in (1, None, {"current": 1}, {"unknown": True}):
            with self.subTest(checked=checked), self.assertRaises(ValueError):
                encode_device_checks({}, checked, THRESHOLDS)
        for value in (True, float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode_device_checks({}, True, {"min_signal_current_a": value})


class ApplyCheckEncodingTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.sample = self.data["samples"][0]
        self.observation = self.sample["source_observation"]

    def apply(self, config=None):
        return apply_op_states(self.data, config)["samples"][0]["devices"]

    def test_default_no_scope_does_not_invent_passes(self):
        devices = self.apply()
        for device in devices:
            self.assertTrue(all(value is None for value in device["op_state"].values()))
            self.assertEqual(device["op_state_provenance"]["checked"], {"current": False, "saturation": False})

    def test_explicit_empty_scope_does_not_invent_passes(self):
        self.observation["checked_devices"] = []
        self.assertTrue(all(value is None for device in self.apply() for value in device["op_state"].values()))

    def test_sample_checklist_and_run_thresholds(self):
        self.observation["checked_devices"] = ["xp"]
        devices = self.apply()
        self.assertEqual(devices[0]["op_state"]["current_pass"], 1)
        self.assertEqual(devices[0]["op_state"]["saturation_pass"], 1)
        self.assertTrue(all(value is None for value in devices[1]["op_state"].values()))
        provenance = devices[0]["op_state_provenance"]
        self.assertEqual(provenance["thresholds"], THRESHOLDS)
        self.assertEqual(provenance["source_run_id"], "run1")
        self.assertIn("provenance.runs[0].constraints.min_signal_current_a",
                      provenance["threshold_sources"]["min_signal_current_a"])

    def test_run_checklist_is_explicit_scope_evidence(self):
        self.data["provenance"]["runs"][0]["signal_devices"] = ["XN"]
        devices = self.apply()
        self.assertTrue(all(value is None for value in devices[0]["op_state"].values()))
        self.assertEqual(devices[1]["op_state"]["current_pass"], 1)

    def test_raw_reason_only_establishes_that_group_was_checked(self):
        self.observation["reasons"] = ["follower_error", "xp:saturation"]
        self.sample["devices"][0]["op"]["saturation_margin_v"] = -0.1
        devices = self.apply()
        self.assertTrue(all(devices[0]["op_state"][key] is None for key in STATE_FEATURES[:3]))
        self.assertEqual(devices[0]["op_state"]["saturation_below"], 1)
        self.assertTrue(all(value is None for value in devices[1]["op_state"].values()))

    def test_failure_label_without_threshold_stays_unknown_unless_missing(self):
        self.data["provenance"]["runs"] = []
        self.observation["reasons"] = ["XP:off_or_missing_current", "XP:saturation"]
        self.sample["devices"][0]["op"].update(id_a=None, saturation_margin_v=-0.1)
        device = self.apply()[0]
        self.assertEqual(device["op_state"]["current_missing"], 1)
        self.assertTrue(all(device["op_state"][key] is None for key in STATE_FEATURES[3:]))
        self.assertEqual(device["op_state_provenance"]["raw_reason_labels"]["saturation"], ["XP:saturation"])

    def test_non_mos_always_unchecked_even_with_raw_zero_or_checked_list(self):
        self.observation["checked_devices"] = ["IREF"]
        self.sample["devices"][2]["op"] = {"id_a": 0, "saturation_margin_v": 0}
        self.sample["devices"][2]["op_target"] = True
        for mode in ("collector_checked", "all_mos"):
            with self.subTest(mode=mode):
                device = self.apply({"mode": mode})[2]
                self.assertTrue(all(value is None for value in device["op_state"].values()))
                self.assertFalse(any(device["op_state_provenance"]["checked"].values()))

    def test_all_mos_is_explicit_derived_opt_in(self):
        result = apply_op_states(self.data, {"mode": "all_mos"})
        for device in result["samples"][0]["devices"][:2]:
            self.assertEqual(device["op_state"]["current_pass"], 1)
            self.assertEqual(device["op_state"]["saturation_pass"], 1)
            self.assertTrue(device["op_state_provenance"]["derived_not_original_collector"])
        self.assertTrue(result["metadata"]["state_encoding"]["derived_not_original_collector"])
        self.assertEqual(result["samples"][0]["source_observation"], self.observation)

    def test_preserve_only_has_no_state_features(self):
        result = apply_op_states(self.data, {"mode": "preserve_only"})
        self.assertEqual(result["metadata"]["state_encoding"]["feature_names"], [])
        self.assertTrue(all(device["op_state"] == {} for device in result["samples"][0]["devices"]))

    def test_all_44_op_fields_signs_and_source_metadata_preserved(self):
        before = copy.deepcopy(self.data)
        result = apply_op_states(self.data, {"mode": "all_mos"})
        self.assertEqual(self.data, before)
        self.assertEqual(result["feature_spec"], before["feature_spec"])
        self.assertEqual(result["provenance"], before["provenance"])
        self.assertEqual(len(result["samples"][0]["devices"][0]["op"]), 44)
        for old, new in zip(before["samples"][0]["devices"], result["samples"][0]["devices"]):
            self.assertEqual(old["op"], new["op"])
            self.assertIsNot(old["op"], new["op"])
        self.assertEqual(result["samples"][0]["source_observation"], before["samples"][0]["source_observation"])
        self.assertEqual(apply_op_states(result, {"mode": "all_mos"}), result)
        json.dumps(result, allow_nan=False)
        validate_dataset(result)

    def test_conflicting_threshold_sources_rejected(self):
        self.observation["thresholds"] = {"saturation_margin_min_v": 0.1}
        with self.assertRaisesRegex(ValueError, "conflicting threshold"):
            self.apply()

    def test_explicit_thresholds_can_fill_gaps_but_not_override(self):
        self.data["provenance"]["runs"][0]["constraints"] = {}
        self.observation["checked_devices"] = ["XP"]
        self.assertEqual(self.apply({"thresholds": THRESHOLDS})[0]["op_state"]["current_pass"], 1)
        self.data["provenance"]["runs"][0]["constraints"] = THRESHOLDS
        with self.assertRaisesRegex(ValueError, "conflicting threshold"):
            self.apply({"thresholds": {"min_signal_current_a": 10}})

    def test_different_runs_join_exact_sample_run_id(self):
        self.observation["checked_devices"] = ["XP"]
        self.data["provenance"]["runs"].insert(0, {"run_id": "other", "constraints": {
            "min_signal_current_a": 100, "saturation_margin_min_v": 100}})
        self.assertEqual(self.apply()[0]["op_state"]["current_pass"], 1)
        self.sample["source_sampling"]["run_id"] = "other"
        self.assertEqual(self.apply()[0]["op_state"]["current_below"], 1)

    def test_missing_or_unmatched_run_id_does_not_borrow_thresholds(self):
        self.observation["checked_devices"] = ["XP"]
        for source_sampling in ({}, {"run_id": "absent"}):
            with self.subTest(source_sampling=source_sampling):
                self.sample["source_sampling"] = source_sampling
                device = self.apply()[0]
                self.assertTrue(all(value is None for value in device["op_state"].values()))
                self.assertTrue(all(device["op_state_provenance"]["checked"].values()))

    def test_combined_provenance_and_ambiguous_same_run_id(self):
        original = self.data.pop("provenance")
        original.update(source_id="source1", circuit_ids=["unit"])
        self.sample["source_id"] = "source1"
        self.data["provenance"] = {"sources": [original]}
        self.observation["checked_devices"] = ["XP"]
        self.assertEqual(self.apply()[0]["op_state"]["current_pass"], 1)
        duplicate = copy.deepcopy(original)
        duplicate["runs"][0]["constraints"]["min_signal_current_a"] = 10
        original["runs"].append(duplicate["runs"][0])
        with self.assertRaisesRegex(ValueError, "ambiguous run"):
            self.apply()

    def test_legacy_combined_sources_never_borrow_another_circuits_run(self):
        first, second = fixture(), fixture()
        first["samples"][0].update(sample_id="A:1", circuit_id="A")
        first["provenance"]["runs"] = []
        second["samples"][0].update(sample_id="B:1", circuit_id="B")
        second["provenance"]["runs"][0]["checked_devices"] = ["XP"]
        combined = copy.deepcopy(first)
        combined["samples"].extend(second["samples"])
        combined["provenance"] = {"sources": [first["provenance"], second["provenance"]]}
        result = apply_op_states(combined)
        for sample in result["samples"]:
            for device in sample["devices"]:
                self.assertTrue(all(value is None for value in device["op_state"].values()))

    def test_scoped_combined_sources_join_source_circuit_and_run(self):
        first, second = fixture(), fixture()
        first["samples"][0].update(sample_id="A:1", circuit_id="A", source_id="sourceA")
        first["provenance"].update(runs=[], source_id="sourceA", circuit_ids=["A"])
        second["samples"][0].update(sample_id="B:1", circuit_id="B", source_id="sourceB")
        second["provenance"].update(source_id="sourceB", circuit_ids=["B"])
        second["provenance"]["runs"][0]["checked_devices"] = ["XP"]
        combined = copy.deepcopy(first)
        combined["samples"].extend(second["samples"])
        combined["provenance"] = {"sources": [first["provenance"], second["provenance"]]}
        result = apply_op_states(combined)
        a, b = result["samples"]
        self.assertTrue(all(value is None for device in a["devices"] for value in device["op_state"].values()))
        self.assertEqual(b["devices"][0]["op_state"]["current_pass"], 1)
        self.assertEqual(b["devices"][0]["op_state_provenance"]["source_id"], "sourceB")
        self.assertIsNone(validate_op_states(result))
        combined["samples"][0]["source_id"] = "sourceB"
        with self.assertRaisesRegex(ValueError, "source_id circuit scope"):
            apply_op_states(combined)

    def test_same_circuit_different_sources_do_not_share_run_settings(self):
        source_a = {"source_id": "a", "circuit_ids": ["unit"], "runs": []}
        source_b = copy.deepcopy(self.data["provenance"])
        source_b.update(source_id="b", circuit_ids=["unit"])
        source_b["runs"][0]["checked_devices"] = ["XP"]
        self.data["provenance"] = {"sources": [source_a, source_b]}
        self.sample["source_id"] = "a"
        self.assertTrue(all(value is None for device in self.apply() for value in device["op_state"].values()))
        self.sample["source_id"] = "b"
        self.assertEqual(self.apply()[0]["op_state"]["current_pass"], 1)

    def test_duplicate_source_ids_rejected_without_choosing_first(self):
        source = copy.deepcopy(self.data["provenance"])
        source.update(source_id="same", circuit_ids=["unit"])
        self.data["provenance"] = {"sources": [source, copy.deepcopy(source)]}
        self.sample["source_id"] = "same"
        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            self.apply()

    def test_unscoped_multicircuit_root_does_not_guess_run_ownership(self):
        other = copy.deepcopy(self.sample)
        other.update(sample_id="B:1", circuit_id="B")
        self.data["samples"].append(other)
        self.data["provenance"]["runs"][0]["checked_devices"] = ["XP"]
        self.assertTrue(all(value is None for device in self.apply() for value in device["op_state"].values()))

    def test_ambiguous_run_checklists_rejected(self):
        first = self.data["provenance"]["runs"][0]
        first["checked_devices"] = ["XP"]
        second = copy.deepcopy(first)
        second["checked_devices"] = ["XN"]
        self.data["provenance"]["runs"].append(second)
        with self.assertRaisesRegex(ValueError, "ambiguous run"):
            self.apply()

    def test_conflicting_checklists_and_reasons_rejected(self):
        self.observation["checked_devices"] = []
        self.observation["signal_devices"] = ["XP"]
        with self.assertRaisesRegex(ValueError, "conflicting.*checklists"):
            self.apply()
        self.observation.pop("signal_devices")
        self.observation["reasons"] = ["XP:saturation"]
        with self.assertRaisesRegex(ValueError, "reason conflicts"):
            self.apply()

    def test_failure_reason_contradicting_computed_pass_rejected(self):
        self.observation["reasons"] = ["XP:saturation"]
        with self.assertRaisesRegex(ValueError, "failure reason conflicts"):
            self.apply()

    def test_config_validation_and_group_selection(self):
        for config in ({"mode": "regions"}, {"features": ["current_pass"]},
                       {"features": ["made_up"]}, {"mode": "preserve_only", "features": STATE_FEATURES},
                       {"mode": "all_mos", "scope": "guess"},
                       {"thresholds": {"min_signal_current": 1e-12}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.apply(config)
        result = apply_op_states(self.data, {"mode": "all_mos", "features": STATE_FEATURES[:3]})
        self.assertEqual(result["metadata"]["state_encoding"]["feature_names"], STATE_FEATURES[:3])
        self.assertEqual(list(result["samples"][0]["devices"][0]["op_state"]), STATE_FEATURES[:3])


class StateValidationTests(unittest.TestCase):
    def setUp(self):
        self.raw = fixture()
        self.config = {"mode": "collector_checked", "features": list(STATE_FEATURES)}
        self.raw["samples"][0]["source_observation"]["checked_devices"] = ["XP"]
        self.data = apply_op_states(self.raw, self.config)

    def assert_invalid(self, data=None, config=None, location=None):
        pattern = "Re-import" if location is None else location + ".*Re-import"
        with self.assertRaisesRegex(ValueError, pattern):
            validate_op_states(self.data if data is None else data, self.config if config is None else config)

    def test_matching_encoding_validates_without_mutation(self):
        before = copy.deepcopy(self.data)
        self.assertIsNone(validate_op_states(self.data, self.config))
        self.assertEqual(before, self.data)
        self.assertIsNone(validate_op_states(json.loads(json.dumps(self.data)), self.config))

    def test_default_config_matches_default_import(self):
        self.assertIsNone(validate_op_states(apply_op_states(self.raw)))

    def test_wrong_mode_rejected_even_when_values_are_all_unknown(self):
        all_mos = apply_op_states(self.raw, {"mode": "all_mos"})
        self.assert_invalid(all_mos, location="metadata.state_encoding")
        unknown = fixture()
        unknown["provenance"] = {}
        for device in unknown["samples"][0]["devices"]:
            device["op"] = {"id_a": 1, "saturation_margin_v": 1}
        all_mos_unknown = apply_op_states(unknown, {"mode": "all_mos"})
        self.assert_invalid(all_mos_unknown, location="metadata.state_encoding")

    def test_requested_threshold_conflict_rejected_with_reimport_guidance(self):
        changed = {**self.config, "thresholds": {"min_signal_current_a": 10}}
        self.assert_invalid(config=changed, location="conflicting threshold")

    def test_changed_explicit_threshold_rejected_even_if_categories_unchanged(self):
        raw = copy.deepcopy(self.raw)
        raw["provenance"]["runs"] = []
        first = {**self.config, "thresholds": {"min_signal_current_a": 1e-12, "saturation_margin_min_v": 0}}
        second = {**self.config, "thresholds": {"min_signal_current_a": 1.5e-12, "saturation_margin_min_v": 0}}
        data = apply_op_states(raw, first)
        self.assertEqual(data["samples"][0]["devices"][0]["op_state"],
                         apply_op_states(raw, second)["samples"][0]["devices"][0]["op_state"])
        self.assert_invalid(data, second, "metadata.state_encoding")

    def test_wrong_features_or_feature_order_rejected(self):
        self.assert_invalid(config={**self.config, "features": STATE_FEATURES[:3]}, location="metadata.state_encoding")
        self.assert_invalid(config={**self.config, "features": list(reversed(STATE_FEATURES))}, location="metadata.state_encoding")

    def test_unencoded_dataset_is_not_silently_accepted(self):
        self.assert_invalid(self.raw, location="metadata.state_encoding")
        raw_unknown = fixture()
        self.assert_invalid(raw_unknown, location="metadata.state_encoding")

    def test_missing_metadata_or_bad_version_rejected(self):
        del self.data["metadata"]["state_encoding"]
        self.assert_invalid(location="metadata.state_encoding")
        self.data = apply_op_states(self.raw, self.config)
        self.data["metadata"]["state_encoding"]["version"] += 1
        self.assert_invalid(location="metadata.state_encoding")

    def test_missing_device_states_and_missing_null_feature_rejected(self):
        for index, key in ((0, "op_state"), (0, "op_state_provenance"), (2, "op_state")):
            with self.subTest(index=index, key=key):
                data = copy.deepcopy(self.data)
                del data["samples"][0]["devices"][index][key]
                self.assert_invalid(data, location=key)
        del self.data["samples"][0]["devices"][1]["op_state"]["current_missing"]
        self.assert_invalid(location="op_state")

    def test_corrupt_values_rejected_including_bool_and_float_aliases(self):
        for value in (None, 0, True, 1.0, "1", 2):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data)
                data["samples"][0]["devices"][0]["op_state"]["current_pass"] = value
                self.assert_invalid(data, location="op_state")

    def test_corrupt_provenance_rejected(self):
        self.data["samples"][0]["devices"][0]["op_state_provenance"]["checked"]["current"] = False
        self.assert_invalid(location="op_state_provenance")

    def test_changed_raw_evidence_requires_reimport(self):
        self.data["samples"][0]["devices"][0]["op"]["id_a"] = 0
        before = copy.deepcopy(self.data)
        self.assert_invalid(location="op_state")
        self.assertEqual(self.data, before)

    def test_disabled_state_request_deliberately_bypasses_existing_encoding(self):
        for config in ({"mode": "preserve_only"}, {"features": []},
                       {"mode": "collector_checked", "features": []}):
            with self.subTest(config=config):
                before = copy.deepcopy(self.data)
                self.assertIsNone(validate_op_states(self.data, config))
                self.assertIsNone(validate_op_states(self.raw, config))
                self.assertEqual(self.data, before)
        broken = {"samples": [], "metadata": {"state_encoding": "ignored while disabled"}}
        self.assertIsNone(validate_op_states(broken, {"features": []}))


if __name__ == "__main__":
    unittest.main()
