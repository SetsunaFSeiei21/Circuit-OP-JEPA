"""Regressions for observation conflicts, deck termination, and import capacity.

The opt-in reader scale check creates 60,000 synthetic sample IDs and 17 OP rows
per sample, with one numeric OP field. It does not simulate circuits or build a
million-device canonical JSON object. Run explicitly with:
RUN_ADAPTER_SCALE_TEST=1 python -m unittest discover -s tests -p test_adapter_audit_fixes.py -v
"""
import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from device_jepa.adapter import (
    AdapterError, _observation, _read_database, adapt_analog_agent, parse_spice_netlist,
)
from device_jepa.cli import main
from test_adapter import fixture


class ObservationMergeTests(unittest.TestCase):
    def merge(self, original, table):
        return _observation(table, {"sample_id": 7, "observation_json": json.dumps(original)})

    def test_conflicting_empty_reasons_are_not_missing(self):
        with self.assertRaisesRegex(AdapterError, "sample 7: conflicting observation reasons"):
            self.merge({"reasons": ["MN:saturation"]}, {"reasons_json": "[]"})

    def test_missing_table_values_do_not_erase_known_evidence(self):
        result = self.merge({"reasons": ["MN:saturation"], "feasible": False, "vout_v": 0.0},
                            {"reasons_json": None, "feasible": None, "vout_v": None})
        self.assertEqual(result["reasons"], ["MN:saturation"])
        self.assertIs(result["feasible"], False)
        self.assertEqual(result["vout_v"], 0.0)

    def test_known_table_values_fill_missing_sample_values(self):
        result = self.merge({"reasons": None, "vout_v": None}, {"reasons_json": "[]", "vout_v": 0.0})
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["vout_v"], 0.0)

    def test_boolean_sqlite_values_and_reason_order_are_equivalent(self):
        result = self.merge({"converged": True, "feasible": False, "reasons": ["b", "a"]},
                            {"converged": 1, "feasible": 0, "reasons_json": '["a","b"]'})
        self.assertEqual(result["converged"], 1)
        self.assertEqual(result["reasons"], ["a", "b"])

    def test_scalar_and_structured_conflicts_are_rejected(self):
        pairs = [({"feasible": False}, {"feasible": 1}),
                 ({"vout_v": 0.0}, {"vout_v": 0.1}),
                 ({"status": "complete"}, {"status": "failed"}),
                 ({"node_voltages_v": {"out": 1}}, {"node_voltages_json": '{"out":2}'}),
                 ({"missing_fields": {}}, {"missing_fields_json": '{"out":true}'})]
        for original, table in pairs:
            with self.subTest(original=original), self.assertRaisesRegex(AdapterError, "conflicting observation"):
                self.merge(original, table)


class AdapterStreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.db = self.root / "sampling_history.sqlite3"

    def change_sql(self, sql, params=()):
        with sqlite3.connect(self.db) as connection:
            connection.execute(sql, params)

    def test_conflicting_duplicate_evidence_is_rejected_on_import(self):
        self.change_sql("UPDATE dc_operating_points SET reasons_json='[]' WHERE sample_id=1")
        with self.assertRaisesRegex(AdapterError, "conflicting observation reasons"):
            adapt_analog_agent(self.root)

    def test_missing_condition_and_bias_fall_back_to_sample_evidence(self):
        original = {"condition": {"TEMP": 27, "CORNER": "tt"},
                    "bias_topology": "unity_gain_follower", "reasons": ["XN:saturation"]}
        self.change_sql("UPDATE samples SET observation_json=?", (json.dumps(original),))
        self.change_sql("UPDATE dc_operating_points SET condition_json=NULL, bias_topology=NULL")
        dataset = adapt_analog_agent(self.root)
        self.assertEqual(dataset["samples"][0]["context"], {
            "TEMP": 27, "CORNER": "tt", "bias_topology": "unity_gain_follower"})

    def test_conflicting_conditions_are_rejected(self):
        self.change_sql("UPDATE samples SET observation_json=? WHERE sample_id=1",
                        (json.dumps({"condition": {"TEMP": 99, "CORNER": "tt"}}),))
        with self.assertRaisesRegex(AdapterError, "conflicting observation condition"):
            adapt_analog_agent(self.root)

    def test_limit_preflight_rejects_without_loading_rows(self):
        # The OP table has 5 rows. The invalid payload must not be touched when
        # its count already exceeds the explicit bound of 4.
        self.change_sql("UPDATE samples SET design_values_json='not JSON'")
        with self.assertRaisesRegex(AdapterError, "dc_device_operating_points has 5 rows.*max_table_rows=4"):
            adapt_analog_agent(self.root, max_table_rows=4)

    def test_inclusive_limit_and_none_have_identical_canonical_output(self):
        bounded = adapt_analog_agent(self.root, max_table_rows=5)
        self.assertEqual(bounded, adapt_analog_agent(self.root, max_table_rows=None))
        self.assertEqual(bounded["provenance"]["source_counts"]["device_op_rows"], 5)

    def test_reader_streams_instead_of_materializing_raw_tables(self):
        with _read_database(self.db, max_table_rows=5) as source:
            self.assertEqual(source["table_counts"]["dc_device_operating_points"], 5)
            self.assertNotIsInstance(source["dc_device_operating_points"], list)
            self.assertNotIsInstance(source["samples"], list)
            self.assertEqual(sum(1 for _ in source["dc_device_operating_points"]), 5)
            self.assertEqual(sum(1 for _ in source["samples"]), 3)

    def test_invalid_limits_rejected(self):
        for limit in (0, -1, True, 1.5, "5"):
            with self.subTest(limit=limit), self.assertRaisesRegex(AdapterError, "positive integer or None"):
                adapt_analog_agent(self.root, max_table_rows=limit)

    def test_orphan_op_before_first_sample_is_not_silently_lost(self):
        self.change_sql("UPDATE dc_device_operating_points SET sample_id=-1 WHERE sample_id=1")
        with self.assertRaisesRegex(AdapterError, "Orphan OP sample_id: -1"):
            adapt_analog_agent(self.root)

    def test_duplicate_and_orphan_observation_rows_rejected(self):
        self.change_sql("INSERT INTO dc_operating_points SELECT * FROM dc_operating_points WHERE sample_id=1")
        with self.assertRaisesRegex(AdapterError, "Duplicate or orphan dc_operating_points"):
            adapt_analog_agent(self.root)

    def test_orphan_observation_after_last_sample_rejected(self):
        self.change_sql("UPDATE dc_operating_points SET sample_id=999 WHERE sample_id=3")
        with self.assertRaisesRegex(AdapterError, "Duplicate or orphan dc_operating_points"):
            adapt_analog_agent(self.root)

    def test_duplicate_sample_ids_rejected(self):
        self.change_sql("UPDATE samples SET sample_id=1 WHERE sample_id=2")
        with self.assertRaisesRegex(AdapterError, "unique sample_id"):
            adapt_analog_agent(self.root)

    def test_missing_op_group_does_not_shift_later_joins(self):
        self.change_sql("DELETE FROM dc_device_operating_points WHERE sample_id=2")
        data = adapt_analog_agent(self.root)
        self.assertTrue(all(v is None for v in data["samples"][1]["devices"][1]["op"].values()))
        self.assertEqual(data["samples"][2]["devices"][1]["op"]["w_um"], 6)

    def test_cli_limit_is_propagated(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"op_state": {"mode": "preserve_only"}}))
        output = self.root / "output.json"
        arguments = ["import_analog_agent", "--input", str(self.root), "--config", str(config), "--output", str(output)]
        with self.assertRaisesRegex(AdapterError, "max_table_rows=4"):
            main(arguments + ["--max-table-rows", "4"])
        self.assertFalse(output.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            main(arguments + ["--max-table-rows", "none"])
        self.assertEqual(len(json.loads(output.read_text())["samples"]), 3)


class DeckTerminationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "Tiny.sp"
        self.dut = ".subckt DUT A B\nR1 A B 1k\n.ends DUT\n"

    def parse(self, text):
        self.path.write_text(text)
        return parse_spice_netlist(self.path)

    def test_end_after_completed_dut_and_comments_is_accepted(self):
        parsed = self.parse(self.dut + ".end\n* comment only\n\n")
        self.assertEqual(parsed["devices"][0]["device_id"], "R1")

    def test_end_before_dut_rejected(self):
        with self.assertRaisesRegex(AdapterError, "requires a completed DUT"):
            self.parse(".end\n" + self.dut)

    def test_content_after_end_rejected(self):
        for suffix in (".param w=2\n", ".end\n", "R2 A B 2k\n"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(AdapterError, "content after .end"):
                self.parse(self.dut + ".end\n" + suffix)


@unittest.skipUnless(os.environ.get("RUN_ADAPTER_SCALE_TEST") == "1", "opt-in 1,020,000-row synthetic reader check")
class ReaderScaleTests(unittest.TestCase):
    def test_sixty_thousand_samples_with_seventeen_mos_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "scale.sqlite3"
            with sqlite3.connect(db) as connection:
                connection.executescript('''
CREATE TABLE dataset_schema (id INTEGER, circuit_name TEXT, parameter_names_json TEXT, metric_names_json TEXT);
INSERT INTO dataset_schema VALUES (1,'synthetic-scale','["W"]','["gain"]');
CREATE TABLE samples (sample_id INTEGER PRIMARY KEY);
WITH RECURSIVE ids(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<60000)
INSERT INTO samples SELECT n FROM ids;
CREATE TABLE dc_device_operating_points (
 sample_id INTEGER, device_name TEXT, device_type TEXT, model_name TEXT,
 primitive_path TEXT, id_a REAL, PRIMARY KEY(sample_id,device_name));
WITH RECURSIVE devices(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM devices WHERE n<17)
INSERT INTO dc_device_operating_points
 SELECT sample_id,printf('M%02d',n),'NMOS','nmos','synthetic',1e-6
 FROM samples CROSS JOIN devices;
''')
            with self.assertRaisesRegex(AdapterError, "1020000 rows.*max_table_rows=1000000"):
                with _read_database(db):
                    self.fail("default safety limit should reject before returning rows")
            with _read_database(db, max_table_rows=1_020_000) as source:
                self.assertEqual(source["table_counts"]["samples"], 60_000)
                self.assertEqual(source["table_counts"]["dc_device_operating_points"], 1_020_000)
                count = 0
                for row in source["dc_device_operating_points"]:
                    self.assertEqual(row["id_a"], 1e-6)
                    count += 1
                self.assertEqual(count, 1_020_000)


if __name__ == "__main__":
    unittest.main()
