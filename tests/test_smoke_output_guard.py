"""Smoke entrypoints must reject existing output before any experiment writes."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from device_jepa.cli import main
from test_adapter import fixture


ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    paths = [root, *sorted(root.rglob("*"))] if root.is_dir() else [root]
    return [(str(path.relative_to(root)), path.is_dir(), path.stat().st_mtime_ns,
             None if path.is_dir() else path.read_bytes()) for path in paths]


def arguments(command, output, config=None, source=None):
    args = [command, "--output", str(output)]
    if config is not None:
        args += ["--config", str(config)]
    if command == "smoke_real":
        args += ["--input", str(source or output.parent / "source")]
    return args


class SmokeOutputGuardTests(unittest.TestCase):
    def assert_rejected_without_changes(self, command, output):
        before = snapshot(output)
        # The output guard must also precede config reads and data import.
        with patch("device_jepa.cli._read", side_effect=AssertionError("read before guard")), \
             patch("device_jepa.cli.synthetic_dataset", side_effect=AssertionError("generated before guard")), \
             patch("device_jepa.adapter.adapt_analog_agent", side_effect=AssertionError("imported before guard")):
            with self.assertRaisesRegex(ValueError, "new or empty directory"):
                main(arguments(command, output, output.parent / "config.json"))
        self.assertEqual(snapshot(output), before)

    def test_data_and_old_report_without_manifest_are_unchanged(self):
        for command in ("smoke", "smoke_real"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "existing"
                output.mkdir()
                (output / "data.json").write_text('{"old": true}\n')
                (output / "test_report.json").write_text('{"old_report": true}\n')
                self.assert_rejected_without_changes(command, output)
                self.assertFalse((output / "manifest.json").exists())

    def test_partial_and_complete_artifacts_are_unchanged(self):
        cases = [("data.json",), ("manifest.json",),
                 ("data.json", "manifest.json", "pretrain_best.pt", "decoder_best.pt", "test_report.json")]
        for command in ("smoke", "smoke_real"):
            for names in cases:
                with self.subTest(command=command, artifacts=names), tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / "existing"
                    output.mkdir()
                    for name in names:
                        (output / name).write_bytes(b"existing artifact must not be modified")
                    self.assert_rejected_without_changes(command, output)

    def test_nonempty_subdirectory_also_blocks_smoke(self):
        for command in ("smoke", "smoke_real"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "existing"
                (output / "other_run").mkdir(parents=True)
                (output / "other_run" / "evidence.txt").write_text("keep")
                self.assert_rejected_without_changes(command, output)

    def test_output_file_is_preserved(self):
        for command in ("smoke", "smoke_real"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "existing_file"
                output.write_text("keep this file")
                self.assert_rejected_without_changes(command, output)

    def run_fresh_smoke(self, command, existing_empty):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "new_parent" / "output"
            if existing_empty:
                output.mkdir(parents=True)
            config = json.loads((ROOT / "configs/smoke.json").read_text())
            config["training"].update(pretrain_epochs=1, decoder_epochs=1)
            config["model"].update(dim=8, heads=2, ffn_dim=16, decoder_ffn_dim=16,
                                   dropout=0., max_devices=128)
            source = root / "source"
            if command == "smoke_real":
                source.mkdir()
                fixture(source)
                # Make every split of this tiny synthetic SQLite fixture usable:
                # all metrics observed and both MOS OP rows present per sample.
                with sqlite3.connect(source / "sampling_history.sqlite3") as connection:
                    connection.execute("UPDATE samples SET metric_values_json='[20, 1000000, 4]'")
                    connection.execute("UPDATE dc_device_operating_points SET saturation_margin_v=-0.1 WHERE device_name='XN'")
                    row = list(connection.execute(
                        "SELECT * FROM dc_device_operating_points WHERE sample_id=2 AND device_name='XP'"
                    ).fetchone())
                    columns = [r[1] for r in connection.execute("PRAGMA table_info(dc_device_operating_points)")]
                    row[columns.index("sample_id")] = 3
                    row[columns.index("w_um")] = 6.
                    connection.execute("INSERT INTO dc_device_operating_points VALUES ("
                                       + ",".join("?" for _ in row) + ")", row)
                config["dataset"].update(train_circuits=["Tiny"], ood_circuits=[])
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            args = arguments(command, output, config_path, source)
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            for name in ("data.json", "manifest.json", "pretrain_best.pt", "decoder_best.pt", "test_report.json"):
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue(json.loads((output / "test_report.json").read_text())["num_samples"])
            before = snapshot(output)
            with self.assertRaisesRegex(ValueError, "new or empty directory"):
                main(args)
            self.assertEqual(snapshot(output), before)

    def test_new_directories_complete_both_smoke_entrypoints(self):
        for command in ("smoke", "smoke_real"):
            with self.subTest(command=command):
                self.run_fresh_smoke(command, existing_empty=False)

    def test_existing_empty_directories_complete_both_smoke_entrypoints(self):
        for command in ("smoke", "smoke_real"):
            with self.subTest(command=command):
                self.run_fresh_smoke(command, existing_empty=True)


if __name__ == "__main__":
    unittest.main()
