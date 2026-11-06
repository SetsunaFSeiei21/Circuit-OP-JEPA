"""Submission mechanics tests use a fake sbatch and never allocate resources."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from final_exam.common import launch


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_submission_fixture(self, sbatch_body="echo 98765"):
        task = self.root / "final_exam/single_circuit"
        task.mkdir(parents=True)
        script = task / "train.sh"
        shutil.copyfile(Path(__file__).with_name("train.sh"), script)
        config = self.root / "final_exam/configs/default.json"
        config.parent.mkdir()
        config.write_text("{}\n")
        database = self.root / "runs/raw/full_data/Opamp1/sampling_history.sqlite3"
        database.parent.mkdir(parents=True)
        database.touch()
        binary = self.root / "bin/sbatch"
        binary.parent.mkdir()
        binary.write_text("#!/usr/bin/env bash\n" + sbatch_body + "\n")
        binary.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = str(binary.parent) + os.pathsep + environment["PATH"]
        environment.update(SLURM_ACCOUNT="b_phzhwu", SLURM_PARTITION="ex01A800",
                           SLURM_CPUS="8", SLURM_GPUS="1", SLURM_MEM="80G")
        return script, task, environment

    def test_dry_run_creates_nothing_and_selects_gpu_smoke_resources(self):
        script, task, environment = self.make_submission_fixture("exit 99")
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        result = subprocess.run(["bash", str(script), "--dry-run", "--smoke"],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")))
        self.assertIn("--cpus-per-task=8", result.stdout)
        self.assertIn("--gres=gpu:1", result.stdout)
        self.assertIn("--time=00:30:00", result.stdout)
        self.assertIn("--partition=ex01A800", result.stdout)
        self.assertIn(str(task / "hpc_logs"), result.stdout)
        self.assertIn("--device cuda", result.stdout)
        self.assertIn("student_predicted_op_distill", result.stdout)

    def test_submit_failure_returns_original_exit_code(self):
        script, task, environment = self.make_submission_fixture("echo 'allocation rejected' >&2\nexit 9")
        result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 9)
        self.assertIn("allocation rejected", result.stderr)
        self.assertNotIn("Submitted job", result.stdout)
        self.assertFalse((task / "results").exists())
        self.assertFalse((task / "logs").exists())

    def test_submission_records_job_and_shared_run_id(self):
        script, task, environment = self.make_submission_fixture("echo '98765;cluster'")
        result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = list(task.glob("hpc_logs/*/submission.txt"))
        self.assertEqual(len(records), 1)
        self.assertIn("job_id=98765;cluster", records[0].read_text())
        self.assertEqual(json.loads((records[0].parent / "submitted_config.json").read_text()), {})
        run_id = records[0].parent.name
        self.assertIn(f"--run-id {run_id}", result.stdout)
        self.assertIn(str(task / "logs" / run_id), result.stdout)
        self.assertIn(str(task / "results" / run_id), result.stdout)
        self.assertIn(str(records[0].parent / "submitted_config.json"), result.stdout)

    def test_all_inputs_checked_before_first_submission(self):
        script, task, environment = self.make_submission_fixture("echo 98765")
        script.write_text(script.read_text().replace("CIRCUITS=(Opamp1)", "CIRCUITS=(Opamp1 MissingCircuit)"))
        result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MissingCircuit", result.stderr)
        self.assertFalse((task / "hpc_logs").exists())

    def test_existing_result_rejected_before_config_or_import(self):
        task = self.root / "single_circuit"
        output = task / "results/occupied"
        output.mkdir(parents=True)
        sentinel = output / "data.json"
        sentinel.write_text("original data")
        with patch.object(launch, "TASK_ROOT", task), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                launch.main(["--circuit", "Opamp1", "--run-id", "occupied",
                             "--config", str(self.root / "absent.json"), "--models", "mlp"])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(sentinel.read_text(), "original data")
        self.assertFalse((task / "logs").exists())

    def test_existing_log_and_escaped_output_are_rejected(self):
        task = self.root / "single_circuit"
        existing = task / "logs/occupied"
        existing.mkdir(parents=True)
        with patch.object(launch, "TASK_ROOT", task):
            with self.assertRaisesRegex(ValueError, "existing logs"):
                launch.new_run_paths("occupied")
        outside = self.root / "outside"
        outside.mkdir()
        (task / "results").symlink_to(outside, target_is_directory=True)
        with patch.object(launch, "TASK_ROOT", task):
            with self.assertRaisesRegex(ValueError, "remain under"):
                launch.new_run_paths("fresh")
        self.assertEqual(list(outside.iterdir()), [])

    def test_process_logging_preserves_stderr_and_failure(self):
        stdout, stderr, logfile = io.StringIO(), io.StringIO(), io.StringIO()
        command = [sys.executable, "-u", "-c",
                   "import sys; print('stdout-value'); print('stderr-value', file=sys.stderr); sys.exit(7)"]
        with patch.object(launch.sys, "stdout", stdout), patch.object(launch.sys, "stderr", stderr):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                launch.run_logged(command, logfile, threading.Lock())
        self.assertEqual(raised.exception.returncode, 7)
        self.assertEqual(stderr.getvalue(), "stderr-value\n")
        self.assertIn("stdout-value\n", stdout.getvalue())
        self.assertIn("stdout-value\n", logfile.getvalue())
        self.assertIn("stderr-value\n", logfile.getvalue())

    def make_raw_fixture(self):
        raw = self.root / "raw/NamedCircuit"
        raw.mkdir(parents=True)
        with sqlite3.connect(raw / "sampling_history.sqlite3") as database:
            database.execute("CREATE TABLE dataset_schema(circuit_name TEXT)")
            database.execute("INSERT INTO dataset_schema VALUES ('NamedCircuit')")
            database.execute("CREATE TABLE samples(sample_id INTEGER)")
            database.execute("INSERT INTO samples VALUES (1)")
        (raw / "NamedCircuit.sp").write_text(".subckt NamedCircuit\n.ends\n")
        config = self.root / "config.json"
        config.write_text(json.dumps({"dataset": {"train_circuits": ["other"], "ood_circuits": ["other"]}}))
        arguments = ["--circuit", "NamedCircuit", "--run-id", "fresh", "--raw-root", str(raw.parent),
                     "--config", str(config), "--models", "mlp", "--seeds", "17", "--device", "cpu"]
        return arguments

    def test_raw_launch_keeps_selection_and_links_logs_to_results(self):
        arguments = self.make_raw_fixture()
        task = self.root / "single_circuit"
        commands = []

        def fake_command(command, log, lock):
            command = list(map(str, command))
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            if "import_analog_agent" in command:
                output.write_text('{"imported":true}')
            else:
                output.mkdir()
                (output / "run.json").write_text('{"status":"complete"}')
                (output / "summary.xlsx").write_bytes(b"fixture workbook")

        with patch.object(launch, "TASK_ROOT", task), \
                patch("final_exam.run.validate_request", side_effect=lambda config, models, *args, **kwargs: (config, models)), \
                patch.object(launch, "run_logged", side_effect=fake_command), \
                contextlib.redirect_stdout(io.StringIO()):
            output = launch.main(arguments)
        record = json.loads((output / "run.json").read_text())
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["models"], ["mlp"])
        self.assertEqual(config["dataset"], {"train_circuits": ["NamedCircuit"], "ood_circuits": []})
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[1][commands[1].index("--log-dir") + 1], str(task / "logs/fresh"))
        self.assertEqual(record["summary_xlsx"], str(output / "training/summary.xlsx"))
        self.assertTrue((task / "logs/fresh/launch.log").exists())

    def test_import_failure_records_failed_status_and_original_code(self):
        arguments = self.make_raw_fixture()
        task = self.root / "single_circuit"
        with patch.object(launch, "TASK_ROOT", task), \
                patch("final_exam.run.validate_request", side_effect=lambda config, models, *args, **kwargs: (config, models)), \
                patch.object(launch, "run_logged", side_effect=subprocess.CalledProcessError(7, ["import"])), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                launch.main(arguments)
        record = json.loads((task / "results/fresh/run.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["exit_code"], 7)
        self.assertIn("CalledProcessError", (task / "logs/fresh/launch.log").read_text())
        self.assertFalse((task / "results/fresh/training").exists())

    def test_real_raw_import_accepts_seed_only_training_config(self):
        """Exercise both child CLIs; final configs must not hit legacy JEPA checks."""
        from tests.test_adapter import fixture

        raw = self.root / "raw/Tiny"
        raw.mkdir(parents=True)
        fixture(raw)
        # Extend the adapter fixture with distinct designs and finite labels so
        # a deterministic grouped split has train, validation and test data.
        with sqlite3.connect(raw / "sampling_history.sqlite3") as database:
            for sample_id in range(1, 13):
                width = 2.0 * sample_id
                metrics = json.dumps([10.0 + sample_id, 1e6 * sample_id, 2.0 + sample_id])
                if sample_id <= 3:
                    database.execute("UPDATE samples SET metric_values_json=? WHERE sample_id=?",
                                     (metrics, sample_id))
                    continue
                database.execute(
                    "INSERT INTO samples SELECT ?, ?, ?, ?, success, run_id, observation_json "
                    "FROM samples WHERE sample_id=2",
                    (sample_id, f"design-{sample_id}", json.dumps([width, 0.5, 3.0, 1e-6]), metrics))
                for table in ("dc_operating_points", "dc_device_operating_points"):
                    columns = [row[1] for row in database.execute(f"PRAGMA table_info({table})")]
                    expressions = ["?" if name == "sample_id" else name for name in columns]
                    parameters = [sample_id]
                    if table == "dc_device_operating_points":
                        expressions[columns.index("w_um")] = "?"
                        parameters.append(width)
                    database.execute(f"INSERT INTO {table} SELECT {','.join(expressions)} "
                                     f"FROM {table} WHERE sample_id=2", parameters)
            # The adapter-only fixture intentionally carries a conflicting raw
            # failure reason. Use consistent collector evidence for this full
            # CLI test, which also applies OP-state validation.
            database.execute("UPDATE samples SET observation_json=?",
                             (json.dumps({"checked_devices": ["XN"], "reasons": []}),))
            database.execute("UPDATE dc_operating_points SET reasons_json='[]'")
        config = json.loads((launch.PROJECT_ROOT / "final_exam/configs/default.json").read_text())
        self.assertEqual(config["training"], {"seed": 17})
        config["dataset"]["split_ratios"] = [0.6, 0.2, 0.2]
        config["model"].update(dim=8, heads=2, encoder_layers=2, cross_layers=1,
                               ffn_dim=16, decoder_ffn_dim=16, decoder_layers=1,
                               max_devices=16, dropout=0.0)
        config["op_transfer"]["batch_size"] = 8
        configuration = self.root / "seed_only.json"
        configuration.write_text(json.dumps(config))
        task = self.root / "single_circuit"
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with patch.object(launch, "TASK_ROOT", task), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                output = launch.main([
                    "--circuit", "Tiny", "--raw-root", str(raw.parent), "--run-id", "real_raw",
                    "--config", str(configuration), "--models", "mlp", "--seeds", "17",
                    "--device", "cpu", "--smoke",
                ])
        except subprocess.CalledProcessError:
            self.fail(stderr.getvalue())
        record = json.loads((output / "run.json").read_text())
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["source_samples"], 12)
        self.assertEqual(len(json.loads((output / "data.json").read_text())["samples"]), 12)
        staged = json.loads((output / "config.json").read_text())
        effective = json.loads((output / "training/config.json").read_text())
        self.assertEqual(staged, effective, "Applying --smoke twice must be idempotent")
        self.assertEqual(effective["training"], {"seed": 17})
        self.assertEqual(effective["op_transfer"]["epochs"]["readout"], 2)
        self.assertTrue((output / "training/summary.xlsx").is_file())
        self.assertTrue((task / "logs/real_raw/launch.log").is_file())
        self.assertTrue((task / "logs/real_raw/experiment.log").is_file())
        self.assertFalse((output / "training/seed_17/shared").exists())


if __name__ == "__main__":
    unittest.main()
