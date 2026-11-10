"""Run real Bash entrypoints from simulated Slurm spool copies, without GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    ("single_circuit", "run.sh", "launch.py"),
    ("single_circuit_pin", "run_pin.sh", "launch_pin.py"),
)


class SlurmEntrypointPinTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.root = self.workspace / "project with spaces"
        self.root.mkdir()
        self.capture = self.workspace / "launch.json"

    def fixture(self, entrypoint, *, copied=True):
        task, runner, launcher = entrypoint
        source = PROJECT_ROOT / "final_exam" / task / runner
        destination = self.root / "final_exam" / task / runner
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        launch = self.root / "final_exam/common" / launcher
        launch.parent.mkdir(parents=True, exist_ok=True)
        launch.write_text(
            "import json, os, pathlib, sys\n"
            "record = dict(cwd=os.getcwd(), argv=sys.argv, "
            "root=os.environ.get('JEPA_PROJECT_ROOT'), "
            "pythonpath=os.environ.get('PYTHONPATH'))\n"
            "pathlib.Path(os.environ['ENTRYPOINT_CAPTURE']).write_text(json.dumps(record))\n"
            "sys.exit(int(os.environ.get('ENTRYPOINT_EXIT_CODE', '0')))\n"
        )
        if copied:
            destination = self.workspace / "var/spool/slurmd/job42/slurm_script"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        environment = os.environ.copy()
        for name in ("JEPA_PROJECT_ROOT", "SLURM_SUBMIT_DIR", "SLURM_JOB_ID",
                     "SLURM_JOB_NODELIST", "JEPA_CONDA_ENV", "JEPA_CONDA_HOME",
                     "ENTRYPOINT_EXIT_CODE"):
            environment.pop(name, None)
        environment.update(JEPA_PYTHON=sys.executable, PYTHONPATH="previous_imports",
                           ENTRYPOINT_CAPTURE=str(self.capture))
        self.capture.unlink(missing_ok=True)
        return destination, launch, environment

    def check_launch(self, script, launcher, environment, *, cwd, exit_code=0):
        arguments = ["--probe", "argument with spaces"]
        result = subprocess.run(["bash", str(script), *arguments], cwd=cwd,
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, exit_code, result.stderr)
        record = json.loads(self.capture.read_text())
        self.assertEqual(record["cwd"], str(self.root))
        self.assertEqual(record["root"], str(self.root))
        self.assertEqual(record["argv"], [str(launcher), *arguments])
        self.assertEqual(record["pythonpath"], str(self.root) + os.pathsep + "previous_imports")
        self.assertIn(f"Project root: {self.root}", result.stdout)

    def test_spooled_script_uses_exported_project_root(self):
        for entry in ENTRYPOINTS:
            with self.subTest(entry=entry):
                script, launcher, environment = self.fixture(entry)
                environment.update(JEPA_PROJECT_ROOT=str(self.root), SLURM_JOB_ID="42",
                                   SLURM_SUBMIT_DIR=str(self.workspace))
                self.check_launch(script, launcher, environment, cwd=self.workspace)

    def test_spooled_script_uses_sbatch_working_directory(self):
        for entry in ENTRYPOINTS:
            with self.subTest(entry=entry):
                script, launcher, environment = self.fixture(entry)
                environment.update(SLURM_JOB_ID="42", SLURM_SUBMIT_DIR=str(self.workspace))
                self.check_launch(script, launcher, environment, cwd=self.root)

    def test_spooled_script_uses_submit_directory_fallback(self):
        for entry in ENTRYPOINTS:
            with self.subTest(entry=entry):
                script, launcher, environment = self.fixture(entry)
                environment.update(SLURM_JOB_ID="42", SLURM_SUBMIT_DIR=str(self.root))
                self.check_launch(script, launcher, environment, cwd=self.workspace)

    def test_direct_local_entrypoint_works_from_another_directory(self):
        for entry in ENTRYPOINTS:
            with self.subTest(entry=entry):
                script, launcher, environment = self.fixture(entry, copied=False)
                self.check_launch(script, launcher, environment, cwd=self.workspace)

    def test_invalid_explicit_root_is_rejected_before_python(self):
        for entry in ENTRYPOINTS:
            with self.subTest(entry=entry):
                script, _, environment = self.fixture(entry)
                environment.update(JEPA_PROJECT_ROOT=str(self.workspace / "missing"),
                                   SLURM_SUBMIT_DIR=str(self.root))
                result = subprocess.run(["bash", str(script)], cwd=self.root,
                                        env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Cannot locate final_exam project root", result.stderr)
                self.assertFalse(self.capture.exists())

    def test_spooled_script_without_a_project_fails_before_python(self):
        for entry in ENTRYPOINTS:
            with self.subTest(entry=entry):
                script, _, environment = self.fixture(entry)
                environment.update(SLURM_JOB_ID="42", SLURM_SUBMIT_DIR=str(self.workspace))
                result = subprocess.run(["bash", str(script)], cwd=self.workspace,
                                        env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Cannot locate final_exam project root", result.stderr)
                self.assertFalse(self.capture.exists())

    def test_python_failure_status_is_preserved(self):
        for entry in ENTRYPOINTS:
            with self.subTest(entry=entry):
                script, launcher, environment = self.fixture(entry)
                environment.update(JEPA_PROJECT_ROOT=str(self.root), ENTRYPOINT_EXIT_CODE="7")
                self.check_launch(script, launcher, environment, cwd=self.workspace, exit_code=7)

    def test_all_submitters_export_root_and_execute_the_spooled_entrypoint(self):
        cases = (
            (ENTRYPOINTS[0], "train.sh", "default.json"),
            (ENTRYPOINTS[1], "train_pin.sh", "default_pin.json"),
            (ENTRYPOINTS[1], "train_op_gate_pin.sh", "op_gate_pin.json"),
        )
        for entry, submitter, config_name in cases:
            with self.subTest(submitter=submitter):
                _, launcher, environment = self.fixture(entry, copied=False)
                script = self.root / "final_exam" / entry[0] / submitter
                shutil.copyfile(PROJECT_ROOT / "final_exam" / entry[0] / submitter, script)
                config = self.root / "final_exam/configs" / config_name
                config.parent.mkdir(exist_ok=True)
                config.write_text("{}\n")
                raw = self.root / "runs/raw/full_data/Opamp1/sampling_history.sqlite3"
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.touch()
                conda = self.workspace / "conda/etc/profile.d/conda.sh"
                conda.parent.mkdir(parents=True, exist_ok=True)
                conda.write_text('conda() { [[ "$1" == activate && "$2" == fixture ]]; }\n')
                sbatch = self.workspace / "bin/sbatch"
                sbatch.parent.mkdir(exist_ok=True)
                sbatch.write_text(
                    f"#!{sys.executable}\n"
                    "import os, pathlib, shutil, subprocess, sys\n"
                    "args = sys.argv[1:]\n"
                    "cwd = next(a.split('=', 1)[1] for a in args if a.startswith('--chdir='))\n"
                    "assert '--export=ALL' in args\n"
                    "assert os.environ.get('JEPA_PROJECT_ROOT') == cwd\n"
                    "index = next(i for i, a in enumerate(args) if a.endswith(('/run.sh', '/run_pin.sh')))\n"
                    "spool = pathlib.Path(os.environ['ENTRYPOINT_CAPTURE']).parent / 'var/spool/slurmd/job99/slurm_script'\n"
                    "spool.parent.mkdir(parents=True, exist_ok=True)\n"
                    "shutil.copyfile(args[index], spool)\n"
                    "env = dict(os.environ, SLURM_SUBMIT_DIR=os.getcwd(), SLURM_JOB_ID='99')\n"
                    "result = subprocess.run(['bash', str(spool), *args[index+1:]], cwd=cwd, env=env, "
                    "capture_output=True, text=True)\n"
                    "sys.stderr.write(result.stderr)\n"
                    "if result.returncode: sys.exit(result.returncode)\n"
                    "print('99')\n"
                )
                sbatch.chmod(0o755)
                environment.update(
                    PATH=str(sbatch.parent) + os.pathsep + environment["PATH"],
                    JEPA_CONDA_HOME=str(self.workspace / "conda"), JEPA_CONDA_ENV="fixture",
                    JEPA_PROJECT_ROOT="stale-parent-root", SLURM_CPUS="8", SLURM_GPUS="1",
                    SLURM_ACCOUNT="b_phzhwu", SLURM_PARTITION="ex01A800",
                )
                result = subprocess.run(["bash", str(script)], cwd=self.workspace,
                                        env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Submitted job 99", result.stdout)
                record = json.loads(self.capture.read_text())
                self.assertEqual(record["root"], str(self.root))
                self.assertEqual(record["cwd"], str(self.root))
                self.assertEqual(record["argv"][0], str(launcher))
                arguments = record["argv"]
                self.assertEqual(arguments[arguments.index("--circuit") + 1], "Opamp1")
                self.assertEqual(arguments[arguments.index("--device") + 1], "cuda")
                snapshot = Path(arguments[arguments.index("--config") + 1])
                self.assertEqual(json.loads(snapshot.read_text()), {})


if __name__ == "__main__":
    unittest.main()
