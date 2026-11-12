"""Probe launch path survives Slurm relocation and never starts training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODE = 'device_op_supervised_gate_fusion_pin_v2'


class GateProbeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'project with spaces'
        self.here = self.root / 'final_exam/single_circuit_pin'
        self.here.mkdir(parents=True)
        for name in ('gate_probe_pin.py', 'gate_version_registry_pin.py', 'staged_gate_registry_pin.py'):
            shutil.copy(ROOT / 'final_exam' / name, self.root / 'final_exam' / name)
        self.script = self.here / 'inspect_op_gate_pin.sh'
        shutil.copy(ROOT / 'final_exam/single_circuit_pin/inspect_op_gate_pin.sh', self.script)
        (self.root / 'scripts/run').mkdir(parents=True)
        shutil.copy(ROOT / 'scripts/run/slurm_run.sh', self.root / 'scripts/run/slurm_run.sh')
        self.source = self.root / 'saved experiment'
        train = self.source / 'training_pin'
        ckpt = train / f'seed_17_pin/{MODE}/checkpoint_pin.pt'
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b'preflight must not deserialize this')
        for name, data in {
            'run_pin.json': dict(granularity='pin', circuit='two_stage_folded_opamp',
                models=[MODE], seeds=[17], dataset_fingerprint='saved_digest'),
            'config_pin.json': dict(dataset=dict(train_circuits=['two_stage_folded_opamp'], ood_circuits=[])),
            'manifest_pin.json': dict(test=['sample_1', 'sample_2']),
        }.items():
            (train / name).write_text(json.dumps(data))
        (self.source / 'data_pin.json').write_text('preflight must not parse full data')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.environment = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                                JEPA_INSPECT_PYTHON=sys.executable)
        self.environment.pop('SOURCE_RUN', None)

    def run_script(self, *args):
        return subprocess.run(['bash', str(self.script), *map(str, args)], cwd=self.tmp.name,
                              env=self.environment, text=True, capture_output=True)

    def test_dry_run_and_help_create_nothing(self):
        before = set(self.root.rglob('*'))
        run = self.run_script(self.source, '--dry-run')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('--cpus-per-task=8', run.stdout)
        self.assertIn('--gres=gpu:1', run.stdout)
        self.assertIn('final_exam/gate_probe_pin.py', run.stdout)
        self.assertNotIn('run_pin.py', run.stdout)
        self.assertEqual(self.run_script('--help').returncode, 0)
        self.assertEqual(before, set(self.root.rglob('*')))

    def test_submit_has_absolute_paths_and_metadata_preflight(self):
        log = self.root / 'submission_arguments.json'
        fake = self.bin / 'sbatch'
        fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
            + 'from pathlib import Path\n'
            + f'Path({str(log)!r}).write_text(json.dumps(dict(args=sys.argv[1:], root=os.environ["CIRCUIT_JEPA_ROOT"])))\n'
            + 'print("12345")\n')
        fake.chmod(0o755)
        run = self.run_script(self.source)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('Submitted probe job 12345', run.stdout)
        command = json.loads(log.read_text())
        self.assertEqual(command['root'], str(self.root))
        args = command['args']
        self.assertIn(str(self.root / 'scripts/run/slurm_run.sh'), args)
        self.assertIn(str(self.root / 'final_exam/gate_probe_pin.py'), args)
        self.assertEqual(args[args.index('--source-run') + 1], str(self.source))
        self.assertEqual(args[args.index('--models') + 1], MODE)
        self.assertEqual(len(list((self.here / 'hpc_logs_pin').glob('*/submission_pin.txt'))), 1)
        self.assertFalse((self.here / 'results_pin').exists())
        self.assertFalse((self.here / 'logs_pin').exists())

    def test_missing_checkpoint_fails_before_submission_or_log_creation(self):
        next(self.source.rglob('checkpoint_pin.pt')).unlink()
        run = self.run_script(self.source)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('checkpoint is missing', run.stderr)
        self.assertFalse((self.here / 'hpc_logs_pin').exists())

    def test_metadata_preflight_runs_without_site_packages(self):
        run = subprocess.run([sys.executable, '-S', str(self.root / 'final_exam/gate_probe_pin.py'),
            '--source-run', str(self.source), '--circuit', 'two_stage_folded_opamp',
            '--models', MODE, '--inspect-source'], text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['seed'], 17)

    def test_spooled_wrapper_uses_exported_root(self):
        (self.root / 'device_jepa').mkdir()
        (self.root / 'device_jepa/cli.py').touch()
        spool = Path(self.tmp.name) / 'spool/job'
        spool.mkdir(parents=True)
        shutil.copy(self.root / 'scripts/run/slurm_run.sh', spool / 'slurm_script')
        env = dict(self.environment, CIRCUIT_JEPA_ROOT=str(self.root), JEPA_CONDA_ENV='')
        run = subprocess.run(['bash', str(spool / 'slurm_script'), sys.executable,
            '-c', 'from pathlib import Path; print(Path.cwd())'], cwd=spool, env=env,
            text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines()[-1], str(self.root))


if __name__ == '__main__':
    unittest.main()
