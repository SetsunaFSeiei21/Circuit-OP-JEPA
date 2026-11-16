"""Submission routing and artifact boundaries; runner preflight is tested separately."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / 'project with spaces'
    here = root / 'final_exam/single_circuit_pin'
    here.mkdir(parents=True)
    (root / 'scripts/run').mkdir(parents=True)
    shutil.copy(ROOT / 'scripts/run/slurm_run.sh', root / 'scripts/run/slurm_run.sh')
    script = here / 'inspect_shared_decoder_pin.sh'
    source = (ROOT / 'final_exam/single_circuit_pin/inspect_shared_decoder_pin.sh').read_text()
    for circuit in ('Opamp1', 'two_stage_folded_opamp'):
        directory = root / circuit
        checkpoint = directory / 'training_pin/seed_17_pin/width_decoder_pin/checkpoint_pin.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'never deserialize in launcher test')
        source = source.replace(f'""  # {circuit}:', f'"{directory}"  # {circuit}:')
    script.write_text(source)
    binary = root / 'bin'
    binary.mkdir()
    sbatch = binary / 'sbatch'
    sbatch.write_text('#!' + sys.executable + '\nimport json,sys\nfrom pathlib import Path\n'
        + f'Path({str(root / "submitted.json")!r}).write_text(json.dumps(sys.argv[1:]))\nprint("123456")\n')
    sbatch.chmod(0o755)
    # A lightweight mock validates that every source reaches the preflight
    # before sbatch/HPC directory creation. Actual metadata validation belongs
    # to shared_decoder_probe tests, not this shell orchestration test.
    inspect = binary / 'inspect_python'
    inspect.write_text('#!' + sys.executable + '\nimport json,sys\nfrom pathlib import Path\n'
        + 'assert "--inspect-source" in sys.argv\n'
        + 'args=sys.argv[1:]\n'
        + 'for i,arg in enumerate(args):\n'
        + '    if arg == "--source":\n'
        + '        path=Path(args[i+1].split("=",1)[1])/"training_pin/seed_17_pin/width_decoder_pin/checkpoint_pin.pt"\n'
        + '        if not path.is_file():\n'
        + '            print("checkpoint is missing: "+str(path),file=sys.stderr); sys.exit(2)\n'
        + f'Path({str(root / "preflight.json")!r}).write_text(json.dumps(args))\n')
    inspect.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               JEPA_INSPECT_PYTHON=str(inspect))

    def run(*args):
        return subprocess.run(['bash', str(script), *args], cwd=tmp_path,
                              env=env, text=True, capture_output=True)
    return root, here, script, run


def test_dry_run_no_files_and_default_protocol(launcher):
    root, here, script, run = launcher
    before = set(root.rglob('*'))
    result = run('--dry-run')
    assert result.returncode == 0, result.stderr
    assert '--cpus-per-task=8' in result.stdout and '--gres=gpu:1' in result.stdout
    assert 'shared_decoder_probe_pin.py' in result.stdout
    assert '--num-samples 128' in result.stdout and '--num-val-samples 128' in result.stdout
    assert '--validation-scope all' in result.stdout and '--component performance' in result.stdout
    assert before == set(root.rglob('*'))
    assert run('--help').returncode == 0


def test_submission_preflight_snapshot_and_paths(launcher):
    root, here, script, run = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    assert 'Submitted shared decoder probe job 123456' in result.stdout
    args = json.loads((root / 'submitted.json').read_text())
    assert str(root / 'scripts/run/slurm_run.sh') in args
    assert str(root / 'final_exam/shared_decoder_probe_pin.py') in args
    assert 'Opamp1=' + str(root / 'Opamp1') in args
    assert 'two_stage_folded_opamp=' + str(root / 'two_stage_folded_opamp') in args
    assert '--inspect-source' not in args
    preflight = json.loads((root / 'preflight.json').read_text())
    assert '--inspect-source' in preflight
    snapshots = list((here / 'hpc_logs_pin').glob('*/submitted_inspect_shared_decoder_pin.sh'))
    assert len(snapshots) == 1 and snapshots[0].read_bytes() == script.read_bytes()
    assert not (here / 'results_pin').exists() and not (here / 'logs_pin').exists()


def test_missing_second_source_stops_before_submission(launcher):
    root, here, script, run = launcher
    next((root / 'two_stage_folded_opamp').rglob('checkpoint_pin.pt')).unlink()
    result = run()
    assert result.returncode != 0 and 'checkpoint is missing' in result.stderr
    assert not (root / 'submitted.json').exists()
    assert not (here / 'hpc_logs_pin').exists()


def test_local_dry_run_and_weights_only_flags(launcher):
    root, here, script, run = launcher
    script.write_text(script.read_text().replace('N_SAMPLES=128', 'N_SAMPLES=0')
                      .replace('N_VAL_SAMPLES=128', 'N_VAL_SAMPLES=0'))
    result = run('--local', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith('bash ')
    assert 'sbatch ' not in result.stdout
    assert '--num-samples 0' in result.stdout and '--num-val-samples 0' in result.stdout
    assert not (root / 'submitted.json').exists() and not (root / 'preflight.json').exists()


def test_invalid_setting_fails_even_in_preview(launcher):
    root, here, script, run = launcher
    script.write_text(script.read_text().replace('VALIDATION_SCOPE="all"', 'VALIDATION_SCOPE="typo"'))
    result = run('--dry-run')
    assert result.returncode != 0 and 'Invalid VALIDATION_SCOPE' in result.stderr
    assert not (here / 'hpc_logs_pin').exists()
