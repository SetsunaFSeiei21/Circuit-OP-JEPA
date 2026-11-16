"""Cluster submission stays read-only and preflights every paired source."""
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
    root = tmp_path/'project with spaces'
    here = root/'final_exam/single_circuit_pin'
    here.mkdir(parents=True)
    for name in ('spectral_probe_pin.py','gate_probe_pin.py'):
        shutil.copy(ROOT/'final_exam'/name, root/'final_exam'/name)
    (root/'scripts/run').mkdir(parents=True)
    shutil.copy(ROOT/'scripts/run/slurm_run.sh',root/'scripts/run/slurm_run.sh')
    script = here/'inspect_spectrum_pin.sh'
    text = (ROOT/'final_exam/single_circuit_pin/inspect_spectrum_pin.sh').read_text()
    for circuit in ('Opamp1','two_stage_folded_opamp'):
        source = root/circuit
        training = source/'training_pin'
        training.mkdir(parents=True)
        for mode in ('device_op_cascade_pin','width_decoder_pin','width_decoder_op_pin'):
            path = training/f'seed_17_pin/{mode}/checkpoint_pin.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'metadata preflight must not deserialize this')
        (source/'data_pin.json').write_text('preflight must not read full data')
        (training/'run_pin.json').write_text(json.dumps(dict(circuit=circuit,granularity='pin',
            models=['device_op_cascade_pin','width_decoder_pin','width_decoder_op_pin'],
            seeds=[17],dataset_fingerprint='fingerprint')))
        (training/'manifest_pin.json').write_text(json.dumps(dict(train=['train0'])))
        (training/'config_pin.json').write_text(json.dumps(dict(dataset=dict(train_circuits=[circuit],ood_circuits=[]))))
    text = text.replace('""  # Opamp1:', f'"{root}/Opamp1"  # Opamp1:')
    text = text.replace('""  # two_stage_folded_opamp:', f'"{root}/two_stage_folded_opamp"  # two_stage_folded_opamp:')
    script.write_text(text)
    binary = root/'bin'
    binary.mkdir()
    sbatch = binary/'sbatch'
    sbatch.write_text('#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\n'
        + f'Path({str(root/"submitted.json")!r}).write_text(json.dumps(sys.argv[1:]))\nprint("123456")\n')
    sbatch.chmod(0o755)
    env = dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],JEPA_INSPECT_PYTHON=sys.executable)
    def run(*args):
        return subprocess.run(['bash',str(script),*args],cwd=tmp_path,env=env,text=True,capture_output=True)
    return root,here,script,run


def test_dry_run_no_files_and_gpu_allocation(launcher):
    root,here,script,run = launcher
    before = set(root.rglob('*'))
    result = run('--dry-run')
    assert result.returncode == 0, result.stderr
    assert '--cpus-per-task=8' in result.stdout and '--gres=gpu:1' in result.stdout
    assert 'spectral_probe_pin.py' in result.stdout and '--num-samples 128' in result.stdout
    assert before == set(root.rglob('*'))
    assert run('--help').returncode == 0


def test_submission_checks_all_sources_and_snapshots_script(launcher):
    root,here,script,run = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    assert 'Submitted spectral probe job 123456' in result.stdout
    arguments = json.loads((root/'submitted.json').read_text())
    assert str(root/'scripts/run/slurm_run.sh') in arguments
    assert str(root/'final_exam/spectral_probe_pin.py') in arguments
    assert 'Opamp1='+str(root/'Opamp1') in arguments
    assert len(list((here/'hpc_logs_pin').glob('*/submitted_inspect_spectrum_pin.sh'))) == 1
    assert not (here/'results_pin').exists() and not (here/'logs_pin').exists()


def test_missing_second_source_checkpoint_never_submits(launcher):
    root,here,script,run = launcher
    next((root/'two_stage_folded_opamp').rglob('checkpoint_pin.pt')).unlink()
    result = run()
    assert result.returncode != 0
    assert 'checkpoint is missing' in result.stderr
    assert not (root/'submitted.json').exists()
    assert not (here/'hpc_logs_pin').exists()
