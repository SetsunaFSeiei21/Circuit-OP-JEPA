"""Submission preserves quoted run lists, separation of logs, and read-only preflight."""
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
    for name in ('prediction_probe_pin.py', 'spectral_probe_pin.py', 'gate_probe_pin.py'):
        shutil.copy(ROOT / 'final_exam' / name, root / 'final_exam' / name)
    (root / 'scripts/run').mkdir(parents=True)
    shutil.copy(ROOT / 'scripts/run/slurm_run.sh', root / 'scripts/run/slurm_run.sh')
    sources = []
    for circuit in ('Opamp1', 'two_stage_folded_opamp'):
        source = root / circuit
        training = source / 'training_pin'
        target = training / 'seed_17_pin/device_op_cascade_pin'
        target.mkdir(parents=True)
        (target / 'checkpoint_pin.pt').write_bytes(b'preflight must not deserialize')
        (target / 'test_pin.json').write_text(json.dumps(dict(split='test', circuit=circuit, metrics={})))
        (source / 'data_pin.json').write_text('preflight must not parse large canonical data')
        (training / 'run_pin.json').write_text(json.dumps(dict(circuit=circuit, granularity='pin',
            models=['device_op_cascade_pin'], seeds=[17], dataset_fingerprint='fingerprint')))
        (training / 'manifest_pin.json').write_text(json.dumps(dict(dataset_fingerprint='fingerprint', ood=[])))
        (training / 'config_pin.json').write_text(json.dumps(dict(dataset=dict(train_circuits=[circuit]))))
        sources.append(source)
    script = here / 'inspect_predictions_pin.sh'
    text = (ROOT / 'final_exam/single_circuit_pin/inspect_predictions_pin.sh').read_text()
    text = text.replace('    ""  # /share/.../results_pin/two_stage_opamp_otaf_lasd_moe_..._pin',
                        '\n'.join(f'    "{source}"' for source in sources))
    script.write_text(text)
    binary = root / 'bin'
    binary.mkdir()
    sbatch = binary / 'sbatch'
    sbatch.write_text('#!' + sys.executable + '\nimport json,sys\nfrom pathlib import Path\n' +
        f'Path({str(root / "submitted.json")!r}).write_text(json.dumps(sys.argv[1:]))\nprint("123456")\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               JEPA_INSPECT_PYTHON=sys.executable, SLURM_GPUS='1', DEVICE='cuda')
    def run(*args):
        return subprocess.run(['bash', str(script), *args], cwd=tmp_path, env=env, text=True, capture_output=True)
    return root, here, sources, run


def test_dry_run_writes_nothing(launcher):
    root, here, sources, run = launcher
    before = set(root.rglob('*'))
    result = run('--dry-run')
    assert result.returncode == 0, result.stderr
    assert '--gres=gpu:1' in result.stdout and '--splits val test' in result.stdout
    assert set(root.rglob('*')) == before
    assert run('--help').returncode == 0


def test_submit_one_job_for_all_sources_and_snapshot(launcher):
    root, here, sources, run = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    args = json.loads((root / 'submitted.json').read_text())
    assert args.count('--source-runs') == 1
    assert all(str(source) in args for source in sources)
    assert str(root / 'scripts/run/slurm_run.sh') in args
    assert str(root / 'final_exam/prediction_probe_pin.py') in args
    assert len(list((here / 'hpc_logs_pin').glob('*/submitted_inspect_predictions_pin.sh'))) == 1
    assert not (here / 'results_pin').exists() and not (here / 'logs_pin').exists()


def test_bad_second_source_blocks_submission(launcher):
    root, here, sources, run = launcher
    (sources[1] / 'training_pin/manifest_pin.json').write_text('{}')
    result = run()
    assert result.returncode != 0 and 'fingerprints disagree' in result.stderr
    assert not (root / 'submitted.json').exists() and not (here / 'hpc_logs_pin').exists()
