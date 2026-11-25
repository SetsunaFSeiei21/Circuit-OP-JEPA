"""Launcher preflight activates the configured environment; no data/GPU import."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_and_real_submission_are_separate_from_sources(tmp_path):
    root = tmp_path/'project with spaces'
    here = root/'final_exam/single_circuit_pin'
    here.mkdir(parents=True)
    (root/'scripts/run').mkdir(parents=True)
    (root/'device_jepa').mkdir()
    (root/'device_jepa/cli.py').write_text('')
    files = ['error_case_probe_pin.py', 'error_case_data_pin.py', 'prediction_probe_pin.py',
             'spectral_probe_pin.py', 'gate_probe_pin.py']
    for name in files:
        shutil.copy(ROOT/'final_exam'/name, root/'final_exam'/name)
    script = here/'inspect_error_cases_pin.sh'
    shutil.copy(ROOT/'final_exam/single_circuit_pin'/script.name, script)
    shutil.copy(ROOT/'scripts/run/slurm_run.sh', root/'scripts/run/slurm_run.sh')
    source = root/'saved experiment'
    training = source/'training_pin'
    models = ['device_op_cascade_pin','lasd_graph_layer_headwise_detach_pin_v3']
    circuit = 'two_stage_opamp_otaf'
    for model in models:
        directory = training/f'seed_29_pin/{model}'
        directory.mkdir(parents=True)
        (directory/'checkpoint_pin.pt').write_bytes(b'never deserialize on login preflight')
        (directory/'test_pin.json').write_text(json.dumps(dict(split='test',circuit=circuit,metrics={})))
    (source/'data_pin.json').write_text('preflight never parses canonical samples')
    (training/'run_pin.json').write_text(json.dumps(dict(granularity='pin',circuit=circuit,models=models,
                                                       seeds=[29],dataset_fingerprint='fp')))
    (training/'manifest_pin.json').write_text(json.dumps(dict(dataset_fingerprint='fp',ood=[])))
    (training/'config_pin.json').write_text(json.dumps(dict(dataset=dict(train_circuits=[circuit]))))
    binary = root/'bin'
    binary.mkdir()
    sbatch = binary/'sbatch'
    sbatch.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\n'+
                     f'Path({str(root/"submitted.json")!r}).write_text(json.dumps(sys.argv[1:]))\nprint("123")\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, SOURCE_RUN=str(source), JEPA_CONDA_ENV='', JEPA_PYTHON=sys.executable,
               PATH=str(binary)+os.pathsep+os.environ['PATH'], DEVICE='cuda', SLURM_GPUS='1')
    before = set(root.rglob('*'))
    result = subprocess.run(['bash',str(script),'--dry-run'],env=env,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--sample-ids 41074 38711 38196' in result.stdout
    assert set(root.rglob('*')) == before
    result = subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads((root/'submitted.json').read_text())
    assert str(source) in args and '--mem=32G' in args and '--seeds' in args
    assert not (here/'results_pin').exists() and not (here/'logs_pin').exists()
    assert list((here/'hpc_logs_pin').glob('*/submitted_inspect_error_cases_pin.sh'))
