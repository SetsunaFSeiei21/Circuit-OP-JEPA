#!/usr/bin/env python3
"""Snapshot, plan and submit a prepare -> bounded model array -> report DAG."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import re
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from final_exam.multi_circuit.config_pin import (read_pin, atomic_pin, digest_pin, model_config_pin,
                                                expand_grid_pin, init_model_pin)


def snapshot_pin(destination):
    conflicts = subprocess.run(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True,capture_output=True,check=True)
    if conflicts.stdout.strip():
        raise ValueError('Resolve Git conflicts before snapshotting: '+conflicts.stdout)
    names = subprocess.run(['git','ls-files','-z','--cached','--others','--exclude-standard'],
                           cwd=ROOT,capture_output=True,check=True).stdout.decode().split('\0')
    allowed = {'device_jepa','final_exam','experiments','scripts','configs','pyproject.toml','requirements.txt'}
    files = {}
    for name in sorted(set(names)):
        path = Path(name)
        if not name or path.parts[0] not in allowed or not (ROOT/path).is_file():
            continue
        if any(p in ('results_pin','logs_pin','hpc_logs_pin','__pycache__') for p in path.parts):
            continue
        if (ROOT/path).is_symlink():
            raise ValueError(f'Source snapshots require regular files: {name}')
        content = (ROOT/path).read_bytes()
        if path.suffix in ('.py','.sh','.json') and any(line.startswith((b'<<<<<<< ',b'>>>>>>> ')) for line in content.splitlines()):
            raise ValueError(f'Unresolved conflict markers in {name}')
        target = destination/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
        target.chmod(0o555 if os.access(ROOT/path,os.X_OK) else 0o444)
        files[name] = hashlib.sha256(content).hexdigest()
    commit = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    return files,commit


def build_plan_pin(models, circuits, config_dir, protocol_path, seeds, search_seed, smoke=False, fixed=False):
    protocol = read_pin(protocol_path)
    if len(circuits)<2 or len(set(circuits)) != len(circuits) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', c) for c in circuits):
        raise ValueError('Select at least two distinct circuit directory names')
    if len(set(models)) != len(models) or not models:
        raise ValueError('Models must be distinct and nonempty')
    protocol['dataset']['train_circuits'] = list(circuits)
    if protocol['dataset'].get('ood_circuits'):
        raise ValueError('This campaign is joint ID, not OOD')
    protocol['seeds'] = seeds if seeds is not None else protocol['seeds']
    protocol['search_seed'] = search_seed if search_seed is not None else protocol['search_seed']
    if (not protocol['seeds'] or len(set(protocol['seeds'])) != len(protocol['seeds']) or
            any(type(s) is not int or not 0 <= s < 2**32 for s in [*protocol['seeds'],protocol['search_seed']])):
        raise ValueError('Use distinct nonnegative 32-bit formal seeds')
    if protocol['search_seed'] in protocol['seeds']:
        raise ValueError('Search seed must differ from fresh formal seeds')
    if protocol['selection'] != 'circuit_macro_val_r2':
        raise ValueError('Selection must be circuit_macro_val_r2')
    if smoke:
        count = protocol['dataset'].get('max_samples_per_circuit')
        protocol['dataset']['max_samples_per_circuit'] = min(count, 100) if type(count) is int else 100
    plan = {}
    for model in models:
        config = model_config_pin(model,config_dir,protocol)
        if fixed:
            trials,skipped = [dict(id='fixed_pin',parameters={},config=config)],[]
        else:
            trials,skipped = expand_grid_pin(model,config,read_pin(Path(config_dir)/'search_spaces'/f'{model}.json'))
        if len(trials)>protocol['max_trials_per_model']:
            raise ValueError(f'{model}: grid exceeds max_trials_per_model; review the budget explicitly')
        plan[model] = dict(config=config,trials=trials,skipped=skipped)
    return protocol,plan


def submit_pin(campaign, record, mode, local, dry_run, max_concurrent, resources):
    source = campaign/'source_pin'
    hpc = Path(record['hpc_dir'])
    submitted = {}
    def runner(action):
        return ['bash',str(source/'final_exam/multi_circuit/run_multi_pin.sh'),str(source),
            record['conda_home'],record['conda_env'],record['code_commit'],action,'--campaign',str(campaign)]
    if local:
        if dry_run:
            print('LOCAL: prepare; model workers sequentially; summary')
            return
        env = os.environ.copy()
        env['JEPA_PYTHON'] = sys.executable
        subprocess.run(runner('prepare'),check=True,env=env)
        failures=[]
        for i in range(len(record['models'])):
            result=subprocess.run(runner('model')+['--index',str(i),'--mode',mode],env=env)
            if result.returncode:failures.append(i)
        subprocess.run(runner('summary'),check=True,env=env)
        if failures:raise RuntimeError(f'Model tasks failed: {failures}; see per-task logs')
        return
    def job(label, action, dependency=None, array=False):
        opts=['sbatch','--parsable',f'--job-name=multi-{label}-pin',f'--chdir={source}',
            f'--output={hpc}/%A_%a_{label}_pin.out',f'--error={hpc}/%A_%a_{label}_pin.err',
            '--nodes=1','--ntasks=1',f'--cpus-per-task={resources["cpus"]}',
            f'--mem={resources["mem"] if label != "summary" else "8G"}',
            f'--time={resources["time"]}','--export=ALL']
        for key in ('account','partition'):
            if resources[key]:opts.append(f'--{key}={resources[key]}')
        if dependency:
            opts.extend(['--dependency='+dependency, '--kill-on-invalid-dep=yes'])
        if array:
            opts.append(f'--array=0-{len(record["models"])-1}%{max_concurrent}')
            if record['device'].startswith('cuda'):opts.append('--gres=gpu:1')
        # sbatch expects a script, not "bash" as executable argument.
        command=opts+runner(action)[1:]
        if action=='model':command+=['--mode',mode]
        print(shlex.join(command),flush=True)
        if dry_run:return label.upper()+'_JOBID'
        result=subprocess.run(command,text=True,capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        jobid=result.stdout.strip().split(';')[0]
        if not jobid.isdigit():raise RuntimeError('Unexpected sbatch output: '+result.stdout)
        submitted[label] = jobid
        atomic_pin(hpc/'submission_latest_pin.json', dict(jobs=submitted, mode=mode,
            array_models={str(i): name for i, name in enumerate(record['models'])}))
        return jobid
    prep=job('prepare','prepare')
    model=job('models','model','afterok:'+prep,True)
    report=job('summary','summary','afterany:'+model)
    if not dry_run:
        atomic_pin(hpc/f'submission_{int(time.time())}_pin.json',dict(prepare=prep,models=model,summary=report,mode=mode))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuits',nargs='+',default=['two_stage_folded_opamp','Opamp1'])
    parser.add_argument('--models',nargs='+',default=['zerosim_supervised_pin','device_op_cascade_pin'])
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--search-seed',type=int)
    parser.add_argument('--config-dir',type=Path,default=ROOT/'final_exam/multi_circuit/configs_pin')
    parser.add_argument('--protocol',type=Path,default=ROOT/'final_exam/multi_circuit/configs_pin/protocol_pin.json')
    parser.add_argument('--raw-root',type=Path,default=ROOT/'runs/raw/full_data')
    parser.add_argument('--canonical-data',type=Path,help='Optional already imported joint data (tests/offline use)')
    parser.add_argument('--campaign',type=Path,help='Resume an existing immutable campaign')
    parser.add_argument('--mode',choices=['all','search','train'],default='all')
    parser.add_argument('--fixed-config',action='store_true')
    parser.add_argument('--init-model')
    parser.add_argument('--max-concurrent',type=int,default=2)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--local',action='store_true')
    parser.add_argument('--account',default=os.environ.get('SLURM_ACCOUNT','b_phzhwu'))
    parser.add_argument('--partition',default=os.environ.get('SLURM_PARTITION','ex01A800'))
    parser.add_argument('--cpus',type=int,default=8)
    parser.add_argument('--mem',default=os.environ.get('SLURM_MEM','80G'))
    parser.add_argument('--time',default=os.environ.get('SLURM_TIME','10-00:00:00'))
    args=parser.parse_args(argv)
    if args.init_model:
        print(*init_model_pin(args.init_model,args.config_dir),sep='\n');return
    if args.max_concurrent<1 or args.cpus<1:raise ValueError('Concurrency and CPU count must be positive')
    if args.fixed_config and args.mode!='train':raise ValueError('--fixed-config requires --mode train')
    if args.campaign:
        campaign=args.campaign.resolve()
        record=read_pin(campaign/'campaign_pin.json')
        plan=read_pin(campaign/'plan_pin.json')
        print('Resuming saved circuits/models/seeds/configs; edits to current bash/configs do not alter this campaign.')
    else:
        protocol,plan=build_plan_pin(args.models,args.circuits,args.config_dir,args.protocol,args.seeds,args.search_seed,args.smoke,args.fixed_config)
        tag=time.strftime('multi_%Y%m%dT%H%M%SZ_',time.gmtime())+str(os.getpid())+'_pin'
        campaign=ROOT/'final_exam/multi_circuit/results_pin'/tag
        record=dict(protocol=protocol,models=args.models,smoke=args.smoke,fixed_config=args.fixed_config,
            device=args.device,raw_root=str(args.raw_root.resolve()),
            canonical_data=str(args.canonical_data.resolve()) if args.canonical_data else None,
            log_dir=str(ROOT/'final_exam/multi_circuit/logs_pin'/tag),
            hpc_dir=str(ROOT/'final_exam/multi_circuit/hpc_logs_pin'/tag),
            conda_home=os.environ.get('JEPA_CONDA_HOME','/share/software/anaconda3'),
            conda_env=os.environ.get('JEPA_CONDA_ENV','newbase'),code_commit='DRY_RUN')
        if not args.dry_run:
            if not args.local and shutil.which('sbatch') is None:raise ValueError('sbatch unavailable; use --local inside an allocation')
            if not args.canonical_data:
                for circuit in args.circuits:
                    path=args.raw_root/circuit/'sampling_history.sqlite3'
                    if not path.is_file():raise ValueError(f'Missing raw source: {path}')
            campaign.mkdir(parents=True)
            files,commit=snapshot_pin(campaign/'source_pin')
            record.update(source_files=files,source_digest=digest_pin(files),code_commit=commit,plan_digest=digest_pin(plan))
            atomic_pin(campaign/'campaign_pin.json',record)
            atomic_pin(campaign/'plan_pin.json',plan)
    counts={m:len(v['trials']) for m,v in plan.items()}
    print(json.dumps(dict(campaign=str(campaign),circuits=record['protocol']['dataset']['train_circuits'],
        grid_trials=counts,search_seed=record['protocol']['search_seed'],formal_seeds=record['protocol']['seeds'],
        total_search_runs=sum(counts.values()) if args.mode!='train' else 0,
        total_formal_runs=len(plan)*len(record['protocol']['seeds']) if args.mode!='search' else 0,
        max_concurrent_model_jobs=args.max_concurrent,smoke=record['smoke']),indent=2),flush=True)
    if not args.dry_run:
        Path(record['hpc_dir']).mkdir(parents=True,exist_ok=True)
        Path(record['log_dir']).mkdir(parents=True,exist_ok=True)
    submit_pin(campaign,record,args.mode,args.local,args.dry_run,args.max_concurrent,
               dict(account=args.account,partition=args.partition,cpus=args.cpus,mem=args.mem,time=args.time))


if __name__=='__main__':
    main()
