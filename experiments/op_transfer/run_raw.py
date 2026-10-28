#!/usr/bin/env python3
"""Select circuit names, import their raw data, and train new OP-transfer models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.op_transfer.run import DEFAULT_MODELS, MODEL_NAMES, positive_int, result_directory


def circuit_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or value in ('.', '..'):
        raise argparse.ArgumentTypeError('Circuit must be a directory name, not a path')
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    names = parser.add_mutually_exclusive_group(required=True)
    names.add_argument('--circuit', type=circuit_name, help='One circuit: fresh ID train/val/test, no OOD')
    names.add_argument('--train-circuits', nargs='+', type=circuit_name, help='ID training circuit names')
    parser.add_argument('--ood-circuits', nargs='+', type=circuit_name, help='Entire held-out circuit names')
    parser.add_argument('--raw-root', default=str(PROJECT_ROOT/'runs/raw/full_data'))
    parser.add_argument('--config', default=str(EXPERIMENT_ROOT/'example_config.json'))
    parser.add_argument('--output', help='New run subdirectory of experiments/op_transfer/results')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(DEFAULT_MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[17])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-table-rows', default='none', help="Importer safety cap; default 'none'")
    for split in ('train','val','test'):
        parser.add_argument('--max-'+split, type=positive_int)
    return parser


def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    try:
        # Guard supplied output before metadata access, directory creation or writes.
        output=result_directory(args.output) if args.output else None
        if args.circuit and args.ood_circuits:
            raise ValueError('Use --train-circuits with --ood-circuits for OOD')
        train=[args.circuit] if args.circuit else args.train_circuits
        ood=args.ood_circuits or []
        if len(set(train+ood)) != len(train+ood):
            raise ValueError('Circuit names must be unique; train and OOD must be disjoint')
        if len(set(args.models)) != len(args.models):
            raise ValueError('Models must be unique')
        if len(set(args.seeds)) != len(args.seeds) or any(s<0 or s>=2**32 for s in args.seeds):
            raise ValueError('Seeds must be unique integers in [0, 2**32)')
        from device_jepa.cli import _max_table_rows
        from device_jepa.model import ModelConfig
        from device_jepa.resident import require_device
        from experiments.op_transfer.engine import settings
        from scripts.run.run_raw_comparison import source_identity
        limit=_max_table_rows(args.max_table_rows)
        config=json.loads(Path(args.config).expanduser().resolve().read_text())
        config['model']=ModelConfig(**config['model']).to_dict()
        settings(config)
        require_device(args.device)
        config.setdefault('dataset',{}).update(train_circuits=train,ood_circuits=ood)
        raw_root=Path(args.raw_root).expanduser().resolve()
        inputs=[]
        source_samples=0
        for name in train+ood:
            directory=(raw_root/name).resolve()
            if not directory.is_relative_to(raw_root):
                raise ValueError(f'Circuit directory escapes raw root: {name}')
            actual,count=source_identity(directory)
            if actual != name:
                raise ValueError(f'Circuit name {name!r} does not match SQLite identity {actual!r}')
            inputs.append(directory)
            source_samples+=count
        if output is None:
            label='_'.join(ood)+'_OOD' if ood else '_'.join(train)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            output=result_directory(EXPERIMENT_ROOT/'results'/f'{label}_{stamp}_{uuid.uuid4().hex[:8]}')
    except (ValueError,OSError,argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True,exist_ok=True)
    config_path,data_path=output/'config.json',output/'data.json'
    config_path.write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    record=dict(status='importing',train_circuits=train,ood_circuits=ood,
        models=args.models,seeds=args.seeds,source_samples=source_samples,
        inputs=[str(p) for p in inputs],training_output=str(output/'results'),
        initialization='new models; no previous experiment checkpoint or manifest')
    def status(value):
        record['status']=value
        (output/'raw_run.json').write_text(json.dumps(record,indent=2)+'\n')
    def run(command):
        print('+ '+' '.join(map(str,command)),flush=True)
        subprocess.run(list(map(str,command)),cwd=PROJECT_ROOT,check=True)
    print(f'New experiment: {output}\nRaw circuits: {train}; OOD: {ood}\nModels: {args.models}',flush=True)
    print('Raw import and new model training execute inside this job allocation.',flush=True)
    try:
        status('importing')
        run([sys.executable,'-m','device_jepa','import_analog_agent','--input',*inputs,
            '--config',config_path,'--output',data_path,'--max-table-rows',
            'none' if limit is None else str(limit)])
        status('training')
        command=[sys.executable,EXPERIMENT_ROOT/'run.py','--data',data_path,'--config',config_path,
            '--output',output/'results','--models',*args.models,'--seeds',*args.seeds,'--device',args.device]
        for split in ('train','val','test'):
            cap=getattr(args,'max_'+split)
            if cap is not None: command.extend(['--max-'+split,cap])
        # No --manifest: the new run creates its own deterministic grouped split.
        run(command)
        report=json.loads((output/'results/run.json').read_text())
        if report.get('status') != 'complete':
            raise ValueError('Training did not produce a complete experiment')
        status('complete')
        print(f'Completed. Excel: {output}/results/summary.xlsx',flush=True)
    except Exception as exc:
        record['failure_type']=type(exc).__name__
        status('failed')
        raise
    return output


if __name__=='__main__':
    main()
