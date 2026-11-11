#!/usr/bin/env python3
"""Recover a staged pin-gate run into a new directory without reimporting raw data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = PROJECT_ROOT / 'final_exam' / 'single_circuit_pin'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_exam.common.launch_pin import directory_name, run_logged, save_json, utc_now
from final_exam.staged_gate_registry_pin import ALL_STAGED_GATE_MODELS_PIN
from final_exam.gate_version_registry_pin import V1_JOINT_GATE_MODELS_PIN


STAGED_MODELS = frozenset(ALL_STAGED_GATE_MODELS_PIN + V1_JOINT_GATE_MODELS_PIN + ('device_op_cascade_pin',))
_SOURCE_IDENTITY = (
    'source_training_output', 'source_run', 'configuration_source', 'manifest_source',
    'data_path', 'dataset_fingerprint', 'circuit', 'models', 'seeds', 'smoke',
    'config_sha256', 'manifest_sha256',
)


def _read_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def inspect_source(source):
    """Read small metadata only; the training runner verifies the full dataset.

    This function intentionally imports neither torch nor the raw-data adapter,
    so login-node submission does not allocate GPUs or parse the large JSON.
    """
    source = Path(source).expanduser().resolve()
    if (source / 'training_pin').is_dir():
        training = source / 'training_pin'
    elif source.name == 'training_pin' and source.is_dir():
        training = source
    else:
        raise ValueError('Resume source must be an existing run directory or its training_pin directory')
    run_root = training.parent
    record = _read_object(training / 'run_pin.json')
    config_path, manifest_path = training / 'config_pin.json', training / 'manifest_pin.json'
    config, manifest = _read_object(config_path), _read_object(manifest_path)
    models, seeds = record.get('models'), record.get('seeds')
    if (not isinstance(models, list) or not models or any(not isinstance(m, str) for m in models)
            or len(set(models)) != len(models) or set(models) - STAGED_MODELS):
        raise ValueError('Resume supports registered staged gates, v1 joint gates and their cascade baseline')
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Source seeds must be unique integers in [0, 2**32)')
    circuit = directory_name(record.get('circuit', ''))
    if (record.get('granularity') != 'pin'
            or config.get('dataset', {}).get('train_circuits') != [circuit]
            or config.get('dataset', {}).get('ood_circuits')):
        raise ValueError('Resume requires a matching single-circuit pin experiment')
    if not all(isinstance(manifest.get(key), list) and manifest[key]
               for key in ('train', 'val', 'test')) or manifest.get('ood'):
        raise ValueError('Resume source manifest must contain train/val/test and no OOD split')
    fingerprint = record.get('dataset_fingerprint')
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError('Source run is missing its dataset fingerprint')
    smoke = record.get('smoke', False)
    if not isinstance(smoke, bool):
        raise ValueError('Source smoke flag must be boolean')

    data_path = run_root / 'data_pin.json'
    if not data_path.is_file():
        pointer = _read_object(run_root / 'data_source_pin.json')
        if pointer.get('dataset_fingerprint') != fingerprint:
            raise ValueError('Canonical data reference fingerprint differs from the source run')
        value = pointer.get('data_path')
        if not isinstance(value, str) or not value:
            raise ValueError('Canonical data reference is missing data_path')
        data_path = Path(value).expanduser()
        if not data_path.is_absolute():
            data_path = run_root / data_path
    data_path = data_path.resolve()
    if not data_path.is_file():
        raise ValueError(f'Original canonical dataset is unavailable: {data_path}')
    return dict(
        source_training_output=str(training), source_run=str(run_root),
        configuration_source=str(config_path), manifest_source=str(manifest_path),
        data_path=str(data_path), dataset_fingerprint=fingerprint, circuit=circuit,
        models=models, seeds=seeds, smoke=smoke,
        config_sha256=_json_digest(config), manifest_sha256=_json_digest(manifest),
        source_status=record.get('status'), source_slurm_job_id=record.get('slurm_job_id'),
    )


def _new_paths(run_id):
    directory_name(run_id)
    if not run_id.endswith('_pin'):
        raise ValueError('Pin experiment run IDs must end in _pin')
    paths = {name: (TASK_ROOT / (name + '_pin') / run_id).resolve()
             for name in ('results', 'logs', 'hpc_logs')}
    for name, path in paths.items():
        if not path.is_relative_to(TASK_ROOT.resolve()):
            raise ValueError(f'{name} directory must remain under {TASK_ROOT}')
    for name in ('results', 'logs'):
        if paths[name].exists():
            raise ValueError(f'Refusing an existing {name} directory: {paths[name]}')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', required=True,
                        help='Existing run root or training_pin directory; source remains unchanged')
    parser.add_argument('--run-id', type=directory_name)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--submitted-source', help='Submission-time source metadata snapshot')
    parser.add_argument('--inspect', action='store_true', help='Print source metadata JSON without writing files')
    args = parser.parse_args(argv)
    try:
        source = inspect_source(args.resume_from)
        if args.submitted_source:
            submitted = _read_object(args.submitted_source)
            if any(source.get(key) != submitted.get(key) for key in _SOURCE_IDENTITY):
                raise ValueError('Resume source changed after submission; submit again after reviewing it')
        if args.inspect:
            print(json.dumps(source, sort_keys=True, allow_nan=False))
            return source
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_id = args.run_id or f'{source["circuit"]}_resume_staged_op_gate_{stamp}_{uuid.uuid4().hex[:8]}_pin'
        paths = _new_paths(run_id)
        config = _read_object(source['configuration_source'])
    except (ValueError, OSError, TypeError, KeyError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))

    output, logs = paths['results'], paths['logs']
    output.mkdir(parents=True, exist_ok=False)
    try:
        logs.mkdir(parents=True, exist_ok=False)
    except BaseException:
        output.rmdir()
        raise
    config_path, training = output / 'config_pin.json', output / 'training_pin'
    record_path = output / 'run_pin.json'
    record = dict(
        run_id=run_id, task='single_circuit_pin', circuit=source['circuit'], status='initializing',
        started_at=utc_now(), resume_from=source['source_training_output'],
        canonical_data=source['data_path'], dataset_fingerprint=source['dataset_fingerprint'],
        configuration_source=source['configuration_source'], effective_config=str(config_path),
        models=source['models'], seeds=source['seeds'], device=args.device, smoke=source['smoke'],
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_node_list=os.environ.get('SLURM_JOB_NODELIST'),
        hpc_logs=str(paths['hpc_logs']), logs=str(logs), launch_log=str(logs / 'launch_pin.log'),
        experiment_log=str(logs / 'experiment_pin.log'), results=str(output), training_output=str(training),
        summary_xlsx=str(training / 'summary_pin.xlsx'),
    )
    try:
        save_json(config_path, config)
        save_json(output / 'resume_source_pin.json', source)
        save_json(output / 'data_source_pin.json', {
            key: source[key] for key in ('data_path', 'dataset_fingerprint', 'source_training_output')
        })
        save_json(record_path, record)
        with (logs / 'launch_pin.log').open('x', buffering=1) as log:
            lock = threading.Lock()
            log.write(f'Resume {run_id} | source {source["source_training_output"]}\n')
            print(f'Resume source: {source["source_training_output"]}\n'
                  f'Reusing canonical JSON: {source["data_path"]}\n'
                  f'HPC logs: {paths["hpc_logs"]}\nExperiment logs: {logs}\nResults: {output}', flush=True)
            record['status'] = 'training'
            save_json(record_path, record)
            command = [
                sys.executable, '-u', PROJECT_ROOT / 'final_exam/run_pin.py',
                '--data', source['data_path'], '--config', config_path, '--output', training,
                '--log-dir', logs, '--manifest', source['manifest_source'],
                '--resume-from', source['source_training_output'],
                '--models', *source['models'], '--seeds', *map(str, source['seeds']),
                '--device', args.device,
            ]
            if source['smoke']:
                command.append('--smoke')
            run_logged(command, log, lock)
            completion = _read_object(training / 'run_pin.json')
            if completion.get('status') != 'complete' or not (training / 'summary_pin.xlsx').is_file():
                raise ValueError('Runner did not produce a complete experiment and summary_pin.xlsx')
            record.update(status='complete', finished_at=utc_now(), exit_code=0)
            save_json(record_path, record)
            print(f'Completed. Excel: {training / "summary_pin.xlsx"}', flush=True)
    except BaseException as error:
        code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        record.update(status='failed', finished_at=utc_now(), exit_code=code,
                      failure_type=type(error).__name__, failure_message=str(error))
        save_json(record_path, record)
        with (logs / 'launch_pin.log').open('a') as log:
            traceback.print_exc(file=log)
        raise
    return output


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        traceback.print_exc()
        sys.exit(error.returncode if error.returncode > 0 else 128 - error.returncode)
