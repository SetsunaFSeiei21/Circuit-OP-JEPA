"""One shared cohort, isolated search trials, fresh formal seeds and safe replay."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager
from final_exam.multi_circuit.config_pin import atomic_pin, read_pin, digest_pin


@contextmanager
def lock_pin(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Task is already running: {path}') from None
        yield


def verify_snapshot_pin(campaign):
    record = read_pin(campaign/'campaign_pin.json')
    if digest_pin(read_pin(campaign/'plan_pin.json')) != record['plan_digest']:
        raise ValueError('Submitted grid/configuration plan changed; use a new campaign')
    import hashlib
    for name, expected in record['source_files'].items():
        path = campaign/'source_pin'/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Submitted source snapshot changed: {name}')
    return record


def verify_shared_pin(campaign):
    ready = read_pin(campaign/'data_pin/ready_pin.json')
    for name, expected in ready['artifact_digests'].items():
        if digest_pin(read_pin(campaign/'data_pin'/name)) != expected:
            raise ValueError(f'Shared split/normalization artifact changed: {name}')
    return ready


def prepare_pin(campaign):
    record = verify_snapshot_pin(campaign)
    root = campaign/'data_pin'
    with lock_pin(root/'prepare_pin.lock'):
        identity = digest_pin({'protocol': record['protocol'], 'source': record['source_digest']})
        ready = root/'ready_pin.json'
        if ready.is_file():
            if read_pin(ready)['identity'] != identity:
                raise ValueError('Prepared data identity changed')
            verify_shared_pin(campaign)
            return
        from device_jepa.data import load_dataset, build_manifest
        from final_exam.multi_circuit.summarize_pin import export_pin  # dependency preflight before long jobs
        from final_exam.multi_circuit.context_pin import MultiCircuitContextPin
        from final_exam.run_pin import validate_request_pin
        import torch
        torch.set_num_threads(1)
        plan = read_pin(campaign/'plan_pin.json')
        # Fail all bad architecture/settings before any long model job starts.
        for model, item in plan.items():
            for trial in item['trials']:
                validate_request_pin(trial['config'], [model], [record['protocol']['search_seed']],
                                     'cpu', record['smoke'], joint=True)
        cfg = copy.deepcopy(next(iter(plan.values()))['config'])
        cfg['model']['max_devices'] = max(t['config']['model']['max_devices'] for v in plan.values() for t in v['trials'])
        atomic_pin(root/'import_config_pin.json', cfg)
        data_path = root/'data_pin.json'
        if not data_path.is_file():
            temporary = root/'data_importing_pin.json'
            if record.get('canonical_data'):
                import shutil
                shutil.copyfile(record['canonical_data'], temporary)
            else:
                command = [sys.executable, '-u', '-m', 'device_jepa', 'import_analog_agent',
                    '--input', *[str(Path(record['raw_root'])/c) for c in cfg['dataset']['train_circuits']],
                    '--config', str(root/'import_config_pin.json'), '--output', str(temporary), '--max-table-rows', 'none']
                subprocess.run(command, check=True)
            temporary.replace(data_path)
        dataset = load_dataset(data_path, sampling_config=cfg['dataset'])
        if {row['circuit_id'] for row in dataset['samples']} != set(cfg['dataset']['train_circuits']):
            raise ValueError('Data must contain exactly the configured joint circuits')
        representatives = {row['circuit_id']: row for row in dataset['samples']}
        maximum_pins = max(len(row.get('ports', {})) + sum(len(d['terminals']) for d in row['devices'])
                           for row in representatives.values())
        for model, item in plan.items():
            if any(t['config']['model']['max_devices'] < maximum_pins for t in item['trials']):
                raise ValueError(f'{model}: max_devices must cover {maximum_pins} pins; no truncation')
        manifest = build_manifest(dataset, config=cfg['dataset'])
        for split in ('train', 'val', 'test'):
            ids = set(manifest[split])
            if {row['circuit_id'] for row in dataset['samples'] if row['sample_id'] in ids} != set(cfg['dataset']['train_circuits']):
                raise ValueError(f'Every circuit needs a nonempty {split} split')
        atomic_pin(root/'manifest_pin.json', manifest)
        ctx = MultiCircuitContextPin(dataset, manifest, cfg, 'cpu', {}, root)
        from device_jepa.comparison import regression_metrics
        eligibility = {}
        for circuit in cfg['dataset']['train_circuits']:
            eligibility[circuit] = {}
            for name, active in zip(ctx.prep['feature_spec']['metric_names'], ctx.metric_mask):
                values = [row.get('performance', {}).get(name) for row in ctx.rows['val'] if row['circuit_id'] == circuit]
                metric = regression_metrics(values, values)
                eligibility[circuit][name] = bool(active and metric['r2'] is not None)
            if not any(eligibility[circuit].values()):
                raise ValueError(f'No eligible validation R2 in {circuit}')
        atomic_pin(ready, dict(identity=identity, dataset_fingerprint=manifest['dataset_fingerprint'],
            artifact_digests={name: digest_pin(read_pin(root/name)) for name in
                ('manifest_pin.json','preprocessing_pin.json','pin_schema_pin.json')},
            validation_r2_eligibility=eligibility,
            split_counts={s: {c: sum(r['circuit_id'] == c for r in rows) for c in cfg['dataset']['train_circuits']}
                          for s, rows in ctx.rows.items()},
            normalization='train only, before expansion to pins', timestamp=time.time()))


def fit_task_pin(task_path):
    task = read_pin(task_path)
    campaign = Path(task['campaign'])
    record = verify_snapshot_pin(campaign)
    verify_shared_pin(campaign)
    from final_exam.run_pin import run_experiment_pin
    run_experiment_pin(campaign/'data_pin/data_pin.json', task['config'], task['output'],
        [task['model']], [task['seed']], record['device'], log_dir=task['logs'],
        manifest_path=campaign/'data_pin/manifest_pin.json', smoke=record['smoke'],
        resume_from=task['resume_from'], joint=True, evaluation_split=task['split'],
        preprocessing_path=campaign/'data_pin/preprocessing_pin.json')


def run_task_pin(campaign, model, seed, config, root, split):
    record = read_pin(campaign/'campaign_pin.json')
    ready = verify_shared_pin(campaign)
    identity = digest_pin(dict(config=config, seed=seed, model=model, split=split,
        source=record['source_digest'], cohort=digest_pin(ready), smoke=record['smoke']))
    status_path = root/'status_pin.json'
    with lock_pin(root/'task_pin.lock'):
        status = read_pin(status_path) if status_path.exists() else dict(identity=identity, attempts=[])
        if status['identity'] != identity:
            raise ValueError('Task config/cohort/code changed; use a new campaign')
        if status.get('status') == 'complete':
            if not Path(status['report']).is_file() or not Path(status['predictions']).is_file():
                raise ValueError('Completed task has missing artifacts')
            return status
        resume = None
        # Try the latest recoverable attempt, preserving all sources read-only.
        for attempt in reversed(status['attempts']):
            source = Path(attempt['output'])
            if all((source/f'{name}_pin.json').is_file() for name in ('run', 'config', 'preprocessing', 'cohort', 'manifest')):
                resume = str(source)
                break
        number = len(status['attempts']) + 1
        destination = root/f'attempt_{number:03d}_pin'
        output = destination/'training_pin'
        relative = root.relative_to(campaign)
        logs = Path(record['log_dir'])/relative/f'attempt_{number:03d}_pin'
        config_path = root/'config_pin.json'
        atomic_pin(config_path, config)
        task = dict(campaign=str(campaign), model=model, seed=seed, config=str(config_path),
            output=str(output), logs=str(logs), resume_from=resume, split=split)
        task_path = destination/'task_pin.json'
        atomic_pin(task_path, task)
        status['attempts'].append(dict(output=str(output), started_unix=time.time(), resume_from=resume))
        status.update(status='running', seed=seed, model=model, split=split)
        atomic_pin(status_path, status)
        logs.mkdir(parents=True, exist_ok=True)
        with (logs/'worker_pin.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', '-m', 'final_exam.multi_circuit.worker_pin',
                                     'fit', '--task', str(task_path)], stdout=log, stderr=subprocess.STDOUT)
        status['attempts'][-1].update(returncode=result.returncode, ended_unix=time.time())
        directory = output/f'seed_{seed}_pin'/model
        report_path = directory/f'{split}_pin.json'
        if result.returncode == 0 and report_path.exists():
            report = read_pin(report_path)
            status.update(status='complete', report=str(report_path), predictions=str(directory/report['predictions_file']),
                          training_output=str(output), macro_r2=report['macro_r2'])
        else:
            status.update(status='failed', log=str(logs/'worker_pin.log'))
        atomic_pin(status_path, status)
        return status


def model_worker_pin(campaign, index, mode):
    record = verify_snapshot_pin(campaign)
    if not (campaign/'data_pin/ready_pin.json').is_file():
        raise ValueError('Shared data preparation is incomplete')
    model = record['models'][index]
    plan = read_pin(campaign/'plan_pin.json')[model]
    selected_path = campaign/'selected_pin'/f'{model}.json'
    with lock_pin(campaign/'locks_pin'/f'{model}.lock'):
        if mode in ('all', 'search'):
            results = []
            for trial in plan['trials']:
                print(f'Search {model} {trial["id"]}', flush=True)
                status = run_task_pin(campaign, model, record['protocol']['search_seed'], trial['config'],
                                      campaign/'search_pin'/model/trial['id'], 'val')
                results.append(dict(trial=trial['id'], parameters=trial['parameters'],
                                    status=status['status'], macro_r2=status.get('macro_r2')))
            atomic_pin(campaign/'search_pin'/model/'trials_pin.json', results)
            # Do not silently pick a winner from an incomplete intended grid.
            if any(r['status'] != 'complete' or r['macro_r2'] is None for r in results):
                raise RuntimeError(f'{model}: incomplete grid; resume failed trials before selecting')
            best = sorted(results, key=lambda r: (-r['macro_r2'], r['trial']))[0]
            winning = next(t for t in plan['trials'] if t['id'] == best['trial'])
            selection = dict(model=model, config=winning['config'], trial=best['trial'],
                validation_macro_r2=best['macro_r2'], selection='validation only', search_seed=record['protocol']['search_seed'],
                source_digest=record['source_digest'], configuration_digest=digest_pin(winning['config']))
            if selected_path.is_file() and read_pin(selected_path) != selection:
                raise ValueError('Locked winning configuration changed; start a new campaign')
            atomic_pin(selected_path, selection)
        if mode == 'search':
            return
        if not selected_path.exists():
            if not record['fixed_config']:
                raise ValueError('Training needs a selected configuration or explicit --fixed-config')
            config = plan['config']
            atomic_pin(selected_path, dict(model=model, config=config, trial=None,
                selection='user fixed, no search', configuration_digest=digest_pin(config), source_digest=record['source_digest']))
        selected = read_pin(selected_path)
        if digest_pin(selected['config']) != selected['configuration_digest']:
            raise ValueError('Locked configuration was modified')
        failures = []
        for seed in record['protocol']['seeds']:
            print(f'Formal training {model} seed={seed}', flush=True)
            status = run_task_pin(campaign, model, seed, selected['config'],
                                 campaign/'training_pin'/f'seed_{seed}_pin'/model, 'test')
            if status['status'] != 'complete':
                failures.append(seed)
        if failures:
            raise RuntimeError(f'{model}: failed formal seeds {failures}; completed seeds are preserved')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'model', 'fit', 'summary'])
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--mode', choices=['all', 'search', 'train'], default='all')
    parser.add_argument('--index', type=int)
    parser.add_argument('--task', type=Path)
    args = parser.parse_args()
    if args.action == 'fit':
        fit_task_pin(args.task)
    elif args.action == 'prepare':
        prepare_pin(args.campaign)
    elif args.action == 'model':
        index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        model_worker_pin(args.campaign, index, args.mode)
    else:
        from final_exam.multi_circuit.summarize_pin import export_pin
        export_pin(args.campaign)


if __name__ == '__main__':
    main()
