#!/usr/bin/env python3
"""Import raw circuits for a single-circuit or explicitly held-out OOD experiment."""
import argparse
from contextlib import closing
import copy
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from device_jepa.cli import _max_table_rows
from device_jepa.methods import add_model_arguments, select_models
from device_jepa.budgets import comparison_epoch_budget, smoke_epochs
from scripts.run.result_paths import create_default_output


def source_identity(directory):
    """Read only small source metadata; the importer performs full validation."""
    database = directory / 'sampling_history.sqlite3'
    if not database.is_file():
        raise ValueError(f'Missing source database: {database}')
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        db.enable_load_extension(False)
        db.execute('PRAGMA query_only = ON')
        db.execute('PRAGMA trusted_schema = OFF')
        for name in ('dataset_schema', 'samples'):
            table = db.execute('SELECT type,sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
            if (table is None or table[0] != 'table'
                    or not re.match(r'\s*CREATE\s+TABLE\b', table[1] or '', re.I)):
                raise ValueError(f'Source {name} must be an ordinary SQLite table')
        rows = db.execute('SELECT circuit_name FROM dataset_schema LIMIT 2').fetchall()
        if len(rows) != 1 or not isinstance(rows[0][0], str) or not rows[0][0].strip():
            raise ValueError('Expected exactly one nonempty source circuit identity')
        circuit = rows[0][0]
        count = db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
    netlist = (directory / (circuit + '.sp')).resolve()
    if not netlist.is_relative_to(directory) or not netlist.is_file():
        raise ValueError(f'Missing or unsafe source netlist for circuit {circuit!r}')
    if count < 1:
        raise ValueError('Source circuit has no samples')
    return circuit, count


def source_plan(directories, train_circuits=None, ood_circuits=None):
    """Resolve roles from actual SQLite identities, without writing or importing rows."""
    if not directories or len(set(directories)) != len(directories):
        raise ValueError('Input directories must be nonempty and unique')
    if (train_circuits is None) != (ood_circuits is None):
        raise ValueError('Specify both --train-circuits and --ood-circuits for OOD')
    if train_circuits is None and len(directories) != 1:
        raise ValueError('Multiple inputs require explicit --train-circuits and --ood-circuits')
    if train_circuits is not None:
        for name, values in (('train_circuits', train_circuits), ('ood_circuits', ood_circuits)):
            if not values or len(set(values)) != len(values) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f'{name} must contain unique, nonempty circuit names')
        if set(train_circuits) & set(ood_circuits):
            raise ValueError('Training and OOD circuits must be disjoint')
    sources = [source_identity(directory) for directory in directories]
    names = [name for name, _ in sources]
    if len(set(names)) != len(names):
        raise ValueError('Each raw input must have a distinct SQLite circuit identity')
    if train_circuits is None:
        train_circuits, ood_circuits = names, []
    elif set(train_circuits) | set(ood_circuits) != set(names):
        raise ValueError('Training and OOD names must exhaustively match the input SQLite circuit identities')
    label = '_'.join(ood_circuits) + '_OOD' if ood_circuits else names[0]
    return (dict(train_circuits=list(train_circuits), ood_circuits=list(ood_circuits)),
            sum(count for _, count in sources), label)


def check_runtime(config, device):
    from device_jepa.model import ModelConfig
    from device_jepa.training import _check_options
    from device_jepa.resident import require_device
    ModelConfig(**config['model'])
    _check_options(config)
    require_device(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='Complete per-circuit raw directories')
    parser.add_argument('--train-circuits', nargs='+', help='Explicit ID circuit names from SQLite (required for OOD)')
    parser.add_argument('--ood-circuits', nargs='+', help='Entire held-out circuit names; must be disjoint from ID')
    parser.add_argument('--output', help='New/empty output root; default: primary checkout '
                        'runs/gpu_full/<circuit>_<YYYYMMDD_HHMMSS> (or JEPA_RESULTS_ROOT)')
    parser.add_argument('--config', default=str(ROOT / 'configs/default.json'),
                        help='Base model/training config; dataset circuit membership is set from SQLite')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    add_model_arguments(parser)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-table-rows', type=_max_table_rows, default=1_000_000,
                        help="Importer row safety limit; raise explicitly or use 'none'")
    parser.add_argument('--smoke', action='store_true', help='JEPA 1+1 epochs, baselines 2; does not subsample data')
    args = parser.parse_args(argv)

    output = Path(args.output).resolve() if args.output is not None else None
    # This entrypoint owns the entire run, including its imported dataset.
    # Check before metadata, config processing, mkdir or any artifact writes.
    if output is not None and output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output must be new/empty; refusing overwrite or stale artifacts')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    directories = [Path(value).resolve() for value in args.input]
    for directory in directories:
        if not directory.is_dir():
            parser.error(f'Input must be a per-circuit directory: {directory}')
    config = copy.deepcopy(json.loads(Path(args.config).read_text()))
    models = select_models(config, args.models)
    membership, count, label = source_plan(directories, args.train_circuits, args.ood_circuits)
    config['dataset'].update(membership)
    if args.smoke:
        smoke_epochs(config)
    comparison_epoch_budget(config, models)
    check_runtime(config, args.device)

    if output is None:
        output = create_default_output(ROOT, label)
    else:
        output.mkdir(parents=True, exist_ok=True)
    config_path, data_path = output / 'config.json', output / 'data.json'
    with config_path.open('x') as handle:
        handle.write(json.dumps(config, indent=2, allow_nan=False) + '\n')
    print(f'Experiment: {label} | source samples: {count} | output: {output}', flush=True)
    print('Circuit membership: ' + json.dumps(membership), flush=True)
    print('Selected models: ' + ', '.join(models), flush=True)
    print('Raw import runs in this process allocation; canonical data still requires full dataset memory.', flush=True)

    def run(command):
        print('+ ' + ' '.join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True)

    limit = 'none' if args.max_table_rows is None else str(args.max_table_rows)
    run([sys.executable, '-m', 'device_jepa', 'import_analog_agent', '--input', *directories,
         '--config', config_path, '--output', data_path, '--max-table-rows', limit])
    # Separate child processes release import/state-encoding memory before
    # training, and release each model's dataset allocation between methods.
    command = [sys.executable, ROOT / 'scripts/run/run_comparison.py', '--data', data_path,
               '--config', config_path, '--output', output / 'results', '--device', args.device,
               '--seeds', *args.seeds, '--models', *models]
    if args.smoke:
        command.append('--smoke')
    run(command)
    summary = output / 'results/comparison/comparison.json'
    if not json.loads(summary.read_text()).get('complete'):
        raise ValueError('Raw comparison did not produce a strictly complete result')
    print(f'Completed experiment {label}: {summary}', flush=True)


if __name__ == '__main__':
    main()
