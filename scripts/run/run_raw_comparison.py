#!/usr/bin/env python3
"""Import one Analog-Agent circuit and run an independent matched experiment."""
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
from device_jepa.methods import MODES, add_model_arguments, select_models
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='One complete per-circuit raw directory')
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
    directory = Path(args.input).resolve()
    if not directory.is_dir():
        parser.error(f'Input must be a per-circuit directory: {directory}')
    config = copy.deepcopy(json.loads(Path(args.config).read_text()))
    models = select_models(config, args.models)
    circuit, count = source_identity(directory)
    config['dataset'].update(train_circuits=[circuit], ood_circuits=[])
    if args.smoke:
        smoke_epochs(config)
    from device_jepa.model import ModelConfig
    from device_jepa.training import _check_options
    ModelConfig(**config['model'])
    _check_options(config)
    comparison_epoch_budget(config, models)
    from device_jepa.resident import require_device
    require_device(args.device)

    if output is None:
        output = create_default_output(ROOT, circuit)
    else:
        output.mkdir(parents=True, exist_ok=True)
    config_path, data_path = output / 'config.json', output / 'data.json'
    with config_path.open('x') as handle:
        handle.write(json.dumps(config, indent=2, allow_nan=False) + '\n')
    print(f'Independent circuit: {circuit} | source samples: {count} | output: {output}', flush=True)
    print('Selected models: ' + ', '.join(models), flush=True)
    print('Raw import runs in this process allocation; canonical data still requires full dataset memory.', flush=True)

    def run(command):
        print('+ ' + ' '.join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True)

    limit = 'none' if args.max_table_rows is None else str(args.max_table_rows)
    run([sys.executable, '-m', 'device_jepa', 'import_analog_agent', '--input', directory,
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
        raise ValueError('Independent comparison did not produce a strictly complete result')
    print(f'Completed independent circuit {circuit}: {summary}', flush=True)


if __name__ == '__main__':
    main()
