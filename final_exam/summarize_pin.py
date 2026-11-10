#!/usr/bin/env python3
"""Export pin-level final-exam results to auditable MAE, RMSE and R2 workbooks.

Only the Python standard library is required. Complete exports validate the
planned model/seed/split matrix and effective configurations before writing.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from device_jepa.excel import Cell, METRICS, SCORES, Sheet, _column, _format_number, _write_workbook
from experiments.op_transfer.summarize import summary_rows as _aggregate_rows


ORACLES = frozenset(('op_oracle_pin', 'design_op_oracle_pin',
                     'device_op_gate_oracle_pin', 'device_op_gate_only_oracle_pin'))

# Keep this small reporting contract independent of Torch and the training
# factories so completed runs can be summarized on a machine without a GPU.
GATE_INPUTS_PIN = {
    'device_op_cascade_pin': ('A', True, 'none'),
    'device_op_gate_pin': ('A', True, 'predicted OP'),
    'device_op_gate_oracle_pin': ('A', True, 'measured OP; missing fields use predicted OP'),
    'device_op_gate_shuffle_pin': ('A', True, 'shuffled training predicted OP'),
    'device_op_gate_only_base_pin': ('B', False, 'none'),
    'device_op_gate_only_pin': ('B', False, 'predicted OP'),
    'device_op_gate_only_oracle_pin': ('B', False, 'measured OP; missing fields use predicted OP'),
    'device_op_gate_only_shuffle_pin': ('B', False, 'shuffled training predicted OP'),
}
SHUFFLE_CONTROLS_PIN = frozenset(mode for mode in GATE_INPUTS_PIN if '_shuffle_' in mode)
STAGED_GATE_INPUTS_PIN = {
    'device_op_distill_gate_pin': ('distilled student', 'gate only'),
    'device_op_distill_gate_decoder_pin': ('distilled student', 'gate and decoder'),
    'device_op_supervised_gate_pin': ('supervised ungated model', 'gate only'),
}
ALL_GATE_REPORTS_PIN = GATE_INPUTS_PIN.keys() | STAGED_GATE_INPUTS_PIN.keys()


def _require_pin_mode(mode):
    if not isinstance(mode, str) or not mode.endswith('_pin') or len(mode) <= 4:
        raise ValueError('Pin reports require every model name to end in _pin')


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, OSError) as exc:
        raise ValueError(f'Cannot read JSON {path}: {exc}') from exc


def _resolve_source(source):
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        candidates = (path / 'results_pin.json', path / 'training_pin' / 'results_pin.json')
        path = next((item for item in candidates if item.is_file()), path / 'results_pin.json')
    if path.name != 'results_pin.json' or not path.is_file():
        raise ValueError(f'No results_pin.json found at {path}')
    return path


def _circuit_identity(record, config):
    report = record['report']
    circuit = record.get('circuit', report.get('circuit'))
    if circuit is not None:
        if not isinstance(circuit, str) or not circuit.strip():
            raise ValueError('Circuit identity must be a nonempty string')
        return circuit
    role = 'ood_circuits' if report.get('split') == 'ood' else 'train_circuits'
    circuits = config.get('dataset', {}).get(role, [])
    if isinstance(circuits, list) and len(circuits) == 1 and isinstance(circuits[0], str):
        return circuits[0]
    return 'ALL_SELECTED_CIRCUITS'


def summary_rows_pin(results, config=None):
    """Aggregate matching seeds, retaining circuit and split identities."""
    if not isinstance(results, list) or not results:
        raise ValueError('Expected a nonempty results_pin.json list')
    groups = defaultdict(list)
    for record in results:
        if not isinstance(record, dict) or not isinstance(record.get('report'), dict):
            raise ValueError('Each result must contain a report object')
        _require_pin_mode(record.get('mode'))
        groups[_circuit_identity(record, config or {})].append(record)
    rows = []
    for circuit, members in sorted(groups.items()):
        # Reuse the existing finite-value, count, seed and OP-input validation.
        normalized = [dict(record, mode=record['mode'][:-4]) for record in members]
        for row in _aggregate_rows(normalized, oracle_modes={mode[:-4] for mode in ORACLES}):
            row['mode'] += '_pin'
            row.update(circuit=circuit, backend='final_exam_pin',
                       scope='pooled' if circuit == 'ALL_SELECTED_CIRCUITS' else 'circuit')
            row['seed_ids'] = sorted(record['seed'] for record in members
                                     if record['mode'] == row['mode']
                                     and record['report']['split'] == row['split'])
            rows.append(row)
    return rows


def _validate_plan(results, run, *, allow_incomplete, finalizing):
    if not isinstance(run, dict):
        raise ValueError('run_pin.json must contain an object')
    models, seeds = run.get('models'), run.get('seeds')
    splits = run.get('evaluation_splits')
    if (not isinstance(models, list) or not models
            or any(not isinstance(mode, str) or not mode for mode in models)
            or len(set(models)) != len(models)):
        raise ValueError('run_pin.json must declare distinct selected models')
    for mode in models:
        _require_pin_mode(mode)
    if run.get('granularity', 'pin') != 'pin':
        raise ValueError('Pin reports cannot mix device-level run metadata')
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('run_pin.json must declare distinct nonnegative integer seeds')
    if (not isinstance(splits, list) or not splits or len(set(splits)) != len(splits)
            or any(split not in ('test', 'ood') for split in splits)):
        raise ValueError('run_pin.json must declare test/OOD evaluation_splits')
    expected = {(mode, seed, split) for mode in models for seed in seeds for split in splits}
    actual = {(record['mode'], record['seed'], record['report']['split']) for record in results}
    schemas = {}
    for record in results:
        report = record['report']
        key = (record.get('circuit', report.get('circuit', run.get('circuit'))), report['split'])
        schema = frozenset(report['metrics'])
        if key in schemas and schemas[key] != schema:
            raise ValueError('Metric schema differs across compared models')
        schemas[key] = schema
    unexpected = actual - expected
    if unexpected:
        raise ValueError(f'Results outside the declared experiment: {sorted(unexpected)}')
    missing = expected - actual
    accepted_statuses = {'complete'}
    if finalizing:
        accepted_statuses.add('ready_to_export')
    complete = not missing and run.get('status') in accepted_statuses
    if not complete and not allow_incomplete:
        raise ValueError(f'Incomplete experiment: status={run.get("status")!r}, '
                         f'missing model/seed/split results={sorted(missing)}')
    # Keep measured-OP references visibly separate at the right of each table.
    ordered_models = [mode for mode in models if mode not in ORACLES]
    ordered_models.extend(mode for mode in models if mode in ORACLES)
    return ordered_models, complete, sorted(missing)


def _validate_stages(payload, results):
    stages = payload.get('unique_stages') if isinstance(payload, dict) else None
    if not isinstance(stages, list):
        raise ValueError('stages_pin.json must contain unique_stages')
    identities = set()
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get('stage_id'), str) or not stage['stage_id']:
            raise ValueError('Each training cost needs a stage_id')
        if stage['stage_id'] in identities:
            raise ValueError(f'Duplicate stage cost: {stage["stage_id"]}')
        identities.add(stage['stage_id'])
        for field in ('seconds', 'optimizer_steps'):
            value = stage.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Stage {stage["stage_id"]}: invalid {field}')
            if field != 'seconds' and int(value) != value:
                raise ValueError(f'Stage {stage["stage_id"]}: {field} must be integral')
        budget_unit = stage.get('budget_unit', 'epochs')
        if budget_unit == 'epochs':
            for field in ('epochs', 'planned_epochs'):
                value = stage.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError(f'Stage {stage["stage_id"]}: invalid {field}')
            if stage['epochs'] > stage['planned_epochs']:
                raise ValueError('Actual stage epochs exceed the configured budget')
        elif budget_unit == 'boosting_rounds_per_metric':
            if any(stage.get(field, 0) is not None for field in ('epochs', 'planned_epochs')):
                raise ValueError('Boosting-round stages must leave epoch fields null')
            planned = stage.get('planned_boosting_rounds')
            rounds = stage.get('boosting_rounds')
            if type(planned) is not int or planned < 1:
                raise ValueError('planned_boosting_rounds must be a positive integer')
            if not isinstance(rounds, dict) or not rounds:
                raise ValueError('boosting_rounds must contain actual rounds per trained metric')
            for metric, count in rounds.items():
                if (not isinstance(metric, str) or not metric or type(count) is not int
                        or not 1 <= count <= planned):
                    raise ValueError('Invalid actual boosting rounds or rounds exceed the configured budget')
        else:
            raise ValueError(f'Unsupported stage budget unit: {budget_unit}')
    for record in results:
        dependencies = record.get('stage_ids')
        if (not isinstance(dependencies, list) or not dependencies
                or any(not isinstance(item, str) for item in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise ValueError('Each result needs distinct nonempty stage_ids')
        absent = set(dependencies) - identities
        if absent:
            raise ValueError(f'Missing costs for dependent stages: {sorted(absent)}')
    return stages


def _flatten(value, prefix=''):
    if isinstance(value, dict) and value:
        for key in sorted(value):
            yield from _flatten(value[key], f'{prefix}.{key}' if prefix else key)
    else:
        yield prefix, value


def _configuration_sheet(directory, config, run, results, *, allow_incomplete):
    sheet = Sheet('Configuration', [26, 15, 38, 54, 80], freeze_rows=4, freeze_columns=2)
    sheet.note('Effective experiment and per-model configurations', 1, 34)
    sheet.note('Saved values include overrides. Refer to the source paths for the full JSON.')
    sheet.note('Results describe the saved run; changing these cells does not retrain models.')
    sheet.add([Cell(value, 3) for value in ('Model / document', 'Seed', 'Source', 'Setting', 'Value')], 32)
    documents = [('experiment', None, directory / 'config_pin.json', config),
                 ('run metadata', None, directory / 'run_pin.json', run)]
    missing = []
    for mode, seed in sorted({(record['mode'], record['seed']) for record in results}):
        path = directory / f'seed_{seed}_pin' / mode / 'config_pin.json'
        if not path.is_file():
            missing.append(str(path))
            sheet.add([mode, seed, str(path), 'MISSING EFFECTIVE CONFIGURATION', None])
            continue
        value = _read_json(path)
        if not isinstance(value, dict) or not value:
            raise ValueError(f'Effective configuration must be a nonempty object: {path}')
        documents.append((mode, seed, path, value))
    if missing and not allow_incomplete:
        raise ValueError(f'Missing per-model effective configurations: {missing}')
    for label, seed, path, value in documents:
        for key, setting in _flatten(value):
            sheet.add([label, seed, str(path), key, setting], 30)
    sheet.filter_range = f'A4:E{len(sheet.rows)}'
    return sheet, missing


def _statistics_sheet(rows):
    fields = ('circuit', 'split', 'mode', 'metric', 'scale', 'requires_op_at_inference',
              'uses_design_parameters', 'seed_ids', 'num_seeds', 'count_per_seed',
              *(score + suffix for score in SCORES for suffix in ('_defined_seeds', '_mean', '_std')))
    sheet = Sheet('Statistics', [30] * len(fields), freeze_rows=4, freeze_columns=4)
    sheet.note('Exact aggregated values in original physical units', 1, 34)
    sheet.note('Standard deviation uses ddof=1 and is blank when fewer than two scores are defined.')
    sheet.note('Mean/std cells in comparison sheets link to this worksheet. R2 is dimensionless.')
    sheet.add([Cell(field, 3) for field in fields], 44)
    for item in rows:
        sheet.add([Cell(item.get(field), 11 if field.endswith(('_mean', '_std')) else 5)
                   for field in fields])
    sheet.filter_range = f'A4:{_column(len(fields))}{len(sheet.rows)}'
    return sheet, fields


def _score_cell(row, score, divisor, fields, source_row, style, decimals):
    if row is None or row[score + '_mean'] is None:
        return Cell('n/a', style)
    average, average_format = _format_number(row[score + '_mean'] / divisor, decimals)
    deviation = row[score + '_std']
    mean_ref = f"'Statistics'!{_column(fields.index(score + '_mean') + 1)}{source_row}"
    std_ref = f"'Statistics'!{_column(fields.index(score + '_std') + 1)}{source_row}"
    deviation_format = _format_number(deviation / divisor, decimals)[1] if deviation is not None else average_format
    formatted_mean = f'TEXT({mean_ref}/{divisor},"{average_format}")'
    formula = (f'IF(ISNUMBER({mean_ref}),{formatted_mean}&'
               f'IF(ISNUMBER({std_ref})," ± "&TEXT({std_ref}/{divisor},"{deviation_format}"),""),"n/a")')
    text = average
    if deviation is not None:
        text += ' ± ' + _format_number(deviation / divisor, decimals)[0]
    defined = row[score + '_defined_seeds']
    if defined < row['num_seeds']:
        suffix = f' [{defined}/{row["num_seeds"]} seeds]'
        text += suffix
        formula += f'&"{suffix}"'
    return Cell(text, style, formula)


def _comparison_sheets(rows, models, fields, complete, decimals):
    indexed = {(row['circuit'], row['split'], row['metric'], row['mode']): (index + 5, row)
               for index, row in enumerate(rows)}
    metric_order = {metric: index for index, metric in enumerate(METRICS)}
    identities = sorted({key[:3] for key in indexed},
                        key=lambda key: (key[0], key[1], metric_order.get(key[2], len(METRICS)), key[2]))
    sheets = []
    for score in SCORES:
        name = 'R2' if score == 'r2' else score.upper()
        sheet = Sheet(name, [26, 12, 25, 16, *([30] * len(models)), 16], freeze_rows=7, freeze_columns=4)
        sheet.note(f'{name} pin-level model comparison', 1, 34)
        sheet.note('STATUS: ' + ('COMPLETE' if complete else 'INCOMPLETE — provisional results, no ranking'), 4, 30)
        sheet.note('Mean ± sample std over seeds. One seed displays the mean alone; its std remains blank.')
        sheet.note('Higher R2 is better; lower MAE/RMSE is better. UGF errors are displayed in MHz.')
        sheet.note('Green marks the best unrounded mean within matching inference-input groups and matched seeds/counts.', height=32)
        if any(mode in STAGED_GATE_INPUTS_PIN for mode in models):
            sheet.note('Staged gates use additional training; compare TrainingCost and GateBeforeAfter. Highlighting is not a compute-matched significance test.', height=30)
        elif any(mode in GATE_INPUTS_PIN for mode in models):
            sheet.note('TRUE OP REFERENCE needs measured OP. SHUFFLE CONTROL is diagnostic and excluded from ranking; see ModelInputs.', height=30)
        else:
            sheet.note('TRUE OP REFERENCE requires measured OP at inference. Highlighting is not a significance test.', height=30)
        labels = [mode + ('\nTRUE OP REFERENCE' if mode in ORACLES else
                          '\nSHUFFLE CONTROL' if mode in SHUFFLE_CONTROLS_PIN else '') for mode in models]
        sheet.add([Cell(value, 3) for value in ('Circuit', 'Split', 'Metric', 'Unit', *labels, 'Labels / seed')], 64)
        for circuit, split, metric in identities:
            label, unit, divisor = METRICS.get(metric, (metric, 'dataset unit', 1))
            if score == 'r2':
                unit, divisor = 'dimensionless', 1
            found = [indexed.get((circuit, split, metric, mode)) for mode in models]
            best_modes = set()
            groups = defaultdict(list)
            for mode, entry in zip(models, found):
                if entry and mode not in SHUFFLE_CONTROLS_PIN:
                    row = entry[1]
                    groups[(row['requires_op_at_inference'], row['uses_design_parameters'])].append((mode, row))
            for members in groups.values():
                counts = {(tuple(row['seed_ids']), row['count_per_seed'], row[score + '_defined_seeds'])
                          for _, row in members}
                eligible = (complete and len(members) > 1 and len(counts) == 1
                            and all(row[score + '_mean'] is not None
                                    and row[score + '_defined_seeds'] == row['num_seeds'] for _, row in members))
                if eligible:
                    values = [row[score + '_mean'] for _, row in members]
                    best = max(values) if score == 'r2' else min(values)
                    best_modes.update(mode for mode, row in members if row[score + '_mean'] == best)
            cells = [circuit, split, label, unit]
            for mode, entry in zip(models, found):
                source_row, row = entry if entry else (None, None)
                cells.append(_score_cell(row, score, divisor, fields, source_row,
                                         9 if mode in best_modes else 7, decimals))
            counts = {entry[1]['count_per_seed'] for entry in found if entry}
            cells.append(next(iter(counts)) if len(counts) == 1 else 'varies')
            sheet.add(cells, 32)
        sheet.filter_range = f'A7:{_column(len(sheet.widths))}{len(sheet.rows)}'
        sheets.append(sheet)
    return sheets


def _per_seed_sheet(results, config):
    fields = ('circuit', 'split', 'model', 'seed', 'metric', 'count', 'mae', 'rmse', 'r2',
              'normalized_rmse', 'requires_measured_op', 'stage_ids')
    sheet = Sheet('PerSeed', [27, 12, 34, 12, 28, 14, 24, 24, 24, 24, 24, 70],
                  freeze_rows=4, freeze_columns=4)
    sheet.note('Per-seed physical-unit metrics', 1, 34)
    sheet.note('Exact source values are retained. UGF values here are in Hz, not the MHz display units.')
    sheet.note('Blank values are undefined or unavailable. Label count refers to the individual metric.')
    sheet.add([Cell(value, 3) for value in fields], 44)
    for record in sorted(results, key=lambda item: (item['mode'], item['seed'], item['report']['split'])):
        report = record['report']
        for metric, values in report['metrics'].items():
            sheet.add([_circuit_identity(record, config), report['split'], record['mode'], record['seed'], metric,
                       values['count'], *(Cell(values.get(score), 11) for score in ('mae', 'rmse', 'r2', 'normalized_rmse')),
                       report['requires_measured_op'], record['stage_ids']])
    sheet.filter_range = f'A4:{_column(len(fields))}{len(sheet.rows)}'
    return sheet


def _model_inputs_sheet(models):
    """Expose the A/B ablations without changing metric aggregation identity."""
    fields = ('Model', 'Study group', 'OP predictor encoder', 'Predicted OP node features',
              'Attention gate source', 'Measured OP required', 'Role / interpretation',
              'Base training', 'Final trainable modules')
    sheet = Sheet('ModelInputs', [48, 18, 30, 28, 58, 26, 80, 42, 34], freeze_rows=5, freeze_columns=1)
    sheet.note('Model inputs and operating-point gate controls', 1, 34)
    sheet.note('A: predicted OP features plus optional gate. B: OP only controls the gate; baseline has no OP dependency.', height=34)
    sheet.note('All new non-baseline gate variants use one frozen DEVICE-level OP predictor per seed, followed by a PIN-level encoder.', height=34)
    sheet.note('Staged variants disable gates during base training and freeze encoder weights during final gate training. Oracle/shuffle variants remain diagnostics.', height=34)
    sheet.add([Cell(value, 3) for value in fields], 44)
    for mode in models:
        if mode in STAGED_GATE_INPUTS_PIN:
            base, final_modules = STAGED_GATE_INPUTS_PIN[mode]
            role = ('Same distilled base checkpoint as the other distilled gate variant'
                    if base == 'distilled student' else 'No teacher or latent distillation dependency')
            sheet.add([mode, 'staged gate', 'device', True, 'predicted OP', False,
                       role, base, final_modules], 46)
        elif mode in GATE_INPUTS_PIN:
            group, features, gate = GATE_INPUTS_PIN[mode]
            predictor = 'none' if mode == 'device_op_gate_only_base_pin' else 'device'
            role = ('Measured OP diagnostic reference; excluded from design-only inference group'
                    if mode in ORACLES else
                    'Diagnostic control; deterministic donors from training predicted OP only'
                    if mode in SHUFFLE_CONTROLS_PIN else
                    'No-gate baseline' if gate == 'none' else 'Predicted OP attention gate')
            sheet.add([mode, group, predictor, features, gate, mode in ORACLES, role,
                       'No separate base stage', 'encoder and decoder' if gate == 'none'
                       else 'encoder, decoder and gate'], 46)
        else:
            sheet.add([mode, 'existing pin model', 'see saved model configuration',
                       None, 'no new OP gate', mode in ORACLES, 'Existing model contract is unchanged',
                       'see saved configuration', 'see stage parameter metadata'], 46)
    sheet.filter_range = f'A5:{_column(len(fields))}{len(sheet.rows)}'
    return sheet


def _gate_comparison_sheet(directory, results, *, allow_incomplete):
    """Show each saved ungated base and its final gated descendant on test only.

    These are per-seed physical-unit values, not extra selected models or an
    independent seed sample. Reading them never chooses a checkpoint.
    """
    fields = ('Model', 'Seed', 'Metric', 'Labels', 'Base training', 'Final trainable modules',
              'Base MAE', 'Final MAE', 'MAE change', 'Base RMSE', 'Final RMSE', 'RMSE change',
              'Base R2', 'Final R2', 'R2 change', 'Base checkpoint')
    sheet = Sheet('GateBeforeAfter', [46, 12, 28, 14, 32, 30, *([24] * 9), 80],
                  freeze_rows=5, freeze_columns=3)
    sheet.note('Ungated base versus final gate model — per seed', 1, 34)
    sheet.note('Both test reports are evaluated after training. Test labels do not select checkpoints or fit transforms.', height=34)
    sheet.note('Change = final minus base. Negative MAE/RMSE and positive R2 changes are improvements; values retain physical units (UGF in Hz).', height=34)
    sheet.note('Distilled gate variants reuse one base per seed; its repeated reference is not another independent run. Extra gate training costs remain separate.', height=34)
    sheet.add([Cell(value, 3) for value in fields], 44)
    missing = []
    for record in sorted(results, key=lambda item: (item['mode'], item['seed'])):
        mode, seed = record['mode'], record['seed']
        if mode not in STAGED_GATE_INPUTS_PIN:
            continue
        path = directory / f'seed_{seed}_pin' / mode / 'gate_comparison_pin.json'
        if not path.is_file():
            if not allow_incomplete:
                raise ValueError(f'Missing staged gate comparison: {path}')
            missing.append(str(path))
            sheet.add([mode, seed, 'MISSING COMPARISON'], 32)
            continue
        comparison = _read_json(path)
        if not isinstance(comparison, dict):
            raise ValueError(f'Invalid gate comparison object: {path}')
        checkpoint = comparison.get('base_checkpoint')
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f'Gate comparison requires a base checkpoint path: {path}')
        base, final = comparison.get('base_test'), comparison.get('post_gate_test')
        for report in (base, final):
            if not isinstance(report, dict) or report.get('split') != 'test':
                raise ValueError(f'Gate comparison requires test reports: {path}')
            summary_rows_pin([dict(record, report=report)])
        if final['metrics'] != record['report']['metrics']:
            raise ValueError(f'Final gate comparison differs from the exported model result: {path}')
        if set(base['metrics']) != set(final['metrics']):
            raise ValueError(f'Base and final gate metric schemas differ: {path}')
        for metric, post in final['metrics'].items():
            before = base['metrics'][metric]
            if before['count'] != post['count']:
                raise ValueError(f'Base and final gate label counts differ: {path}')
            values = []
            for score in ('mae', 'rmse', 'r2'):
                old, new = before.get(score), post.get(score)
                change = None if old is None or new is None else new - old
                values.extend(Cell(value, 11) for value in (old, new, change))
            base_name, final_modules = STAGED_GATE_INPUTS_PIN[mode]
            sheet.add([mode, seed, metric, post['count'], base_name, final_modules,
                       *values, checkpoint], 36)
    sheet.filter_range = f'A5:{_column(len(fields))}{len(sheet.rows)}'
    return sheet, missing


def _cost_sheet(stages, results):
    fields = ('Stage ID', 'Used by models', 'Planned epochs', 'Actual epochs', 'Budget unit',
              'Planned boosting rounds per metric', 'Actual boosting rounds per metric',
              'Seconds', 'Optimizer steps', 'Examples seen', 'Parameters',
              'Trainable parameters', 'Device', 'Selection')
    sheet = Sheet('TrainingCost', [48, 64, 20, 20, 36, 28, 70, 22, 22, 22, 22, 24, 18, 80],
                  freeze_rows=5, freeze_columns=1)
    sheet.note('Training costs — each stage ID counted once', 1, 34)
    sheet.note('Shared teachers, OP predictors and ungated bases are counted once per seed; final gate stages add cost.', height=32)
    sheet.note('Epochs do not imply equal compute. XGBoost reports boosting rounds per metric, with blank epoch cells.', height=32)
    sheet.note('TOTAL unique stage seconds = ' + str(sum(stage['seconds'] for stage in stages))
               + '; optimizer steps = ' + str(sum(stage['optimizer_steps'] for stage in stages)))
    sheet.add([Cell(value, 3) for value in fields], 44)
    users = defaultdict(set)
    for record in results:
        for stage_id in record['stage_ids']:
            users[stage_id].add(record['mode'])
    for stage in stages:
        sheet.add([stage['stage_id'], ', '.join(sorted(users[stage['stage_id']])), stage['planned_epochs'],
                   stage['epochs'], stage.get('budget_unit', 'epochs'), stage.get('planned_boosting_rounds'),
                   stage.get('boosting_rounds'), Cell(stage['seconds'], 11), stage['optimizer_steps'], stage.get('examples_seen'),
                   stage.get('parameters'), stage.get('trainable_parameters'), stage.get('device'),
                   stage.get('selection')], 32)
    sheet.filter_range = f'A5:{_column(len(fields))}{len(sheet.rows)}'
    return sheet


def export_results_pin(source, output=None, *, overwrite=False, allow_incomplete=False, finalizing=False, decimals=4):
    """Validate and export a completed run; finalizing accepts ready_to_export.

    An explicit provisional export is possible with allow_incomplete=True. Its
    status is prominent and every comparison highlight is disabled.
    """
    if type(decimals) is not int or not 1 <= decimals <= 10:
        raise ValueError('decimals must be an integer between 1 and 10')
    source = _resolve_source(source)
    directory = source.parent
    output = Path(output).expanduser().resolve() if output is not None else directory / 'summary_pin.xlsx'
    if output.suffix.lower() != '.xlsx' or not output.stem.endswith('_pin'):
        raise ValueError('Pin workbook output must end in _pin.xlsx')
    if output.exists() and not overwrite:
        raise ValueError(f'Excel output already exists; choose a new filename: {output}')
    results, config, run = (_read_json(path) for path in
                            (source, directory / 'config_pin.json', directory / 'run_pin.json'))
    if not isinstance(config, dict) or not config:
        raise ValueError('config_pin.json must contain the effective experiment configuration')
    rows = summary_rows_pin(results, config)
    models, complete, missing = _validate_plan(results, run, allow_incomplete=allow_incomplete, finalizing=finalizing)
    stages = _validate_stages(_read_json(directory / 'stages_pin.json'), results)
    configuration, missing_configs = _configuration_sheet(directory, config, run, results,
                                                           allow_incomplete=allow_incomplete)
    complete = complete and not missing_configs
    if missing:
        configuration.note('MISSING RESULTS: ' + json.dumps(missing), 4, 42)
    gate_comparison = None
    if any(mode in STAGED_GATE_INPUTS_PIN for mode in models):
        gate_comparison, missing_gate_comparisons = _gate_comparison_sheet(
            directory, results, allow_incomplete=allow_incomplete)
        complete = complete and not missing_gate_comparisons
        if missing_gate_comparisons:
            configuration.note('MISSING GATE COMPARISONS: ' + json.dumps(missing_gate_comparisons), 4, 42)
    statistics, fields = _statistics_sheet(rows)
    sheets = _comparison_sheets(rows, models, fields, complete, decimals)
    sheets.extend((_per_seed_sheet(results, config), statistics, _cost_sheet(stages, results), configuration))
    if any(mode in ALL_GATE_REPORTS_PIN for mode in models):
        sheets.append(_model_inputs_sheet(models))
    if gate_comparison is not None:
        sheets.append(gate_comparison)
    # The reusable writer validates all XML and writes atomically. Publishing
    # with link prevents concurrent exporters from overwriting an existing file.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.final-exam-report-pin-', dir=output.parent) as temporary:
        prepared = Path(temporary) / 'report_pin.xlsx'
        _write_workbook(prepared, sheets)
        if overwrite:
            os.replace(prepared, output)
        else:
            try:
                os.link(prepared, output)
            except FileExistsError as exc:
                raise ValueError(f'Excel output already exists: {output}') from exc
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Run directory, raw-run directory, or results_pin.json')
    parser.add_argument('--output', help='New .xlsx filename, default <training-run>/summary_pin.xlsx')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace the derived workbook')
    parser.add_argument('--allow-incomplete', action='store_true', help='Export provisional results without highlighting')
    args = parser.parse_args(argv)
    try:
        path = export_results_pin(args.source, args.output, overwrite=args.overwrite,
                              allow_incomplete=args.allow_incomplete)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(path)


if __name__ == '__main__':
    main()
