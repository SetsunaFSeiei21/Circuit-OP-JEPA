"""Paired frozen-branch diagnostics; no registry or production comparison changes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from device_jepa.excel import export_summary


SHARED = ('source_checkpoint_sha256', 'source_mode', 'pretrain_seed',
          'source_model_digest', 'target_scaler_digest', 'input_transform_digest',
          'partition_fingerprint', 'input_op_fingerprint', 'data_fingerprint', 'train_ids',
          'labeled_train_ids', 'metric_names', 'readout_policy', 'shared_pretraining_cost')
SCORES = ('mae', 'rmse', 'r2')
IDENTITY = ('mode', 'backend', 'split', 'scope', 'circuit', 'metric', 'scale')
SEED_NOTE = ('Standard deviations describe decoder training seeds on ONE fixed '
             'pretraining checkpoint, not independent pretraining runs. '
             'Validation reports use the same partition used for checkpoint selection. '
             'OP probe requires DC OP at inference; design probe does not. '
             'Both readouts use topology-eligible device tokens and exclude the graph token. '
             'This is a paired representation diagnostic, not a production model leaderboard.')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _config(report):
    config = dict(report['training_config'])
    config.pop('seed', None)
    return config


def _targets(report):
    ids = report['evaluation_ids']
    predictions = report['predictions']
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or not isinstance(predictions, list) or len(predictions) != len(ids)):
        raise ValueError('Evaluation IDs and predictions must describe unique matching samples')
    if [row['sample_id'] for row in predictions] != ids:
        raise ValueError('Prediction IDs differ from ordered evaluation IDs')
    return [(row['sample_id'], row['circuit_id'], row['target']) for row in predictions]


def _validate_metric(metric):
    if type(metric.get('count')) is not int or metric['count'] < 0:
        raise ValueError('Metric count must be a nonnegative integer')
    for score in SCORES:
        value = metric.get(score)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('Metrics must be finite numbers or null')
        if score != 'r2':
            if (metric['count'] > 0) != (value is not None) or (value is not None and value < 0):
                raise ValueError('Metric error/count mismatch')
        elif metric['count'] < 2 and value is not None:
            raise ValueError('R2 is undefined with fewer than two observations')


def build_summary(reports):
    """Validate source/cohort pairing before computing decoder-seed statistics."""
    if not isinstance(reports, list) or not reports:
        raise ValueError('Expected nonempty frozen branch probe reports')
    pairs, references, seeds, runs_by_branch = {}, {}, set(), {}
    anchor = None
    per_seed = []
    for report in reports:
        required = (*SHARED, 'branch', 'mode', 'backend', 'decoder_seed', 'split',
                    'decoder_initial_digest', 'training_config', 'encoder_frozen_verified',
                    'frozen_model_digest', 'requires_op_at_inference', 'uses_design_parameters',
                    'evaluation_ids', 'predictions', 'metrics', 'by_circuit', 'num_samples',
                    'decoder_digest', 'selection', 'early_stopping', 'training_cost')
        if not isinstance(report, dict) or any(key not in report for key in required):
            raise ValueError('Missing frozen branch probe report provenance')
        branch, seed, split = report['branch'], report['decoder_seed'], report['split']
        if report.get('experiment_kind') != 'frozen_branch_probe' or branch not in ('design', 'op'):
            raise ValueError('Only frozen branch probe reports are accepted')
        if report['mode'] != f'{branch}_probe' or report['backend'] != 'frozen_branch_probe_v1':
            raise ValueError('Probe mode/backend contradicts branch identity')
        if type(seed) is not int or report.get('seed', seed) != seed or split not in ('val', 'test', 'ood'):
            raise ValueError('Invalid decoder seed or evaluation split')
        if report['training_config'].get('seed') != seed:
            raise ValueError('Decoder training config seed differs from report seed')
        if report['selection'].get('split') != 'val' or report.get('selection_used_ood') is not False:
            raise ValueError('Probe checkpoints must be selected exclusively by ID validation')
        if report['requires_op_at_inference'] is not (branch == 'op'):
            raise ValueError('OP inference requirement contradicts branch')
        if report['uses_design_parameters'] is not (branch == 'design'):
            raise ValueError('Design parameter requirement contradicts branch')
        if (report['encoder_frozen_verified'] is not True
                or report['frozen_model_digest'] != report['source_model_digest']):
            raise ValueError('Frozen source model integrity was not verified')
        if report['readout_policy'] != 'topology_eligible_device_tokens_no_graph_v1':
            raise ValueError('Unsupported paired readout policy')
        protocol = {key: report[key] for key in SHARED}
        protocol['training_config'] = _config(report)
        signed = _canonical(protocol)
        if anchor is None:
            anchor = signed
        elif signed != anchor:
            raise ValueError('Mixed source checkpoints, preprocessing, partitions or decoder training protocol')
        key = (seed, split)
        pair = pairs.setdefault(key, {})
        if branch in pair:
            raise ValueError('Duplicate branch/decoder seed/split report')
        pair[branch] = report
        seeds.add(seed)
        targets = _targets(report)
        if report['num_samples'] != len(targets):
            raise ValueError('Evaluation sample count differs from prediction records')
        run_key = (seed, branch)
        run_identity = _canonical({name: report.get(name) for name in
                                   ('decoder_digest', 'decoder_initial_digest', 'epoch', 'selection',
                                    'early_stopping', 'training_cost', 'shared_pretraining_cost')})
        if run_key in runs_by_branch and runs_by_branch[run_key] != run_identity:
            raise ValueError('Evaluation splits refer to different decoder checkpoints or training runs')
        runs_by_branch[run_key] = run_identity
        target_signature = _canonical(targets)
        if split in references and references[split] != target_signature:
            raise ValueError('Evaluation cohort or targets differ across branches/decoder seeds')
        references[split] = target_signature
        if set(report['metrics']) != set(report['metric_names']):
            raise ValueError('Metric schema differs from source preprocessing')
        actual_circuits = {row[1] for row in targets}
        if set(report['by_circuit']) != actual_circuits:
            raise ValueError('Per-circuit reports differ from evaluation membership')
        for scope, circuit, metrics in [('pooled', '__all__', report['metrics']),
                                        *[('circuit', name, values) for name, values in report['by_circuit'].items()]]:
            if set(metrics) != set(report['metric_names']):
                raise ValueError('Per-circuit metric schema mismatch')
            for name, values in metrics.items():
                _validate_metric(values)
                observed = sum(target.get(name) is not None for _, c, target in targets
                               if scope == 'pooled' or c == circuit)
                if values['count'] != observed:
                    raise ValueError('Reported metric count differs from evaluation targets')
                per_seed.append(dict(mode=report['mode'], backend=report['backend'], branch=branch,
                    split=split, scope=scope, circuit=circuit, metric=name, scale='physical',
                    decoder_seed=seed, count=values['count'], **{s: values[s] for s in SCORES}))
    for split in references:
        if {seed for seed, subset in pairs if subset == split} != seeds:
            raise ValueError('Evaluation splits must have matching decoder seed sets')
    for (seed, split), pair in pairs.items():
        if set(pair) != {'design', 'op'}:
            raise ValueError(f'Missing paired design/OP report for seed {seed}, split {split}')
        if pair['design']['decoder_initial_digest'] != pair['op']['decoder_initial_digest']:
            raise ValueError('Paired branches did not start from identical decoder parameters')

    grouped = defaultdict(list)
    for row in per_seed:
        grouped[tuple(row[k] for k in IDENTITY)].append(row)
    summary = []
    first = reports[0]
    for key, rows in sorted(grouped.items()):
        counts = {row['count'] for row in rows}
        if len(counts) != 1:
            raise ValueError('Label counts differ across decoder seeds')
        item = dict(zip(IDENTITY, key))
        branch = rows[0]['branch']
        item.update(branch=branch, num_seeds=len(rows), count_per_seed=counts.pop(),
                    requires_op_at_inference=branch == 'op', uses_design_parameters=branch == 'design',
                    source_mode=first['source_mode'], fixed_pretrain_seed=first['pretrain_seed'],
                    seed_axis='decoder_only', source_checkpoint_sha256=first['source_checkpoint_sha256'])
        for score in SCORES:
            values = [row[score] for row in rows if row[score] is not None]
            item.update({score+'_mean': mean(values) if values else None,
                         score+'_std': stdev(values) if len(values) > 1 else None,
                         score+'_defined_seeds': len(values)})
        summary.append(item)
    index = {(r['decoder_seed'], r['split'], r['scope'], r['circuit'], r['metric'], r['branch']): r
             for r in per_seed}
    deltas = []
    for row in per_seed:
        if row['branch'] != 'design':
            continue
        key = tuple(row[k] for k in ('decoder_seed', 'split', 'scope', 'circuit', 'metric'))
        op = index[(*key, 'op')]
        item = dict(zip(('decoder_seed', 'split', 'scope', 'circuit', 'metric'), key))
        item.update(scale='physical', count=row['count'], direction='op_minus_design')
        for score in SCORES:
            item[score+'_delta'] = (op[score] - row[score]
                                   if op[score] is not None and row[score] is not None else None)
        deltas.append(item)
    delta_groups = defaultdict(list)
    for row in deltas:
        delta_groups[tuple(row[k] for k in ('split', 'scope', 'circuit', 'metric', 'scale'))].append(row)
    paired = []
    for key, rows in sorted(delta_groups.items()):
        item = dict(zip(('split', 'scope', 'circuit', 'metric', 'scale'), key))
        item.update(num_decoder_seeds=len(rows), direction='op_minus_design')
        for score in SCORES:
            values = [r[score+'_delta'] for r in rows if r[score+'_delta'] is not None]
            item.update({score+'_delta_mean': mean(values) if values else None,
                         score+'_delta_std': stdev(values) if len(values) > 1 else None,
                         score+'_defined_seeds': len(values)})
        paired.append(item)
    runs = []
    for seed in sorted(seeds):
        pair = next(pair for (s, _), pair in pairs.items() if s == seed)
        for branch in ('design', 'op'):
            report = pair[branch]
            runs.append({key: report.get(key) for key in
                         ('branch', 'decoder_seed', 'training_cost', 'early_stopping', 'selection')})
    return dict(schema_version=1, experiment_kind='frozen_branch_probe', complete=True,
                source_checkpoint_sha256=first['source_checkpoint_sha256'], source_mode=first['source_mode'],
                pretrain_seed=first['pretrain_seed'], decoder_seeds=sorted(seeds),
                readout_policy=first['readout_policy'], seed_axis='decoder_only_fixed_pretraining_checkpoint',
                evaluation_splits=sorted(references), protocol=json.loads(anchor), note=SEED_NOTE,
                summary=summary, per_seed=per_seed, paired_deltas=deltas, paired_summary=paired,
                decoder_runs=runs, shared_pretraining_cost=first.get('shared_pretraining_cost'))


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _label_decoder_seeds(path):
    """Specialize generic workbook labels without changing the shared exporter."""
    replacements = {
        'Circuit model comparison': 'Frozen branch diagnostic — one pretraining checkpoint',
        'Mean ± sample standard deviation across training seeds (ddof=1).':
            'Mean ± sample SD across DECODER seeds (ddof=1); ONE fixed pretraining checkpoint.',
        'Training seeds': 'Decoder seeds',
    }
    with ZipFile(path) as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with NamedTemporaryFile(dir=path.parent, suffix='.xlsx', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, 'w') as target:
            for info, content in entries:
                if info.filename.startswith('xl/worksheets/'):
                    value = content.decode('utf-8')
                    for before, after in replacements.items():
                        value = value.replace(before, after)
                    content = value.encode('utf-8')
                target.writestr(info, content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_reports(reports, output):
    result = build_summary(reports)  # Validate everything before creating files.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('summary', result['summary']), ('per_seed', result['per_seed']),
                       ('paired_deltas', result['paired_deltas']), ('paired_summary', result['paired_summary'])]:
        _csv(output / f'{name}.csv', rows)
    workbook = output / 'model_comparison.xlsx'
    export_summary(result['summary'], workbook, source=f'{output / "summary.json"} | {SEED_NOTE}',
                   metadata=result, models=['design_probe', 'op_probe'])
    _label_decoder_seeds(workbook)
    result['artifacts'] = {name: str(output / name) for name in
                           ('summary.json', 'summary.csv', 'per_seed.csv', 'paired_deltas.csv',
                            'paired_summary.csv', 'model_comparison.xlsx')}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Re-export paired branch probe reports; no training.')
    parser.add_argument('--input', required=True, help='One branch_probe run directory')
    parser.add_argument('--output', help='Defaults to INPUT/comparison; must remain in branch_probe/results')
    args = parser.parse_args(argv)
    root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / 'comparison'
    if RESULTS_ROOT.resolve() != RESULTS_ROOT or not output.is_relative_to(RESULTS_ROOT):
        parser.error('Output must be within experiments/branch_probe/results')
    try:
        reports = [json.loads(path.read_text()) for path in sorted(root.glob('seed_*/*/*_report.json'))]
        result = summarize_reports(reports, output)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(result['artifacts']['model_comparison.xlsx'])


if __name__ == '__main__':
    main()
