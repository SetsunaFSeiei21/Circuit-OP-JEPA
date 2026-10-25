"""Strict, dependency-free comparison of matched evaluation reports.

Never pool physical units across metrics or pool validation, test and OOD rows.
Seed summaries are unweighted means/sample SDs of per-seed metrics, not a new
pooled R². Missing labels are excluded; nonfinite predictions are errors.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

MODES = ('zerosim_pin', 'zerosim_device', 'jepa_zerosim')


def regression_metrics(target, prediction):
    if len(target) != len(prediction):
        raise ValueError('Target/prediction length mismatch')
    pairs = []
    for y, p in zip(target, prediction):
        if y is None or not math.isfinite(y):
            continue
        if p is None or not math.isfinite(p):
            raise ValueError('Nonfinite prediction for an observed target')
        pairs.append((float(y), float(p)))
    if not pairs:
        return dict(count=0, mae=None, rmse=None, r2=None)
    ys, ps = zip(*pairs)
    errors = [p-y for y, p in pairs]
    if any(not math.isfinite(e) for e in errors):
        raise ValueError('Physical prediction error exceeds finite reporting range')

    def rms(values):
        scale = max(abs(v) for v in values)
        if scale == 0:
            return 0.
        return scale * math.sqrt(math.fsum((v / scale) ** 2 for v in values) / len(values))

    error_rms = rms(errors)
    center = mean(ys)
    deviations = [y - center for y in ys]
    if any(not math.isfinite(v) for v in deviations):
        raise ValueError('Physical target variation exceeds finite reporting range')
    target_rms = rms(deviations)
    r2 = None
    if len(pairs) > 1 and target_rms > 0:
        ratio = error_rms / target_rms
        r2 = 1 - ratio * ratio
        if not math.isfinite(r2):
            raise ValueError('R2 exceeds finite reporting range')
    return dict(count=len(pairs), mae=mean(abs(e) for e in errors),
                rmse=error_rms, r2=r2)



def _identity(report):
    p = report['provenance']
    required = ('partition_fingerprint', 'input_op_fingerprint', 'target_scaler_digest', 'input_transform_digest',
                'labeled_train_ids', 'train_ids')
    for key in required:
        if key not in p:
            raise ValueError(f'Missing comparison provenance: {key}')
    data_id = p.get('dataset_fingerprint', p.get('performance_fingerprint'))
    if not data_id:
        raise ValueError('Missing full dataset/performance fingerprint')
    return (data_id, report.get('data_fingerprint'), *(json.dumps(p[k], sort_keys=True) for k in required),
            tuple(report['metric_names']), bool(report['synthetic']))


def _records(report):
    records = report['predictions']
    if not records:
        raise ValueError('Cannot compare an empty evaluation split')
    ids = [(r['circuit_id'], r['sample_id']) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate evaluation sample IDs')
    if len(ids) != report['num_samples']:
        raise ValueError('Report sample count does not match predictions')
    metrics = report['metric_names']
    for r in records:
        for key in ('target', 'prediction'):
            if set(r[key]) != set(metrics):
                raise ValueError(f'{key} fields do not match metric_names')
        for metric in metrics:
            y = r['target'][metric]
            ny = r.get('normalized_target', {}).get(metric)
            if y is not None and math.isfinite(y) and (ny is None or not math.isfinite(ny)):
                raise ValueError('Normalized prediction/target missing for observed label')
            if (y is None or not math.isfinite(y)) and ny is not None and math.isfinite(ny):
                raise ValueError('Physical and normalized missing-label masks differ')
    return sorted(records, key=lambda r: (r['circuit_id'], r['sample_id']))


def compare_reports(reports, require_all_modes=True):
    if not reports:
        raise ValueError('No reports supplied')
    identity = _identity(reports[0])
    cohorts, seen, per_seed, costs = {}, set(), [], []
    mode_seeds = defaultdict(set)
    backends, mode_configs, architectures = {}, {}, {}
    supervised_budget = None
    for report in reports:
        if _identity(report) != identity:
            raise ValueError('Incompatible data, split, labels, metric names or target preprocessing')
        p = report['provenance']
        mode, seed, split = p['mode'], p['seed'], report['split']
        if split not in ('test', 'ood', 'val', 'train'):
            raise ValueError('Unknown evaluation split')
        if mode not in MODES:
            raise ValueError(f'Unknown mode: {mode}')
        backend = p.get('backend')
        if not backend:
            raise ValueError('Missing backend provenance')
        if mode in backends and backends[mode] != backend:
            raise ValueError('Cannot mix backend implementations within a mode')
        backends[mode] = backend
        config = p.get('training_config')
        if not isinstance(config, dict):
            raise ValueError('Missing training_config provenance')
        keys = ('decoder_lr', 'batch_size', 'weight_decay', 'grad_clip')
        if any(k not in config for k in (*keys, 'decoder_epochs')):
            raise ValueError('Incomplete supervised training budget')
        epochs = config.get('baseline_epochs', config['decoder_epochs']) if mode != 'jepa_zerosim' else config['decoder_epochs']
        budget = (epochs, *(config[k] for k in keys))
        if supervised_budget is not None and supervised_budget != budget:
            raise ValueError('Incompatible supervised training budget')
        supervised_budget = budget
        mode_config = {k: v for k, v in config.items() if k != 'seed'}
        if mode in mode_configs and mode_configs[mode] != mode_config:
            raise ValueError('Training configuration differs across seeds within a mode')
        mode_configs[mode] = mode_config
        architecture = p.get('model_config')
        if not isinstance(architecture, dict):
            raise ValueError('Missing model_config provenance')
        if mode in architectures and architectures[mode] != architecture:
            raise ValueError('Model configuration differs across seeds within a mode')
        architectures[mode] = architecture
        key = mode, seed, split
        if key in seen:
            raise ValueError(f'Duplicate mode/seed/split: {key}')
        seen.add(key)
        mode_seeds[split, mode].add(seed)
        records = _records(report)
        targets = [(r['circuit_id'], r['sample_id'], r['target'], r.get('normalized_target')) for r in records]
        if split in cohorts and cohorts[split] != targets:
            raise ValueError('Evaluation IDs or targets differ within the same split')
        cohorts[split] = targets
        for circuit in ['__all__'] + sorted({r['circuit_id'] for r in records}):
            subset = [r for r in records if circuit == '__all__' or r['circuit_id'] == circuit]
            for metric in report['metric_names']:
                for scale, ykey, pkey in [('physical', 'target', 'prediction'),
                                          ('normalized', 'normalized_target', 'normalized_prediction')]:
                    if scale == 'normalized' and not all(ykey in r and pkey in r for r in subset):
                        raise ValueError('Normalized prediction/target records are required for comparable metrics')
                    scores = regression_metrics([r[ykey][metric] for r in subset], [r[pkey][metric] for r in subset])
                    per_seed.append(dict(mode=mode, backend=p.get('backend', 'unspecified'), seed=seed,
                                         split=split, circuit=circuit, metric=metric, scale=scale, **scores))
        costs.append(dict(mode=mode, seed=seed, split=split, timing=p.get('timing', {}),
                          parameter_count=p.get('parameter_count')))
    # Device supervised and JEPA must differ in optimization, not architecture.
    # The pin reproduction may differ only in node-budget and terminal-role bias.
    if 'zerosim_device' in architectures and 'jepa_zerosim' in architectures:
        if architectures['zerosim_device'] != architectures['jepa_zerosim']:
            raise ValueError('Model configuration differs between device and JEPA modes')
    common_architectures = [{k: v for k, v in cfg.items() if k not in ('max_devices', 'terminal_bias')}
                            for cfg in architectures.values()]
    if any(cfg != common_architectures[0] for cfg in common_architectures[1:]):
        raise ValueError('Model configuration differs across comparison modes')
    if require_all_modes:
        for split in cohorts:
            sets = [mode_seeds[split, mode] for mode in MODES]
            if not sets[0] or any(s != sets[0] for s in sets):
                raise ValueError(f'All three modes must have matching seed sets for {split}: {sets}')
    groups = defaultdict(list)
    for row in per_seed:
        groups[tuple(row[k] for k in ('mode', 'backend', 'split', 'circuit', 'metric', 'scale'))].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(('mode', 'backend', 'split', 'circuit', 'metric', 'scale'), key))
        item.update(num_seeds=len(rows), count_per_seed=rows[0]['count'])
        for metric in ('mae', 'rmse', 'r2'):
            values = [r[metric] for r in rows if r[metric] is not None]
            item[metric+'_defined_seeds'] = len(values)
            item[metric+'_mean'] = mean(values) if values else None
            item[metric+'_std'] = stdev(values) if len(values)>1 else None
        summary.append(item)
    return dict(schema_version=1, synthetic=reports[0]['synthetic'], per_seed=per_seed,
                summary=summary, run_costs=costs,
                warning='Smoke/toy results validate mechanics only, not generalization. R2 undefined for fewer than two observations or constant targets. Physical units are dataset units. Normalized metrics use training-only target transforms. Costs are per run, repeated by evaluation split; do not sum duplicate split costs.')


def _csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-partial', action='store_true', help='Allow missing modes/seeds; still enforce provenance')
    args = parser.parse_args(argv)
    result = compare_reports([json.loads(Path(p).read_text()) for p in args.reports], not args.allow_partial)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    _csv(out/'per_seed.csv', result['per_seed'])
    _csv(out/'summary.csv', result['summary'])
    _csv(out/'costs.csv', [{**r, 'timing': json.dumps(r['timing'], sort_keys=True),
                            'parameter_count': json.dumps(r['parameter_count'], sort_keys=True)} for r in result['run_costs']])
    (out/'README.txt').write_text(result['warning']+'\nsummary.csv: one row per mode/backend/split/circuit/metric/scale. Across-seed SD uses ddof=1; blank with one seed. Never compare different units or different splits as one score.\n')
    print(f'Wrote matched comparison to {out}')


if __name__ == '__main__':
    main()
