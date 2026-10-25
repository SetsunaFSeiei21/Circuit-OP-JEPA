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

LEGACY_MODES = ('zerosim_pin', 'zerosim_device', 'jepa_zerosim')
OP_MODES = ('zerosim_pin_op', 'zerosim_device_op')
MODES = (*LEGACY_MODES, *OP_MODES)
OP_PAIRS = (('pin', 'zerosim_pin', 'zerosim_pin_op'),
            ('device', 'zerosim_device', 'zerosim_device_op'))


def _information_contract(report):
    """Make extra information and unmeasured acquisition costs explicit.

    Old three-mode reports have an unambiguous no-OP-at-inference contract.
    New OP-input reports must declare their requirement rather than silently
    masquerading as zero-simulation inference.
    """
    p = report['provenance']
    mode = p['mode']
    required = mode in OP_MODES
    declared = p.get('requires_op_at_inference', False)
    if type(declared) is not bool or declared != required:
        raise ValueError(f'Incorrect or missing requires_op_at_inference for {mode}')
    training_use = 'input' if required else ('pretraining_target' if mode == 'jepa_zerosim' else 'none')
    inference_use = 'input' if required else 'none'
    info = p.get('op_information', {})
    if not isinstance(info, dict):
        raise ValueError('op_information must be a dictionary')
    for key, value in (('training_use', training_use), ('inference_use', inference_use)):
        if key in info and info[key] != value:
            raise ValueError(f'Incorrect OP {key} for {mode}')
    if required and info.get('state_features'):
        raise ValueError('Continuous OP-input baselines must not consume discrete OP states')
    if training_use != 'none' and 'scaler_fitted_sample_ids' in info:
        if sorted(info['scaler_fitted_sample_ids']) != sorted(p['train_ids']):
            raise ValueError('OP scaler must be fitted on the shared ID training samples only')
    seconds = info.get('acquisition_cost_seconds')
    status = info.get('acquisition_cost_status', 'not_measured')
    if status not in ('not_measured', 'measured'):
        raise ValueError('OP acquisition cost status must be not_measured or measured')
    if seconds is not None:
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                or status != 'measured' or not info.get('acquisition_cost_scope')):
            raise ValueError('Measured OP acquisition cost requires finite seconds, measured status and scope')
    elif status == 'measured':
        raise ValueError('Measured OP acquisition cost requires seconds')
    return dict(requires_op_at_inference=required, op_training_use=training_use,
                op_inference_use=inference_use,
                op_source=info.get('source'), op_feature_names=info.get('feature_names'),
                op_state_features=info.get('state_features'), op_state_config=info.get('state_config'),
                op_eligibility=info.get('eligibility'), op_evaluation_coverage=info.get('evaluation_coverage'),
                op_availability_requirement=('continuous DC OP for every inference sample (missing fields remain masked)'
                                             if required else ('DC OP during JEPA pretraining only; no OP at inference'
                                                               if mode == 'jepa_zerosim' else 'no OP model input')),
                op_acquisition_cost_seconds=seconds, op_acquisition_cost_status=status,
                op_acquisition_cost_scope=info.get('acquisition_cost_scope'))


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


def compare_reports(reports, require_all_modes=True, expected_modes=None):
    if not reports:
        raise ValueError('No reports supplied')
    supplied_modes = {r['provenance']['mode'] for r in reports}
    # Preserve strict comparison of pre-existing three-mode result sets. Once
    # an OP-input report is supplied, all five are expected unless the caller
    # explicitly declares a different cohort (e.g. a two-mode ablation).
    if expected_modes is None:
        expected_modes = LEGACY_MODES if supplied_modes <= set(LEGACY_MODES) else MODES
    expected_modes = tuple(expected_modes)
    if not expected_modes or len(expected_modes) != len(set(expected_modes)) or any(m not in MODES for m in expected_modes):
        raise ValueError('Expected modes must be a nonempty unique selection of supported modes')
    if not supplied_modes <= set(expected_modes):
        raise ValueError('Reports contain modes outside the declared comparison cohort')
    identity = _identity(reports[0])
    cohorts, seen, per_seed, costs = {}, set(), [], []
    mode_seeds = defaultdict(set)
    backends, mode_configs, architectures, mode_op_protocols = {}, {}, {}, {}
    supervised_budget, op_transform = None, None
    for report in reports:
        if _identity(report) != identity:
            raise ValueError('Incompatible data, split, labels, metric names or target preprocessing')
        p = report['provenance']
        mode, seed, split = p['mode'], p['seed'], report['split']
        if split not in ('test', 'ood', 'val', 'train'):
            raise ValueError('Unknown evaluation split')
        if mode not in MODES:
            raise ValueError(f'Unknown mode: {mode}')
        information = _information_contract(report)
        # Continuous OP scaling alone does not identify the JEPA target branch:
        # categorical state channels/configuration may change its input width
        # while leaving the structure encoder and raw dataset unchanged.
        op_protocol = {k: information[k] for k in ('op_source', 'op_feature_names',
                       'op_state_features', 'op_state_config', 'op_eligibility')}
        if mode in mode_op_protocols and mode_op_protocols[mode] != op_protocol:
            raise ValueError('OP information protocol differs across seeds or splits within a mode')
        mode_op_protocols[mode] = op_protocol
        if mode in OP_MODES and not p.get('op_input_transform_digest'):
            raise ValueError('Missing continuous OP input scaler provenance')
        if information['op_training_use'] != 'none' and p.get('op_input_transform_digest'):
            if op_transform is not None and op_transform != p['op_input_transform_digest']:
                raise ValueError('Incompatible continuous OP input scalers')
            op_transform = p['op_input_transform_digest']
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
                                         split=split, circuit=circuit, metric=metric, scale=scale,
                                         requires_op_at_inference=information['requires_op_at_inference'],
                                         op_training_use=information['op_training_use'], **scores))
        costs.append(dict(mode=mode, seed=seed, split=split, timing=p.get('timing', {}),
                          parameter_count=p.get('parameter_count'), **information))
    # Device supervised and JEPA must differ in optimization, not architecture.
    # The pin reproduction may differ only in node-budget and terminal-role bias.
    if 'zerosim_device' in architectures and 'jepa_zerosim' in architectures:
        if architectures['zerosim_device'] != architectures['jepa_zerosim']:
            raise ValueError('Model configuration differs between device and JEPA modes')
    for _, baseline_mode, op_mode in OP_PAIRS:
        if baseline_mode in architectures and op_mode in architectures and architectures[baseline_mode] != architectures[op_mode]:
            raise ValueError(f'Model configuration differs within OP/no-OP pair {baseline_mode}/{op_mode}')
    if 'zerosim_device_op' in architectures and 'jepa_zerosim' in architectures:
        if architectures['zerosim_device_op'] != architectures['jepa_zerosim']:
            raise ValueError('Model configuration differs between device OP and JEPA modes')
    common_architectures = [{k: v for k, v in cfg.items() if k not in ('max_devices', 'terminal_bias')}
                            for cfg in architectures.values()]
    if any(cfg != common_architectures[0] for cfg in common_architectures[1:]):
        raise ValueError('Model configuration differs across comparison modes')
    if require_all_modes:
        for split in cohorts:
            sets = [mode_seeds[split, mode] for mode in expected_modes]
            if not sets[0] or any(s != sets[0] for s in sets):
                raise ValueError(f'All declared modes must have matching seed sets for {split}: {dict(zip(expected_modes, sets))}')
    groups = defaultdict(list)
    for row in per_seed:
        groups[tuple(row[k] for k in ('mode', 'backend', 'split', 'circuit', 'metric', 'scale'))].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(('mode', 'backend', 'split', 'circuit', 'metric', 'scale'), key))
        item.update(num_seeds=len(rows), count_per_seed=rows[0]['count'],
                    requires_op_at_inference=rows[0]['requires_op_at_inference'],
                    op_training_use=rows[0]['op_training_use'])
        for metric in ('mae', 'rmse', 'r2'):
            values = [r[metric] for r in rows if r[metric] is not None]
            item[metric+'_defined_seeds'] = len(values)
            item[metric+'_mean'] = mean(values) if values else None
            item[metric+'_std'] = stdev(values) if len(values)>1 else None
        summary.append(item)
    paired_per_seed, paired_summary = _paired_deltas(per_seed)
    complete = all(mode_seeds[split, expected_modes[0]] and
                   all(mode_seeds[split, mode] == mode_seeds[split, expected_modes[0]] for mode in expected_modes)
                   for split in cohorts)
    return dict(schema_version=2, synthetic=reports[0]['synthetic'],
                expected_modes=list(expected_modes), compared_modes=[m for m in MODES if m in supplied_modes],
                complete=complete, per_seed=per_seed, summary=summary, run_costs=costs,
                paired_per_seed=paired_per_seed, paired_summary=paired_summary,
                warning='Smoke/toy results validate mechanics only, not generalization. R2 undefined for fewer than two observations or constant targets. Physical units are dataset units. Normalized metrics use training-only target transforms. Costs are per run, repeated by evaluation split; do not sum duplicate split costs. OP-input baselines require extra DC OP information at inference, unlike no-OP baselines and JEPA (training-only OP). OP acquisition/simulation cost is unknown unless explicitly measured, never assumed zero or included in model phase timing. Label/epoch parity does not imply equal information, parameters or total compute. Paired deltas are OP minus no-OP: negative MAE/RMSE and positive R2 favor OP; these are descriptive, not significance tests.')


def _paired_deltas(per_seed):
    """Within-seed contrasts; never subtract unpaired across-seed means."""
    index = {(r['mode'], *(r[k] for k in ('seed', 'split', 'circuit', 'metric', 'scale'))): r for r in per_seed}
    paired = []
    for granularity, baseline_mode, op_mode in OP_PAIRS:
        for row in per_seed:
            if row['mode'] != baseline_mode:
                continue
            key = tuple(row[k] for k in ('seed', 'split', 'circuit', 'metric', 'scale'))
            op = index.get((op_mode, *key))
            if op is None:
                continue
            item = dict(node_granularity=granularity, baseline_mode=baseline_mode, op_mode=op_mode,
                        **dict(zip(('seed', 'split', 'circuit', 'metric', 'scale'), key)),
                        count=row['count'], baseline_requires_op_at_inference=False, op_requires_op_at_inference=True)
            for metric in ('mae', 'rmse', 'r2'):
                baseline_value, op_value = row[metric], op[metric]
                item[f'no_op_{metric}'], item[f'op_{metric}'] = baseline_value, op_value
                delta = op_value - baseline_value if baseline_value is not None and op_value is not None else None
                if delta is not None and not math.isfinite(delta):
                    raise ValueError('Paired delta exceeds finite reporting range')
                item[f'{metric}_delta'] = delta
            paired.append(item)
    groups = defaultdict(list)
    fields = ('node_granularity', 'baseline_mode', 'op_mode', 'split', 'circuit', 'metric', 'scale')
    for row in paired:
        groups[tuple(row[k] for k in fields)].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(fields, key))
        item.update(num_paired_seeds=len(rows), count_per_seed=rows[0]['count'])
        for metric in ('mae', 'rmse', 'r2'):
            values = [r[f'{metric}_delta'] for r in rows if r[f'{metric}_delta'] is not None]
            for prefix in ('no_op', 'op'):
                scores = [r[f'{prefix}_{metric}'] for r in rows if r[f'{prefix}_{metric}'] is not None]
                item[f'{prefix}_{metric}_mean'] = mean(scores) if scores else None
            item[f'{metric}_delta_defined_seeds'] = len(values)
            item[f'{metric}_delta_mean'] = mean(values) if values else None
            item[f'{metric}_delta_std'] = stdev(values) if len(values) > 1 else None
        summary.append(item)
    return paired, summary


def _csv(path, rows):
    if not rows:
        # Reprocessing a legacy cohort into an existing comparison directory
        # must not leave stale OP-pair files from an earlier five-mode cohort.
        path.unlink(missing_ok=True)
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
    parser.add_argument('--modes', nargs='+', choices=MODES,
                        help='Explicit expected cohort; default is legacy three for legacy-only reports, otherwise five')
    args = parser.parse_args(argv)
    result = compare_reports([json.loads(Path(p).read_text()) for p in args.reports], not args.allow_partial, args.modes)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    _csv(out/'per_seed.csv', result['per_seed'])
    _csv(out/'summary.csv', result['summary'])
    _csv(out/'paired_per_seed.csv', result['paired_per_seed'])
    _csv(out/'paired_summary.csv', result['paired_summary'])
    json_columns = ('timing', 'parameter_count', 'op_feature_names', 'op_state_features',
                    'op_state_config', 'op_evaluation_coverage')
    _csv(out/'costs.csv', [{k: json.dumps(v, sort_keys=True) if k in json_columns and v is not None else v
                          for k, v in r.items()} for r in result['run_costs']])
    (out/'README.txt').write_text(result['warning']+'\nsummary.csv: one row per mode/backend/split/circuit/metric/scale. Across-seed SD uses ddof=1; blank with one seed. Never compare different units or different splits as one score.\npaired_per_seed.csv and paired_summary.csv: only matched seeds within pin/device OP/no-OP pairs. Deltas are OP minus no-OP. Missing/undefined values are blank, not zero. These files are omitted if no pairs exist.\n')
    print(f'Wrote matched comparison to {out}')


if __name__ == '__main__':
    main()
