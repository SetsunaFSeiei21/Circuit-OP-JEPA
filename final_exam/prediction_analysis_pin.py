"""Statistics for saved predictions; no fitting, seed selection or label filtering.

Formal paired units are training seeds on an identical saved cohort. Test rows
are not treated as independent training repetitions. Bootstrap intervals describe
training-seed variation conditional on this fixed dataset/split.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
import math

import numpy as np

METRICS = ('mae', 'rmse', 'r2', 'normalized_mse')
IDENTITY = ('source_id', 'cohort_id', 'circuit', 'model', 'seed', 'split', 'metric')


def stable_seed(value, seed=17):
    text = json.dumps([seed, value], sort_keys=True, ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def metrics_pin(target, prediction, normalized_error):
    """Undefined R² (n<2 or constant target) stays None, never forced to 0/1."""
    from device_jepa.comparison import regression_metrics
    y, p, ne = (np.asarray(a, dtype=np.float64) for a in (target, prediction, normalized_error))
    if y.ndim != 1 or y.shape != p.shape or y.shape != ne.shape:
        raise ValueError('Prediction/target/error arrays must be matching vectors')
    if not all(np.isfinite(a).all() for a in (y, p, ne)):
        raise ValueError('Observed predictions and targets must be finite')
    result = regression_metrics(y.tolist(), p.tolist())
    result['normalized_mse'] = float(np.mean(ne ** 2)) if len(y) else None
    if not len(y):
        return result
    error = p - y
    absolute = np.abs(error)
    result.update(bias=float(error.mean()), median_absolute_error=float(np.median(absolute)))
    for quantile in (50, 90, 95, 99, 100):
        result[f'absolute_error_p{quantile}'] = float(np.percentile(absolute, quantile))
    # Scale before squaring to avoid overflow in physical units.
    scale = float(absolute.max())
    squared = (absolute / scale) ** 2 if scale else np.zeros(len(y))
    ordered = np.sort(squared)[::-1]
    for percent in (1, 5):
        k = max(1, math.ceil(len(y) * percent / 100))
        result[f'top_{percent}pct_count'] = k
        result[f'top_{percent}pct_sse_fraction'] = float(ordered[:k].sum() / ordered.sum()) if scale else 0.
    return result


def training_thresholds_pin(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Training labels must be finite vectors')
    if not len(values):
        return None
    return dict(count=len(values), minimum=float(values.min()), maximum=float(values.max()),
                **{f'p{q:02d}': float(np.percentile(values, q)) for q in (1, 5, 95, 99)})


def target_strata_pin(target, thresholds):
    """Five disjoint strata. Cutoffs are fitted on saved labeled TRAIN rows only."""
    target = np.asarray(target)
    if thresholds is None:
        return np.full(len(target), 'no_training_labels', dtype=object)
    cutoffs = [thresholds[f'p{q:02d}'] for q in (1, 5, 95, 99)]
    names = np.array(['below_train_p01', 'train_p01_p05', 'train_p05_p95',
                      'train_p95_p99', 'at_or_above_train_p99'])
    return names[np.searchsorted(cutoffs, target, side='right')]


def mean_std_pin(values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    return dict(defined_seeds=len(values), mean=float(values.mean()) if len(values) else None,
                std=float(values.std(ddof=1)) if len(values) > 1 else None)


def seed_summary_pin(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ('cohort_id', 'circuit', 'model', 'split', 'metric'))].append(row)
    result = []
    for key, group in sorted(groups.items()):
        seeds = [r['seed'] for r in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Multiple checkpoints for the same cohort/model/seed; choose one source run')
        item = dict(zip(('cohort_id', 'circuit', 'model', 'split', 'metric'), key))
        item.update(seeds=sorted(seeds), num_seeds=len(seeds), count_per_seed={str(r['seed']): r['count'] for r in group})
        for metric in METRICS:
            stats = mean_std_pin([r.get(metric) for r in group])
            item.update({f'{metric}_{k}': v for k, v in stats.items()})
            item[f'{metric}_mean_std'] = (f'{stats["mean"]:.6g} ± {stats["std"]:.6g}'
                if stats['std'] is not None else f'{stats["mean"]:.6g} (one seed)' if stats['mean'] is not None else None)
        result.append(item)
    return result


def sign_flip_pin(differences, *, seed=17, draws=20000):
    """Two-sided paired sign-flip test under seed-difference sign exchangeability."""
    d = np.asarray(differences, dtype=np.float64)
    if len(d) < 2:
        return None, 'insufficient_paired_seeds'
    if not np.isfinite(d).all():
        raise ValueError('Nonfinite paired seed differences')
    if np.all(d == 0):
        return 1., 'exact_seed_sign_flip'
    d = d / np.max(np.abs(d))
    observed = abs(float(d.mean()))
    if len(d) <= 16:
        extreme = sum(abs(sum(x * y for x, y in zip(signs, d)) / len(d)) >= observed - 1e-12
                      for signs in itertools.product((-1, 1), repeat=len(d)))
        return extreme / 2 ** len(d), 'exact_seed_sign_flip'
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, draws, 256):
        signs = rng.choice([-1., 1.], size=(min(256, draws - start), len(d)))
        extreme += int(np.count_nonzero(np.abs((signs * d).mean(1)) >= observed - 1e-12))
    return (extreme + 1) / (draws + 1), 'monte_carlo_seed_sign_flip'


def holm_pin(values):
    """Holm family-wise correction; undefined tests remain undefined."""
    output = [None] * len(values)
    order = sorted((p, i) for i, p in enumerate(values) if p is not None)
    previous = 0.
    for rank, (p, i) in enumerate(order):
        previous = max(previous, min(1., p * (len(order) - rank)))
        output[i] = previous
    return output


def paired_seed_summary_pin(rows, *, bootstrap=2000, seed=17):
    groups = defaultdict(list)
    keys = ('cohort_id', 'circuit', 'model', 'baseline', 'split', 'metric')
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        for metric in METRICS:
            selected = [r for r in group if r.get(f'{metric}_delta') is not None]
            d = np.array([r[f'{metric}_delta'] for r in selected], dtype=np.float64)
            item = dict(zip(keys, key), statistic=metric, paired_seeds=[r['seed'] for r in selected],
                        num_paired_seeds=len(d), direction='positive_favors_model' if metric == 'r2' else 'negative_favors_model')
            item.update({f'delta_{k}': v for k, v in mean_std_pin(d).items()})
            rng = np.random.default_rng(stable_seed([key, metric], seed))
            ci = (None, None)
            if len(d) > 1 and bootstrap:
                means = np.mean(rng.choice(d, size=(bootstrap, len(d)), replace=True), axis=1)
                ci = tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))
            p, method = sign_flip_pin(d, seed=stable_seed([key, metric], seed))
            item.update(ci95_low=ci[0], ci95_high=ci[1], p_value=p, test=method,
                        bootstrap_unit='paired_training_seed_fixed_dataset', bootstrap_draws=bootstrap,
                        low_seed_count=len(d) < 5)
            output.append(item)
    families = defaultdict(list)
    for i, row in enumerate(output):
        families[(row['cohort_id'], row['circuit'], row['split'])].append(i)
    for indices in families.values():
        adjusted = holm_pin([output[i]['p_value'] for i in indices])
        for i, p in zip(indices, adjusted):
            output[i]['p_holm'] = p
            output[i]['holm_family'] = 'all_models_metrics_statistics_within_cohort_circuit_split'
    return output


def compare_series_pin(series, baseline, *, bootstrap=2000, seed=17, top_errors=20):
    """Pair exact identities; never match rows by position or cherry-pick seeds."""
    lookup = {}
    for item in series:
        key = tuple(item[k] for k in ('cohort_id', 'circuit', 'model', 'seed', 'split', 'metric'))
        if key in lookup:
            raise ValueError('Ambiguous duplicate model/seed/cohort checkpoints; remove duplicate source runs')
        lookup[key] = item
    comparisons, status, cases = [], [], []
    for key, candidate in lookup.items():
        if candidate['model'] == baseline:
            continue
        ref_key = (key[0], key[1], baseline, *key[3:])
        reference = lookup.get(ref_key)
        identity = {k: candidate[k] for k in IDENTITY}
        identity['baseline'] = baseline
        if reference is None:
            status.append(dict(identity, status='unpaired', reason='No completed baseline with identical cohort/seed/split/metric'))
            continue
        left = {sid: i for i, sid in enumerate(reference['ids'])}
        right = {sid: i for i, sid in enumerate(candidate['ids'])}
        if left.keys() != right.keys():
            status.append(dict(identity, status='unpaired', reason='Observed target sample IDs differ; no intersection-only comparison'))
            continue
        ids = sorted(left)
        a, b = [left[s] for s in ids], [right[s] for s in ids]
        if not np.array_equal(reference['target'][a], candidate['target'][b]):
            raise ValueError('Paired sample IDs have different physical labels')
        row = dict(identity, baseline_source_id=reference['source_id'], count=len(ids))
        for metric in METRICS:
            x, y = reference['stats'].get(metric), candidate['stats'].get(metric)
            row.update({f'{metric}_baseline': x, f'{metric}_model': y,
                        f'{metric}_delta': y - x if x is not None and y is not None else None})
        comparisons.append(row)
        status.append(dict(identity, status='paired', reason='Exact sample IDs, target values, dataset/split and normalization'))
        if ids and top_errors:
            delta = candidate['normalized_error'][b] ** 2 - reference['normalized_error'][a] ** 2
            for direction, order in [('largest_degradation', np.argsort(-delta, kind='stable')),
                                     ('largest_improvement', np.argsort(delta, kind='stable'))]:
                for i in order[:top_errors]:
                    cases.append(dict(identity, baseline_source_id=reference['source_id'], category=direction,
                        sample_id=ids[i], target=float(candidate['target'][b[i]]),
                        baseline_prediction=float(reference['prediction'][a[i]]),
                        model_prediction=float(candidate['prediction'][b[i]]),
                        normalized_squared_error_delta=float(delta[i])))
    return comparisons, paired_seed_summary_pin(comparisons, bootstrap=bootstrap, seed=seed), status, cases
