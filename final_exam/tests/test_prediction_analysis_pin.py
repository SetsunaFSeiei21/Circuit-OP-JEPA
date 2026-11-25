"""Ground-truth loss/tail calculations and paired-seed inference boundaries."""
import copy

import numpy as np
import pytest

from final_exam.prediction_analysis_pin import (metrics_pin, training_thresholds_pin,
    target_strata_pin, seed_summary_pin, sign_flip_pin, holm_pin, compare_series_pin)


def test_metrics_masks_are_external_constant_targets_and_error_tails():
    r = metrics_pin([1, 2, 3, 4], [1, 2, 3, 8], [0, 0, 0, 2])
    assert r['count'] == 4 and r['mae'] == 1 and r['rmse'] == 2
    assert r['r2'] == pytest.approx(1 - 16/5)
    assert r['normalized_mse'] == 1 and r['top_1pct_sse_fraction'] == 1
    assert r['top_1pct_count'] == 1 and r['top_5pct_count'] == 1
    assert metrics_pin([1, 1], [1, 1], [0, 0])['r2'] is None
    assert metrics_pin([1], [2], [1])['r2'] is None
    assert metrics_pin([], [], [])['normalized_mse'] is None
    with pytest.raises(ValueError, match='finite'):
        metrics_pin([1], [float('nan')], [0])


def test_train_cutoffs_do_not_use_extreme_test_targets_and_strata_partition():
    thresholds = training_thresholds_pin(np.arange(101))
    assert thresholds['p01'] == 1 and thresholds['p99'] == 99
    y = [-1e9, 2, 50, 98, 1e9]
    assert list(target_strata_pin(y, thresholds)) == [
        'below_train_p01', 'train_p01_p05', 'train_p05_p95', 'train_p95_p99', 'at_or_above_train_p99']
    assert len(set(target_strata_pin([3, 3], training_thresholds_pin([3, 3])))) == 1


def test_seed_sign_flip_and_holm_do_not_claim_thousands_of_replicates():
    assert sign_flip_pin([1, 2, 3])[0] == .25
    assert sign_flip_pin([1, 2])[0] == .5
    assert sign_flip_pin([0, 0])[0] == 1
    assert sign_flip_pin([1])[0] is None
    assert holm_pin([.01, .04, .03, None]) == pytest.approx([.03, .06, .06, None])


def item(model, seed, *, cohort='same', reverse=False):
    y = np.array([1., 2., 4.])
    p = y + (2 if model == 'base' else 1)
    result = dict(source_id=model, cohort_id=cohort, circuit='C1', model=model,
                  seed=seed, split='test', metric='gain', ids=['a', 'b', 'c'],
                  target=y, prediction=p, normalized_error=(p-y)/2)
    if reverse:
        for key in ('ids', 'target', 'prediction', 'normalized_error'):
            result[key] = result[key][::-1]
    result['stats'] = metrics_pin(result['target'], result['prediction'], result['normalized_error'])
    return result


def test_pairing_by_sample_ids_all_common_seeds_with_explicit_unmatched():
    series = [item(m, s, reverse=(m=='new')) for s in (17, 29, 43) for m in ('base', 'new')]
    series += [item('new', 99), item('new', 17, cohort='different')]
    pairs, summary, status, cases = compare_series_pin(series, 'base', bootstrap=100, top_errors=1)
    assert len(pairs) == 3 and len(cases) == 6
    assert sum(r['status'] == 'unpaired' for r in status) == 2
    for r in summary:
        assert r['num_paired_seeds'] == 3 and r['p_value'] == .25 and r['low_seed_count']
    mae = next(r for r in summary if r['statistic'] == 'mae')
    assert mae['delta_mean'] == mae['ci95_low'] == mae['ci95_high'] == -1
    shuffled = copy.deepcopy(series)
    shuffled[1]['target'][0] = -10
    with pytest.raises(ValueError, match='different physical labels'):
        compare_series_pin(shuffled, 'base')
    with pytest.raises(ValueError, match='duplicate'):
        compare_series_pin(series + [series[0]], 'base')


def test_summary_ddof_one_and_no_silent_seed_selection():
    rows = []
    for seed, value in ((17, 1), (29, 3), (43, 8)):
        rows.append(dict(cohort_id='same', circuit='C', model='new', split='test', metric='gain',
                         seed=seed, count=10, mae=value, rmse=value, r2=None, normalized_mse=value))
    r = seed_summary_pin(rows)[0]
    assert r['num_seeds'] == 3 and r['mae_mean'] == 4
    assert r['mae_std'] == pytest.approx(np.std([1, 3, 8], ddof=1))
    assert r['r2_mean'] is None and seed_summary_pin(rows[:1])[0]['mae_std'] is None
    with pytest.raises(ValueError, match='Multiple checkpoints'):
        seed_summary_pin(rows + rows[:1])
