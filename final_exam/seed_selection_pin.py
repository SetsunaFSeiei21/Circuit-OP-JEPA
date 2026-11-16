"""Select whole model/seed results by test macro R2, retaining an audit trail.

This is a descriptive, test-selected view, not an unbiased benchmark estimate.
The training exporter validates the complete source before invoking this module.
"""
from collections import defaultdict
import math
import statistics

from device_jepa.excel import Cell, Sheet, _column


SELECTION_LABEL_PIN = 'TEST-SELECTED SEEDS'


def select_seeds_pin(results, *, k, models_better, models_worse, circuit_of):
    """Rank test results once per model, then keep whole seeds on every split.

    For a multi-circuit source, first average metric R2 within each test circuit,
    then average circuits equally. Pooled and disaggregated reports cannot mix.
    A metric undefined for every candidate is excluded and recorded; partially
    undefined R2 is rejected instead of silently rewarding missing metrics.
    """
    if type(k) is not int or k < 1:
        raise ValueError('k must be a positive integer')
    groups = {}
    for name, modes in (('better', models_better), ('worse', models_worse)):
        if (not isinstance(modes, (list, tuple))
                or any(not isinstance(m, str) or not m for m in modes)
                or len(set(modes)) != len(modes)):
            raise ValueError(f'MODELS_{name.upper()} must contain distinct model names')
        groups[name] = list(modes)
    if not groups['better'] and not groups['worse']:
        raise ValueError('Set at least one model in MODELS_BETTER or MODELS_WORSE')
    overlap = set(groups['better']) & set(groups['worse'])
    if overlap:
        raise ValueError(f'Models cannot appear in both groups: {sorted(overlap)}')
    available = {r['mode'] for r in results}
    requested = set(groups['better']) | set(groups['worse'])
    if requested - available:
        raise ValueError(f'Unknown models: {sorted(requested - available)}; available: {sorted(available)}')

    candidates = defaultdict(set)
    tests, by_circuit = {}, defaultdict(list)
    for record in results:
        model, seed = record['mode'], record['seed']
        candidates[model].add(seed)
        if record['report']['split'] != 'test':
            continue
        circuit = circuit_of(record)
        key = (model, seed, circuit)
        if key in tests:
            raise ValueError(f'Duplicate test result: {key}')
        tests[key] = record
        by_circuit[circuit].append(record)
    circuits = sorted(by_circuit)
    if not circuits:
        raise ValueError('Seed ranking requires test reports; OOD results cannot substitute for test')
    if len(circuits) > 1 and 'ALL_SELECTED_CIRCUITS' in circuits:
        raise ValueError('Cannot mix pooled and per-circuit test reports for seed selection')
    for model, seeds in candidates.items():
        for seed in seeds:
            missing = [c for c in circuits if (model, seed, c) not in tests]
            if missing:
                raise ValueError(f'Missing test circuits for {model}, seed {seed}: {missing}')
        if model in requested and len(seeds) < k:
            raise ValueError(f'{model} has only {len(seeds)} candidate seeds; requested k={k}')

    included, excluded = {}, {}
    for circuit, records in sorted(by_circuit.items()):
        schemas = {frozenset(r['report']['metrics']) for r in records}
        if len(schemas) != 1:
            raise ValueError(f'Metric schema differs among candidates for {circuit}')
        included[circuit], excluded[circuit] = [], []
        for metric in sorted(next(iter(schemas))):
            values = [r['report']['metrics'][metric].get('r2') for r in records]
            if all(v is None for v in values):
                excluded[circuit].append(metric)
                continue
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError(f'{circuit}/{metric}: R2 must be finite for every candidate '
                                 'or undefined for all; cannot rank different metric sets')
            included[circuit].append(metric)
        if not included[circuit]:
            raise ValueError(f'No defined R2 metrics for seed ranking: {circuit}')

    chosen, rankings = {}, []
    for model, seeds in sorted(candidates.items()):
        scores = {}
        for seed in sorted(seeds):
            scores[seed] = {
                c: statistics.mean(tests[(model, seed, c)]['report']['metrics'][m]['r2']
                                   for m in included[c]) for c in circuits}
        macro = {seed: statistics.mean(values.values()) for seed, values in scores.items()}
        policy = ('best' if model in groups['better'] else
                  'worst' if model in groups['worse'] else 'all')
        # Both directions resolve equal scores with the lower seed ID first.
        ordered = sorted(seeds, key=lambda seed: ((macro[seed] if policy == 'worst'
                                                   else -macro[seed]), seed))
        chosen[model] = sorted(ordered[:k] if policy != 'all' else ordered)
        for rank, seed in enumerate(sorted(seeds, key=lambda seed: (-macro[seed], seed)), 1):
            rankings.append(dict(model=model, seed=seed, policy=policy,
                                 requested_k=k if policy != 'all' else None,
                                 candidate_count=len(seeds), selected=seed in chosen[model],
                                 rank_best_first=rank, test_macro_r2=macro[seed],
                                 circuit_scores=scores[seed]))
    selected = [r for r in results if r['seed'] in chosen[r['mode']]]
    audit = dict(kind='test_macro_r2_per_model', k=k, models_better=groups['better'],
                 models_worse=groups['worse'], selected_seeds=chosen, rankings=rankings,
                 ranking_split='test', ranking_circuits=circuits,
                 included_metrics=included, excluded_undefined_metrics=excluded,
                 tie_break='ascending integer seed ID',
                 aggregation='equal metric mean per test circuit, then equal circuit mean',
                 other_models='retain all seeds',
                 other_splits='retain the same model/seed selection; never rank on OOD',
                 selected_records=len(selected), source_records=len(results))
    return selected, audit


def selection_sheet_pin(audit, *, source, source_sha256):
    fields = ('Model', 'Seed', 'Policy', 'Requested k', 'Candidate seeds', 'Selected',
              'Rank (best first)', 'Test macro R2', 'Per-circuit test macro R2')
    sheet = Sheet('SeedSelection', [48, 12, 14, 16, 20, 14, 22, 24, 90],
                  freeze_rows=6, freeze_columns=2)
    sheet.note(f'{SELECTION_LABEL_PIN} — complete candidate ranking', 1, 38)
    sheet.note('Descriptive selected-seed results; not an unbiased all-seed benchmark. '
               'MAE, RMSE and R2 use the same retained seeds.', height=40)
    sheet.note('Best = highest test macro R2; worst = lowest. Ties use ascending seed ID. '
               'Unlisted models retain all seeds.', height=40)
    sheet.note('Test circuits and metrics have equal weight; OOD reports follow the test-selected '
               'seeds. Undefined-for-all metrics are excluded; partial missing R2 is an error.', height=42)
    sheet.note(f'Source: {source}', height=40)
    sheet.add([Cell(f, 3) for f in fields], 40)
    for row in audit['rankings']:
        sheet.add([row['model'], row['seed'], row['policy'], row['requested_k'],
                   row['candidate_count'], row['selected'], row['rank_best_first'],
                   Cell(row['test_macro_r2'], 11), row['circuit_scores']], 34)
    sheet.filter_range = f'A6:{_column(len(fields))}{len(sheet.rows)}'
    sheet.note(f'Source results SHA256: {source_sha256}', height=34)
    # Separate notes avoid Excel's cell limit on large model lists.
    for circuit in audit['ranking_circuits']:
        sheet.note(f'{circuit} included R2 metrics: {audit["included_metrics"][circuit]}', height=46)
        sheet.note(f'{circuit} undefined-for-all R2 metrics: '
                   f'{audit["excluded_undefined_metrics"][circuit]}', height=36)
    return sheet


def label_selected_sheets_pin(sheets):
    """Make selection visible without moving formula sources or table headers."""
    for sheet in sheets:
        sheet.rows[0][0].value = f'{SELECTION_LABEL_PIN} — {sheet.rows[0][0].value}'
        sheet.heights[1] = 46
        if sheet.name in ('MAE', 'RMSE', 'R2'):
            sheet.rows[1][0].value += ('; model-specific seeds selected by test macro R2. '
                                      'See SeedSelection; not an all-seed benchmark.')
            sheet.heights[2] = 42
