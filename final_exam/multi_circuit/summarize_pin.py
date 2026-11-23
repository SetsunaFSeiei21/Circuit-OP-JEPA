"""One workbook writer. Means/std are over seeds, never over test samples."""
from collections import defaultdict
import itertools
import math
from pathlib import Path
import statistics
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from final_exam.multi_circuit.config_pin import read_pin, atomic_pin


def aggregate_pin(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row['circuit'], row['metric'])].append(row)
    output = []
    for (model, circuit, metric), items in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        value = dict(model=model, circuit=circuit, metric=metric, completed_seeds=len(items),
                     seeds=','.join(str(r['seed']) for r in items),
                     valid_count_min=min(r['count'] for r in items), valid_count_max=max(r['count'] for r in items))
        for key in ('mae', 'rmse', 'r2', 'normalized_rmse'):
            numbers = [r[key] for r in items if r.get(key) is not None]
            mean = statistics.mean(numbers) if numbers else None
            std = statistics.stdev(numbers) if len(numbers) > 1 else None
            value.update({key+'_mean': mean, key+'_std': std, key+'_seed_count': len(numbers),
                          key+'_mean±std': 'NA' if mean is None else f'{mean:.6g} ± {std:.6g}' if std is not None else f'{mean:.6g} ± NA'})
        output.append(value)
    return output


def paired_pin(details, baseline):
    groups = defaultdict(dict)
    for row in details:
        groups[(row['view'], row['circuit'], row['metric'])][(row['model'], row['seed'])] = row
    result = []
    for (view, circuit, metric), values in sorted(groups.items()):
        for model in sorted({m for m, s in values if m != baseline}):
            for stat in ('mae', 'rmse', 'r2'):
                seeds = sorted(s for m, s in values if m == model and (baseline, s) in values
                    and values[(m, s)][stat] is not None and values[(baseline, s)][stat] is not None)
                if not seeds:
                    continue
                delta = np.array([values[(model, s)][stat] - values[(baseline, s)][stat] for s in seeds])
                delta *= 1 if stat == 'r2' else -1
                mean = float(delta.mean())
                p = None
                if len(delta) <= 16:
                    null = [abs(float(np.mean(delta * signs))) for signs in itertools.product((-1, 1), repeat=len(delta))]
                    p = sum(x >= abs(mean)-1e-15 for x in null)/len(null)
                result.append(dict(view=view, circuit=circuit, metric=metric, statistic=stat,
                    baseline=baseline, model=model, paired_seeds=','.join(map(str, seeds)), n=len(delta),
                    mean_improvement=mean, std_improvement=float(delta.std(ddof=1)) if len(delta)>1 else None,
                    exact_sign_flip_p=p, assumptions='paired seeds; symmetric null; exploratory; five pairs minimum two-sided p=0.0625'))
    eligible = sorted((r for r in result if r['exact_sign_flip_p'] is not None), key=lambda r: r['exact_sign_flip_p'])
    previous = 0.
    for i, row in enumerate(eligible):
        previous = max(previous, min(1., (len(eligible)-i)*row['exact_sign_flip_p']))
        row['holm_adjusted_p_all_workbook_comparisons'] = previous
    return result


def sheet_pin(book, title, rows):
    sheet = book.create_sheet(title)
    if not rows:
        sheet.append(['No completed records'])
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    sheet.append(keys)
    for row in rows:
        sheet.append([row.get(k) if not isinstance(row.get(k), (list, dict)) else str(row[k]) for k in keys])
    sheet.freeze_panes = 'D2'
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.fill = PatternFill('solid', fgColor='15395B')
        cell.font = Font(color='FFFFFF', bold=True)
    for column in sheet.columns:
        sheet.column_dimensions[column[0].column_letter].width = min(58, max(14, max(len(str(c.value or '')) for c in column[:200])+2))
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = '0.000000'


def export_pin(campaign):
    from final_exam.multi_circuit.worker_pin import lock_pin
    campaign = Path(campaign).resolve()
    with lock_pin(campaign/'reports_pin/report_pin.lock'):
        record = read_pin(campaign/'campaign_pin.json')
        details, statuses, trials, bests, costs = [], [], [], [], []
        reference_cohorts = {}
        for model in record['models']:
            path = campaign/'selected_pin'/f'{model}.json'
            if path.exists():
                best = read_pin(path)
                bests.append(dict(model=model, selection=best['selection'], trial=best.get('trial'),
                    validation_macro_r2=best.get('validation_macro_r2'), config_path=str(path),
                    configuration_digest=best['configuration_digest']))
            trial_file = campaign/'search_pin'/model/'trials_pin.json'
            if trial_file.exists():
                trials.extend(dict(model=model, **r) for r in read_pin(trial_file))
                for trial in read_pin(trial_file):
                    state = read_pin(campaign/'search_pin'/model/trial['trial']/'status_pin.json')
                    if state['status'] == 'complete':
                        for stage in read_pin(Path(state['training_output'])/'stages_pin.json'):
                            costs.append({**stage, 'model': model, 'phase': 'search', 'trial': trial['trial']})
            for seed in record['protocol']['seeds']:
                status_path = campaign/'training_pin'/f'seed_{seed}_pin'/model/'status_pin.json'
                status = read_pin(status_path) if status_path.exists() else {'status':'not_started'}
                statuses.append(dict(model=model, seed=seed, status=status['status'], status_path=str(status_path)))
                if status['status'] != 'complete':
                    continue
                report = read_pin(status['report'])
                # Check paired cohorts and required per-sample artifacts, independent of row order.
                with np.load(status['predictions'], allow_pickle=False) as arrays:
                    cohort = set(zip(arrays['circuit_id'].tolist(), arrays['sample_id'].tolist()))
                    if len(cohort) != len(arrays['sample_id']):
                        raise ValueError('Duplicate test sample identities')
                    key = tuple(arrays['metric_names'])
                    if key in reference_cohorts and reference_cohorts[key] != cohort:
                        raise ValueError('Formal model/seed test cohorts differ')
                    reference_cohorts[key] = cohort
                for view, section in [('PerCircuit', report['per_circuit']), ('PooledTest', {'ALL': report['pooled']}),
                                       ('CircuitMacro', {'MACRO': report['circuit_macro']})]:
                    for circuit, metrics in section.items():
                        for metric, values in metrics.items():
                            details.append(dict(view=view, model=model, seed=seed, circuit=circuit, metric=metric, **values))
                stage_file = Path(status['training_output'])/'stages_pin.json'
                seen = set()
                for stage in read_pin(stage_file):
                    if stage['stage_id'] in seen:
                        continue
                    seen.add(stage['stage_id'])
                    costs.append({**stage, 'model': model, 'seed': seed, 'phase': 'formal'})
        summaries = {view: aggregate_pin([r for r in details if r['view'] == view]) for view in ('PerCircuit','PooledTest','CircuitMacro')}
        paired = paired_pin(details, record['protocol'].get('baseline', 'device_op_cascade_pin'))
        reportdir = campaign/'reports_pin'
        atomic_pin(reportdir/'summary_pin.json', dict(**summaries, SeedDetails=details, SearchTrials=trials,
            BestConfigs=bests, TrainingCost=costs, RunStatus=statuses, PairedComparison=paired,
            protocol=record['protocol'], smoke=record['smoke'], source_digest=record['source_digest']))
        book = Workbook()
        book.remove(book.active)
        for title, rows in {**summaries, 'SeedDetails':details,'SearchTrials':trials,'BestConfigs':bests,
                            'TrainingCost':costs,'RunStatus':statuses,'PairedComparison':paired}.items():
            sheet_pin(book, title, rows)
        sheet_pin(book, 'Protocol', [dict(setting=k, value=v) for k,v in record['protocol'].items()] + [
            dict(setting='smoke', value=record['smoke']), dict(setting='std', value='sample standard deviation across seeds, ddof=1; n<2 is NA'),
            dict(setting='pooled', value='physical predictions concatenated within seed, metrics recomputed before seed aggregation'),
            dict(setting='incomplete', value='all completed seeds shown; use completed_seeds/RunStatus; no best/worst seed picking'),
            dict(setting='source_digest', value=record['source_digest'])])
        target = reportdir/'summary_pin.xlsx'
        temporary = target.with_suffix('.tmp.xlsx')
        book.save(temporary)
        temporary.replace(target)
        print(f'Workbook: {target}', flush=True)
        return summaries
