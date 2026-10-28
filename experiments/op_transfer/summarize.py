#!/usr/bin/env python3
"""Export OP-transfer results to a model x metric Excel comparison (mean ± std)."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from device_jepa.excel import SCORES, export_summary, prepare_rows


def summary_rows(results):
    """Keep test/OOD separate; never invent per-circuit scores from pooled ones."""
    if not isinstance(results, list) or not results:
        raise ValueError('Expected nonempty results.json records')
    groups, seen, metric_sets = defaultdict(list), set(), {}
    for record in results:
        if not isinstance(record, dict) or not isinstance(record.get('report'), dict):
            raise ValueError('Each result must contain a report object')
        mode, seed, report = record.get('mode'), record.get('seed'), record['report']
        split = report.get('split')
        if not isinstance(mode, str) or not mode or type(seed) is not int or seed < 0:
            raise ValueError('Result mode and seed are invalid')
        if split not in ('test', 'ood') or not isinstance(report.get('metrics'), dict):
            raise ValueError('Expected test/OOD metric reports')
        cohort = (mode, split)
        names = frozenset(report['metrics'])
        if cohort in metric_sets and metric_sets[cohort] != names:
            raise ValueError('Metric schema differs across seeds; compare compatible reports')
        metric_sets[cohort] = names
        identity = (mode, seed, split)
        if identity in seen:
            raise ValueError(f'Duplicate model/seed/split result: {identity}')
        seen.add(identity)
        requires_op = report.get('requires_measured_op')
        if type(requires_op) is not bool or requires_op != (mode == 'op_oracle'):
            raise ValueError('Measured OP input contract disagrees with the model name')
        for metric, values in report['metrics'].items():
            if not isinstance(metric, str) or not metric or not isinstance(values, dict):
                raise ValueError('Malformed metric report')
            count = values.get('count')
            if type(count) is not int or count < 0:
                raise ValueError('Metric count must be a nonnegative integer')
            for score in SCORES:
                value = values.get(score)
                if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
                    raise ValueError(f'{score} must be finite or null')
                if value is not None and count == 0:
                    raise ValueError('Defined score cannot have zero observed labels')
            groups[(mode, split, metric)].append((seed, values, requires_op))
    rows = []
    for (mode, split, metric), members in sorted(groups.items()):
        counts = {values['count'] for _, values, _ in members}
        if len(counts) != 1:
            raise ValueError(f'Label counts differ across seeds for {(mode, split, metric)}; compare matched cohorts')
        item = dict(mode=mode, backend='op_transfer', split=split, scope='pooled',
                    circuit='ALL_SELECTED_CIRCUITS', metric=metric, scale='physical',
                    num_seeds=len(members), count_per_seed=counts.pop(),
                    requires_op_at_inference=members[0][2], uses_design_parameters=mode != 'op_oracle')
        for score in SCORES:
            values = [entry[score] for _, entry, _ in members if entry.get(score) is not None]
            item[score + '_defined_seeds'] = len(values)
            item[score + '_mean'] = statistics.mean(values) if values else None
            item[score + '_std'] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(item)
    return prepare_rows(rows)


def export_results(source, output=None):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        source = source / 'results.json'
    if not source.is_file():
        raise ValueError(f'No results.json file found: {source}')
    output = Path(output).expanduser().resolve() if output is not None else source.with_name('summary.xlsx')
    if output.exists():
        raise ValueError(f'Excel output already exists; choose a new filename: {output}')
    if output.suffix.lower() != '.xlsx':
        raise ValueError('Excel output must end in .xlsx')
    rows = summary_rows(json.loads(source.read_text()))
    run_path = source.with_name('run.json')
    metadata = {}
    if run_path.is_file():
        run = json.loads(run_path.read_text())
        metadata['complete'] = run.get('status') == 'complete'
    return export_summary(rows, output, source=str(source), metadata=metadata,
                          split='all', scope='pooled', scale='physical')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Run directory or results.json')
    parser.add_argument('--output', help='New .xlsx filename; default <run>/summary.xlsx')
    args = parser.parse_args(argv)
    try:
        output = export_results(args.source, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == '__main__':
    main()
