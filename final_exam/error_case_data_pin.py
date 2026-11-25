"""Read-only raw evidence and empirical training references for selected cases."""
from __future__ import annotations

from contextlib import closing
import json
import math
from pathlib import Path
import re
import sqlite3


def finite_pin(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def select_cases_pin(dataset, circuit, sample_ids):
    """Match ORIGINAL source_sample_id, never array position or sample_index."""
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Provide distinct original sample IDs')
    found = {}
    for row in dataset['samples']:
        sid = row.get('source_sample_id')
        if row['circuit_id'] == circuit and sid in sample_ids:
            if sid in found:
                raise ValueError(f'Ambiguous source sample ID: {sid}')
            found[sid] = row
    missing = set(sample_ids) - found.keys()
    if missing:
        raise ValueError(f'IDs absent from the saved sampled dataset: {sorted(missing)}; do not resample')
    return [found[sid] for sid in sample_ids]


class TrainReferencePin:
    """Empirical physical-value percentiles; exact device identity, train only."""
    def __init__(self, rows):
        import numpy as np
        columns = {}
        for row in rows:
            for field, value in row.get('raw_design_parameters', {}).items():
                if finite_pin(value):
                    columns.setdefault(('design', '', field), []).append(value)
            for device in row['devices']:
                identity = (device['device_id'], device['device_type'], device.get('model_name', ''))
                for category in ('parameters', 'op'):
                    for field, value in device.get(category, {}).items():
                        if finite_pin(value):
                            columns.setdefault((category, identity, field), []).append(value)
        self.columns = {k: np.sort(np.asarray(v, dtype=np.float64)) for k, v in columns.items()}
        self.summaries = {k: self.summary(v) for k, v in self.columns.items()}

    @staticmethod
    def summary(values):
        import numpy as np
        if len(values) == 0:
            return dict(train_count=0)
        qs = np.quantile(values, [0., .01, .05, .5, .95, .99, 1.])
        return dict(train_count=len(values), **dict(zip(
            ('train_min', 'train_p01', 'train_p05', 'train_p50', 'train_p95', 'train_p99', 'train_max'),
            map(float, qs))))

    @staticmethod
    def percentile(values, value):
        import numpy as np
        if value is None or not math.isfinite(value) or not len(values):
            return None
        # Midrank: constant training fields place an equal observation at 50%.
        left, right = np.searchsorted(values, value, side='left'), np.searchsorted(values, value, side='right')
        return float((left + right) * 50. / len(values))

    def describe(self, category, device, field, value):
        identity = ((device['device_id'], device['device_type'], device.get('model_name', ''))
                    if device else '')
        key = (category, identity, field)
        values = self.columns.get(key, ())
        result = dict(self.summaries.get(key, dict(train_count=0)))
        result['train_percentile'] = self.percentile(values, value)
        result['outside_train_range'] = (bool(value < values[0] or value > values[-1])
            if finite_pin(value) and len(values) else None)
        return result


def _decode(row):
    result = {}
    for key, value in dict(row).items():
        if isinstance(value, bytes):
            result[key] = dict(binary_bytes=len(value), exported=False)
        elif key.endswith('_json') and isinstance(value, str):
            try:
                result[key] = json.loads(value)
            except ValueError:
                result[key] = value
        else:
            result[key] = value
    return result


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: ('[redacted]' if re.search(r'password|secret|credential|api[_-]?key|access[_-]?token', k, re.I)
                    else _clean(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def artifact_candidates_pin(value, root, sid, prefix=''):
    """Inventory explicit metadata paths only; do not guess which trace is a case."""
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(artifact_candidates_pin(child, root, sid, f'{prefix}.{key}'.strip('.')))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            result.extend(artifact_candidates_pin(child, root, sid, f'{prefix}[{i}]'))
    elif isinstance(value, str) and re.search(r'path|file|dir|wave|trace|artifact', prefix, re.I):
        if value and '\n' not in value and len(value) < 4096 and '://' not in value:
            path = Path(value).expanduser()
            path = path if path.is_absolute() else root / path
            try:
                exists, is_file = path.exists(), path.is_file()
            except OSError:
                exists, is_file = False, False
            result.append(dict(source_sample_id=sid, metadata_key=prefix, path=str(path),
                exists=exists, is_file=is_file, status='candidate_only_not_verified_as_waveform'))
    return result


def raw_evidence_pin(raw_dir, circuit, cases):
    """Small parameterized SQLite queries in a read-only snapshot, no simulation."""
    root = Path(raw_dir).expanduser().resolve()
    database = root / 'sampling_history.sqlite3'
    details, checks, candidates = {}, [], []
    info = dict(directory=str(root), database=str(database), status='missing_database', tables=[])
    if not database.is_file():
        return info, details, checks, candidates
    ids = [row['source_sample_id'] for row in cases]
    placeholders = ','.join('?' for _ in ids)
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as con:
        con.row_factory = sqlite3.Row
        con.enable_load_extension(False)
        con.execute('PRAGMA query_only=ON')
        con.execute('PRAGMA trusted_schema=OFF')
        con.execute('BEGIN')
        tables = {r['name']: dict(r) for r in con.execute('SELECT name,type,sql FROM sqlite_master')}
        info['tables'] = [dict(name=k, type=v['type']) for k, v in tables.items() if not k.startswith('sqlite_')]
        selected = {}
        for name in ('dataset_schema', 'samples', 'dc_device_operating_points', 'dc_operating_points'):
            if name not in tables:
                selected[name] = []
                continue
            if tables[name]['type'] != 'table' or not re.match(r'\s*CREATE\s+TABLE\b', tables[name]['sql'] or '', re.I):
                raise ValueError(f'Expected ordinary source table: {name}')
            query = (f'SELECT * FROM {name} LIMIT 2' if name == 'dataset_schema' else
                     f'SELECT * FROM {name} WHERE sample_id IN ({placeholders})')
            selected[name] = [_clean(_decode(r)) for r in con.execute(query, () if name == 'dataset_schema' else ids)]
        if len(selected['dataset_schema']) != 1 or selected['dataset_schema'][0].get('circuit_name') != circuit:
            raise ValueError('Raw database circuit differs from the saved experiment')
        schema = selected['dataset_schema'][0]
        info.update(status='read_only_selected_rows', schema=schema)
        def vector(value, names):
            if isinstance(value, dict):
                return value
            if isinstance(value, list) and len(value) == len(names):
                return dict(zip(names, value))
            return {}
        for case in cases:
            sid = case['source_sample_id']
            rows = {name: [r for r in selected[name] if r.get('sample_id') == sid]
                    for name in ('samples', 'dc_device_operating_points', 'dc_operating_points')}
            details[str(sid)] = rows
            candidates.extend(artifact_candidates_pin(rows, root, sid))
            if len(rows['samples']) != 1:
                checks.append(dict(source_sample_id=sid, field='samples', status='missing_or_duplicate',
                                   raw_count=len(rows['samples'])))
                continue
            raw = rows['samples'][0]
            comparisons = [('design_key', case.get('design_key', case.get('design_id')), raw.get('design_key'))]
            design = vector(raw.get('design_values_json'), schema.get('parameter_names_json', []))
            metrics = vector(raw.get('metric_values_json'), schema.get('metric_names_json', []))
            for label, left, right in (('design', case.get('raw_design_parameters', {}), design),
                                       ('performance', case.get('performance', {}), metrics)):
                comparisons.extend((f'{label}.{k}', v, right.get(k)) for k, v in left.items())
            device_rows = {}
            for row in rows['dc_device_operating_points']:
                name = str(row.get('device_name', '')).lower()
                if name in device_rows:
                    raise ValueError(f'Duplicate raw OP device: {sid} {name}')
                device_rows[name] = row
            for device in case['devices']:
                raw_op = device_rows.get(device['device_id'].lower(), {})
                comparisons.extend((f"op.{device['device_id']}.{k}", v, raw_op.get(k))
                                   for k, v in device.get('op', {}).items())
            for field, saved, raw_value in comparisons:
                equal = (math.isclose(saved, raw_value, rel_tol=1e-10, abs_tol=0.)
                         if finite_pin(saved) and finite_pin(raw_value) else saved == raw_value)
                checks.append(dict(source_sample_id=sid, field=field, saved_value=saved,
                                   raw_value=raw_value, status='matched' if equal else 'mismatch'))
        info['case_values_match'] = bool(checks) and all(r['status'] == 'matched' for r in checks)
    return info, details, checks, candidates


def waveform_evidence_pin(manifest_path, circuit, sample_ids, output, plots=True):
    """Plot explicitly mapped headered CSV/TSV/whitespace traces, without SPICE.

    Finite differences are diagnostics, NOT the source simulator's SR metric.
    An absent trace is never reconstructed from scalar SR labels or DC OP.
    """
    import csv
    import numpy as np
    from final_exam.gate_probe_pin import file_digest
    statuses, figures, entries = [], [], []
    output = Path(output)
    if manifest_path:
        path = Path(manifest_path).expanduser().resolve()
        content = json.loads(path.read_text(encoding='utf-8'))
        if content.get('version') != 1 or content.get('circuit') != circuit or not isinstance(content.get('waveforms'), list):
            raise ValueError('Waveform manifest needs version=1, matching circuit, and waveforms list')
        entries = content['waveforms']
        seen = set()
        for entry in entries:
            sid = entry.get('source_sample_id')
            key = (sid, entry.get('path'))
            if type(sid) is not int or sid not in sample_ids or key in seen:
                raise ValueError('Waveform identity is unknown/duplicated; use ORIGINAL source_sample_id')
            seen.add(key)
        base = path.parent
    else:
        base = output
    for sid in sample_ids:
        mapped = [e for e in entries if e['source_sample_id'] == sid and e.get('path')]
        if not mapped:
            statuses.append(dict(source_sample_id=sid, status='not_supplied',
                reason='No explicitly sample-mapped saved trace. Scalar metrics/DC OP cannot recover a waveform.'))
        for number, entry in enumerate(mapped):
            record = dict(source_sample_id=sid, path=entry['path'],
                provenance=entry.get('provenance', 'unspecified'), identity_basis='user_manifest_explicit_mapping')
            try:
                source = Path(entry['path']).expanduser()
                source = source if source.is_absolute() else base / source
                if not source.is_file():
                    raise ValueError(f'Waveform file missing: {source}')
                if source.stat().st_size > 64 * 1024**2:
                    raise ValueError('Trace exceeds 64 MiB; export a bounded CSV trace')
                delimiter = entry.get('delimiter', ',')
                if delimiter != 'whitespace' and (not isinstance(delimiter, str) or len(delimiter) != 1):
                    raise ValueError('delimiter must be one character or whitespace')
                with source.open(encoding='utf-8-sig', newline='') as handle:
                    reader = ((line.strip().split() for line in handle if line.strip() and not line.lstrip().startswith('#'))
                              if delimiter == 'whitespace' else csv.reader(handle, delimiter=delimiter))
                    header = next(reader)
                    if len(set(header)) != len(header):
                        raise ValueError('Duplicate waveform column names; give each column a unique header')
                    columns = entry.get('voltage_columns', ['vout_v'])
                    if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
                        raise ValueError('voltage_columns must be a nonempty unique list')
                    indices = [header.index(entry.get('time_column', 'time_s'))] + [header.index(c) for c in columns]
                    data = np.asarray([[float(row[i]) for i in indices] for row in reader if row], dtype=np.float64)
                if data.ndim != 2 or len(data) < 3 or not np.isfinite(data).all():
                    raise ValueError('Need at least three finite trace points')
                ts, vs = entry.get('time_scale_to_s', 1.), entry.get('voltage_scale_to_v', 1.)
                if not finite_pin(ts) or not finite_pin(vs) or ts <= 0 or vs <= 0:
                    raise ValueError('Time/voltage scales must be finite and positive')
                time, voltage = data[:, 0] * ts, data[:, 1:] * vs
                if not np.isfinite(time).all() or not np.isfinite(voltage).all():
                    raise ValueError('Scaled waveform overflows')
                dt = np.diff(time)
                if (dt < 0).any() or not (dt > 0).any():
                    raise ValueError('Time must be nondecreasing with at least one positive interval')
                positive = dt > 0
                slope = np.diff(voltage, axis=0)[positive] / dt[positive, None] / 1e6
                if not np.isfinite(slope).all():
                    raise ValueError('Finite-difference slope overflows')
                stem = f'case_{sid}_trace_{number}_pin'
                dest = output / 'waveforms_pin'
                dest.mkdir(exist_ok=True)
                with (dest / f'{stem}.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['time_s', *[f'{c} [V]' for c in columns]])
                    writer.writerows(np.column_stack((time, voltage)))
                for index, column in enumerate(columns):
                    statuses.append(dict(record, status='loaded', column=column, exported_column=f'{column} [V]', points=len(time),
                        time_min_s=float(time[0]), time_max_s=float(time[-1]),
                        duplicate_time_intervals=int((dt == 0).sum()),
                        voltage_min_v=float(voltage[:, index].min()), voltage_max_v=float(voltage[:, index].max()),
                        diagnostic_slope_min_v_us=float(slope[:, index].min()),
                        diagnostic_slope_max_v_us=float(slope[:, index].max()),
                        slope_definition='adjacent finite differences on dt>0; not the source SR extraction',
                        sha256=file_digest(source), exported_csv=f'waveforms_pin/{stem}.csv'))
                if plots:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
                    for index, column in enumerate(columns):
                        axes[0].plot(time * 1e6, voltage[:, index], label=column, lw=1)
                        axes[1].plot(((time[1:] + time[:-1]) / 2)[positive] * 1e6, slope[:, index], label=column, lw=1)
                    axes[0].set(title=f'{circuit} | source sample {sid} | {record["provenance"]}', ylabel='Voltage (V)')
                    axes[1].set(xlabel='Time (us)', ylabel='Finite difference (V/us)', title='Diagnostic slope; not a reproduction of source SR labels')
                    for ax in axes:
                        ax.grid(alpha=.2)
                        ax.legend()
                    filename = f'plots_pin/{stem}.png'
                    fig.savefig(output / filename, dpi=150)
                    plt.close(fig)
                    figures.append(dict(label=f'Waveform | {sid}', path=filename))
            except (OSError, ValueError, KeyError, IndexError, StopIteration, TypeError, UnicodeError) as exc:
                statuses.append(dict(record, status='unavailable_or_invalid', reason=str(exc)))
    return statuses, figures
