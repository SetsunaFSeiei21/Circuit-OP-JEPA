"""Streaming diagnostic CSVs, compact Excel summaries and an offline gate viewer."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from device_jepa.excel import Cell, Sheet, _column, _write_workbook

SCORE_FIELDS = ('model', 'seed', 'sample_id', 'circuit', 'device_id', 'device_type',
    'query_role', 'key_role', 'query_net', 'key_net', 'head', 'gate_version', 'attention_scope',
    'applied', 'protection_reason', 'gate_logit', 'gate_score', 'attention_bias', 'effective_multiplier')
OP_FIELDS = ('model', 'sample_id', 'device_id', 'field', 'supported', 'normalized_gate_op', 'relative_gate_op')
GROUP_FIELDS = ('model', 'device_id', 'device_type', 'query_role', 'key_role', 'head')
SUMMARY_FIELDS = (*GROUP_FIELDS, 'samples', 'gate_mean', 'gate_std', 'gate_min', 'gate_max',
                  'fraction_below_0_1', 'fraction_below_0_5', 'fraction_above_0_99',
                  'bias_mean', 'bias_min', 'bias_max', 'effective_multiplier_mean')


class Moments:
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = 0.
        self.minimum, self.maximum = math.inf, -math.inf

    def add(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)

    @property
    def std(self):
        return math.sqrt(max(0., self.m2 / self.n)) if self.n else None


class GateStats:
    def __init__(self):
        self.score, self.bias, self.multiplier = Moments(), Moments(), Moments()
        self.below_01 = self.below_05 = self.above_099 = 0

    def add(self, row):
        value = row['gate_score']
        self.score.add(value)
        self.bias.add(row['attention_bias'])
        self.multiplier.add(row['effective_multiplier'])
        self.below_01 += value < .1
        self.below_05 += value < .5
        self.above_099 += value > .99

    def values(self):
        n = self.score.n
        return (n, self.score.mean, self.score.std, self.score.minimum, self.score.maximum,
                self.below_01 / n, self.below_05 / n, self.above_099 / n,
                self.bias.mean, self.bias.minimum, self.bias.maximum, self.multiplier.mean)


def table_sheet(name, fields, rows, note):
    sheet = Sheet(name, [48 if f in ('model', 'Model') else 30 if 'id' in f.lower() else 22
                        for f in fields], freeze_rows=3, freeze_columns=1)
    sheet.note(name + ' — OP gate diagnostics', 1, 30)
    sheet.note(note, 2, 44)
    sheet.add([Cell(field, 3) for field in fields])
    for row in rows:
        sheet.add(row)
    sheet.filter_range = f'A3:{_column(len(fields))}{len(sheet.rows)}'
    return sheet


class GateReport:
    """Write full detail incrementally; only the HTML/Excel preview is bounded."""
    preview_limit = 4000

    def __init__(self, output, viewer_ids):
        self.output = Path(output)
        self.viewer_ids = set(viewer_ids)
        self.models = {}
        self.groups = defaultdict(GateStats)
        self.overall = defaultdict(GateStats)
        self.viewer = defaultdict(lambda: defaultdict(dict))
        self.preview = []
        self.total_rows = 0
        self.protected_rows = defaultdict(int)
        self.handles = []

    def _writer(self, name, fields):
        handle = (self.output / name).open('x', newline='', encoding='utf-8-sig')
        self.handles.append(handle)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        return writer

    def __enter__(self):
        try:
            self.scores = self._writer('gate_scores_pin.csv', SCORE_FIELDS)
            self.ops = self._writer('gate_op_inputs_pin.csv', OP_FIELDS)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.close()

    def add_model(self, mode, info):
        self.models[mode] = info

    def add_scores(self, rows):
        for row in rows:
            self.scores.writerow(row)
            self.total_rows += 1
            if len(self.preview) < self.preview_limit:
                self.preview.append([row[field] for field in SCORE_FIELDS])
            if row['applied']:
                self.groups[tuple(row[field] for field in GROUP_FIELDS)].add(row)
                self.overall[row['model']].add(row)
            else:
                self.protected_rows[row['model']] += 1
            if row['sample_id'] in self.viewer_ids:
                device = self.viewer[row['model']][row['sample_id']].setdefault(row['device_id'],
                    dict(device_type=row['device_type'], pairs=[], op=[]))
                device['pairs'].append([row[k] for k in ('query_role', 'key_role', 'head',
                    'gate_score', 'attention_bias', 'effective_multiplier', 'applied', 'protection_reason')])

    def add_op_inputs(self, mode, rows, field_names, captured):
        for b, sample in enumerate(rows):
            for owner, dev in enumerate(sorted(sample['devices'], key=lambda value: value['device_id'])):
                if dev['device_type'].lower() not in ('nmos', 'pmos'):
                    continue
                for f, name in enumerate(field_names):
                    supported = bool(captured['supported'][b, owner, f])
                    value = float(captured['values'][b, owner, f]) if supported else 0.
                    relative = (float(captured['relative'][b, owner, f])
                                if captured['relative'] is not None else None)
                    self.ops.writerow(dict(model=mode, sample_id=sample['sample_id'], device_id=dev['device_id'],
                        field=name, supported=supported, normalized_gate_op=value, relative_gate_op=relative))
                    if sample['sample_id'] in self.viewer_ids:
                        device = self.viewer[mode][sample['sample_id']].get(dev['device_id'])
                        if device is not None:
                            device['op'].append([name, value, relative, supported])

    def finish(self, record):
        group_rows = [(*key, *stats.values()) for key, stats in sorted(self.groups.items())]
        with (self.output / 'gate_summary_pin.csv').open('x', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(SUMMARY_FIELDS)
            writer.writerows(group_rows)
        model_rows = []
        for mode, info in self.models.items():
            stats = self.overall.get(mode)
            model_rows.append([mode, info['gate_active'], info['gate_version'], info['attention_scope'],
                ', '.join(map(str, info['self_attention_layers_1based'])), info['heads'],
                stats.score.n if stats else 0, self.protected_rows[mode],
                stats.score.mean if stats else None, stats.score.std if stats else None,
                stats.score.minimum if stats else None, stats.score.maximum if stats else None,
                stats.bias.mean if stats else None, stats.multiplier.mean if stats else None])
        sheets = [
            table_sheet('Models', ('Model', 'Gate active', 'Version', 'Scope', 'SA layers (1-based)',
                'Heads', 'Applied rows', 'Protected rows', 'Gate mean', 'Gate std', 'Gate min', 'Gate max',
                'Bias mean', 'Effective multiplier mean'), model_rows,
                'Statistics include applied MOS pairs only. Gate scores are not softmax attention or physical connection probabilities.'),
            table_sheet('ByDeviceHead', SUMMARY_FIELDS, group_rows,
                'Across selected samples for a fixed MOS / query / key / head. Population std. Protected pairs are excluded.'),
            table_sheet('ScorePreview', SCORE_FIELDS, self.preview,
                f'First {len(self.preview)} of {self.total_rows} rows; full precision and all samples are in gate_scores_pin.csv. Raw scores may exist on protected pairs; applied bias is zero.'),
            table_sheet('SelectedSamples', ('sample_id', 'split', 'sampling_seed'),
                [(sid, record['split'], record['sampling_seed']) for sid in record['selected_sample_ids']],
                'Same selected IDs for every model. Sampling is from saved checkpoint cohorts without replacement.'),
        ]
        _write_workbook(self.output / 'gate_scores_pin.xlsx', sheets)
        payload = dict(circuit=record['circuit'], split=record['split'],
            total_samples=len(record['selected_sample_ids']), viewer_samples=len(self.viewer_ids),
            seed=record['checkpoint_seed'], models={name: {k: value for k, value in info.items() if k != 'config'}
                for name, info in self.models.items()}, data=self.viewer)
        template = (Path(__file__).parent / 'templates/gate_viewer_pin.html').read_text(encoding='utf-8')
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        encoded = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        (self.output / 'gate_viewer_pin.html').write_text(template.replace('__GATE_DATA_PIN__', encoded), encoding='utf-8')
        record['reports'] = dict(detail_rows=self.total_rows, excel_preview_rows=len(self.preview),
                                 aggregate_groups=len(group_rows), html_samples=len(self.viewer_ids))
