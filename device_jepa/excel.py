"""Portable Excel presentation of already-computed comparison statistics.

Only Python's standard library is needed, including on a login node without
PyTorch or Excel packages. The small OOXML writer is deliberately limited to
this report: two worksheets, typed sources, linked display formulas and styles.
It never recomputes statistics or combines circuits, splits or backends.
"""
import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
SCORES = ('mae', 'rmse', 'r2')
IDENTITY = ('mode', 'backend', 'split', 'scope', 'circuit', 'metric', 'scale')
COUNTS = ('num_seeds', 'count_per_seed', *(s + '_defined_seeds' for s in SCORES))
NUMBERS = tuple(s + suffix for s in SCORES for suffix in ('_mean', '_std'))
BOOLEANS = ('requires_op_at_inference', 'uses_design_parameters')
METRICS = {
    'DC_Gain_dB': ('DC Gain', 'dB', 1),
    'UGF_Hz': ('UGF', 'MHz', 1_000_000),
    'Phase_Margin_deg': ('Phase Margin', 'deg', 1),
    'CMRR_dB': ('CMRR', 'dB', 1),
    'PSRR_Plus_dB': ('PSRR+', 'dB', 1),
    'PSRR_Minus_dB': ('PSRR-', 'dB', 1),
    'Slew_Rise_V_us': ('Slew Rise', 'V/us', 1),
    'Slew_Fall_V_us': ('Slew Fall', 'V/us', 1),
    'Power_Quiescent_uW': ('Quiescent Power', 'uW', 1),
}


def _number(value, field):
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise ValueError(f'{field} must be numeric, not boolean')
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid {field}: {value!r}') from exc
    if not math.isfinite(number):
        raise ValueError(f'{field} must be finite or blank')
    if field in COUNTS:
        if number < 0 or not number.is_integer():
            raise ValueError(f'{field} must be a nonnegative integer')
        return int(number)
    return number


def prepare_rows(rows):
    """Validate CSV/JSON rows before creating or replacing any workbook."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty comparison summary')
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set((*IDENTITY, *COUNTS, *NUMBERS)) - row.keys():
            raise ValueError('Missing required comparison summary fields')
        item = dict(row)
        for field in IDENTITY:
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f'{field} must be a nonempty string')
        key = tuple(item[field] for field in IDENTITY)
        if key in seen:
            raise ValueError(f'Duplicate summary identity: {key}')
        seen.add(key)
        for field in (*COUNTS, *NUMBERS):
            item[field] = _number(item[field], field)
        if any(item[field] is None for field in COUNTS) or item['num_seeds'] < 1:
            raise ValueError('Summary counts must be present and num_seeds positive')
        for score in SCORES:
            count = item[score + '_defined_seeds']
            average, std = item[score + '_mean'], item[score + '_std']
            if count > item['num_seeds'] or (count > 0) != (average is not None):
                raise ValueError(f'Inconsistent {score} mean/defined seed count')
            if (count > 1) != (std is not None) or (std is not None and std < 0):
                raise ValueError(f'Inconsistent {score} std/defined seed count')
            if score != 'r2' and average is not None and average < 0:
                raise ValueError(f'{score} must be nonnegative')
        for field in BOOLEANS:
            if field not in item or item[field] is None or item[field] == '':
                continue  # Older summaries do not have the design-input field.
            if type(item[field]) is bool:
                continue
            if isinstance(item[field], str) and item[field].lower() in ('true', 'false'):
                item[field] = item[field].lower() == 'true'
            else:
                raise ValueError(f'Invalid boolean {field}')
        result.append(item)
    return result


def load_summary(source):
    """Accept a CSV/JSON, comparison directory, results directory or raw run."""
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        candidates = [directory / name
                      for directory in (path, path / 'comparison', path / 'results/comparison')
                      for name in ('comparison.json', 'summary.csv')]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError('No comparison.json or summary.csv found in the supplied run')
    metadata = {}
    if path.suffix.lower() == '.json':
        metadata = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(metadata, dict):
            raise ValueError('Expected a comparison.json object')
        rows = metadata.get('summary')
    elif path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('Missing or duplicate CSV headers')
            rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError('CSV row width does not match its header')
    else:
        raise ValueError('Input must be summary.csv, comparison.json or an experiment directory')
    return prepare_rows(rows), metadata, path


def _column(index):
    result = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


@dataclass
class Cell:
    value: object = None
    style: int = 0
    formula: str | None = None


class Sheet:
    def __init__(self, name, widths, freeze_rows=7, freeze_columns=2):
        self.name, self.widths = name, widths
        self.freeze_rows, self.freeze_columns = freeze_rows, freeze_columns
        self.rows, self.heights, self.merges = [], {}, []
        self.filter_range = None

    def add(self, values=(), height=23):
        self.rows.append([v if isinstance(v, Cell) else Cell(v) for v in values])
        self.heights[len(self.rows)] = height
        return len(self.rows)

    def note(self, value, style=2, height=27):
        row = self.add([Cell(value, style)], height)
        self.merges.append(f'A{row}:{_column(len(self.widths))}{row}')


def _element(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attributes.items()})


def _xml(root):
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def _text(value):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value)
    if len(text) > 32767 or any(ord(c) < 32 and c not in '\t\n\r' for c in text):
        raise ValueError('Cell text exceeds Excel limits or contains XML control characters')
    return text


def _sheet_xml(sheet):
    if len(sheet.rows) > 1_048_576 or len(sheet.widths) > 16_384:
        raise ValueError('Report exceeds Excel worksheet limits')
    root = ET.Element('worksheet', xmlns=NS)
    _element(root, 'dimension', ref=f'A1:{_column(len(sheet.widths))}{len(sheet.rows)}')
    view = _element(_element(root, 'sheetViews'), 'sheetView', workbookViewId=0, showGridLines=0)
    top_left = f'{_column(sheet.freeze_columns + 1)}{sheet.freeze_rows + 1}'
    _element(view, 'pane', xSplit=sheet.freeze_columns, ySplit=sheet.freeze_rows,
             topLeftCell=top_left, activePane='bottomRight', state='frozen')
    _element(view, 'selection', pane='bottomRight', activeCell=top_left, sqref=top_left)
    _element(root, 'sheetFormatPr', defaultRowHeight=23)
    columns = _element(root, 'cols')
    for index, width in enumerate(sheet.widths, 1):
        _element(columns, 'col', min=index, max=index, width=width, customWidth=1)
    data = _element(root, 'sheetData')
    for row_index, values in enumerate(sheet.rows, 1):
        row = _element(data, 'row', r=row_index, ht=sheet.heights[row_index], customHeight=1)
        for col_index, cell in enumerate(values, 1):
            node = _element(row, 'c', r=f'{_column(col_index)}{row_index}', s=cell.style)
            value = cell.value
            if cell.formula is not None:
                node.set('t', 'str')
                _element(node, 'f').text = cell.formula
                _element(node, 'v').text = _text(value)
            elif value is None:
                continue
            elif type(value) is bool:
                node.set('t', 'b')
                _element(node, 'v').text = str(int(value))
            elif isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise ValueError('Nonfinite numeric workbook cell')
                _element(node, 'v').text = str(value)
            else:
                node.set('t', 'inlineStr')  # Never interpret source text as a formula.
                text = _element(_element(node, 'is'), 't')
                text.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                text.text = _text(value)
    if sheet.filter_range:
        _element(root, 'autoFilter', ref=sheet.filter_range)
    if sheet.merges:
        merges = _element(root, 'mergeCells', count=len(sheet.merges))
        for ref in sheet.merges:
            _element(merges, 'mergeCell', ref=ref)
    _element(root, 'pageMargins', left=.3, right=.3, top=.5, bottom=.5, header=.2, footer=.2)
    return _xml(root)


def _styles_xml():
    root = ET.Element('styleSheet', xmlns=NS)
    _element(_element(root, 'numFmts', count=1), 'numFmt', numFmtId=164, formatCode='0.0000000000E+00')
    fonts = _element(root, 'fonts', count=6)
    for size, color, bold in ((11, '243247', False), (17, '233C62', True),
                              (11, 'FFFFFF', True), (10, '5C697D', False),
                              (11, '243247', True), (11, '17623B', True)):
        font = _element(fonts, 'font')
        _element(font, 'sz', val=size)
        _element(font, 'color', rgb='FF' + color)
        _element(font, 'name', val='Arial')
        if bold:
            _element(font, 'b')
    fills = _element(root, 'fills', count=6)
    for value in ('none', 'gray125', '233C62', 'F2F5FA', 'DDE6F2', 'E2F0E8'):
        pattern = _element(_element(fills, 'fill'), 'patternFill',
                           patternType=value if len(value) != 6 else 'solid')
        if len(value) == 6:
            _element(pattern, 'fgColor', rgb='FF' + value)
            _element(pattern, 'bgColor', indexed=64)
    border = _element(_element(root, 'borders', count=1), 'border')
    for side in ('left', 'right', 'top', 'bottom', 'diagonal'):
        _element(border, side)
    _element(_element(root, 'cellStyleXfs', count=1), 'xf', numFmtId=0, fontId=0, fillId=0, borderId=0)
    specifications = [
        (0, 0, 0, 'left'), (1, 0, 0, 'left'), (3, 0, 0, 'left'),
        (2, 2, 0, 'center'), (4, 4, 0, 'left'),
        (0, 0, 0, 'left'), (0, 3, 0, 'left'),
        (0, 0, 0, 'right'), (0, 3, 0, 'right'), (5, 5, 0, 'right'),
        (0, 0, 3, 'right'), (0, 0, 164, 'right'), (0, 0, 0, 'center'),
    ]
    styles = _element(root, 'cellXfs', count=len(specifications))
    for font, fill, number_format, alignment in specifications:
        style = _element(styles, 'xf', numFmtId=number_format, fontId=font, fillId=fill,
                         borderId=0, xfId=0, applyAlignment=1, applyNumberFormat=1)
        _element(style, 'alignment', horizontal=alignment, vertical='center', wrapText=1)
    _element(_element(root, 'cellStyles', count=1), 'cellStyle', name='Normal', xfId=0, builtinId=0)
    return _xml(root)


def _write_workbook(path, sheets):
    """Build and validate XML in memory, then replace the derived report atomically."""
    path = Path(path)
    if path.suffix.lower() != '.xlsx':
        raise ValueError('Excel output must have an .xlsx extension')
    parts = {'xl/styles.xml': _styles_xml()}
    workbook = ET.Element('workbook', {'xmlns': NS, 'xmlns:r': REL})
    _element(_element(workbook, 'bookViews'), 'workbookView', activeTab=0)
    names = _element(workbook, 'sheets')
    relationships = ET.Element('Relationships', xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    types = ET.Element('Types', xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
    _element(types, 'Default', Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
    _element(types, 'Default', Extension='xml', ContentType='application/xml')
    for index, sheet in enumerate(sheets, 1):
        _element(names, 'sheet', name=sheet.name, sheetId=index, **{'r:id': f'rId{index}'})
        target = f'worksheets/sheet{index}.xml'
        _element(relationships, 'Relationship', Id=f'rId{index}', Type=REL + '/worksheet', Target=target)
        parts['xl/' + target] = _sheet_xml(sheet)
        _element(types, 'Override', PartName='/xl/' + target,
                 ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
    _element(workbook, 'calcPr', calcMode='auto', fullCalcOnLoad=1)
    _element(relationships, 'Relationship', Id=f'rId{len(sheets) + 1}', Type=REL + '/styles', Target='styles.xml')
    for name, kind in (('workbook', 'sheet.main'), ('styles', 'styles')):
        _element(types, 'Override', PartName=f'/xl/{name}.xml',
                 ContentType=f'application/vnd.openxmlformats-officedocument.spreadsheetml.{kind}+xml')
    root_relationships = ET.Element('Relationships', xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    _element(root_relationships, 'Relationship', Id='rId1', Type=REL + '/officeDocument', Target='xl/workbook.xml')
    parts.update({'[Content_Types].xml': _xml(types), '_rels/.rels': _xml(root_relationships),
                  'xl/workbook.xml': _xml(workbook), 'xl/_rels/workbook.xml.rels': _xml(relationships)})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.stem + '-', suffix='.xlsx', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            with ZipFile(handle, 'w', ZIP_DEFLATED) as archive:
                for name, content in parts.items():
                    archive.writestr(name, content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _format_number(value, decimals):
    # Preserve small nonzero results instead of displaying misleading zeros.
    scientific = value != 0 and (abs(value) < 10 ** -decimals or abs(value) >= 1e7)
    return (f'{value:.{decimals}E}' if scientific else f'{value:.{decimals}f}',
            '0.' + '0' * decimals + ('E+00' if scientific else ''))


def _display_cell(row, score, divisor, fields, source_row, decimals, style):
    if row is None or row[score + '_mean'] is None:
        return Cell('n/a', style)
    mean = row[score + '_mean'] / divisor
    std = row[score + '_std']
    average, mean_format = _format_number(mean, decimals)
    deviation, std_format = ('n/a', '0.' + '0' * decimals) if std is None else _format_number(std / divisor, decimals)
    mean_ref = f"'数据明细'!{_column(fields.index(score + '_mean') + 1)}{source_row}"
    std_ref = f"'数据明细'!{_column(fields.index(score + '_std') + 1)}{source_row}"
    formula = (f'IF(ISNUMBER({mean_ref}),TEXT({mean_ref}/{divisor},"{mean_format}")&" ± "&'
               f'IF(ISNUMBER({std_ref}),TEXT({std_ref}/{divisor},"{std_format}"),"n/a"),"n/a")')
    return Cell(average + ' ± ' + deviation, style, formula)


def export_summary(rows, output, *, source='', metadata=None, split='all', scale='physical',
                   scope='circuit', models=None, decimals=4):
    rows = prepare_rows(rows)
    if split not in ('all', 'train', 'val', 'test', 'ood') or scale not in ('physical', 'normalized') or scope not in ('circuit', 'pooled'):
        raise ValueError('Invalid split, scale or scope')
    if type(decimals) is not int or not 1 <= decimals <= 10:
        raise ValueError('decimals must be between 1 and 10')
    if models is not None and (not models or len(set(models)) != len(models)):
        raise ValueError('Model selection must be nonempty and unique')
    selected = [(index, row) for index, row in enumerate(rows)
                if row['scale'] == scale and row['scope'] == scope and (split == 'all' or row['split'] == split)
                and (models is None or row['mode'] in models)]
    if not selected:
        raise ValueError('No summary rows match the requested selection')
    if models and set(models) - {row['mode'] for _, row in selected}:
        raise ValueError('Selected model is absent from the requested results')
    fields = list(dict.fromkeys(field for row in rows for field in row))
    data = Sheet('数据明细', [30 if field in IDENTITY else 22 for field in fields], freeze_columns=1)
    data.note('Original comparison summary', 1, 34)
    data.note(f'Source: {source}', height=38)
    data.note(f'All {len(rows)} source rows retained, including unselected splits/scales/models.')
    data.note('Numeric values retain source units and precision. Blank statistics remain blank.')
    data.note('Use filters below to inspect individual circuits, pooled results and defined seed counts.')
    data.add()
    data.add([Cell(field, 3) for field in fields], 48)
    for row in rows:
        data.add([Cell(row.get(field), 11 if field in NUMBERS else 10 if field in COUNTS else 5) for field in fields])
    data.filter_range = f'A7:{_column(len(fields))}{len(data.rows)}'
    identities = sorted({(row['mode'], row['backend']) for _, row in selected},
                        key=lambda key: ((models.index(key[0]) if models else 0), key))
    labels = [mode if sum(m == mode for m, _ in identities) == 1 else mode + '\n' + backend
              for mode, backend in identities]
    summary = Sheet('模型对比', [25, 20] + [30] * len(identities) + [18], freeze_rows=13)
    summary.note('Circuit model comparison', 1, 34)
    summary.note('Mean ± sample standard deviation across training seeds (ddof=1).')
    summary.note('Lower MAE/RMSE and higher R² are better. n/a = missing/undefined; one seed has no std.')
    summary.note('Physical UGF errors: Hz → MHz for BOTH mean and std. R² is dimensionless.')
    summary.note(f'Source: {source}', height=38)
    complete = (metadata or {}).get('complete')
    status = ('COMPLETE (as declared by comparison.json)' if complete is True else
              'INCOMPLETE (as declared by comparison.json)' if complete is False else
              'UNKNOWN (CSV does not verify run compatibility)')
    summary.note('Comparison status: ' + status)
    summary.note('Green = best unrounded mean at export, only with matching counts. Not a significance test.', height=30)
    summary.add()
    groups = defaultdict(dict)
    for index, row in selected:
        groups[(row['split'], row['circuit'])][(row['mode'], row['backend'], row['metric'])] = (index, row)
    metric_order = {name: index for index, name in enumerate(METRICS)}
    split_order = {'test': 0, 'ood': 1, 'val': 2, 'train': 3}
    for (subset, circuit), group in sorted(groups.items(), key=lambda item: (split_order.get(item[0][0], 4), item[0])):
        summary.note(f'{circuit} | split={subset} | scope={scope} | scale={scale}', 4, 29)
        for field, title in (('num_seeds', 'Training seeds'), ('requires_op_at_inference', 'OP at inference'),
                             ('uses_design_parameters', 'Design parameters')):
            cells = [Cell(title, 2), Cell()]
            for identity in identities:
                values = {r.get(field) for _, r in group.values() if (r['mode'], r['backend']) == identity}
                value = next(iter(values)) if len(values) == 1 else 'varies' if values else 'n/a'
                if value is None or value == '':
                    value = 'unknown'
                elif type(value) is bool:
                    value = 'Yes' if value else 'No'
                cells.append(Cell(value, 12))
            summary.add(cells)
        summary.add([Cell(value, 3) for value in ['Circuit metric', 'Score / unit', *labels, 'Labels / seed']], 48)
        metrics = sorted({key[2] for key in group}, key=lambda m: (metric_order.get(m, len(METRICS)), m))
        for metric_index, metric in enumerate(metrics):
            label, unit, divisor = METRICS.get(metric, (metric, 'dataset unit', 1))
            if scale == 'normalized':
                unit, divisor = 'normalized', 1
            found = [group.get((*identity, metric)) for identity in identities]
            counts = {item[1]['count_per_seed'] for item in found if item}
            for score in SCORES:
                score_divisor = 1 if score == 'r2' else divisor
                values = [item[1][score + '_mean'] if item else None for item in found]
                comparable = len(values) > 1 and all(value is not None for value in values) and len({
                    (item[1]['num_seeds'], item[1]['count_per_seed'], item[1][score + '_defined_seeds'])
                    for item in found if item}) == 1
                best = (max(values) if score == 'r2' else min(values)) if comparable else None
                cells = [Cell(label, 6 if metric_index % 2 == 0 else 5),
                         Cell('R²' if score == 'r2' else f'{score.upper()} ({unit})', 6 if metric_index % 2 == 0 else 5)]
                for item, value in zip(found, values):
                    style = 9 if comparable and value == best else 8 if metric_index % 2 == 0 else 7
                    index, row = item if item else (None, None)
                    cells.append(_display_cell(row, score, score_divisor, fields,
                                               index + 8 if index is not None else None, decimals, style))
                cells.append(Cell(next(iter(counts)) if len(counts) == 1 else 'varies', 10))
                summary.add(cells, 25)
        summary.add()
    return _write_workbook(output, [summary, data])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export model MAE/RMSE/R² mean ± std to Excel; no training or GPU needed.')
    parser.add_argument('--input', required=True, help='Experiment/results/comparison directory, summary.csv or comparison.json')
    parser.add_argument('--output', help='Default: model_comparison.xlsx beside the source summary')
    parser.add_argument('--split', choices=('all', 'test', 'ood', 'val', 'train'), default='all')
    parser.add_argument('--scale', choices=('physical', 'normalized'), default='physical')
    parser.add_argument('--scope', choices=('circuit', 'pooled'), default='circuit')
    parser.add_argument('--models', nargs='+', help='Optional ordered model subset; otherwise include all present models')
    parser.add_argument('--decimals', type=int, default=4)
    args = parser.parse_args(argv)
    try:
        rows, metadata, path = load_summary(args.input)
        output = Path(args.output) if args.output else path.with_name('model_comparison.xlsx')
        result = export_summary(rows, output, source=str(path), metadata=metadata, split=args.split,
                                scale=args.scale, scope=args.scope, models=args.models, decimals=args.decimals)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Excel comparison: {result.resolve()}')


if __name__ == '__main__':
    main()
