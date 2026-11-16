#!/usr/bin/env python3
"""Repair explicitly selected raw sources with missing micrometer geometry units.

Only scale=1 -> scale=1u is supported. All observed W/L values in the checked
scope must disagree before repair, with the reported OP/input ratios equal to
1e-6 and no multiplicity mismatch. Existing consistent scale=1u sources are
unchanged. All input directories are preflighted before any writes. Exact-byte
backups are retained; a failed postcheck restores every file changed by this
invocation. SQLite, parameter files and canonical datasets are never modified.

Default checks cover the first 100 samples, not the entire dataset. Full import
continues to enforce geometry consistency on every selected sample.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from device_jepa.adapter import _lines, evaluate_spice_expression
from scripts.run.check_raw_geometry import inspect_geometry


def _plan(directory, max_samples):
    directory = Path(directory).resolve()
    report = inspect_geometry(directory, max_samples=max_samples)
    scale = report['geometry_scale_to_m']
    path = (directory / (report['circuit'] + '.sp')).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f'Netlist escapes input directory: {path}')
    original = path.read_bytes()
    plan = dict(directory=directory, path=path, original=original,
                report=report, changed=False, backup=None)
    if (math.isclose(scale, 1e-6, rel_tol=1e-12)
            and report['status'] == 'consistent_observed_geometry'):
        return plan
    if not math.isclose(scale, 1., rel_tol=1e-12):
        raise ValueError(f'{directory.name}: only scale=1 -> 1u is supported; got {scale}')
    if report['mismatches']['m'] or any(
            not report['comparisons'][field]
            or report['mismatches'][field] != report['comparisons'][field]
            for field in ('W_m', 'L_m')):
        raise ValueError(f'{directory.name}: not a uniform W/L unit mismatch; refusing repair')
    if not report['examples'] or any(
            item['op_over_input_ratio'] is None
            or not math.isclose(item['op_over_input_ratio'], 1e-6, rel_tol=1e-7)
            for item in report['examples']):
        raise ValueError(f'{directory.name}: OP/input ratios must be 1e-6; refusing repair')
    text = original.decode('utf-8')
    lines = text.splitlines(keepends=True)
    directives = [(n, line) for n, line in _lines(path)
                  if re.match(r'(?i)^\.options?\s', line)]
    if len(directives) > 1:
        raise ValueError(f'{path}: multiple option directives require manual review')
    if directives:
        number, logical = directives[0]
        physical = lines[number - 1]
        active = re.split(r'[;$]', physical, maxsplit=1)[0]
        if active.strip() != logical:
            raise ValueError(f'{path}: continued option directives require manual review')
        matches = list(re.finditer(r'(?i)\bscale\s*=\s*([^\s;$]+)', active))
        if len(matches) != 1 or not math.isclose(
                evaluate_spice_expression(matches[0][1]), 1., rel_tol=1e-12):
            raise ValueError(f'{path}: complex scale expression requires manual review')
        # Reject a spaced expression such as "scale=1 + 0" rather than replacing
        # only its first operand. Preserve all unrelated whitespace/comments.
        if active[matches[0].end():].strip():
            raise ValueError(f'{path}: complex scale expression requires manual review')
        start, end = matches[0].span(1)
        lines[number - 1] = physical[:start] + '1u' + physical[end:]
        updated = ''.join(lines)
    else:
        newline = '\r\n' if '\r\n' in text else '\n'
        updated = ('* Explicit geometry units: micrometers' + newline
                   + '.option scale=1u' + newline + text)
    plan['updated'] = updated.encode('utf-8')
    return plan


def _atomic_write(path, content):
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), mode)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fix_geometry_scale(directories, max_samples=100):
    """Return concise reports; retain backups and roll back on postcheck failure."""
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError('max_samples must be a positive integer')
    directories = [Path(directory).resolve() for directory in directories]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError('Input directories must be nonempty and unique')
    plans = [_plan(directory, max_samples) for directory in directories]
    try:
        for plan in plans:
            if 'updated' not in plan:
                continue
            path = plan['path']
            if path.read_bytes() != plan['original']:
                raise ValueError(f'{path}: source changed after preflight; refusing repair')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            backup = path.with_name(path.name + '.bak_' + stamp)
            with backup.open('xb') as handle:
                os.fchmod(handle.fileno(), stat.S_IMODE(path.stat().st_mode))
                handle.write(plan['original'])
                handle.flush()
                os.fsync(handle.fileno())
            plan['backup'] = backup
            _atomic_write(path, plan['updated'])
            plan['changed'] = True
            after = inspect_geometry(plan['directory'], max_samples=max_samples)
            if (after['status'] != 'consistent_observed_geometry'
                    or not math.isclose(after['geometry_scale_to_m'], 1e-6, rel_tol=1e-12)):
                raise ValueError(f'{path}: postcheck failed; netlist and OP still disagree')
            plan['report'] = after
    except BaseException as error:
        rollback_errors = []
        for plan in reversed(plans):
            if plan['changed']:
                try:
                    _atomic_write(plan['path'], plan['original'])
                except OSError as rollback_error:
                    rollback_errors.append(f"{plan['path']}: {rollback_error}; backup={plan['backup']}")
        if rollback_errors:
            raise RuntimeError('Repair failed and rollback needs manual recovery: '
                               + '; '.join(rollback_errors)) from error
        if any(plan['changed'] for plan in plans) and isinstance(error, Exception):
            raise ValueError(f'{error}; all changed netlists restored; backups retained') from error
        raise
    return [dict(input=str(plan['directory']), circuit=plan['report']['circuit'],
                 status='fixed' if plan['changed'] else 'already_consistent',
                 netlist=str(plan['path']),
                 backup=None if plan['backup'] is None else str(plan['backup']),
                 geometry_scale_to_m=plan['report']['geometry_scale_to_m'],
                 samples_scanned=plan['report']['samples_scanned'],
                 source_samples=plan['report']['source_samples'],
                 scan_complete=plan['report']['scan_complete'],
                 mismatches=plan['report']['mismatches']) for plan in plans]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='Raw circuit directories to repair')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Check first N samples before and after repair (default: 100)')
    args = parser.parse_args(argv)
    if args.max_samples < 1:
        parser.error('--max-samples must be positive')
    try:
        reports = fix_geometry_scale(args.input, max_samples=args.max_samples)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f'Geometry scale repair failed: {error}', file=sys.stderr)
        return 1
    print(json.dumps(reports, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
