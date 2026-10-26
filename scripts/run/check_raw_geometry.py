#!/usr/bin/env python3
"""Read-only, streaming W/L/m comparison before canonical import or training."""
import argparse
from itertools import groupby, islice
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from device_jepa.adapter import (AdapterError, _json, _names, _number, _read_database,
    _resolve_parameters, _vector, evaluate_spice_expression, parse_spice_netlist)

FIELDS = (('w_um', 'W_m', 'w', 1e-6), ('l_um', 'L_m', 'l', 1e-6),
          ('multiplicity', 'm', 'm', 1.))


def inspect_geometry(directory, max_samples=None, max_examples=8):
    directory = Path(directory).resolve()
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError('max_samples must be a positive integer or None')
    # The source cursors stream; no canonical dataset or model tensors are made.
    # Removing the table count bound here does not relax any physical checks.
    with _read_database(directory / 'sampling_history.sqlite3', max_table_rows=None) as source:
        if len(source['dataset_schema']) != 1:
            raise AdapterError('Expected exactly one dataset_schema row')
        schema = source['dataset_schema'][0]
        circuit = schema['circuit_name']
        if not isinstance(circuit, str) or not circuit.strip():
            raise AdapterError('Source circuit_name must be nonempty')
        parsed = parse_spice_netlist(circuit + '.sp', root=directory)
        names = _names(_json(schema['parameter_names_json'], 'parameter_names_json'), 'parameter names')
        inventory = {d['device_id'].lower(): d for d in parsed['devices']}
        result = dict(circuit=circuit, input=str(directory), geometry_scale_to_m=parsed['geometry_scale_to_m'],
                      source_samples=source['table_counts']['samples'], samples_scanned=0,
                      mos_devices_scanned=0, comparisons={f[1]: 0 for f in FIELDS},
                      missing_geometry={f[1]: 0 for f in FIELDS},
                      mismatches={f[1]: 0 for f in FIELDS}, examples=[])
        groups = iter(groupby(source['dc_device_operating_points'], key=lambda r: r['sample_id']))
        pending, previous = next(groups, None), object()
        for row in islice(source['samples'], max_samples):
            sample_id = row['sample_id']
            if sample_id == previous:
                raise AdapterError('Duplicate sample_id')
            previous = sample_id
            result['samples_scanned'] += 1
            design = _vector(row['design_values_json'], names, f'sample {sample_id} design', missing=False)
            environment = _resolve_parameters(parsed['parameter_defaults'], design)
            observed = {}
            if pending is not None and pending[0] == sample_id:
                for op in pending[1]:
                    name = op['device_name']
                    if not isinstance(name, str) or name.lower() not in inventory:
                        raise AdapterError(f'OP device is absent from DUT: {name}')
                    name = name.lower()
                    device = inventory[name]
                    if name in observed or not device['op_target']:
                        raise AdapterError(f'Duplicate or ineligible OP device: {name}')
                    if (str(op['device_type']).upper() != device['device_type']
                            or str(op['model_name']).lower() != device['model_name'].lower()):
                        raise AdapterError(f'OP model/type mismatch for {name}')
                    observed[name] = op
                pending = next(groups, None)
            for name, device in inventory.items():
                if not device['op_target']:
                    continue
                result['mos_devices_scanned'] += 1
                op = observed.get(name, {})
                for raw_key, field, expression_key, factor in FIELDS:
                    scale = parsed['geometry_scale_to_m'] if expression_key in ('w', 'l') else 1.
                    expected = evaluate_spice_expression(device['parameter_expressions'][expression_key], environment) * scale
                    if not math.isfinite(expected) or expected <= 0:
                        raise AdapterError(f'sample {sample_id} {name}: W/L/m must be finite and positive')
                    value = _number(op.get(raw_key), f'{sample_id}.{name}.{raw_key}')
                    if value is None:
                        result['missing_geometry'][field] += 1
                        continue
                    actual = value * factor
                    result['comparisons'][field] += 1
                    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-18):
                        result['mismatches'][field] += 1
                        if len(result['examples']) < max_examples:
                            ratio = actual / expected
                            result['examples'].append(dict(sample_id=sample_id, device=device['device_id'],
                                field=field, input_value=expected, op_field=raw_key, op_raw_value=value,
                                op_converted_value=actual,
                                op_over_input_ratio=ratio if math.isfinite(ratio) else None))
        result['scan_complete'] = result['samples_scanned'] == result['source_samples']
        if result['scan_complete'] and pending is not None:
            raise AdapterError(f'Orphan OP sample_id: {pending[0]}')
        if any(result['mismatches'].values()):
            result['status'] = 'mismatch'
        elif not all(result['comparisons'][field] for field in ('W_m', 'L_m')):
            result['status'] = 'unverifiable'
        else:
            result['status'] = 'consistent_observed_geometry'
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='Per-circuit raw directories')
    parser.add_argument('--max-samples', type=int, help='Check the first N samples; omit to scan all')
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max-samples must be positive')
    reports = []
    for directory in args.input:
        try:
            reports.append(inspect_geometry(directory, args.max_samples))
        except (OSError, ValueError, sqlite3.Error) as exc:
            reports.append(dict(input=directory, status='error', error=str(exc)))
    print(json.dumps(reports, indent=2, allow_nan=False))
    return 0 if all(r['status'] == 'consistent_observed_geometry' for r in reports) else 1


if __name__ == '__main__':
    sys.exit(main())
