#!/usr/bin/env python3
"""Trace original sample IDs to designs, measured/predicted OP, and saved traces.

No training, resampling, preprocessing fit, SPICE execution or source writes.
Only the selected cases and a bounded train OP reference are sent to the GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK = ROOT / 'final_exam/single_circuit_pin'

from final_exam.gate_probe_pin import file_digest
from final_exam.prediction_probe_pin import (inspect_sources_pin, checkpoint_selection_pin,
                                            infer_split_pin, digest_pin)
from final_exam.spectral_probe_pin import write_json
from final_exam.error_case_data_pin import (TrainReferencePin, select_cases_pin, finite_pin,
                                          raw_evidence_pin, waveform_evidence_pin)


def inspect_case_source_pin(source_run, circuit, models, seeds):
    sources, inventory = inspect_sources_pin([source_run], models, seeds)
    source = sources[0]
    if source['circuit'] != circuit:
        raise ValueError(f"Requested circuit {circuit} != saved circuit {source['circuit']}")
    found = {(r['model'], r['seed']) for r in source['checkpoints']}
    expected = {(m, s) for m in models for s in seeds}
    if expected - found:
        raise ValueError(f'Missing completed model/seed pairs: {sorted(expected - found)}')
    return source, inventory


def infer_op_cases_pin(model, saved, rows, batch_size, device):
    """Physical-device order is exactly PinResidentDataset's device_id sort."""
    import torch
    from device_jepa.tensors import denormalize
    from final_exam.data_pin import PinResidentDataset
    predictor = getattr(model, 'predictor', None)
    if predictor is None or not hasattr(predictor, 'predict_devices'):
        raise ValueError('This diagnostic needs a deployed model with predict_devices OP predictor')
    prep = saved['preprocessing']
    names = list(predictor.op_names)
    device_prep = prep['device_preprocessing']
    if names != device_prep['feature_spec']['op_names']:
        raise ValueError('Predictor OP fields disagree with saved device preprocessing')
    result = [None] * len(rows)
    for start in range(0, len(rows), 128):
        part = rows[start:start + 128]
        cache = PinResidentDataset(part, prep, 'cpu', include_op=False, include_performance=False,
                                   max_tokens=saved['model_config']['max_devices'], verbose=False)
        for indices, cpu_batch in cache.batches(batch_size):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cpu_batch.items()}
            with torch.no_grad():
                predicted, support = predictor.predict_devices(batch)
                if predicted.shape != support.shape or predicted.shape[:2] != batch['owner_mask'].shape:
                    raise ValueError('OP output does not use the canonical device axis')
                if not torch.isfinite(predicted[support]).all():
                    raise ValueError('Nonfinite supported OP prediction')
                physical = denormalize(torch.where(support, predicted, 0.), device_prep['op'])
            predicted, physical, support = (x.detach().cpu().numpy() for x in (predicted, physical, support))
            for b, local in enumerate(indices.tolist()):
                devices = sorted(part[local]['devices'], key=lambda d: d['device_id'])
                if int(cpu_batch['owner_mask'][b].sum()) != len(devices):
                    raise ValueError('Canonical device identity/order mismatch')
                result[start + local] = {d['device_id']: dict(normalized=predicted[b, i],
                    physical=physical[b, i], supported=support[b, i]) for i, d in enumerate(devices)}
        del cache
    return names, result


def op_reference_pin(rows, predictions, names, statistics):
    """Reference OP error distributions use a documented subset of TRAIN only."""
    import numpy as np
    from device_jepa.data import transform_value
    errors = {}
    for sample, predicted in zip(rows, predictions):
        for device in sample['devices']:
            values = predicted[device['device_id']]
            for j, field in enumerate(names):
                truth = device.get('op', {}).get(field)
                if values['supported'][j] and finite_pin(truth):
                    target = transform_value(truth, statistics, j)
                    error = float(values['normalized'][j]) - target
                    errors.setdefault((device['device_id'], field), []).append(error)
    return {key: np.asarray(values, dtype=np.float64) for key, values in errors.items()}


def run_error_case_probe_pin(*, source_run, circuit, sample_ids, models, seeds=(29,),
        split='test', raw_dir=None, waveform_manifest=None, reference_samples=128,
        reference_seed=17, batch_size=32, device='cuda', plots=True, output=None, log_dir=None):
    import numpy as np
    import torch
    from device_jepa.data import load_dataset, verify_manifest, transform_value
    from device_jepa.resident import require_device
    from device_jepa.training import state_digest
    from final_exam.run_pin import load_checkpoint_pin
    from final_exam.logging_utils import log_to
    from final_exam.common.provenance import code_revision
    from final_exam.error_case_report_pin import finish_case_report_pin

    source, inventory = inspect_case_source_pin(source_run, circuit, models, seeds)
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val or test')
    for name, value, minimum in (('reference_samples', reference_samples, 0),
            ('reference_seed', reference_seed, 0), ('batch_size', batch_size, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if any(type(s) is not int or s < 0 for s in sample_ids) or not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('sample_ids must be distinct nonnegative ORIGINAL source sample IDs')
    torch.set_num_threads(1)
    target_device = require_device(device)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'{circuit}_error_cases_{stamp}_{uuid.uuid4().hex[:8]}_pin'
    output = Path(output or TASK / 'results_pin' / run_id).expanduser().resolve()
    logs = Path(log_dir or TASK / 'logs_pin' / output.name).expanduser().resolve()
    raw_dir = Path(raw_dir or ROOT / 'runs/raw/full_data' / circuit).expanduser().resolve()
    for dest in (output, logs):
        if dest.exists():
            raise ValueError(f'Choose a fresh output/log directory: {dest}')
        for protected in (Path(source['source_run']), Path(source['data']).parent, raw_dir):
            if dest.is_relative_to(protected) or protected.is_relative_to(dest):
                raise ValueError('Diagnostic output must be separate from saved experiment/data/raw source')
    if output.is_relative_to(logs) or logs.is_relative_to(output):
        raise ValueError('Outputs and logs must be separate')
    if plots:
        from final_exam.spectral_report_pin import require_plotting_pin
        require_plotting_pin()
    output.mkdir(parents=True)
    logs.mkdir(parents=True)
    (output / 'plots_pin').mkdir()
    record = dict(version=1, status='running', circuit=circuit, original_sample_ids=list(sample_ids),
        source_run=source['source_run'], data_path=source['data'], dataset_fingerprint=source['dataset_fingerprint'],
        models=list(models), seeds=list(seeds), split=split, code_commit=code_revision(ROOT),
        python=sys.version, torch=str(torch.__version__), device=str(target_device),
        reference_samples=reference_samples, reference_seed=reference_seed, batch_size=batch_size,
        retrained=False, resampled=False, preprocessing_refitted=False, simulator_executed=False,
        checkpoints=[], limitations=[
            'Raw source/CSV observations are diagnostic evidence, never forward model inputs.',
            'Empirical physical-value training percentiles are not probabilities of invalid data.',
            'Finite-difference waveform slopes do not reproduce unspecified original SR extraction rules.',
            'Training reference OP errors are in-sample diagnostics, not held-out accuracy.'])
    write_json(output / 'probe_pin.json', record)
    tables = {key: [] for key in ('Cases', 'Performance', 'Parameters', 'OPComparison', 'OPSummary',
                                 'OPReference', 'PredictorChecks', 'RawChecks', 'Waveforms', 'ArtifactCandidates')}
    tables['Inventory'] = inventory
    try:
        with log_to(logs / 'error_case_probe_pin.log'):
            print(f'Loading saved dataset and verifying manifest: {source["data"]}', flush=True)
            dataset = load_dataset(source['data'], sampling_config=source['manifest'].get('sampling'))
            verify_manifest(dataset, source['manifest'])
            cases = select_cases_pin(dataset, circuit, sample_ids)
            by_id = {row['sample_id']: row for row in dataset['samples']}
            if not {c['sample_id'] for c in cases} <= set(source['manifest'][split]):
                raise ValueError(f'Some requested original IDs are not in saved {split}; select the correct run/split')
            details = dict(circuit=circuit, selected_cases=cases, raw_source={}, raw_rows={})
            raw_info, raw_rows, raw_checks, candidates = raw_evidence_pin(raw_dir, circuit, cases)
            details.update(raw_source=raw_info, raw_rows=raw_rows)
            # Preserve the small source templates as evidence, never execute them.
            details['raw_source_files'] = []
            for name in (f'{circuit}.sp', f'{circuit}_params.sp', 'circuit_metadata.json'):
                path = raw_dir / name
                if path.is_file():
                    item = dict(path=str(path), bytes=path.stat().st_size)
                    if path.stat().st_size <= 2 * 1024**2:
                        try:
                            item.update(content=path.read_text(encoding='utf-8'), sha256=file_digest(path))
                        except UnicodeError:
                            item['status'] = 'not_utf8'
                    else:
                        item['status'] = 'omitted_over_2_mib'
                    details['raw_source_files'].append(item)
            tables['RawChecks'], tables['ArtifactCandidates'] = raw_checks, candidates
            record['raw_source'] = raw_info
            write_json(output / 'case_details_pin.json', details)
            write_json(output / 'manifest_pin.json', source['manifest'])
            write_json(output / 'source_config_pin.json', source['config'])
            for case in cases:
                tables['Cases'].append(dict(circuit=circuit, sample_id=case['sample_id'],
                    source_sample_id=case['source_sample_id'], split=split, design_key=case.get('design_key'),
                    op_status=case.get('op_status'), source_observation=case.get('source_observation'),
                    context=case.get('context'), source_sampling=case.get('source_sampling')))
            reference = None
            previous_selection = None
            predictor_checks = {}
            for entry in source['checkpoints']:
                model_name, seed = entry['model'], entry['seed']
                print(f'Inspecting {model_name} seed={seed} | cases={sample_ids}', flush=True)
                checkpoint_sha = file_digest(entry['checkpoint'])
                model, saved = load_checkpoint_pin(entry['checkpoint'], target_device)
                if saved.get('mode') != model_name or saved.get('seed') != seed:
                    raise ValueError('Model/seed directory disagrees with checkpoint')
                selection = checkpoint_selection_pin(saved, source)
                if not {c['sample_id'] for c in cases} <= set(selection[split]):
                    raise ValueError('Requested cases are outside actual checkpoint evaluation cohort')
                if previous_selection is not None and selection != previous_selection:
                    raise ValueError('Compared checkpoints used different actual cohorts')
                previous_selection = selection
                prep = saved['preprocessing']
                if prep.get('raw_feature_spec', prep['feature_spec']) != dataset['feature_spec']:
                    raise ValueError('Saved preprocessing and canonical raw feature schemas differ')
                if saved.get('requires_measured_op', getattr(model, 'requires_measured_op', False)):
                    raise ValueError('Use deployed predicted-OP models, not measured-OP/oracle models')
                train_rows = [by_id[sid] for sid in selection['train']]
                if reference is None:
                    reference = TrainReferencePin(train_rows)
                    for case in cases:
                        identity = dict(circuit=circuit, source_sample_id=case['source_sample_id'])
                        for field, value in case.get('raw_design_parameters', {}).items():
                            tables['Parameters'].append(dict(identity, category='raw_design', device='', field=field,
                                value=value, unit='source parameter unit; see netlist scale',
                                **reference.describe('design', None, field, value)))
                        for d in sorted(case['devices'], key=lambda x: x['device_id']):
                            for field, value in d.get('parameters', {}).items():
                                tables['Parameters'].append(dict(identity, category='physical_device', device=d['device_id'],
                                    device_type=d['device_type'], model_name=d.get('model_name'), field=field, value=value,
                                    unit=dataset['feature_spec'].get('parameter_units', {}).get(field, ''),
                                    **reference.describe('parameters', d, field, value)))
                model.eval()
                before = state_digest(model)
                predictor = getattr(model, 'predictor', None)
                if predictor is None:
                    raise ValueError('Selected performance model has no OP predictor')
                predictor_digest = state_digest(predictor)
                op_prep = prep['device_preprocessing']
                predictor_identity = digest_pin(dict(weights=predictor_digest, preprocessing=op_prep))
                key = (seed, predictor_identity)
                first = predictor_checks.setdefault(key, model_name)
                tables['PredictorChecks'].append(dict(model=model_name, seed=seed,
                    predictor_state_digest=predictor_digest, predictor_and_preprocessing_digest=predictor_identity,
                    same_predictor_as=first if first != model_name else None,
                    explanation='Equal digest includes OP weights and saved device preprocessing; not just architecture.'))
                perf = infer_split_pin(model, saved, cases, batch_size=batch_size, device=target_device)
                metric_names = prep['feature_spec']['metric_names']
                for i, case in enumerate(cases):
                    for j, name in enumerate(metric_names):
                        observed = bool(perf['mask'][i, j])
                        truth = float(perf['target'][i, j]) if observed else None
                        pred = float(perf['prediction'][i, j])
                        tables['Performance'].append(dict(model=model_name, seed=seed, circuit=circuit,
                            source_sample_id=case['source_sample_id'], sample_id=case['sample_id'], split=split,
                            metric=name, unit=dataset['feature_spec'].get('metric_units', {}).get(name, ''),
                            observed=observed, target=truth, prediction=pred,
                            residual=pred-truth if observed else None,
                            normalized_residual=float(perf['normalized_prediction'][i,j]-perf['normalized_target'][i,j]) if observed else None))
                names, case_predictions = infer_op_cases_pin(model, saved, cases, batch_size, target_device)
                rng = np.random.default_rng(reference_seed)
                indices = np.sort(rng.choice(len(train_rows), min(reference_samples, len(train_rows)), replace=False))
                controls = [train_rows[i] for i in indices]
                if controls:
                    _, control_predictions = infer_op_cases_pin(model, saved, controls, batch_size, target_device)
                    reference_errors = op_reference_pin(controls, control_predictions, names, op_prep['op'])
                else:
                    reference_errors = {}
                for (dev, field), errors in reference_errors.items():
                    tables['OPReference'].append(dict(model=model_name, seed=seed, device=dev, field=field,
                        reference_split='train', count=len(errors), normalized_rmse=float(np.sqrt(np.mean(errors**2))),
                        abs_normalized_error_p50=float(np.quantile(np.abs(errors), .5)),
                        abs_normalized_error_p95=float(np.quantile(np.abs(errors), .95)),
                        abs_normalized_error_p99=float(np.quantile(np.abs(errors), .99))))
                for case, predicted in zip(cases, case_predictions):
                    identity = dict(model=model_name, seed=seed, circuit=circuit, source_sample_id=case['source_sample_id'])
                    case_errors = []
                    for d in sorted(case['devices'], key=lambda x: x['device_id']):
                        values = predicted[d['device_id']]
                        fields = list(dict.fromkeys([*d.get('op', {}), *names]))
                        # Unsupported passive devices do not have a physical OP prediction.
                        if not d.get('op') and not values['supported'].any():
                            continue
                        for field in fields:
                            j = names.index(field) if field in names else None
                            supported = j is not None and bool(values['supported'][j])
                            truth = d.get('op', {}).get(field)
                            physical = float(values['physical'][j]) if supported else None
                            normalized_truth = transform_value(truth, op_prep['op'], j) if supported and finite_pin(truth) else None
                            error = float(values['normalized'][j])-normalized_truth if normalized_truth is not None else None
                            controls_abs = np.sort(np.abs(reference_errors.get((d['device_id'], field), [])))
                            if error is not None:
                                case_errors.append(error)
                            tables['OPComparison'].append(dict(identity, device=d['device_id'], device_type=d['device_type'],
                                model_name=d.get('model_name'), terminals=d.get('terminals'), field=field,
                                unit=dataset['feature_spec'].get('op_units', {}).get(field, ''),
                                observed=finite_pin(truth), supported=supported, true_op=truth, predicted_op=physical,
                                residual=physical-truth if physical is not None and finite_pin(truth) else None,
                                normalized_true=normalized_truth,
                                normalized_predicted=float(values['normalized'][j]) if supported else None,
                                normalized_residual=error,
                                reference_error_count=len(controls_abs),
                                abs_error_train_percentile=TrainReferencePin.percentile(controls_abs, abs(error) if error is not None else None),
                                **reference.describe('op', d, field, truth)))
                    tables['OPSummary'].append(dict(identity, comparable_fields=len(case_errors),
                        normalized_rmse=float(np.sqrt(np.mean(np.square(case_errors)))) if case_errors else None,
                        max_abs_normalized_error=float(np.max(np.abs(case_errors))) if case_errors else None))
                if state_digest(model) != before or file_digest(entry['checkpoint']) != checkpoint_sha:
                    raise AssertionError('Read-only diagnostic modified a model/source checkpoint')
                record['checkpoints'].append(dict(model=model_name, seed=seed, checkpoint=entry['checkpoint'],
                    checkpoint_sha256=checkpoint_sha, predictor_digest=predictor_digest,
                    reference_sample_ids=[r['sample_id'] for r in controls], preprocessing=prep,
                    selected_sample_ids=selection))
                write_json(output / 'probe_pin.json', record)
                del model, saved, perf, case_predictions
                gc.collect()
                if target_device.type == 'cuda':
                    torch.cuda.empty_cache()
            tables['Waveforms'], wave_figures = waveform_evidence_pin(
                waveform_manifest, circuit, sample_ids, output, plots)
            template = dict(version=1, circuit=circuit, waveforms=[dict(source_sample_id=sid,
                path='', time_column='time_s', voltage_columns=['vout_v'], delimiter=',',
                time_scale_to_s=1., voltage_scale_to_v=1., provenance='original_saved_or_rerun_specify')
                for sid in sample_ids])
            write_json(output / 'waveform_manifest_template_pin.json', template)
            record['status'] = 'complete'
            record['waveforms_loaded'] = sum(r['status'] == 'loaded' for r in tables['Waveforms'])
            finish_case_report_pin(output, tables, record, plots, wave_figures)
            write_json(output / 'probe_pin.json', record)
            print(f'Complete: {output}', flush=True)
            print(f'Raw source: {raw_info["status"]}; matched={raw_info.get("case_values_match")}; '
                  f'waveform channels={record["waveforms_loaded"]}', flush=True)
            return output
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'probe_pin.json', record)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True)
    p.add_argument('--circuit', required=True)
    p.add_argument('--sample-ids', nargs='+', type=int, default=[41074, 38711, 38196])
    p.add_argument('--models', nargs='+', default=['device_op_cascade_pin', 'lasd_graph_layer_headwise_detach_pin_v3'])
    p.add_argument('--seeds', nargs='+', type=int, default=[29])
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--raw-dir')
    p.add_argument('--waveform-manifest')
    p.add_argument('--reference-samples', type=int, default=128)
    p.add_argument('--reference-seed', type=int, default=17)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--output')
    p.add_argument('--log-dir')
    p.add_argument('--inspect-source', action='store_true')
    args = vars(p.parse_args())
    inspect_only = args.pop('inspect_source')
    args['plots'] = not args.pop('no_plots')
    if inspect_only:
        source, inventory = inspect_case_source_pin(args['source_run'], args['circuit'], args['models'], args['seeds'])
        print(json.dumps(dict(circuit=source['circuit'], data=source['data'], checkpoints=inventory), indent=2))
        return
    run_error_case_probe_pin(**args)


if __name__ == '__main__':
    main()
