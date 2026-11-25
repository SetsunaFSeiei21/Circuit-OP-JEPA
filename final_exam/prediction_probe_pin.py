#!/usr/bin/env python3
"""Replay completed single-circuit pin checkpoints on their saved val/test rows.

No training, resampling, preprocessing fitting, checkpoint selection or source
writes. Login-node --inspect-source needs only Python's standard library.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK = ROOT / 'final_exam/single_circuit_pin'

from final_exam.gate_probe_pin import identifier, read_object, file_digest
from final_exam.spectral_probe_pin import write_json


def digest_pin(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inspect_sources_pin(source_runs, models=(), seeds=()):
    """Inventory completed model + seed pairs; an unfinished RUN is acceptable."""
    if not source_runs or any(not str(x).strip() for x in source_runs):
        raise ValueError('Fill SOURCE_RUNS with result roots or training_pin directories')
    if len(set(models)) != len(models) or len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate model/seed filters')
    for mode in models:
        identifier(mode)
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Seeds must be integers in [0, 2**32)')
    sources, inventory, seen = [], [], set()
    for source in source_runs:
        path = Path(source).expanduser().resolve()
        training = path / 'training_pin' if (path / 'training_pin/run_pin.json').is_file() else path
        if training in seen:
            raise ValueError(f'Duplicate source directory: {training}')
        seen.add(training)
        record = read_object(training / 'run_pin.json')
        config = read_object(training / 'config_pin.json')
        manifest = read_object(training / 'manifest_pin.json')
        cohort_path = training / 'cohort_pin.json'
        cohort = read_object(cohort_path) if cohort_path.is_file() else None
        circuit = record.get('circuit')
        identifier(circuit)
        if (record.get('granularity') != 'pin' or record.get('joint') or
                config.get('dataset', {}).get('train_circuits') != [circuit] or manifest.get('ood')):
            raise ValueError('Supply single_circuit_pin runs (several circuits/runs are allowed in the list)')
        fingerprint = record.get('dataset_fingerprint')
        if not fingerprint or manifest.get('dataset_fingerprint') != fingerprint:
            raise ValueError(f'Saved run/manifest fingerprints disagree: {training}')
        run_root = training.parent
        data = run_root / 'data_pin.json'
        if not data.is_file():
            pointer = read_object(run_root / 'data_source_pin.json')
            if pointer.get('dataset_fingerprint') != fingerprint or not pointer.get('data_path'):
                raise ValueError('Resume data reference is missing or has a different fingerprint')
            data = Path(pointer['data_path']).expanduser()
            if not data.is_absolute():
                data = run_root / data
        data = data.resolve()
        if not data.is_file():
            raise ValueError(f'Saved canonical data is missing: {data}')
        source_id = 'source_' + digest_pin(str(training))[:12]
        chosen = []
        for model in record.get('models', []):
            identifier(model)
            if models and model not in models:
                continue
            for seed in record.get('seeds', []):
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError('Invalid saved seed')
                if seeds and seed not in seeds:
                    continue
                directory = training / f'seed_{seed}_pin' / model
                checkpoint = directory / ('checkpoint_pin.json' if model == 'xgboost_pin' else 'checkpoint_pin.pt')
                report_path = directory / 'test_pin.json'
                entry = dict(source_id=source_id, circuit=circuit, model=model, seed=seed,
                             checkpoint=str(checkpoint), source_run=str(run_root))
                if not checkpoint.is_file() or not report_path.is_file():
                    inventory.append(dict(entry, status='skipped_incomplete', reason='Need both checkpoint and completed test report'))
                    continue
                report = read_object(report_path)
                if report.get('split') != 'test' or report.get('circuit') != circuit or 'metrics' not in report:
                    raise ValueError(f'Invalid completed test report: {report_path}')
                entry.update(report=report, report_path=str(report_path))
                chosen.append(entry)
                inventory.append(dict((k, v) for k, v in entry.items() if k not in ('report',)))
                inventory[-1]['status'] = 'completed'
        sources.append(dict(source_id=source_id, circuit=circuit, source_run=str(run_root),
            training=str(training), data=str(data), manifest=manifest, config=config,
            cohort=cohort,
            dataset_fingerprint=fingerprint, source_code_commit=record.get('code_commit'), checkpoints=chosen))
    if not any(s['checkpoints'] for s in sources):
        raise ValueError('No completed model/seed checkpoint + test report matches the filters')
    return sources, inventory


def checkpoint_selection_pin(saved, source):
    if saved.get('manifest') != source['manifest']:
        raise ValueError('Checkpoint manifest differs from the source run')
    # Native tree metadata predates selected_sample_ids. Use the run's actual
    # saved cohort, never assume an older smoke run used the whole manifest.
    selection = saved.get('selected_sample_ids')
    if selection is None and saved.get('format') == 'final_exam_xgboost_pin_v1':
        selection = source.get('cohort')
    if not isinstance(selection, dict):
        raise ValueError('Checkpoint lacks its actual saved sample cohort')
    result = {}
    for split in ('train', 'val', 'test'):
        ids = selection.get(split)
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or not set(ids) <= set(source['manifest'][split])):
            raise ValueError(f'Invalid actual checkpoint cohort: {split}')
        result[split] = sorted(ids)
    labeled = selection.get('labeled_train')
    if (not isinstance(labeled, list) or not labeled or len(set(labeled)) != len(labeled)
            or not set(labeled) <= set(result['train']) & set(source['manifest']['labeled_train'])):
        raise ValueError('Invalid actual checkpoint labeled training cohort')
    result['labeled_train'] = sorted(labeled)
    prep = saved['preprocessing']
    if prep.get('granularity') != 'pin':
        raise ValueError('Expected saved pin preprocessing')
    if not set(prep.get('fitted_sample_ids', [])) <= set(source['manifest']['train']):
        raise ValueError('Preprocessing was fitted outside the saved training split')
    return result


def training_protocol_pin(saved):
    """Keep repeated seeds from different hyperparameter experiments separate."""
    config = json.loads(json.dumps(saved['config']))
    config.get('training', {}).pop('seed', None)
    # Selection of other models is orchestration, not this model's settings.
    extra = config.get('final_exam_pin', {})
    extra.pop('models', None)
    # Overrides are already applied to saved model_config/config.model.
    extra.pop('model_overrides', None)
    if saved['mode'] != 'xgboost_pin':
        extra.pop('xgboost', None)
    return dict(config=config, options=saved.get('options'),
                input_contract=saved.get('input_contract_pin'))


def infer_split_pin(model, saved, rows, *, batch_size, device, chunk_size=256):
    """CPU stores at most chunk_size rows; only the current batch is on GPU."""
    import numpy as np
    import torch
    from device_jepa.tensors import denormalize
    from final_exam.data_pin import PinResidentDataset, group_keys_pin
    prep = saved['preprocessing']
    measured = bool(saved.get('requires_measured_op', getattr(model, 'requires_measured_op', False)))
    count = len(prep['feature_spec']['metric_names'])
    arrays = {key: np.empty((len(rows), count), dtype=(bool if key == 'mask' else np.float64))
              for key in ('target', 'prediction', 'normalized_target', 'normalized_prediction', 'mask')}
    cfg = saved.get('model_config', saved['config']['model'])
    for start in range(0, len(rows), chunk_size):
        part = rows[start:start + chunk_size]
        cache = PinResidentDataset(part, prep, 'cpu', include_op=measured, include_performance=True,
            performance_metric_mask=[n > 0 for n in prep['performance']['count']],
            performance_dtype=torch.float64, max_tokens=cfg['max_devices'], verbose=False)
        for indices, cpu_batch in cache.batches(batch_size):
            local = indices.tolist()
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cpu_batch.items()}
            batch['_sample_ids'] = [part[i]['sample_id'] for i in local]
            batch['_group_keys'] = [group_keys_pin(part[i], prep['feature_spec']['terminal_names'],
                                      prep.get('input_contract_version', 1)) for i in local]
            forbidden = {'performance', 'performance_mask', 'physical_performance'}
            if not measured:
                forbidden |= {'op', 'op_mask', 'op_target_mask', 'owner_op', 'owner_op_mask',
                              'owner_op_target_mask', 'op_state', 'op_state_mask'}
            inputs = {k: v for k, v in batch.items() if k not in forbidden}
            # no_grad matches the training evaluator, including attention backend
            # behavior; labels are held outside the forward input dictionary.
            with torch.no_grad():
                prediction = model(inputs).double()
                if prediction.shape != batch['performance'].shape or not torch.isfinite(prediction).all():
                    raise ValueError('Checkpoint produced invalid predictions')
                physical = denormalize(prediction, prep['performance'])
            destinations = start + np.asarray(local)
            for key, value in (('prediction', physical), ('target', batch['physical_performance']),
                               ('normalized_prediction', prediction), ('normalized_target', batch['performance']),
                               ('mask', batch['performance_mask'])):
                arrays[key][destinations] = value.detach().cpu().numpy()
        del cache
    return arrays


def run_prediction_probe_pin(*, source_runs, models=(), seeds=(), baseline='device_op_cascade_pin',
        splits=('val', 'test'), batch_size=32, device='cuda', bootstrap=2000,
        analysis_seed=17, top_errors=20, max_plot_points=5000, plots=True, output=None, log_dir=None):
    import numpy as np
    import torch
    from device_jepa.data import load_dataset, verify_manifest
    from device_jepa.resident import require_device
    from device_jepa.training import state_digest
    from final_exam.logging_utils import log_to
    from final_exam.run_pin import load_checkpoint_pin
    from final_exam.common.provenance import code_revision
    from final_exam.prediction_analysis_pin import (metrics_pin, training_thresholds_pin, target_strata_pin,
        seed_summary_pin, compare_series_pin)
    from final_exam.prediction_report_pin import PredictionReportPin

    sources, inventory = inspect_sources_pin(source_runs, models, seeds)
    identifier(baseline)
    if not splits or len(set(splits)) != len(splits) or set(splits) - {'val', 'test'}:
        raise ValueError('Choose val and/or test, without duplicates')
    for key, value, minimum in [('batch_size', batch_size, 1), ('bootstrap', bootstrap, 0),
            ('analysis_seed', analysis_seed, 0), ('top_errors', top_errors, 0), ('max_plot_points', max_plot_points, 1)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    target_device = require_device(device)
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'prediction_probe_{stamp}_{uuid.uuid4().hex[:8]}_pin'
    output = Path(output or TASK / 'results_pin' / run_id).resolve()
    logs = Path(log_dir or TASK / 'logs_pin' / output.name).resolve()
    for destination in (output, logs):
        if destination.exists():
            raise ValueError(f'Choose a fresh result/log directory: {destination}')
        for source in sources:
            protected = Path(source['source_run'])
            if destination.is_relative_to(protected) or protected.is_relative_to(destination):
                raise ValueError('Diagnostic outputs must be separate from source experiments')
    if output.is_relative_to(logs) or logs.is_relative_to(output):
        raise ValueError('Result and log directories must be separate')
    if plots:
        from final_exam.spectral_report_pin import require_plotting_pin
        require_plotting_pin()
    output.mkdir(parents=True)
    logs.mkdir(parents=True)
    record = dict(version=1, status='running', baseline=baseline, splits=list(splits),
        source_runs=[s['source_run'] for s in sources], filters=dict(models=list(models), seeds=list(seeds)),
        batch_size=batch_size, device=str(target_device), bootstrap=bootstrap, analysis_seed=analysis_seed,
        top_errors=top_errors, max_plot_points=max_plot_points, plots=plots,
        resampled=False, refit_preprocessing=False, retrained=False, checkpoint_modified=False,
        inference_samples='all_actual_saved_checkpoint_val_test_rows',
        pairing='exact_dataset_split_actual_cohort_performance_normalization_and_seed',
        formal_replicate='training_seed_conditional_on_fixed_data_split', checkpoints=[],
        analysis_code_commit=code_revision(ROOT), python_version=sys.version,
        torch_version=str(torch.__version__), cuda_build=torch.version.cuda,
        replay_tolerance=dict(rtol=1e-4, atol=1e-6))
    write_json(output / 'probe_pin.json', record)
    tables = {name: [] for name in ('Metrics', 'Summary', 'Aggregate', 'ValTestGap', 'Strata',
              'Thresholds', 'PairedSeeds', 'PairedSummary', 'PairStatus', 'PairedCases', 'LargestErrors')}
    tables['Inventory'] = inventory
    series, unique_models, protocols = [], set(), {}
    try:
        with log_to(logs / 'prediction_probe_pin.log'), PredictionReportPin(output, plots, max_plot_points, analysis_seed) as report:
            for source in sources:
                if not source['checkpoints']:
                    continue
                print(f'Loading {source["circuit"]}: {source["data"]}', flush=True)
                dataset = load_dataset(source['data'], sampling_config=source['manifest'].get('sampling'))
                verify_manifest(dataset, source['manifest'])
                by_id = {row['sample_id']: row for row in dataset['samples']}
                source_out = output / 'sources_pin' / source['source_id']
                source_out.mkdir(parents=True)
                write_json(source_out / 'manifest_pin.json', source['manifest'])
                write_json(source_out / 'config_pin.json', source['config'])
                write_json(source_out / 'source_pin.json', {k: source[k] for k in
                    ('source_run', 'training', 'data', 'circuit', 'dataset_fingerprint', 'source_code_commit')})
                source_labeled = set(source['manifest'].get('labeled_train', []))
                for split in ('train', 'val', 'test'):
                    for sid in source['manifest'][split]:
                        report.membership(dict(source_id=source['source_id'], circuit=source['circuit'], split=split,
                            sample_id=sid, source_sample_id=by_id[sid].get('source_sample_id'),
                            labeled_train=sid in source_labeled))
                for entry in source['checkpoints']:
                    mode, seed = entry['model'], entry['seed']
                    print(f'Replaying {source["circuit"]} {mode} seed={seed}', flush=True)
                    model, saved = load_checkpoint_pin(entry['checkpoint'], target_device)
                    if saved.get('mode') != mode or saved.get('seed') != seed:
                        raise ValueError('Checkpoint model/seed metadata differs from directory')
                    selection = checkpoint_selection_pin(saved, source)
                    saved.pop('state_dict', None)
                    prep = saved['preprocessing']
                    if prep.get('raw_feature_spec', prep['feature_spec']) != dataset['feature_spec']:
                        raise ValueError('Saved preprocessing feature schema differs from canonical data')
                    names = prep['feature_spec']['metric_names']
                    cohort = digest_pin(dict(dataset=source['dataset_fingerprint'],
                        partitions={k: sorted(source['manifest'].get(k, [])) for k in ('train', 'val', 'test', 'labeled_train')},
                        selected=selection, metrics=names, normalization=prep['performance']))[:20]
                    unique_key = (cohort, source['circuit'], mode, seed)
                    if unique_key in unique_models:
                        raise ValueError('Multiple completed checkpoints for the same cohort/model/seed; choose one source run')
                    unique_models.add(unique_key)
                    protocol = training_protocol_pin(saved)
                    protocol_id = digest_pin(protocol)
                    protocol_key = (cohort, source['circuit'], mode)
                    if protocol_key in protocols and protocols[protocol_key] != protocol_id:
                        raise ValueError('Same model has different saved training settings across seeds; analyze those experiments separately')
                    protocols[protocol_key] = protocol_id
                    model.eval()
                    before = state_digest(model)
                    checkpoint_id = digest_pin([source['source_id'], mode, seed])[:16]
                    checkpoint_out = source_out / f'seed_{seed}_pin' / mode
                    checkpoint_out.mkdir(parents=True)
                    write_json(checkpoint_out / 'selection_pin.json', selection)
                    write_json(checkpoint_out / 'training_protocol_pin.json', protocol)
                    labeled = set(selection['labeled_train'])
                    thresholds = {name: training_thresholds_pin([by_id[sid]['performance'][name]
                        for sid in selection['train'] if sid in labeled and by_id[sid]['performance'].get(name) is not None])
                        for name in names}
                    identity = dict(source_id=source['source_id'], cohort_id=cohort, circuit=source['circuit'], model=mode, seed=seed)
                    for name, threshold in thresholds.items():
                        tables['Thresholds'].append(dict(identity, metric=name, fit_split='saved_labeled_train', **(threshold or {})))
                    info = dict(identity, checkpoint=entry['checkpoint'], checkpoint_sha256=file_digest(entry['checkpoint']),
                        training_protocol_id=protocol_id,
                        requires_measured_op=bool(saved.get('requires_measured_op', getattr(model, 'requires_measured_op', False))),
                        selection_file=str((checkpoint_out / 'selection_pin.json').relative_to(output)))
                    record['checkpoints'].append(info)
                    for split in splits:
                        ids = selection[split]
                        rows = [by_id[sid] for sid in ids]
                        values = infer_split_pin(model, saved, rows, batch_size=batch_size, device=target_device)
                        normalized_errors = values['normalized_prediction'] - values['normalized_target']
                        valid = values['mask']
                        total_mse = float(np.mean(normalized_errors[valid] ** 2)) if valid.any() else None
                        total = dict(identity, split=split, count=len(ids), observed_labels=int(valid.sum()), normalized_mse=total_mse)
                        if split == 'test':
                            previous = entry['report'].get('normalized_mse')
                            total.update(saved_test_normalized_mse=previous,
                                replay_mse_difference=total_mse - previous if total_mse is not None and previous is not None else None)
                            if previous is None or total_mse is None:
                                total['replay_status'] = 'no_comparable_saved_mse'
                            else:
                                matched = np.isclose(total_mse, previous, **record['replay_tolerance'])
                                matched = bool(matched and entry['report'].get('count') == len(ids))
                                total['replay_status'] = 'matched_tolerance' if matched else 'mismatch_check_before_interpreting'
                                if not matched:
                                    print(f'WARNING: {mode} seed={seed} test replay differs from saved report: {total}', flush=True)
                        tables['Aggregate'].append(total)
                        for j, name in enumerate(names):
                            observed = valid[:, j]
                            item = dict(identity, split=split, metric=name,
                                ids=[sid for i, sid in enumerate(ids) if observed[i]],
                                target=values['target'][observed, j], prediction=values['prediction'][observed, j],
                                normalized_error=normalized_errors[observed, j])
                            stats = metrics_pin(item['target'], item['prediction'], item['normalized_error'])
                            item['stats'] = stats
                            series.append(item)
                            metric_identity = dict(identity, split=split, metric=name)
                            metric_row = dict(metric_identity, **stats)
                            if split == 'test':
                                original = entry['report']['metrics'].get(name, {})
                                for key in ('mae', 'rmse', 'r2'):
                                    a, b = original.get(key), stats.get(key)
                                    metric_row.update({f'saved_test_{key}': a,
                                        f'replay_{key}_difference': b - a if a is not None and b is not None else None})
                            tables['Metrics'].append(metric_row)
                            strata = target_strata_pin(item['target'], thresholds[name])
                            for group in sorted(set(strata)):
                                selected = strata == group
                                tables['Strata'].append(dict(metric_identity, stratum=group,
                                    **metrics_pin(item['target'][selected], item['prediction'][selected], item['normalized_error'][selected])))
                            for i, sid in enumerate(ids):
                                live = bool(observed[i])
                                row = dict(metric_identity, sample_id=sid, source_sample_id=rows[i].get('source_sample_id'), observed=live,
                                    target=float(values['target'][i, j]) if live else None,
                                    prediction=float(values['prediction'][i, j]),
                                    residual=float(values['prediction'][i, j] - values['target'][i, j]) if live else None,
                                    normalized_target=float(values['normalized_target'][i, j]) if live else None,
                                    normalized_prediction=float(values['normalized_prediction'][i, j]),
                                    normalized_residual=float(normalized_errors[i, j]) if live else None)
                                report.prediction(row)
                            errors = np.abs(item['prediction'] - item['target'])
                            for rank, i in enumerate(np.argsort(-errors, kind='stable')[:top_errors], 1):
                                tables['LargestErrors'].append(dict(metric_identity, rank=rank, sample_id=item['ids'][i],
                                    source_sample_id=by_id[item['ids'][i]].get('source_sample_id'),
                                    target=float(item['target'][i]), prediction=float(item['prediction'][i]),
                                    absolute_error=float(errors[i]), normalized_squared_error=float(item['normalized_error'][i] ** 2),
                                    target_stratum=str(strata[i])))
                            report.plot(item, checkpoint_id, thresholds[name])
                        print(f'  {split}: {len(ids)} samples, normalized MSE={total_mse}', flush=True)
                    if state_digest(model) != before:
                        raise ValueError('Inference changed model weights/buffers')
                    del model, saved
                    gc.collect()
                    if target_device.type == 'cuda':
                        torch.cuda.empty_cache()
                    write_json(output / 'probe_pin.json', record)
                del dataset, by_id
                gc.collect()
            tables['Summary'] = seed_summary_pin(tables['Metrics'])
            paired = compare_series_pin(series, baseline, bootstrap=bootstrap, seed=analysis_seed, top_errors=top_errors)
            for name, rows in zip(('PairedSeeds', 'PairedSummary', 'PairStatus', 'PairedCases'), paired):
                tables[name] = rows
            metrics_by_key = {tuple(r[k] for k in ('source_id', 'circuit', 'model', 'seed', 'metric', 'split')): r for r in tables['Metrics']}
            for key, test in metrics_by_key.items():
                if key[-1] != 'test' or (*key[:-1], 'val') not in metrics_by_key:
                    continue
                val = metrics_by_key[(*key[:-1], 'val')]
                row = {k: test[k] for k in ('source_id', 'cohort_id', 'circuit', 'model', 'seed', 'metric')}
                for metric in ('mae', 'rmse', 'r2', 'normalized_mse'):
                    a, b = val.get(metric), test.get(metric)
                    row.update({f'val_{metric}': a, f'test_{metric}': b,
                                f'test_minus_val_{metric}': b - a if a is not None and b is not None else None})
                tables['ValTestGap'].append(row)
            report.finish(tables, record)
            record['status'] = 'complete'
            write_json(output / 'probe_pin.json', record)
            print(f'Excel: {output / "prediction_summary_pin.xlsx"}\nViewer: {output / "prediction_viewer_pin.html"}', flush=True)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        write_json(output / 'probe_pin.json', record)
        with (logs / 'prediction_probe_pin.log').open('a', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-runs', nargs='+', required=True)
    parser.add_argument('--models', nargs='*', default=[])
    parser.add_argument('--seeds', nargs='*', type=int, default=[])
    parser.add_argument('--baseline', default='device_op_cascade_pin')
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--analysis-seed', type=int, default=17)
    parser.add_argument('--top-errors', type=int, default=20)
    parser.add_argument('--max-plot-points', type=int, default=5000)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--log-dir')
    parser.add_argument('--inspect-source', action='store_true')
    args = parser.parse_args(argv)
    if args.inspect_source:
        sources, inventory = inspect_sources_pin(args.source_runs, args.models, args.seeds)
        print(json.dumps(dict(sources=[dict(circuit=s['circuit'], source_run=s['source_run'], data=s['data'])
                                      for s in sources], inventory=inventory), ensure_ascii=False, indent=2))
        return
    options = vars(args)
    options.pop('inspect_source')
    options['plots'] = not options.pop('no_plots')
    run_prediction_probe_pin(**options)


if __name__ == '__main__':
    main()
