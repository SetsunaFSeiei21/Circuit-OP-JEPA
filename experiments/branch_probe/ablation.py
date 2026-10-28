"""Controlled readout architecture x feature scaling on BOTH frozen branches.

Only ID train/val are used. Fixed device slots, cached features, labels, minibatch
order and optimization settings are shared by all cells of the experiment.
"""
import copy
import hashlib
import math
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from device_jepa.early_stopping import EarlyStopping, SUPERVISED_OBJECTIVE
from device_jepa.model import ModelConfig, make_decoder, masked_mse
from device_jepa.resident import BatchProgress, require_device
from device_jepa.training import seed_everything, state_digest
from experiments.branch_probe.diagnostics import (extract_features, fit_targets,
    score, select_rows, write_csv)
from experiments.branch_probe.probe import _json, _json_digest


HEADS = ('attention', 'mlp')
NORMALIZATIONS = ('raw', 'standardized')


def fit_feature_scaler(train, floor=1e-4):
    """Fit one mean/std per FIXED device and coordinate, using train only."""
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('std_floor must be finite and positive')
    if train.ndim != 3 or len(train) < 2 or not bool(torch.isfinite(train).all()):
        raise ValueError('Expected finite [>=2 samples, devices, features]')
    variance, mean = torch.var_mean(train.double(), dim=0, unbiased=False)
    std = variance.sqrt()
    return dict(mean=mean, scale=std.clamp_min(floor), std=std, std_floor=floor,
                fitted_samples=len(train))


def transform_memory(memory, normalization, scaler):
    if normalization == 'raw':
        return memory
    if normalization != 'standardized':
        raise ValueError('Unknown normalization')
    result = ((memory.double() - scaler['mean']) / scaler['scale']).float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Nonfinite standardized memory')
    return result


def make_readout(kind, cfg, devices, metrics, hidden):
    if kind == 'attention':
        return make_decoder(cfg, metrics)
    if kind == 'mlp':
        return nn.Sequential(nn.Flatten(1), nn.Linear(devices * cfg.dim, hidden),
                             nn.GELU(), nn.Linear(hidden, metrics))
    raise ValueError('Unknown readout architecture')


def forward_readout(head, kind, memory):
    if kind == 'attention':
        # extract_features already retained only eligible, aligned device slots.
        padding = torch.zeros(memory.shape[:2], device=memory.device, dtype=torch.bool)
        return head(memory, padding)
    return head(memory)


@torch.no_grad()
def predict(head, kind, memory, batch_size):
    head.eval()
    result = torch.cat([forward_readout(head, kind, chunk)
                        for chunk in memory.split(batch_size)]).double()
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Nonfinite readout predictions')
    return result


def fit_readout(kind, cfg, x, y, mask, val, val_y, val_mask, *, seed=17,
                epochs=50, batch_size=256, hidden=128, lr=1e-3, weight_decay=1e-4,
                grad_clip=1., patience=0, progress_every=25, label=None):
    """One training loop for both heads; patience=0 executes the full budget."""
    seed_everything(seed)
    head = make_readout(kind, cfg, x.shape[1], y.shape[1], hidden).to(x.device)
    initial_digest = state_digest(head)
    # Reset after initialization so paired scaling cells also share dropout RNG.
    seed_everything(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    stopping = EarlyStopping(dict(early_stopping=dict(enabled=patience > 0,
        patience=max(1, patience), min_delta=0., min_epochs=1)), epochs, SUPERVISED_OBJECTIVE)
    history, best, steps = [], None, 0
    order_digest = hashlib.sha256()
    label = label or kind
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        head.train()
        order = list(range(len(x)))
        random.Random(seed + epoch).shuffle(order)
        cpu_indices = torch.tensor(order, dtype=torch.int64)
        order_digest.update(cpu_indices.numpy().tobytes())
        order = cpu_indices.to(x.device)
        total_loss = torch.zeros((), dtype=torch.float64, device=x.device)
        observations = 0
        progress = BatchProgress(f'{label} epoch {epoch}/{epochs}',
                                 math.ceil(len(x) / batch_size), progress_every, x.device)
        for step, start in enumerate(range(0, len(x), batch_size), 1):
            idx = order[start:start + batch_size]
            count = int(mask[idx].sum())
            if count:
                loss = masked_mse(forward_readout(head, kind, x[idx]), y[idx].float(), mask[idx])
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite readout training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.detach().double() * count
                observations += count
                steps += 1
            progress.update(step)
        value = float(masked_mse(predict(head, kind, val, batch_size), val_y, val_mask))
        history.append(dict(epoch=epoch, train_online_normalized_mse=float(total_loss / observations),
                            val_normalized_mse=value, seconds=time.perf_counter() - started))
        print(f'{label}: epoch {epoch}/{epochs} | val normalized MSE={value:.7g}', flush=True)
        if stopping.update(value, epoch):
            best = copy.deepcopy(head.state_dict())
        if stopping.should_stop:
            stopping.announce(label)
            break
    head.load_state_dict(best, strict=True)
    a, b = predict(head, kind, x, batch_size), predict(head, kind, val, batch_size)
    metadata = dict(initial_digest=initial_digest, selected_digest=state_digest(head),
                    parameter_count=sum(p.numel() for p in head.parameters()),
                    optimizer_steps=steps, sample_order_sha256=order_digest.hexdigest(),
                    early_stopping=stopping.summary(), history=history)
    return a, b, metadata, {k: v.detach().cpu() for k, v in best.items()}


def summarize_readouts(reports):
    """Aggregate decoder seeds, keeping physical metrics and splits separate."""
    groups, pairs = {}, {}
    for report in reports:
        for split in ('train', 'val'):
            for metric, result in report[split]['metrics'].items():
                key = (report['source'], report['head'], report['normalization'], split, metric)
                groups.setdefault(key, []).append(result)
                pair = (report['source'], report['head'], report['seed'], split, metric)
                pairs.setdefault(pair, {})[report['normalization']] = result
    summary, deltas = [], []
    for key, values in sorted(groups.items()):
        counts = {v['count'] for v in values}
        if len(counts) != 1:
            raise ValueError('Different evaluation cohorts across decoder seeds')
        row = dict(zip(('source', 'head', 'normalization', 'split', 'metric'), key))
        row.update(count=values[0]['count'], decoder_seeds=len(values))
        for name in ('mae', 'rmse', 'r2', 'normalized_rmse'):
            observed = [v[name] for v in values if v[name] is not None]
            row.update({f'{name}_n': len(observed),
                        f'{name}_mean': statistics.mean(observed) if observed else None,
                        f'{name}_std': statistics.stdev(observed) if len(observed) > 1 else None})
        summary.append(row)
    for key, cells in sorted(pairs.items()):
        if set(cells) != set(NORMALIZATIONS):
            raise ValueError('Missing cell in raw/standardized pair')
        row = dict(zip(('source', 'head', 'seed', 'split', 'metric'), key))
        for name in ('mae', 'rmse', 'r2', 'normalized_rmse'):
            a, b = cells['raw'][name], cells['standardized'][name]
            row[name + '_standardized_minus_raw'] = b - a if a is not None and b is not None else None
        deltas.append(row)
    return summary, deltas


def validate_options(*, seeds, sample_seed, epochs, batch_size, encoder_batch_size,
                     hidden, lr, weight_decay, grad_clip, patience, std_floor, progress_every):
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in [*seeds, sample_seed])):
        raise ValueError('Seeds must be unique integers in [0, 2**32); sample_seed must be in that range')
    for name, value in dict(epochs=epochs, batch_size=batch_size, encoder_batch_size=encoder_batch_size,
                            hidden=hidden, progress_every=progress_every).items():
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(patience) is not int or patience < 0:
        raise ValueError('patience must be >=0 (0 disables early stopping)')
    for name, value in dict(lr=lr, weight_decay=weight_decay, grad_clip=grad_clip, std_floor=std_floor).items():
        if (type(value) not in (float, int) or not math.isfinite(value) or value < 0
                or (name != 'weight_decay' and value == 0)):
            raise ValueError(f'Invalid {name}')


def run_ablation(context, output, *, circuit=None, seeds=(17,), sample_seed=17,
                 max_train=4096, max_val=1024, encoder_batch_size=32, batch_size=256,
                 epochs=50, hidden=128, lr=1e-3, weight_decay=1e-4, grad_clip=1.,
                 patience=0, std_floor=1e-4, progress_every=25, device='cuda'):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be new or empty; refusing to overwrite ablation')
    settings = dict(epochs=epochs, batch_size=batch_size, encoder_batch_size=encoder_batch_size,
                    hidden=hidden, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
                    patience=patience, std_floor=std_floor, progress_every=progress_every)
    validate_options(seeds=seeds, sample_seed=sample_seed, **settings)
    device = require_device(device)
    seed_everything(sample_seed)
    cfg = ModelConfig(**context['source']['model_config'])
    rows, identity = select_rows(context, circuit, max_train, max_val, sample_seed)
    print('Fitting shared target transforms on selected ID train only', flush=True)
    prep = fit_targets(context, rows['train'], device)
    active = [name for name, count in zip(prep['feature_spec']['metric_names'], prep['performance']['count']) if count]
    if not any(r['performance'].get(name) is not None for r in rows['val'] for name in active):
        raise ValueError('No validation labels for metrics observed in selected train')
    plan = dict(experiment_kind='frozen_readout_2x2', version=1, complete=False,
                source_checkpoint=context['source_checkpoint'], source_checkpoint_sha256=context['source_sha256'],
                source_model_digest=context['source_model_digest'], source_mode=context['source'].get('mode', 'jepa_zerosim'),
                source_model_config=cfg.to_dict(), pretrain_seed=context['source']['options']['training']['seed'],
                partition_fingerprint=context['source']['partition_fingerprint'],
                input_op_fingerprint=context['source']['input_op_fingerprint'],
                dataset_fingerprint=context['manifest']['dataset_fingerprint'],
                sample_ids={s: [r['sample_id'] for r in rs] for s, rs in rows.items()},
                decoder_seeds=list(seeds), sample_seed=sample_seed, settings=settings, **identity,
                device=str(device), readout_policy='fixed_eligible_device_slots_no_graph',
                factors=dict(source=['design', 'op'], head=list(HEADS), normalization=list(NORMALIZATIONS)),
                target_scaler_digest=_json_digest(prep['performance']), evaluation_splits=['train', 'val'],
                seed_interpretation='decoder seeds conditional on one pretrained checkpoint and one fixed subset',
                budget_policy='equal_executed_epochs' if patience == 0 else 'equal_max_epochs_with_early_stopping')
    output.mkdir(parents=True, exist_ok=True)
    _json(output / 'run.json', plan)
    _json(output / 'preprocessing.json', prep)
    print(f'2x2 readout ablation: {output} | {len(rows["train"])} train / {len(rows["val"])} val', flush=True)
    features, target, mask, physical, frozen = extract_features(
        context, rows['train'] + rows['val'], prep, device, encoder_batch_size)
    del features['raw_op']  # This experiment changes only readout architecture and scaling.
    n = len(rows['train'])
    y, vy, m, vm = target[:n], target[n:], mask[:n], mask[n:]
    scalers = {branch: fit_feature_scaler(memory[:n], std_floor) for branch, memory in features.items()}
    torch.save({branch: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in scaler.items()}
                for branch, scaler in scalers.items()}, output / 'feature_scalers.pt')
    mean = (y * m).sum(0) / m.sum(0).clamp_min(1)
    _json(output / 'train_mean.json', dict(train=score(mean.expand_as(y), y, m, prep, physical[:n]),
                                          val=score(mean.expand_as(vy), vy, vm, prep, physical[n:])))
    _json(output / 'frozen_encoder.json', dict(source_checkpoint_sha256=context['source_sha256'], **frozen))
    reports, initial_digests, order_digests = [], {}, {}
    for branch in ('design', 'op'):
        for normalization in NORMALIZATIONS:
            memory = transform_memory(features[branch], normalization, scalers[branch])
            for seed in seeds:
                for kind in HEADS:
                    label = f'{branch}/{kind}/{normalization}/seed_{seed}'
                    started = time.perf_counter()
                    a, b, training, state = fit_readout(kind, cfg, memory[:n], y, m, memory[n:], vy, vm,
                        seed=seed, epochs=epochs, batch_size=batch_size, hidden=hidden, lr=lr,
                        weight_decay=weight_decay, grad_clip=grad_clip, patience=patience,
                        progress_every=progress_every, label=label)
                    key = (kind, seed)
                    expected = initial_digests.setdefault(key, training['initial_digest'])
                    if training['initial_digest'] != expected:
                        raise AssertionError('Paired head initializations differ')
                    if patience == 0:
                        budget = (training['sample_order_sha256'], training['optimizer_steps'])
                        if order_digests.setdefault(seed, budget) != budget:
                            raise AssertionError('Fixed-budget sample order or optimizer steps differ')
                    cell = output / f'seed_{seed}' / branch / kind / normalization
                    cell.mkdir(parents=True)
                    report = dict(source=branch, head=kind, normalization=normalization, seed=seed,
                        seconds=time.perf_counter() - started, training=training,
                        source_checkpoint_sha256=context['source_sha256'],
                        target_scaler_digest=plan['target_scaler_digest'],
                        train=score(a, y, m, prep, physical[:n]), val=score(b, vy, vm, prep, physical[n:]))
                    _json(cell / 'report.json', report)
                    _json(cell / 'history.json', training['history'])
                    torch.save(dict(experiment_kind=plan['experiment_kind'], head_state=state,
                        head_digest=training['selected_digest'], head=kind, source=branch,
                        normalization=normalization, seed=seed, model_config=cfg.to_dict(), hidden=hidden,
                        metric_names=prep['feature_spec']['metric_names'], device_ids=identity['device_ids'],
                        best_epoch=training['early_stopping']['best_epoch'],
                        source_checkpoint_sha256=context['source_sha256'],
                        target_scaler_digest=plan['target_scaler_digest'],
                        shared_files_relative_to_run=['run.json', 'preprocessing.json', 'feature_scalers.pt']),
                        cell / 'head_best.pt')
                    reports.append(report)
                    _json(output / 'readouts.json', reports)
                    print(f'{label}: selected val MSE={report["val"]["normalized_mse"]:.7g}', flush=True)
            del memory
    table = [dict(source=r['source'], head=r['head'], normalization=r['normalization'], seed=r['seed'],
                  split=split, metric=name, **metric)
             for r in reports for split in ('train', 'val') for name, metric in r[split]['metrics'].items()]
    summary, deltas = summarize_readouts(reports)
    write_csv(output / 'readouts.csv', table)
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'paired_deltas.csv', deltas)
    _json(output / 'summary.json', dict(experiment_kind=plan['experiment_kind'],
        seed_interpretation=plan['seed_interpretation'], summary=summary, paired_deltas=deltas))
    plan.update(complete=True, encoder_frozen_verified=frozen['encoder_frozen_verified'],
                initialization_pairing_verified=True, fixed_budget_verified=patience == 0)
    _json(output / 'run.json', plan)
    print(f'Complete. Share run.json, readouts.json and summary.csv from {output}', flush=True)
    return reports
