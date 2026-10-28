"""Within-device variation and simple readouts of a fixed pretrained circuit.

Diagnostic only: flatten aligned device slots, keep the encoders frozen, and
use the original ID train/val membership. No test/OOD fitting or evaluation.
"""
import copy
import csv
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from device_jepa.data import is_op_target
from device_jepa.early_stopping import EarlyStopping, SUPERVISED_OBJECTIVE
from device_jepa.model import masked_mse
from device_jepa.resident import (BatchProgress, ResidentDataset, _reserve,
                                  _topology_key, device_statistics, require_device)
from device_jepa.tensors import denormalize
from device_jepa.training import _metrics, seed_everything, state_digest
from experiments.branch_probe.probe import (_json, _snapshot, _verify_frozen,
                                            branch_memory, reconstruct_model)


def select_rows(context, circuit=None, max_train=4096, max_val=1024, seed=17):
    """Choose deterministic, label-independent subsets and require fixed slots."""
    circuits = sorted({r['circuit_id'] for r in context['split_rows']['train']})
    if circuit is None:
        if len(circuits) != 1:
            raise ValueError('Multiple training circuits; specify --circuit: ' + ', '.join(circuits))
        circuit = circuits[0]
    if circuit not in circuits:
        raise ValueError('--circuit must belong to the original ID training split')
    selected, available = {}, {}
    for split, limit in (('train', max_train), ('val', max_val)):
        if limit is not None and (type(limit) is not int or limit < 2):
            raise ValueError('Sample limits must be >=2 or none')
        rows = sorted((r for r in context['split_rows'][split] if r['circuit_id'] == circuit),
                      key=lambda r: r['sample_id'])
        available[split] = len(rows)
        if limit is not None and len(rows) > limit:
            rows = sorted(random.Random(seed).sample(rows, limit), key=lambda r: r['sample_id'])
        if len(rows) < 2:
            raise ValueError(f'Need at least two {split} samples for circuit {circuit}')
        selected[split] = rows
    first = selected['train'][0]
    devices = sorted(first['devices'], key=lambda d: d['device_id'])
    key = _topology_key(first, devices)
    eligible = [is_op_target(d) for d in devices]
    for row in selected['train'] + selected['val']:
        ordered = sorted(row['devices'], key=lambda d: d['device_id'])
        if (_topology_key(row, ordered) != key or [is_op_target(d) for d in ordered] != eligible):
            raise ValueError('Fixed-slot diagnostics require the same device IDs, models, ports and topology')
    if not any(eligible):
        raise ValueError('No eligible devices for branch diagnostics')
    return selected, dict(circuit=circuit, available_samples=available,
                         device_ids=[d['device_id'] for d in devices if is_op_target(d)])


def fit_targets(context, rows, device):
    """Keep pretrained input transforms; fit labels only on selected train rows."""
    prep = copy.deepcopy(context['prep'])
    prep['performance'] = device_statistics(
        [(r['circuit_id'], r['performance']) for r in rows],
        prep['feature_spec']['metric_names'], prep['transform_config']['performance'],
        prep['transform_config']['weighting'], device)
    prep['performance_fitted_sample_ids'] = [r['sample_id'] for r in rows]
    prep['fit_ids']['performance'] = list(prep['performance_fitted_sample_ids'])
    if not any(prep['performance']['count']):
        raise ValueError('Selected training subset has no performance labels')
    return prep


def embedding_variation(memory, device_ids):
    """Population variance across designs at each FIXED device/coordinate.

Double precision and centered reductions prevent cancellation of small changes
around a large, device-specific mean. No scaling is applied to this diagnostic.
"""
    if memory.ndim != 3 or memory.shape[0] < 2 or memory.shape[1] != len(device_ids):
        raise ValueError('Expected [>=2 designs, aligned devices, features]')
    if not bool(torch.isfinite(memory).all()):
        raise ValueError('Nonfinite diagnostic features')
    z = memory.double()
    variance, means = torch.var_mean(z, dim=0, unbiased=False)
    within = variance.mean()
    between = means.var(dim=0, unbiased=False).mean()
    total = within + between
    per_device = []
    for i, name in enumerate(device_ids):
        std = variance[i].sqrt()
        per_device.append(dict(device_id=name, std_rms=float(variance[i].mean().sqrt()),
                               std_median=float(std.median()), std_max=float(std.max()),
                               fraction_std_below_1e_6=float((std < 1e-6).double().mean())))
    return dict(samples=z.shape[0], devices=z.shape[1], dimensions=z.shape[2],
                within_device_std_rms=float(within.sqrt()),
                between_device_std_rms=float(between.sqrt()),
                pooled_token_std_rms=float(total.sqrt()),
                within_fraction_of_total_variance=float(within / total) if float(total) > 0 else None,
                per_device=per_device)


def extract_features(context, rows, prep, device, batch_size):
    model = reconstruct_model(context['source']).to(device).requires_grad_(False).eval()
    before = _snapshot(model)
    active = [count > 0 for count in prep['performance']['count']]
    resident = ResidentDataset(rows, prep, device, include_op=True, include_op_states=True,
                               include_performance=True, performance_metric_mask=active,
                               performance_dtype=torch.float64, max_tokens=model.encoder.cfg.max_devices,
                               label='branch diagnostics')
    eligible = resident.tables['op_target_mask'][0]
    if not bool((resident.tables['op_target_mask'] == eligible).all()):
        raise ValueError('Eligibility changed across diagnostic examples')
    count, dim = int(eligible.sum()), model.encoder.cfg.dim
    _reserve(device, 2 * len(rows) * count * dim * 4)
    features = {branch: torch.empty(len(rows), count, dim, device=device) for branch in ('design', 'op')}
    progress = BatchProgress('extract both frozen branches', math.ceil(len(rows) / batch_size), 25, device)
    with torch.no_grad():
        for step, (indices, batch) in enumerate(resident.batches(batch_size), 1):
            for branch in ('design', 'op'):
                memory, padding = branch_memory(model, batch, branch, batch['op_target_mask'])
                if bool(padding[:, eligible].any()):
                    raise ValueError('An eligible device was padded')
                features[branch][indices] = memory[:, eligible]
            progress.update(step)
    _verify_frozen(model, before)
    # Raw OP is a diagnostic control with the same continuous fields and state
    # inputs as the OP branch. Masks stay explicit; no geometry is appended.
    op, observed = resident.tables['op'][:, eligible], resident.tables['op_mask'][:, eligible]
    raw_parts = [op, observed.float()]
    if 'op_state' in resident.tables:
        raw_parts += [resident.tables['op_state'][:, eligible],
                      resident.tables['op_state_mask'][:, eligible].float()]
    features['raw_op'] = torch.cat(raw_parts, dim=-1)
    valid = observed.any(-1)
    metadata = dict(encoder_frozen_verified=True, frozen_model_digest=state_digest(model),
                    eligible_devices=valid.numel(), observed_devices=int(valid.sum()),
                    all_op_missing_samples=int((~valid.any(-1)).sum()),
                    alignment_mse_observed_devices=(float((features['design'][valid].double()
                                                          - features['op'][valid].double()).square().mean())
                                                    if bool(valid.any()) else None))
    return (features, resident.tables['performance'], resident.tables['performance_mask'],
            resident.tables['physical_performance'], metadata)


def standardize_features(train, val, floor=1e-4):
    """Train-only scaling; the floor prevents amplifying float32 rounding noise."""
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('Feature std floor must be finite and positive')
    a, b = train.flatten(1).double(), val.flatten(1).double()
    variance, mean = torch.var_mean(a, dim=0, unbiased=False)
    std = variance.sqrt()
    scale = std.clamp_min(floor)
    x, v = ((a - mean) / scale).float(), ((b - mean) / scale).float()
    if not bool(torch.isfinite(x).all() and torch.isfinite(v).all()):
        raise ValueError('Nonfinite standardized diagnostic features')
    return x, v, dict(columns=x.shape[1], std_floor=floor,
                      columns_below_floor=int((std < floor).sum()),
                      train_std_min=float(std.min()), train_std_max=float(std.max()))


def ridge_predict(x, y, mask, val, alpha=1.0):
    """Exact linear ridge: sum squared error + alpha * ||W||^2; free intercept.

Group metrics with identical observation masks. Solve on the requested device,
in float64; use the smaller primal/dual system. Alpha is fixed, never tuned on
test data, and this baseline does not depend on neural optimizer convergence.
"""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Ridge alpha must be finite and positive')
    x, y, val = x.double(), y.double(), val.double()
    weights = x.new_zeros(x.shape[1], y.shape[1])
    intercept = y.new_zeros(y.shape[1])
    groups = {}
    for j in range(y.shape[1]):
        key = tuple(mask[:, j].tolist())
        if any(key):
            groups.setdefault(key, []).append(j)
    for index, (key, metrics) in enumerate(groups.items(), 1):
        print(f'  ridge observation group {index}/{len(groups)}', flush=True)
        selected = torch.tensor(key, device=x.device, dtype=torch.bool)
        a, target = x[selected], y[selected][:, metrics]
        mx, my = a.mean(0), target.mean(0)
        a, target = a - mx, target - my
        size = min(a.shape)
        _reserve(x.device, 5 * size * size * 8)
        if a.shape[1] <= a.shape[0]:
            gram, rhs = a.T @ a, a.T @ target
            gram.diagonal().add_(alpha)
            w = torch.cholesky_solve(rhs, torch.linalg.cholesky(gram))
        else:
            gram = a @ a.T
            gram.diagonal().add_(alpha)
            w = a.T @ torch.cholesky_solve(target, torch.linalg.cholesky(gram))
        weights[:, metrics], intercept[metrics] = w, my - mx @ w
    return x @ weights + intercept, val @ weights + intercept


def mlp_predict(x, y, mask, val, val_y, val_mask, *, seed=17, epochs=50,
                batch_size=256, hidden=128, lr=1e-3, patience=10):
    seed_everything(seed)
    head = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.GELU(), nn.Linear(hidden, y.shape[1])).to(x.device)
    initial_digest = state_digest(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    training = dict(early_stopping=dict(enabled=True, patience=patience, min_delta=0., min_epochs=1))
    stopping = EarlyStopping(training, epochs, SUPERVISED_OBJECTIVE)
    history, best, steps = [], None, 0
    for epoch in range(1, epochs + 1):
        head.train()
        order = list(range(len(x)))
        random.Random(seed + epoch).shuffle(order)
        order = torch.tensor(order, device=x.device)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if not bool(mask[idx].any()):
                continue
            loss = masked_mse(head(x[idx]), y[idx].float(), mask[idx])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite MLP training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
        head.eval()
        with torch.no_grad():
            pred = torch.cat([head(chunk) for chunk in val.split(batch_size)]).double()
            value = float(masked_mse(pred, val_y, val_mask))
        history.append(dict(epoch=epoch, val_normalized_mse=value))
        print(f'  MLP {epoch}/{epochs}: val normalized MSE={value:.7g}', flush=True)
        if stopping.update(value, epoch):
            best = copy.deepcopy(head.state_dict())
        if stopping.should_stop:
            stopping.announce('diagnostic MLP')
            break
    head.load_state_dict(best, strict=True)
    head.eval()
    with torch.no_grad():
        predict = lambda data: torch.cat([head(chunk) for chunk in data.split(batch_size)]).double()
        a, b = predict(x), predict(val)
    return a, b, dict(initial_digest=initial_digest, selected_digest=state_digest(head),
                      parameter_count=sum(p.numel() for p in head.parameters()),
                      optimizer_steps=steps, early_stopping=stopping.summary(), history=history)


def score(prediction, target, mask, prep, physical_target):
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError('Nonfinite readout prediction')
    physical_p = denormalize(prediction, prep['performance'])
    physical_y = physical_target
    metrics = _metrics(physical_p, physical_y, mask, prep, prediction, target)
    for j, name in enumerate(prep['feature_spec']['metric_names']):
        selected = mask[:, j]
        metrics[name].update(prediction_std=(float(physical_p[selected, j].std(unbiased=False))
                                              if bool(selected.any()) else None),
                             target_std=(float(physical_y[selected, j].std(unbiased=False))
                                         if bool(selected.any()) else None))
    return dict(normalized_mse=float(masked_mse(prediction, target, mask)), metrics=metrics)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostics(context, output, *, circuit=None, seed=17, max_train=4096,
                    max_val=1024, batch_size=256, encoder_batch_size=32, epochs=50,
                    hidden=128, lr=1e-3, patience=10, ridge_alpha=1., std_floor=1e-4,
                    device='cuda', stats_only=False):
    output, device = Path(output), require_device(device)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be new or empty; refusing to overwrite diagnostics')
    for name, value in (('batch_size', batch_size), ('encoder_batch_size', encoder_batch_size),
                        ('epochs', epochs), ('hidden', hidden), ('patience', patience)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name, value in (('lr', lr), ('ridge_alpha', ridge_alpha), ('std_floor', std_floor)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be in [0, 2**32)')
    seed_everything(seed)
    rows, identity = select_rows(context, circuit, max_train, max_val, seed)
    print('Fitting diagnostic target transforms on selected ID train only', flush=True)
    prep = fit_targets(context, rows['train'], device)
    active = [n for n, c in zip(prep['feature_spec']['metric_names'], prep['performance']['count']) if c]
    if not any(r['performance'].get(n) is not None for r in rows['val'] for n in active):
        raise ValueError('No validation labels for metrics observed in selected train')
    plan = dict(experiment_kind='frozen_branch_diagnostics', version=1, complete=False,
                source_checkpoint=context['source_checkpoint'], source_checkpoint_sha256=context['source_sha256'],
                source_model_digest=context['source_model_digest'], source_mode=context['source'].get('mode', 'jepa_zerosim'),
                source_model_config=context['source']['model_config'],
                partition_fingerprint=context['source']['partition_fingerprint'],
                input_op_fingerprint=context['source']['input_op_fingerprint'],
                dataset_fingerprint=context['manifest']['dataset_fingerprint'],
                pretrain_seed=context['source']['options']['training']['seed'], seed=seed,
                sample_ids={s: [r['sample_id'] for r in rs] for s, rs in rows.items()},
                readout_policy='fixed_device_id_flatten_no_graph', device=str(device),
                stats_only=stats_only, **identity,
                settings=dict(batch_size=batch_size, encoder_batch_size=encoder_batch_size, epochs=epochs,
                              hidden=hidden, lr=lr, patience=patience, ridge_alpha=ridge_alpha, std_floor=std_floor))
    output.mkdir(parents=True, exist_ok=True)
    _json(output / 'run.json', plan)
    _json(output / 'preprocessing.json', prep)
    print(f'Diagnostics: {output} | {len(rows["train"])} train, {len(rows["val"])} val | {device}', flush=True)
    features, target, mask, physical, frozen = extract_features(context, rows['train'] + rows['val'], prep,
                                                              device, encoder_batch_size)
    n = len(rows['train'])
    variation, csv_rows = {}, []
    for branch in ('design', 'op'):
        variation[branch] = {}
        for split, memory in (('train', features[branch][:n]), ('val', features[branch][n:])):
            stats = embedding_variation(memory, identity['device_ids'])
            variation[branch][split] = stats
            csv_rows.extend(dict(branch=branch, split=split, **item) for item in stats['per_device'])
            print(f'{branch} {split}: within-device std RMS={stats["within_device_std_rms"]:.6g}, '
                  f'between-device std RMS={stats["between_device_std_rms"]:.6g}', flush=True)
    diagnostics = dict(source_checkpoint_sha256=context['source_sha256'], **frozen, variation=variation)
    _json(output / 'diagnostics.json', diagnostics)
    write_csv(output / 'device_variation.csv', csv_rows)
    reports = []
    if not stats_only:
        y, vy, m, vm = target[:n], target[n:], mask[:n], mask[n:]
        mean = (y * m).sum(0) / m.sum(0).clamp_min(1)

        def add(source, head, train_pred, val_pred, **metadata):
            report = dict(source=source, head=head, **metadata,
                          train=score(train_pred, y, m, prep, physical[:n]),
                          val=score(val_pred, vy, vm, prep, physical[n:]))
            reports.append(report)
            _json(output / 'readouts.json', reports)
            print(f'{source}/{head}: train={report["train"]["normalized_mse"]:.7g}, '
                  f'val={report["val"]["normalized_mse"]:.7g}', flush=True)

        add('train_mean', 'constant', mean.expand_as(y), mean.expand_as(vy))
        for branch in ('design', 'op', 'raw_op'):
            x, v, scaling = standardize_features(features[branch][:n], features[branch][n:], std_floor)
            print(f'{branch}: {x.shape[1]} flattened inputs; {scaling["columns_below_floor"]} below std floor', flush=True)
            started = time.perf_counter()
            a, b = ridge_predict(x, y, m, v, ridge_alpha)
            add(branch, 'ridge', a, b, scaling=scaling, seconds=time.perf_counter()-started)
            started = time.perf_counter()
            a, b, training = mlp_predict(x, y, m, v, vy, vm, seed=seed, epochs=epochs,
                                         batch_size=batch_size, hidden=hidden, lr=lr, patience=patience)
            add(branch, 'mlp', a, b, scaling=scaling, training=training, seconds=time.perf_counter()-started)
        table = [dict(source=r['source'], head=r['head'], split=split, metric=name, **metric)
                 for r in reports for split in ('train', 'val') for name, metric in r[split]['metrics'].items()]
        write_csv(output / 'readouts.csv', table)
    plan['complete'] = True
    _json(output / 'run.json', plan)
    files = 'diagnostics.json' if stats_only else 'diagnostics.json and readouts.json'
    print(f'Complete. Share {files} from {output}', flush=True)
    return diagnostics, reports
