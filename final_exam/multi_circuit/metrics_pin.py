"""Circuit-equal validation selection; physical-unit pooled test evaluation."""
from pathlib import Path
import numpy as np
import torch
from device_jepa.training import _metrics
from experiments.op_transfer.engine import denormalize, masked_mse


def summarize_arrays_pin(pred, truth, mask, circuits, prep, normalized_pred=None, normalized_truth=None):
    pred, truth, mask = (torch.as_tensor(x) for x in (pred, truth, mask))
    if normalized_pred is not None:
        normalized_pred, normalized_truth = map(torch.as_tensor, (normalized_pred, normalized_truth))
    circuits = np.asarray(circuits)
    per = {}
    for circuit in sorted(set(circuits.tolist())):
        idx = torch.from_numpy(circuits == circuit)
        per[circuit] = _metrics(pred[idx], truth[idx], mask[idx], prep,
            normalized_pred[idx] if normalized_pred is not None else None,
            normalized_truth[idx] if normalized_truth is not None else None)
    pooled = _metrics(pred, truth, mask, prep, normalized_pred, normalized_truth)
    macro = {}
    for metric in pooled:
        macro[metric] = {'count': sum(v[metric]['count'] for v in per.values())}
        for stat in ('mae', 'rmse', 'r2', 'normalized_rmse'):
            values = [v[metric][stat] for v in per.values() if v[metric][stat] is not None]
            macro[metric][stat] = float(np.mean(values)) if values else None
            macro[metric][stat + '_circuits'] = len(values)
    circuit_scores = {}
    for circuit, metrics in per.items():
        eligible = [v['r2'] for v in metrics.values() if v['r2'] is not None]
        if eligible:
            circuit_scores[circuit] = float(np.mean(eligible))
    selection = float(np.mean(list(circuit_scores.values()))) if len(circuit_scores) == len(per) else None
    return dict(per_circuit=per, pooled=pooled, circuit_macro=macro,
                macro_r2=selection, circuit_r2=circuit_scores,
                selection_policy='mean over eligible metrics within circuit, then equal mean over every circuit',
                undefined_r2='exclude constant/less-than-two-observed targets; never replace by zero')


def evaluate_joint_pin(ctx, model, split, options, *, measured_op=False, destination=None):
    model.eval()
    predictions, targets, truths, masks, circuits, samples = [], [], [], [], [], []
    with torch.no_grad():
        for batch in ctx.batches(split, options['batch_size'], op=measured_op):
            pred = model(batch)
            if not bool(torch.isfinite(pred).all()):
                raise FloatingPointError('Non-finite joint prediction')
            predictions.append(pred.detach().double().cpu())
            targets.append(batch['performance'].detach().double().cpu())
            truths.append(batch['physical_performance'].detach().double().cpu())
            masks.append(batch['performance_mask'].detach().cpu())
            circuits.extend(batch['_circuit_ids'])
            samples.extend(batch['_sample_ids'])
    pred, target, raw, mask = map(torch.cat, (predictions, targets, truths, masks))
    physical = denormalize(pred, ctx.prep['performance'])
    report = summarize_arrays_pin(physical, raw, mask, circuits, ctx.prep, pred, target)
    report.update(split=split, count=len(samples), requires_measured_op=measured_op,
                  metrics=report['pooled'], normalized_mse=float(masked_mse(pred, target, mask)))
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f'{split}_predictions_pin.npz'
        temp = path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, prediction=physical.numpy(), truth=raw.numpy(), mask=mask.numpy(),
            normalized_prediction=pred.numpy(), normalized_truth=target.numpy(),
            circuit_id=np.asarray(circuits, dtype=str), sample_id=np.asarray(samples, dtype=str),
            metric_names=np.asarray(ctx.prep['feature_spec']['metric_names'], dtype=str))
        temp.replace(path)
        report['predictions_file'] = path.name
    return report


def negative_macro_r2_pin(ctx, model, options, *, measured_op=False):
    value = evaluate_joint_pin(ctx, model, 'val', options, measured_op=measured_op)['macro_r2']
    if value is None:
        raise ValueError('Every validation circuit needs at least one nonconstant metric with >=2 labels')
    return -value
