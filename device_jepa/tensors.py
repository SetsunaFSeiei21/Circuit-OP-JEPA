"""Mask-aware batches: design inputs, OP targets and labels stay separated."""
import torch

from .data import is_op_target, transform_value, inverse_transform_value


def collate_samples(samples, preprocessing, include_op=True, include_performance=False):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    spec = preprocessing["feature_spec"]
    pn, on, mn, tn = (spec[k] for k in ("parameter_names", "op_names", "metric_names", "terminal_names"))
    b, n = len(samples), max(len(s["devices"]) for s in samples)
    p, o, m, t = len(pn), len(on), len(mn), len(tn)
    batch = {
        "device_types": torch.zeros(b, n, dtype=torch.long),
        "device_mask": torch.zeros(b, n, dtype=torch.bool),
        "parameters": torch.zeros(b, n, p),
        "parameter_mask": torch.zeros(b, n, p, dtype=torch.bool),
        "adjacency": torch.zeros(b, n, n, dtype=torch.bool),
        "edge_roles": torch.zeros(b, n, n, t * t),
        "port_roles": torch.zeros(b, n, t),
        "sample_ids": [s["sample_id"] for s in samples],
        "device_ids": [],
    }
    if include_op:
        batch.update(op=torch.zeros(b, n, o), op_mask=torch.zeros(b, n, o, dtype=torch.bool),
                     op_target_mask=torch.zeros(b, n, dtype=torch.bool))
        states = preprocessing.get("op_state_features", [])
        if states:
            batch.update(op_state=torch.zeros(b, n, len(states)),
                         op_state_mask=torch.zeros(b, n, len(states), dtype=torch.bool))
    if include_performance:
        if "performance" not in preprocessing:
            raise ValueError("Performance statistics must be fitted on ID train before using labels")
        batch.update(performance=torch.zeros(b, m), performance_mask=torch.zeros(b, m, dtype=torch.bool))
    vocab = preprocessing["device_vocab"]
    terminal_index = {name: i for i, name in enumerate(tn)}
    for bi, sample in enumerate(samples):
        # Stable canonical ordering, independent of incoming JSON record order.
        devices = sorted(sample["devices"], key=lambda d: d["device_id"])
        batch["device_ids"].append([d["device_id"] for d in devices])
        batch["device_mask"][bi, :len(devices)] = True
        nets = {}
        external_nets = set(sample.get("ports", {}).values())
        for ni, device in enumerate(devices):
            batch["device_types"][bi, ni] = vocab.get(device["device_type"], vocab["<UNK>"])
            _fill(batch["parameters"][bi, ni], batch["parameter_mask"][bi, ni],
                  device.get("parameters", {}), pn, preprocessing["parameters"])
            if include_op:
                batch["op_target_mask"][bi, ni] = is_op_target(device)
                _fill(batch["op"][bi, ni], batch["op_mask"][bi, ni],
                      device.get("op", {}), on, preprocessing["op"])
                for si, name in enumerate(preprocessing.get("op_state_features", [])):
                    value = device.get("op_state", {}).get(name)
                    if value is not None:
                        if value not in (0, 1):
                            raise ValueError(f"OP state {name} must be categorical 0/1/null")
                        batch["op_state"][bi, ni, si] = value
                        batch["op_state_mask"][bi, ni, si] = True
            for terminal, net in device["terminals"].items():
                ti = terminal_index[terminal]
                nets.setdefault(net, []).append((ni, ti))
                if net in external_nets:
                    batch["port_roles"][bi, ni, ti] = 1
        for endpoints in nets.values():
            for ni, ti in endpoints:
                for nj, tj in endpoints:
                    batch["adjacency"][bi, ni, nj] = True
                    batch["edge_roles"][bi, ni, nj, ti * t + tj] = 1
        if include_performance:
            _fill(batch["performance"][bi], batch["performance_mask"][bi],
                  sample.get("performance", {}), mn, preprocessing["performance"])
    return batch


def _fill(values, mask, mapping, names, stats):
    for i, name in enumerate(names):
        value = mapping.get(name)
        if value is not None:
            normalized = transform_value(float(value), stats, i)
            if not torch.isfinite(torch.tensor(normalized, dtype=values.dtype)):
                raise ValueError(f"Non-finite standardized feature {name}; inspect units/transforms")
            values[i], mask[i] = normalized, True


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def denormalize(prediction, statistics):
    """Evaluation-only inverse in float64, using the stable scalar transform.

    Reporting must not cast physical scales/offsets into float32: that can erase
    small target variation, underflow tiny scales, or overflow sinh before its
    physical scale is applied. This deliberately detached CPU scalar path also
    handles finite scaled-sinh results whose unscaled sinh would overflow.
    """
    if prediction.ndim < 1 or prediction.shape[-1] != len(statistics["mean"]):
        raise ValueError("Prediction shape does not match target statistics")
    flat = prediction.detach().to(device="cpu", dtype=torch.float64).reshape(-1, prediction.shape[-1])
    values = [[inverse_transform_value(value, statistics, j) for j, value in enumerate(row)]
              for row in flat.tolist()]
    return torch.tensor(values, dtype=torch.float64, device=prediction.device).reshape(prediction.shape)


def evaluation_targets(samples, preprocessing):
    """Physical truths come from original data, never a float32 round trip."""
    names, stats = preprocessing["feature_spec"]["metric_names"], preprocessing["performance"]
    raw, normalized = [], []
    for sample in samples:
        values = [sample["performance"].get(name) for name in names]
        raw.append([float(value) if value is not None else 0. for value in values])
        normalized.append([transform_value(float(value), stats, j) if value is not None else 0.
                           for j, value in enumerate(values)])
    return torch.tensor(raw, dtype=torch.float64), torch.tensor(normalized, dtype=torch.float64)
