"""Mask-aware batches: design inputs, OP targets and labels stay separated."""
import torch

from .data import (device_category, is_op_target, parameter_applicability,
                   transform_value, inverse_transform_value, canonical_model_name,
                   canonical_port_name)


def _identity_vocabularies(preprocessing):
    """Contract 3 adds explicit physical-model and named-port identities.

    Legacy tensorization stays unchanged so old checkpoints retain their actual
    input semantics. New checkpoints must never silently fall back to that path.
    """
    contract = preprocessing.get("input_contract_version", 1)
    if type(contract) is not int or contract not in (1, 2, 3):
        raise ValueError("Unsupported tensor input contract version")
    if contract < 3:
        if "model_vocab" in preprocessing or "port_role_vocab" in preprocessing:
            raise ValueError("Legacy input contracts cannot declare contract 3 identity vocabularies")
        return None, None
    models, ports = preprocessing.get("model_vocab"), preprocessing.get("port_role_vocab")
    for name, vocab, reserved in (
            ("model_vocab", models, {"<PAD>": 0, "<UNK>": 1, "<NONE>": 2}),
            ("port_role_vocab", ports, {"<PAD>": 0, "<UNK>": 1})):
        if (not isinstance(vocab, dict) or any(type(v) is not int for v in vocab.values())
                or any(vocab.get(k) != v for k, v in reserved.items())
                or sorted(vocab.values()) != list(range(len(vocab)))):
            raise ValueError(f"Input contract 3 requires a valid {name}")
    return models, ports


def _model_identity(device, model_vocab):
    name = canonical_model_name(device) if device is not None else None
    return model_vocab["<NONE>"] if name is None else model_vocab.get(name, model_vocab["<UNK>"])


def _named_port_nets(sample, port_vocab):
    nets = {}
    for name, net in sample.get("ports", {}).items():
        index = port_vocab.get(canonical_port_name(name), port_vocab["<UNK>"])
        nets.setdefault(net, set()).add(index)
    return nets


def _performance_metric_mask(selection, num_metrics, include_performance):
    """Validate an explicit label selection before reading or converting labels.

    Training may select only metrics supported by its fitted targets. The
    default remains all metrics so reporting does not silently hide labels.
    """
    if selection is None:
        return None
    if not include_performance:
        raise ValueError("performance_metric_mask requires include_performance=True")
    if isinstance(selection, torch.Tensor):
        if selection.dtype != torch.bool or selection.ndim != 1:
            raise ValueError("performance_metric_mask must be a one-dimensional boolean mask")
        selection = selection.detach().cpu().tolist()
    if (not isinstance(selection, (list, tuple)) or len(selection) != num_metrics
            or any(type(value) is not bool for value in selection)):
        raise ValueError("performance_metric_mask must contain one boolean per metric")
    return tuple(selection)


def collate_samples(samples, preprocessing, include_op=True, include_performance=False,
                    include_op_states=True, *, performance_metric_mask=None, include_parameters=True):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    spec = preprocessing["feature_spec"]
    pn, on, mn, tn = (spec[k] for k in ("parameter_names", "op_names", "metric_names", "terminal_names"))
    b, n = len(samples), max(len(s["devices"]) for s in samples)
    p, o, m, t = len(pn), len(on), len(mn), len(tn)
    selected_metrics = _performance_metric_mask(performance_metric_mask, m, include_performance)
    batch = {
        "device_types": torch.zeros(b, n, dtype=torch.long),
        "device_categories": torch.zeros(b, n, dtype=torch.long),
        "device_mask": torch.zeros(b, n, dtype=torch.bool),
        "adjacency": torch.zeros(b, n, n, dtype=torch.bool),
        "edge_roles": torch.zeros(b, n, n, t * t),
        "port_roles": torch.zeros(b, n, t),
        "sample_ids": [s["sample_id"] for s in samples],
        "device_ids": [],
    }
    if include_parameters:
        batch.update(parameters=torch.zeros(b, n, p),
                     parameter_mask=torch.zeros(b, n, p, dtype=torch.bool),
                     parameter_applicability=torch.zeros(b, n, p, dtype=torch.bool))
    model_vocab, port_vocab = _identity_vocabularies(preprocessing)
    if model_vocab is not None:
        batch.update(device_models=torch.zeros(b, n, dtype=torch.long),
                     named_port_roles=torch.zeros(b, n, t * len(port_vocab)))
    if include_op:
        batch.update(op=torch.zeros(b, n, o), op_mask=torch.zeros(b, n, o, dtype=torch.bool),
                     op_target_mask=torch.zeros(b, n, dtype=torch.bool))
        states = preprocessing.get("op_state_features", []) if include_op_states else []
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
        named_ports = _named_port_nets(sample, port_vocab) if port_vocab is not None else {}
        for ni, device in enumerate(devices):
            batch["device_types"][bi, ni] = vocab.get(device["device_type"], vocab["<UNK>"])
            batch["device_categories"][bi, ni] = preprocessing.get("device_category_vocab", {}).get(device_category(device), 0)
            if model_vocab is not None:
                batch["device_models"][bi, ni] = _model_identity(device, model_vocab)
            if include_parameters:
                batch["parameter_applicability"][bi, ni] = torch.tensor(parameter_applicability(device, pn))
                _fill(batch["parameters"][bi, ni], batch["parameter_mask"][bi, ni],
                      device.get("parameters", {}), pn, preprocessing["parameters"])
            if include_op:
                batch["op_target_mask"][bi, ni] = is_op_target(device)
                if is_op_target(device):
                    _fill(batch["op"][bi, ni], batch["op_mask"][bi, ni],
                          device.get("op", {}), on, preprocessing["op"])
                for si, name in enumerate(states if is_op_target(device) else []):
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
                if port_vocab is not None:
                    for role in named_ports.get(net, ()):
                        batch["named_port_roles"][bi, ni, ti * len(port_vocab) + role] = 1
        for endpoints in nets.values():
            for ni, ti in endpoints:
                for nj, tj in endpoints:
                    batch["adjacency"][bi, ni, nj] = True
                    batch["edge_roles"][bi, ni, nj, ti * t + tj] = 1
        if include_performance:
            _fill(batch["performance"][bi], batch["performance_mask"][bi],
                  sample.get("performance", {}), mn, preprocessing["performance"],
                  feature_mask=selected_metrics)
    return batch


def _fill(values, mask, mapping, names, stats, *, feature_mask=None):
    for i, name in enumerate(names):
        # Inactive labels must not undergo normalization or float32 conversion:
        # even a finite physical value can overflow before the loss masks it.
        if feature_mask is not None and not feature_mask[i]:
            continue
        value = mapping.get(name)
        if value is not None:
            normalized = transform_value(float(value), stats, i)
            if not torch.isfinite(torch.tensor(normalized, dtype=values.dtype)):
                raise ValueError(f"Non-finite standardized feature {name}; inspect units/transforms")
            values[i], mask[i] = normalized, True


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _device_denormalize(prediction, statistics):
    # Keep physical inversion on the GPU, including extreme finite scales.
    values = prediction.detach().to(dtype=torch.float64)
    kwargs = dict(dtype=torch.float64, device=prediction.device)
    mean, std = (torch.tensor(statistics[k], **kwargs) for k in ('mean', 'std'))
    transformed = values * std + mean
    magnitude = torch.maximum(std.abs(), mean.abs())
    alternate = (values * (std / magnitude) + mean / magnitude) * magnitude
    transformed = torch.where(torch.isfinite(transformed), transformed, alternate)
    scale = torch.tensor(statistics.get('asinh_scale', [1.]*len(statistics['mean'])), **kwargs)
    kinds = torch.tensor([kind == 'asinh' for kind in statistics.get('kind', ['linear']*len(statistics['mean']))],
                         device=prediction.device)
    large = torch.copysign((transformed.abs() + scale.log() - 0.6931471805599453).exp(), transformed)
    sinh = torch.sinh(transformed.clamp(-709.999, 709.999)) * scale
    physical = torch.where(kinds, torch.where(transformed.abs() < 710., sinh, large), transformed)
    if not bool(torch.isfinite(physical).all()):
        raise ValueError('inverse transform exceeds finite physical range')
    return physical


def denormalize(prediction, statistics):
    """Evaluation-only inverse in float64, using the stable scalar transform.

    Reporting must not cast physical scales/offsets into float32: that can erase
    small target variation, underflow tiny scales, or overflow sinh before its
    physical scale is applied. This deliberately detached CPU scalar path also
    handles finite scaled-sinh results whose unscaled sinh would overflow.
    """
    if prediction.ndim < 1 or prediction.shape[-1] != len(statistics["mean"]):
        raise ValueError("Prediction shape does not match target statistics")
    if prediction.device.type == 'cuda':
        return _device_denormalize(prediction, statistics)
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
