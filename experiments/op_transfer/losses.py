"""Train-only latent calibration and device-conditional experimental objectives.

Keys identify a circuit, topology and physical device, never a sizing sample.
The scalar variance penalty is a diagnostic constraint, not evidence that a
representation contains useful physical information.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch


def _check_memory(memory, padding, keys):
    if memory.ndim != 3 or padding.shape != memory.shape[:2]:
        raise ValueError("Expected memory [batch, graph+devices, hidden] and matching padding")
    if padding.dtype != torch.bool or len(keys) != memory.shape[0]:
        raise ValueError("Padding must be boolean; provide device keys for every sample")
    if memory.shape[1] < 2 or memory.shape[2] < 1:
        raise ValueError("Memory must contain graph and device tokens")
    lengths = []
    for row_keys in keys:
        if len(row_keys) > memory.shape[1] - 1:
            raise ValueError("More keys than device slots")
        if any(not isinstance(key, str) or not key for key in row_keys):
            raise ValueError("Every device key must be a nonempty string")
        if len(set(row_keys)) != len(row_keys):
            raise ValueError("Device keys must be unique within each sample")
        lengths.append(len(row_keys))
    # Homogeneous topology batches have one key for every slot, so no device
    # check is needed. Mixed padded lengths require a single reduction rather
    # than synchronizing CUDA once for every sample in the minibatch.
    if any(length < memory.shape[1] - 1 for length in lengths):
        sizes = torch.tensor(lengths, device=padding.device)
        slots = torch.arange(memory.shape[1] - 1, device=padding.device)
        without_key = slots.unsqueeze(0) >= sizes.unsqueeze(1)
        if bool((without_key & ~padding[:, 1:]).any()):
            raise ValueError("Every non-padding device needs its circuit/topology/device key")
    if not bool(torch.isfinite(memory[~padding]).all()):
        raise ValueError("Nonfinite active latent feature")


def _groups(keys, valid):
    indices = defaultdict(list)
    for row, row_keys in enumerate(keys):
        for column, key in enumerate(row_keys):
            indices[key].append((row, column))
    for key, positions in indices.items():
        index = torch.tensor(positions, dtype=torch.long, device=valid.device)
        index = index[valid[index[:, 0], index[:, 1]]]
        yield key, index


class LatentScaler:
    """Streaming per-device train moments with explicit unseen-device fallback.

    Graph tokens use global graph moments. Known device slots use their own
    training moments. An unseen device/topology uses global *device* moments,
    never validation or OOD moments. This fallback supports inference but is
    not a claim that a fixed-topology calibration generalizes to unseen graphs.
    Accumulation uses float64 Chan/Welford moments to retain tiny residuals.
    """

    VERSION = 1

    def __init__(self, floor=1e-4):
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("Latent scale floor must be finite and positive")
        self.floor = float(floor)
        self.groups = {}
        self.graph = None
        self.global_device = None
        self.hidden_dim = None
        self.fitted_samples = 0
        self.fitted_tokens = 0
        self.finalized = False
        self.last_fallback_tokens = 0

    @staticmethod
    def _merge(old, values):
        if not values.numel():
            return old
        values = values.detach().to(torch.float64)
        count = values.shape[0]
        mean = values.mean(0)
        m2 = (values - mean).square().sum(0)
        if old is None:
            return {"count": count, "mean": mean, "m2": m2}
        old_mean, old_m2 = old["mean"].to(mean.device), old["m2"].to(mean.device)
        total = old["count"] + count
        delta = mean - old_mean
        return {"count": total, "mean": old_mean + delta * (count / total),
                "m2": old_m2 + m2 + delta.square() * (old["count"] * count / total)}

    def update(self, memory, padding, keys, *, partition="train"):
        if partition != "train":
            raise ValueError("Latent scaler can only fit the training partition")
        if self.finalized:
            raise RuntimeError("Latent scaler is frozen after finalize")
        _check_memory(memory, padding, keys)
        if self.hidden_dim is not None and self.hidden_dim != memory.shape[-1]:
            raise ValueError("Latent width changed during fitting")
        self.hidden_dim = memory.shape[-1]
        self.graph = self._merge(self.graph, memory[:, 0][~padding[:, 0]])
        active = ~padding[:, 1:]
        self.global_device = self._merge(self.global_device, memory[:, 1:][active])
        for key, index in _groups(keys, active):
            if index.shape[0]:
                values = memory[index[:, 0], index[:, 1] + 1]
                self.groups[key] = self._merge(self.groups.get(key), values)
        self.fitted_samples += memory.shape[0]
        self.fitted_tokens += int(active.sum())
        return self

    def finalize(self):
        if self.graph is None or self.global_device is None or not self.groups:
            raise ValueError("Cannot finalize a scaler without observed graph and device tokens")
        for stats in (self.graph, self.global_device, *self.groups.values()):
            stats["std"] = (stats["m2"] / stats["count"]).clamp_min(0).sqrt()
            stats["scale"] = stats["std"].clamp_min(self.floor)
        self.finalized = True
        return self

    def _ready(self):
        if not self.finalized:
            raise RuntimeError("Finalize latent scaler before use")

    def to(self, device):
        """Move frozen moments once; subsequent forwards do no host transfers.

        The scaler deliberately is not an nn.Module, so model.to() does not
        visit it. transform()/scale_for_key() call this lazily after checkpoint
        restore. Moving statistics never refits or changes their precision.
        """
        self._ready()
        device = torch.device(device)
        if self.graph["mean"].device == device:
            return self
        for stats in (self.graph, self.global_device, *self.groups.values()):
            for key, value in stats.items():
                if torch.is_tensor(value):
                    stats[key] = value.to(device=device)
        return self

    def scale_for_key(self, key, reference):
        self._ready()
        self.to(reference.device)
        stats = self.groups.get(key, self.global_device)
        return stats["scale"].to(device=reference.device, dtype=reference.dtype)

    def transform(self, memory, padding, keys):
        self._ready()
        self.to(memory.device)
        _check_memory(memory, padding, keys)
        if memory.shape[-1] != self.hidden_dim:
            raise ValueError("Latent width does not match fitted scaler")
        # Double arithmetic preserves sub-milliscale variation around large
        # identity means; the model receives its original dtype afterwards.
        result = torch.zeros_like(memory, dtype=torch.float64)
        graph_values = torch.where(padding[:, :1, None], 0., memory[:, :1]).double()
        result[:, 0] = ((graph_values[:, 0] - self.graph["mean"].to(memory.device))
                        / self.graph["scale"].to(memory.device))
        fallback = 0
        for key, index in _groups(keys, ~padding[:, 1:]):
            if not index.shape[0]:
                continue
            stats = self.groups.get(key, self.global_device)
            fallback += index.shape[0] if key not in self.groups else 0
            values = memory[index[:, 0], index[:, 1] + 1].double()
            result[index[:, 0], index[:, 1] + 1] = (
                (values - stats["mean"].to(memory.device)) / stats["scale"].to(memory.device))
        result = result.masked_fill(padding.unsqueeze(-1), 0.).to(memory.dtype)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Latent scaling overflow; inspect training feature scale")
        self.last_fallback_tokens = fallback
        return result

    def state_dict(self):
        self._ready()
        def cpu(stats):
            return {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in stats.items()}
        return {"version": self.VERSION, "floor": self.floor, "hidden_dim": self.hidden_dim,
                "fitted_samples": self.fitted_samples, "fitted_tokens": self.fitted_tokens,
                "graph": cpu(self.graph), "global_device": cpu(self.global_device),
                "groups": {k: cpu(v) for k, v in self.groups.items()},
                "fit_partition": "train", "unknown_device_policy": "global_train_device_moments"}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != cls.VERSION or state.get("fit_partition") != "train":
            raise ValueError("Unsupported or non-training latent scaler state")
        if state.get("unknown_device_policy") != "global_train_device_moments":
            raise ValueError("Unsupported unknown-device scaling policy")
        obj = cls(state["floor"])
        obj.hidden_dim = state["hidden_dim"]
        obj.fitted_samples = state["fitted_samples"]
        obj.fitted_tokens = state["fitted_tokens"]
        obj.graph, obj.global_device, obj.groups = state["graph"], state["global_device"], state["groups"]
        for stats in (obj.graph, obj.global_device, *obj.groups.values()):
            if stats["count"] < 1:
                raise ValueError("Invalid latent scaler observation count")
            for name in ("mean", "m2", "std", "scale"):
                value = stats[name]
                if value.shape != (obj.hidden_dim,) or not bool(torch.isfinite(value).all()):
                    raise ValueError("Invalid latent scaler moments")
            if bool((stats["scale"] < obj.floor).any()) or bool((stats["m2"] < 0).any()):
                raise ValueError("Invalid latent scaler scale or variance")
        obj.finalized = True
        return obj


def _check_alignment(predicted, target, valid, keys):
    if predicted.shape != target.shape or predicted.ndim != 3 or valid.shape != predicted.shape[:2]:
        raise ValueError("Expected matched device latents [batch, devices, hidden] and valid mask")
    padding = torch.cat((torch.zeros(valid.shape[0], 1, dtype=torch.bool, device=valid.device), ~valid), 1)
    graph = predicted.new_zeros(predicted.shape[0], 1, predicted.shape[-1])
    _check_memory(torch.cat((graph, predicted), 1), padding, keys)
    _check_memory(torch.cat((graph, target), 1), padding, keys)


def grouped_alignment_loss(predicted, target, valid, keys, *, invariance=25., variance=25.,
                           covariance=1., variance_floor=1.):
    """Direct alignment plus within-device variance/covariance on RAW latents.

    Regularization averages over eligible repeated device groups, not over
    different transistor identities. Singleton groups still contribute to
    alignment but cannot estimate design-dependent variance. Both branches
    retain gradients; callers decide whether their target branch is frozen.
    """
    _check_alignment(predicted, target, valid, keys)
    weights = (invariance, variance, covariance, variance_floor)
    if any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError("Loss weights and variance floor must be finite and nonnegative")
    if not bool(valid.any()):
        raise ValueError("No eligible paired device latents")
    mse = (predicted[valid] - target[valid]).square().mean()
    var_terms, cov_terms = [], []
    singleton_groups, repeated_tokens = 0, 0
    for _, index in _groups(keys, valid):
        if index.shape[0] < 2:
            singleton_groups += int(index.shape[0] == 1)
            continue
        repeated_tokens += index.shape[0]
        for branch in (predicted, target):
            z = branch[index[:, 0], index[:, 1]]
            centered = z - z.mean(0)
            var_terms.append(torch.relu(variance_floor - (centered.square().mean(0) + 1e-4).sqrt()).mean())
            matrix = centered.T @ centered / (z.shape[0] - 1)
            off = matrix - torch.diag_embed(matrix.diagonal())
            cov_terms.append(off.square().sum() / z.shape[-1])
    if not var_terms:
        raise ValueError("Within-device objective needs >=2 samples of at least one same device; use grouped batches")
    var, cov = torch.stack(var_terms).mean(), torch.stack(cov_terms).mean()
    total = invariance * mse + variance * var + covariance * cov
    return total, {"alignment_mse": mse.detach(), "variance": var.detach(), "covariance": cov.detach(),
                   "observed_devices": int(valid.sum()), "repeated_groups": len(var_terms) // 2,
                   "singleton_groups": singleton_groups, "regularized_tokens": repeated_tokens}


def residual_alignment_loss(predicted, target, padding, keys, scaler, *, valid=None):
    """Directly compare independently centered, train-scaled device residuals.

    Inputs include the graph token, which this objective excludes. Centering
    is within same-device batch groups on each branch separately. The scale
    is frozen train-only teacher scale, never fitted to validation/OOD data.
    Teacher targets are fully detached. This loss alone cannot exclude jointly
    constant representations; use the explicit grouped/OP-aux ablations to
    diagnose anchoring rather than silently combine unrelated objectives.
    """
    _check_memory(predicted, padding, keys)
    _check_memory(target, padding, keys)
    if predicted.shape != target.shape:
        raise ValueError("Residual alignment requires identical encoder widths and token counts")
    active = ~padding[:, 1:]
    if valid is not None:
        if valid.shape != active.shape or valid.dtype != torch.bool:
            raise ValueError("Invalid residual device eligibility mask")
        active = active & valid
    terms, count = [], 0
    for key, index in _groups(keys, active):
        if index.shape[0] < 2:
            continue
        a = predicted[index[:, 0], index[:, 1] + 1]
        b = target[index[:, 0], index[:, 1] + 1].detach()
        scale = scaler.scale_for_key(key, a)
        a = (a - a.mean(0)) / scale
        b = (b - b.mean(0)) / scale
        terms.append((a - b).square().mean())
        count += index.shape[0]
    if not terms:
        raise ValueError("Residual alignment needs repeated eligible device groups")
    loss = torch.stack(terms).mean()
    return loss, {"residual_mse": loss.detach(), "repeated_groups": len(terms), "regularized_tokens": count}


def masked_field_mse(prediction, target, mask):
    """Average observed MSE per OP field, then equally over supported fields.

    Missing values are selected out BEFORE arithmetic, so NaNs/infinite
    placeholders cannot poison otherwise finite supervised fields.
    """
    if prediction.shape != target.shape or mask.shape != target.shape or prediction.ndim < 2:
        raise ValueError("OP prediction, target and mask shapes must match")
    if mask.dtype != torch.bool:
        raise ValueError("OP supervision mask must be boolean")
    terms = []
    for field in range(prediction.shape[-1]):
        selected = mask[..., field]
        if bool(selected.any()):
            p, y = prediction[..., field][selected], target[..., field][selected]
            if not bool(torch.isfinite(p).all() & torch.isfinite(y).all()):
                raise ValueError("Nonfinite observed OP prediction or target")
            terms.append((p - y).square().mean())
    if not terms:
        raise ValueError("No observed supported OP fields in batch")
    return torch.stack(terms).mean()
