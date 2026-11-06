"""Independent, masked XGBoost regressors for one fixed device topology.

The tree feature vector consists of the normalized parameters, observation masks
and applicability masks in the canonical device order. The input topology is
fixed by the single-circuit runner. Measured OP and performance labels never
enter feature construction. Native XGBoost model files are independently usable.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEFAULTS = dict(num_boost_round=500, eta=0.05, max_depth=6,
                min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0,
                reg_lambda=1.0, reg_alpha=0.0, max_bin=256)


def tree_settings(config):
    raw = dict(config.get("final_exam", {}).get("xgboost", {}))
    if "learning_rate" in raw:
        if "eta" in raw:
            raise ValueError("Use only one of xgboost eta and learning_rate")
        raw["eta"] = raw.pop("learning_rate")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown XGBoost settings: {sorted(unknown)}")
    options = {**DEFAULTS, **raw}
    for key in ("num_boost_round", "max_depth", "max_bin"):
        if type(options[key]) is not int or options[key] < (2 if key == "max_bin" else 1):
            raise ValueError(f"XGBoost {key} must be a positive integer")
    for key in ("eta", "min_child_weight", "subsample", "colsample_bytree",
                "reg_lambda", "reg_alpha"):
        value = options[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"XGBoost {key} must be finite and nonnegative")
    for key in ("eta", "subsample", "colsample_bytree"):
        if not 0 < options[key] <= 1:
            raise ValueError(f"XGBoost {key} must be in (0, 1]")
    return options


def validate_tree_config(config):
    """Validate configuration without importing the optional tree dependency."""
    return tree_settings(config)


def device_features(batch, num_devices=None):
    """Return one vector per circuit from ordered device design features only."""
    parameters = batch["parameters"]
    mask = batch["parameter_mask"].bool()
    applicable = batch["parameter_applicability"].bool()
    devices = batch["device_mask"].bool()
    if parameters.ndim != 3 or mask.shape != parameters.shape or applicable.shape != parameters.shape:
        raise ValueError("XGBoost parameter tensors must share shape [batch, devices, fields]")
    if devices.shape != parameters.shape[:2] or not bool(devices.all()):
        raise ValueError("XGBoost requires a fixed single-circuit device count without padding")
    if num_devices is not None and parameters.shape[1] != num_devices:
        raise ValueError("XGBoost inference device count differs from training")
    observed = mask & applicable
    values = torch.where(observed, parameters, 0.0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("XGBoost observed parameters must be finite")
    features = torch.cat((values, observed.to(values.dtype), applicable.to(values.dtype)), dim=-1)
    return features.flatten(1).contiguous()


def _host(tensor):
    return tensor.detach().float().cpu().numpy()


def _backend(booster, device):
    """XGBoost warns and falls back to CPU on some builds; reject that case."""
    actual = json.loads(booster.save_config())["learner"]["generic_param"].get("device", "cpu")
    if device.type == "cuda" and not actual.startswith("cuda"):
        raise RuntimeError(f"Requested CUDA XGBoost but its active backend is {actual}")
    return actual


class TreeRegressor(nn.Module):
    """Torch-shaped prediction interface around one native booster per metric."""
    def __init__(self, boosters, metric_names, num_devices, parameter_names, device="cpu"):
        super().__init__()
        self.boosters = list(boosters)
        self.metric_names = list(metric_names)
        self.num_devices = int(num_devices)
        self.parameter_names = list(parameter_names)
        self.execution_device = torch.device(device)
        if len(self.boosters) != len(self.metric_names):
            raise ValueError("One XGBoost model slot is required per metric")

    def forward(self, batch):
        import xgboost as xgb

        features = device_features(batch, self.num_devices)
        if features.shape[1] != self.num_devices * len(self.parameter_names) * 3:
            raise ValueError("XGBoost parameter schema differs from its checkpoint")
        # Host DMatrix construction is explicit, while the saved/requested CUDA
        # backend performs boosting and prediction. No measured values are read.
        matrix = xgb.DMatrix(_host(features))
        predictions = np.zeros((features.shape[0], len(self.metric_names)), dtype=np.float32)
        for index, booster in enumerate(self.boosters):
            if booster is not None:
                predictions[:, index] = booster.predict(matrix, validate_features=True)
                _backend(booster, self.execution_device)
        return torch.from_numpy(predictions).to(features.device)


def _collect(ctx, split, batch_size):
    features, targets, masks = [], [], []
    num_devices = None
    for batch in ctx.batches(split, batch_size, op=False):
        current = batch["device_mask"].shape[1]
        if num_devices is None:
            num_devices = current
        features.append(_host(device_features(batch, num_devices)))
        targets.append(_host(batch["performance"]))
        masks.append(batch["performance_mask"].detach().cpu().numpy())
    if not features:
        raise ValueError(f"XGBoost requires a nonempty {split} split")
    return np.concatenate(features), np.concatenate(targets), np.concatenate(masks), num_devices


def fit_xgboost(ctx, config, options, seed, directory):
    """Fit each target on its observed train rows and select using validation."""
    import xgboost as xgb

    tree_options = tree_settings(config)
    requested_device = torch.device(ctx.device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available() or not xgb.build_info().get("USE_CUDA", False):
            raise RuntimeError("CUDA XGBoost requires a CUDA-enabled build and visible GPU")
    directory = Path(directory)
    destination = directory / "checkpoint.json"
    if destination.exists() or list(directory.glob("metric_*.ubj")):
        raise ValueError("Refusing to overwrite an existing XGBoost checkpoint")
    start = time.perf_counter()
    train_x, train_y, train_mask, num_devices = _collect(ctx, "train", options["batch_size"])
    val_x, val_y, val_mask, val_devices = _collect(ctx, "val", options["batch_size"])
    if val_devices != num_devices:
        raise ValueError("XGBoost train and validation device counts differ")
    metric_names = ctx.prep["feature_spec"]["metric_names"]
    parameters = {key: value for key, value in tree_options.items() if key != "num_boost_round"}
    parameters.update(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                      device=str(requested_device), nthread=options.get("torch_threads", 1))
    directory.mkdir(parents=True, exist_ok=True)
    boosters, history, selections = [], {}, []
    rounds_run = {}
    for index, name in enumerate(metric_names):
        observed = train_mask[:, index]
        if not bool(observed.any()):
            boosters.append(None)
            selections.append(dict(metric=name, trained=False, reason="no_training_labels"))
            continue
        validation = val_mask[:, index]
        dtrain = xgb.DMatrix(train_x[observed], label=train_y[observed, index])
        evals = [(dtrain, "train")]
        if bool(validation.any()):
            dval = xgb.DMatrix(val_x[validation], label=val_y[validation, index])
            evals.append((dval, "val"))
        patience = options.get("patience", 10) if bool(validation.any()) else 0
        progress_every = options.get("progress_every", 100)

        class CheckDevice(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if epoch == 0:
                    _backend(model, requested_device)
                if epoch == 0 or (epoch + 1) % progress_every == 0:
                    current = evals_log.get("val", evals_log["train"])["rmse"][-1]
                    print(f"xgboost {name}: round {epoch+1}/{tree_options['num_boost_round']} "
                          f"{'val' if validation.any() else 'train'}_rmse={current:.6g}", flush=True)
                return False

        evaluation = {}
        booster = xgb.train({**parameters, "seed": seed + index}, dtrain,
            num_boost_round=tree_options["num_boost_round"], evals=evals,
            early_stopping_rounds=patience or None, evals_result=evaluation,
            verbose_eval=False, callbacks=[CheckDevice()])
        actual_rounds = booster.num_boosted_rounds()
        # Disabling early stopping changes runtime, not validation selection.
        # Truncate to the exact validation best even after a fixed round budget.
        best_rounds = (int(np.argmin(evaluation["val"]["rmse"])) + 1
                       if bool(validation.any()) else actual_rounds)
        if best_rounds < actual_rounds:
            booster = booster[:best_rounds]
        booster.set_param({"device": str(requested_device)})
        backend = _backend(booster, requested_device)
        booster.save_model(directory / f"metric_{index}.ubj")
        boosters.append(booster)
        history[name] = evaluation
        rounds_run[name] = actual_rounds
        selections.append(dict(metric=name, trained=True, file=f"metric_{index}.ubj",
            selected_rounds=best_rounds, actual_rounds=actual_rounds, backend=backend,
            train_count=int(observed.sum()), val_count=int(validation.sum()),
            selection="val_rmse" if bool(validation.any()) else "fixed_round_budget",
            early_stopping_enabled=bool(patience)))
        print(f"xgboost {name}: selected {best_rounds}/{actual_rounds} rounds", flush=True)
        (directory / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    if not any(booster is not None for booster in boosters):
        raise ValueError("XGBoost has no observed training metric")
    model = TreeRegressor(boosters, metric_names, num_devices,
                          ctx.prep["feature_spec"]["parameter_names"], requested_device).eval()
    metadata = dict(format="final_exam_xgboost_v1", metric_names=metric_names,
        num_devices=num_devices, parameter_names=model.parameter_names,
        metrics=selections, settings=tree_options, seed=seed,
        device=str(requested_device), xgboost_version=xgb.__version__,
        input_contract="fixed device order: parameters + observed mask + applicability mask",
        preprocessing=ctx.prep, manifest=ctx.full_manifest, config=config,
        host_staging="NumPy DMatrix ingestion; requested backend performs training/prediction",
        inference_requires_op=False)
    destination.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    cost = dict(stage_id="xgboost", optimizer_steps=0, examples_seen=int(train_mask.sum()),
        epochs=None, planned_epochs=None, budget_unit="boosting_rounds_per_metric",
        boosting_rounds=rounds_run, planned_boosting_rounds=tree_options["num_boost_round"],
        seconds=time.perf_counter()-start, parameters=None, trainable_parameters=None,
        device=str(requested_device), selection=selections)
    (directory / "cost.json").write_text(json.dumps(cost, indent=2, allow_nan=False) + "\n")
    return model, cost


def load_xgboost(path, device="cpu"):
    """Restore native XGBoost files and return the model and saved metadata."""
    import xgboost as xgb

    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.json"
    metadata = json.loads(path.read_text())
    if metadata.get("format") != "final_exam_xgboost_v1":
        raise ValueError("Unsupported XGBoost checkpoint format")
    target_device = torch.device(device)
    if target_device.type == "cuda" and (not torch.cuda.is_available() or not xgb.build_info().get("USE_CUDA", False)):
        raise RuntimeError("Requested CUDA XGBoost checkpoint load without CUDA support")
    boosters = []
    if len(metadata["metrics"]) != len(metadata["metric_names"]):
        raise ValueError("XGBoost checkpoint metric count is inconsistent")
    for index, item in enumerate(metadata["metrics"]):
        if item["metric"] != metadata["metric_names"][index]:
            raise ValueError("XGBoost checkpoint metric order is inconsistent")
        if not item["trained"]:
            boosters.append(None)
            continue
        expected_file = f"metric_{index}.ubj"
        if item["file"] != expected_file:
            raise ValueError("XGBoost checkpoint contains an invalid model filename")
        booster = xgb.Booster()
        booster.load_model(path.parent / expected_file)
        booster.set_param({"device": str(target_device)})
        _backend(booster, target_device)
        boosters.append(booster)
    model = TreeRegressor(boosters, metadata["metric_names"], metadata["num_devices"],
                          metadata["parameter_names"], target_device).eval()
    return model, metadata
