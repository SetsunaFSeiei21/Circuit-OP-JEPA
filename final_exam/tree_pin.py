"""Masked XGBoost regressors on one fixed pin-level circuit topology.

Actual observed device terminals and separate named external ports form the
feature rows. Design values repeat per owner pin; pin identities and topology
are explicit. No measured OP/label values or their masks enter the feature
vector. Native per-metric boosters and a strict pin schema are independently
reloadable. This is a tabular pin adaptation, not a message-passing tree model.
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


def tree_settings_pin(config):
    raw = dict(config.get("final_exam_pin", {}).get("xgboost", {}))
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


def validate_tree_config_pin(config):
    """Validate configuration without importing the optional tree dependency."""
    return tree_settings_pin(config)


# Only static, input-side quantities may determine the saved pin topology.
# Owner OP/labels and their missingness are deliberately absent.
_TOPOLOGY_PIN = (
    "device_types", "device_categories", "device_models", "device_mask",
    "adjacency", "edge_roles", "port_roles", "named_port_roles",
    "owner_index", "pin_to_device", "owner_mask", "owner_device_types", "owner_device_models",
    "pin_terminal_index", "pin_net_index",
)


def _require_pin_preprocessing(preprocessing):
    if preprocessing.get("granularity") != "pin" or "pin_vocab" not in preprocessing:
        raise ValueError("XGBoost_pin requires train-fitted pin preprocessing")
    if preprocessing.get("device_vocab") != preprocessing["pin_vocab"]:
        raise ValueError("Pin token vocabulary differs from saved type schema")


def feature_schema_pin(preprocessing):
    """Column meanings in the exact concatenation order used by PinFeatures."""
    _require_pin_preprocessing(preprocessing)
    spec = preprocessing['feature_spec']
    names = [f'{kind}:{name}' for kind in ('parameter','observed','applicable')
             for name in spec['parameter_names']]
    for label, key in (('pin_type','device_vocab'),('category','device_category_vocab'),
                       ('physical_model','model_vocab')):
        vocabulary = preprocessing.get(key, {})
        names.extend(f'{label}:{name}' for name,index in sorted(vocabulary.items(),key=lambda item:item[1]))
    terminals = spec['terminal_names']
    names.extend(f'port_role:{name}' for name in terminals)
    ports = [name for name,index in sorted(preprocessing.get('port_role_vocab',{}).items(),
                                         key=lambda item:item[1])]
    names.extend(f'named_port:{terminal}:{port}' for terminal in terminals for port in ports)
    names.append('normalized_adjacency_degree')
    names.extend(f'normalized_edge_incidence:{source}:{target}'
                 for source in terminals for target in terminals)
    return dict(version=1, ordering='canonical pin-major, then per_pin_columns', per_pin_columns=names)


def pin_features_pin(batch, preprocessing, num_pins=None):
    """Flatten actual pin/port feature rows, not the original device rows.

    Owner design values and masks repeat on physical pins. Independent named
    external ports have no design values. Categorical pin/terminal identities,
    physical model and named-port incidence use the same one-hot feature kernel
    as neural pin baselines. Degree and directional terminal-pair incidence
    counts append topology descriptors per pin. Fixed-order topology is checked
    separately before training/prediction.
    """
    from final_exam.models.common_pin import PinFeatures

    _require_pin_preprocessing(preprocessing)
    parameters = batch["parameters"]
    observed = batch["parameter_mask"].bool()
    applicable = batch["parameter_applicability"].bool()
    pins = batch["device_mask"].bool()
    if parameters.ndim != 3 or observed.shape != parameters.shape or applicable.shape != parameters.shape:
        raise ValueError("XGBoost_pin tensors must share shape [batch, pins, fields]")
    if pins.shape != parameters.shape[:2] or not bool(pins.all()):
        raise ValueError("XGBoost_pin requires a fixed pin count without padding")
    if num_pins is not None and parameters.shape[1] != num_pins:
        raise ValueError("XGBoost_pin inference pin count differs from training")
    safe = dict(batch)
    safe["parameter_mask"] = observed & applicable
    tokens = PinFeatures(preprocessing)(safe)
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("XGBoost_pin observed features must be finite")
    count = pins.shape[1]
    degree = batch["adjacency"].to(tokens.dtype).sum(-1, keepdim=True) / max(1, count)
    role_incidence = batch["edge_roles"].to(tokens.dtype).sum(2) / max(1, count)
    values = torch.cat((tokens, degree, role_incidence), dim=-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("XGBoost_pin topology features must be finite")
    return values.flatten(1).contiguous()


def _topology_schema_pin(batch):
    schema = {}
    for key in _TOPOLOGY_PIN:
        if key not in batch:
            raise ValueError(f"XGBoost_pin needs canonical topology field {key}")
        value = batch[key]
        if not torch.equal(value, value[:1].expand_as(value)):
            raise ValueError("XGBoost_pin requires one fixed pin topology/order")
        schema[key] = dict(dtype=str(value.dtype).removeprefix("torch."),
                           shape=list(value.shape[1:]), values=value[0].detach().cpu().tolist())
    keys = batch.get("_group_keys")
    if not keys or any(list(row) != list(keys[0]) for row in keys):
        raise ValueError("XGBoost_pin requires canonical pin identity group keys")
    return dict(tensors=schema, token_group_keys=list(keys[0]))


def _check_topology_pin(batch, schema, buffers=None):
    keys = batch.get("_group_keys")
    if not keys or any(list(row) != schema["token_group_keys"] for row in keys):
        raise ValueError("XGBoost_pin input pin identities/topology differ from its checkpoint")
    for key, expected in schema["tensors"].items():
        value = batch.get(key)
        if value is None or list(value.shape[1:]) != expected["shape"]:
            raise ValueError(f"XGBoost_pin topology shape differs for {key}")
        ref = (buffers[key] if buffers is not None else
               torch.tensor(expected["values"], dtype=getattr(torch, expected["dtype"])))
        ref = ref.to(value.device)
        if value.dtype != ref.dtype or not bool((value == ref).all()):
            raise ValueError(f"XGBoost_pin fixed topology/order differs for {key}")


def _host(tensor):
    return tensor.detach().float().cpu().numpy()


def _backend(booster, device):
    """XGBoost warns and falls back to CPU on some builds; reject that case."""
    actual = json.loads(booster.save_config())["learner"]["generic_param"].get("device", "cpu")
    if device.type == "cuda" and not actual.startswith("cuda"):
        raise RuntimeError(f"Requested CUDA XGBoost but its active backend is {actual}")
    return actual


class TreeRegressorPin(nn.Module):
    """Native per-metric boosters with a strict fixed pin feature contract."""
    def __init__(self, boosters, metric_names, preprocessing, num_pins, topology, device="cpu"):
        super().__init__()
        _require_pin_preprocessing(preprocessing)
        self.boosters = list(boosters)
        self.metric_names = list(metric_names)
        self.preprocessing = preprocessing
        self.num_pins = int(num_pins)
        self.parameter_names = list(preprocessing["feature_spec"]["parameter_names"])
        self.execution_device = torch.device(device)
        self.topology = topology
        for key, item in topology["tensors"].items():
            self.register_buffer("_topology_" + key,
                torch.tensor(item["values"], dtype=getattr(torch, item["dtype"])), persistent=False)
        self.to(self.execution_device)
        if len(self.boosters) != len(self.metric_names):
            raise ValueError("One XGBoost model slot is required per metric")

    def forward(self, batch):
        import xgboost as xgb

        features = pin_features_pin(batch, self.preprocessing, self.num_pins)
        _check_topology_pin(batch, self.topology,
                           {key: getattr(self,"_topology_"+key) for key in self.topology["tensors"]})
        # Host DMatrix staging is explicit; requested CUDA backend still performs
        # boosting/prediction. No measured OP or performance values enter X.
        matrix = xgb.DMatrix(_host(features))
        predictions = np.zeros((features.shape[0], len(self.metric_names)), dtype=np.float32)
        for index, booster in enumerate(self.boosters):
            if booster is not None:
                predictions[:, index] = booster.predict(matrix, validate_features=True)
                _backend(booster, self.execution_device)
        return torch.from_numpy(predictions).to(features.device)


def _collect_pin(ctx, split, batch_size, topology=None):
    features, targets, masks = [], [], []
    num_pins = None
    for batch in ctx.batches(split, batch_size, op=False):
        current = batch["device_mask"].shape[1]
        if num_pins is None:
            num_pins = current
        if topology is None:
            topology = _topology_schema_pin(batch)
        _check_topology_pin(batch, topology)
        features.append(_host(pin_features_pin(batch, ctx.prep, num_pins)))
        targets.append(_host(batch["performance"]))
        masks.append(batch["performance_mask"].detach().cpu().numpy())
    if not features:
        raise ValueError(f"XGBoost_pin requires a nonempty {split} split")
    return np.concatenate(features), np.concatenate(targets), np.concatenate(masks), num_pins, topology


def fit_xgboost_pin(ctx, config, options, seed, directory):
    """Fit each target on its observed train rows and select using validation."""
    import xgboost as xgb

    tree_options = tree_settings_pin(config)
    requested_device = torch.device(ctx.device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available() or not xgb.build_info().get("USE_CUDA", False):
            raise RuntimeError("CUDA XGBoost requires a CUDA-enabled build and visible GPU")
    directory = Path(directory)
    destination = directory / "checkpoint_pin.json"
    if destination.exists() or list(directory.glob("metric_*_pin.ubj")):
        raise ValueError("Refusing to overwrite an existing XGBoost checkpoint")
    start = time.perf_counter()
    train_x, train_y, train_mask, num_pins, topology = _collect_pin(ctx, "train", options["batch_size"])
    val_x, val_y, val_mask, val_pins, _ = _collect_pin(ctx, "val", options["batch_size"], topology)
    if val_pins != num_pins:
        raise ValueError("XGBoost_pin train and validation pin counts differ")
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
                    print(f"xgboost_pin {name}: round {epoch+1}/{tree_options['num_boost_round']} "
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
        booster.save_model(directory / f"metric_{index}_pin.ubj")
        boosters.append(booster)
        history[name] = evaluation
        rounds_run[name] = actual_rounds
        selections.append(dict(metric=name, trained=True, file=f"metric_{index}_pin.ubj",
            selected_rounds=best_rounds, actual_rounds=actual_rounds, backend=backend,
            train_count=int(observed.sum()), val_count=int(validation.sum()),
            selection="val_rmse" if bool(validation.any()) else "fixed_round_budget",
            early_stopping_enabled=bool(patience)))
        print(f"xgboost_pin {name}: selected {best_rounds}/{actual_rounds} rounds", flush=True)
        (directory / "history_pin.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    if not any(booster is not None for booster in boosters):
        raise ValueError("XGBoost has no observed training metric")
    model = TreeRegressorPin(boosters, metric_names, ctx.prep, num_pins, topology, requested_device).eval()
    metadata = dict(format="final_exam_xgboost_pin_v1", metric_names=metric_names,
        granularity="pin", mode="xgboost_pin", num_pins=num_pins,
        parameter_names=model.parameter_names, topology=topology, feature_count=train_x.shape[1],
        feature_schema=feature_schema_pin(ctx.prep),
        metrics=selections, settings=tree_options, seed=seed,
        device=str(requested_device), xgboost_version=xgb.__version__,
        input_contract="fixed pin/port order: design values/masks, pin/type/model/terminal/port identity, degree and terminal-pair incidence",
        preprocessing=ctx.prep, manifest=ctx.full_manifest, config=config,
        host_staging="NumPy DMatrix ingestion; requested backend performs training/prediction",
        inference_requires_op=False)
    destination.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    cost = dict(stage_id="xgboost_pin", optimizer_steps=0, examples_seen=int(train_mask.sum()),
        epochs=None, planned_epochs=None, budget_unit="boosting_rounds_per_metric",
        boosting_rounds=rounds_run, planned_boosting_rounds=tree_options["num_boost_round"],
        seconds=time.perf_counter()-start, parameters=None, trainable_parameters=None,
        device=str(requested_device), selection=selections)
    (directory / "cost_pin.json").write_text(json.dumps(cost, indent=2, allow_nan=False) + "\n")
    return model, cost


def load_xgboost_pin(path, device="cpu"):
    """Restore native XGBoost files and return the model and saved metadata."""
    import xgboost as xgb

    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint_pin.json"
    metadata = json.loads(path.read_text())
    if metadata.get("format") != "final_exam_xgboost_pin_v1":
        raise ValueError("Unsupported XGBoost checkpoint format")
    expected_schema = feature_schema_pin(metadata['preprocessing'])
    if (metadata.get('feature_schema') != expected_schema or metadata['feature_count'] !=
            metadata['num_pins']*len(expected_schema['per_pin_columns'])):
        raise ValueError('XGBoost_pin checkpoint feature schema/order is inconsistent')
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
        expected_file = f"metric_{index}_pin.ubj"
        if item["file"] != expected_file:
            raise ValueError("XGBoost checkpoint contains an invalid model filename")
        booster = xgb.Booster()
        booster.load_model(path.parent / expected_file)
        booster.set_param({"device": str(target_device)})
        _backend(booster, target_device)
        boosters.append(booster)
    model = TreeRegressorPin(boosters, metadata["metric_names"], metadata["preprocessing"],
                             metadata["num_pins"], metadata["topology"], target_device).eval()
    return model, metadata
