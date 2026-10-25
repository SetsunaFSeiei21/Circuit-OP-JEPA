"""Clean-room supervised ZeroSim-style references, not official ZeroSim code.

All references optimize the input encoder and metric-query decoder end to end.
They use exactly the canonical experiment manifest and input/label scalers used
by the JEPA pipeline. No OP value, OP mask, OP state or OP representation enters
the original two models. The explicit *_op variants add continuous DC OP and
observation masks at BOTH training and inference; they do not use OP states.

The pin reference expands each *observed terminal* into a token, repeats its
parent device's named design parameters, and prepends independent external-port
tokens. Pin cliques inside devices, same-net pin connections and port-to-pin
connections are explicit. This is a clean-room architectural reference, not an
untouched reproduction or a claim of numerical parity with upstream ZeroSim.
"""
import copy
import math
from pathlib import Path
import time

import torch
from torch import nn

from .data import (device_category, fit_preprocessor, is_op_target, parameter_applicability,
                   partition_fingerprint, samples_for_split, verify_manifest)
from .model import DeviceEncoder, OperatingPointDeviceEncoder, ModelConfig, make_decoder, masked_mse
from .tensors import _fill, collate_samples, denormalize, to_device, evaluation_targets
from .training import (_batches, _check_dataset_options, _check_options, _metrics,
                       _provenance, _save_json,
                       load_checkpoint, prediction_records, report_provenance,
                       seed_everything, state_digest)


BASELINE_MODES = ("zerosim_device", "zerosim_pin", "zerosim_pin_op", "zerosim_device_op")
PIN_MODES = ("zerosim_pin", "zerosim_pin_op")
OP_INPUT_MODES = ("zerosim_pin_op", "zerosim_device_op")


def _check_mode(mode):
    if mode not in BASELINE_MODES:
        raise ValueError(f"Unknown supervised baseline mode {mode!r}; expected {BASELINE_MODES}")


def _pin_key(device_type, terminal):
    # Pin and port namespaces cannot collide, even with unusual source names.
    return f"pin:{device_type}:{terminal}"


def fit_pin_vocabulary(dataset, manifest):
    """Fit terminal-type and named-port token IDs on ALL ID train rows only.

    PAD=0 and UNK=1 are distinct. An unseen type/terminal combination or external
    port name maps to UNK, not padding. No test or OOD token grows the vocabulary.
    Parameter/label statistics remain device-based, rather than being refitted
    on repeated pin values.
    """
    verify_manifest(dataset, manifest)
    train = samples_for_split(dataset, manifest, "train")
    keys = {_pin_key(d["device_type"], terminal)
            for sample in train for d in sample["devices"] for terminal in d["terminals"]}
    keys.update(f"port:{port}" for sample in train for port in sample.get("ports", {}))
    return {"<PAD>": 0, "<UNK>": 1, **{key: i + 2 for i, key in enumerate(sorted(keys))}}


def _pin_tokens(sample, terminal_names):
    """Return ephemeral tensorization descriptors, without altering raw data."""
    tokens = [{"token_id": f"port:{name}", "key": f"port:{name}", "net": net,
               "terminal": None, "device_id": None, "parameters": {}, "device": None}
              for name, net in sorted(sample.get("ports", {}).items())]
    for device in sorted(sample["devices"], key=lambda d: d["device_id"]):
        for terminal in terminal_names:
            if terminal in device["terminals"]:
                tokens.append({"token_id": f"pin:{device['device_id']}:{terminal}",
                               "key": _pin_key(device["device_type"], terminal),
                               "net": device["terminals"][terminal], "terminal": terminal,
                               "device_id": device["device_id"],
                               "device": device,
                               "parameters": device.get("parameters", {})})
    return tokens


def collate_pin_samples(samples, preprocessing, pin_vocab,
                        include_performance=True, max_tokens=None, include_op=False):
    """Tensorize canonical device samples as ports followed by device/pin tokens.

    ``max_tokens`` (the model config's ``max_devices``) includes external ports
    but excludes the encoder's auxiliary graph token. Overflow raises; there is
    never truncation. Ports carry no device parameters. Their role is represented
    solely by their train-vocabulary token, so ``port_roles`` is deliberately
    zero rather than adding a second, pin-specific feature channel. Directional
    terminal edge metadata is available for the optional terminal-bias extension.
    The strict pin reference disables that extension by default.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    if pin_vocab.get("<PAD>") != 0 or pin_vocab.get("<UNK>") != 1:
        raise ValueError("Pin vocabulary requires distinct PAD=0 and UNK=1")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("max_tokens must be a positive integer")
    spec = preprocessing["feature_spec"]
    pn, mn, tn = (spec[k] for k in ("parameter_names", "metric_names", "terminal_names"))
    rows = [_pin_tokens(sample, tn) for sample in samples]
    b, n, p, m, t = len(rows), max(map(len, rows)), len(pn), len(mn), len(tn)
    if max_tokens is not None and n > max_tokens:
        raise ValueError(f"{n} pin/port tokens exceed max_tokens={max_tokens} (max_devices); no truncation is allowed")
    batch = {"device_types": torch.zeros(b, n, dtype=torch.long),
             "device_categories": torch.zeros(b, n, dtype=torch.long),
             "device_mask": torch.zeros(b, n, dtype=torch.bool),
             "parameters": torch.zeros(b, n, p),
             "parameter_mask": torch.zeros(b, n, p, dtype=torch.bool),
             "parameter_applicability": torch.zeros(b, n, p, dtype=torch.bool),
             "adjacency": torch.zeros(b, n, n, dtype=torch.bool),
             "edge_roles": torch.zeros(b, n, n, t * t),
             "port_roles": torch.zeros(b, n, t),
             "sample_ids": [sample["sample_id"] for sample in samples],
             "device_ids": [[token["token_id"] for token in row] for row in rows]}
    if include_op:
        o = len(spec["op_names"])
        batch.update(op=torch.zeros(b, n, o), op_mask=torch.zeros(b, n, o, dtype=torch.bool),
                     op_target_mask=torch.zeros(b, n, dtype=torch.bool))
    if include_performance:
        if "performance" not in preprocessing:
            raise ValueError("Performance statistics must be fitted on ID train before using labels")
        batch.update(performance=torch.zeros(b, m), performance_mask=torch.zeros(b, m, dtype=torch.bool))
    ti = {name: i for i, name in enumerate(tn)}
    for bi, (sample, tokens) in enumerate(zip(samples, rows)):
        batch["device_mask"][bi, :len(tokens)] = True
        for ni, token in enumerate(tokens):
            batch["device_types"][bi, ni] = pin_vocab.get(token["key"], pin_vocab["<UNK>"])
            parent = token["device"]
            category = device_category(parent) if parent is not None else "port"
            batch["device_categories"][bi, ni] = preprocessing.get("device_category_vocab", {}).get(category, 0)
            if parent is not None:
                batch["parameter_applicability"][bi, ni] = torch.tensor(parameter_applicability(parent, pn))
            if include_op and parent is not None and is_op_target(parent):
                batch["op_target_mask"][bi, ni] = True
                _fill(batch["op"][bi, ni], batch["op_mask"][bi, ni],
                      parent.get("op", {}), spec["op_names"], preprocessing["op"])
            _fill(batch["parameters"][bi, ni], batch["parameter_mask"][bi, ni],
                  token["parameters"], pn, preprocessing["parameters"])
            for nj, other in enumerate(tokens):
                both_pins = token["terminal"] is not None and other["terminal"] is not None
                same_device = both_pins and token["device_id"] == other["device_id"]
                # Independent ports are not directly joined to each other, even
                # if they name the same net. Both connect to that net's pins.
                same_net = (token["net"] == other["net"] and
                            (token["terminal"] is not None or other["terminal"] is not None))
                if ni == nj or same_device or same_net:
                    batch["adjacency"][bi, ni, nj] = True
                    if both_pins:
                        batch["edge_roles"][bi, ni, nj, ti[token["terminal"]] * t + ti[other["terminal"]]] = 1
        if include_performance:
            _fill(batch["performance"][bi], batch["performance_mask"][bi],
                  sample.get("performance", {}), mn, preprocessing["performance"])
    return batch


class SupervisedPerformanceModel(nn.Module):
    """Unlike the JEPA readout stage, both modules stay trainable and in train mode."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, batch):
        memory, padding = self.encoder(batch)
        return self.decoder(memory, padding)


def _make_model(mode, model_config, preprocessing, pin_vocab=None):
    _check_mode(mode)
    spec = preprocessing["feature_spec"]
    if mode in PIN_MODES:
        if pin_vocab is None:
            raise ValueError("Pin reference requires its saved train-only pin vocabulary")
    num_types = len(pin_vocab) if mode in PIN_MODES else len(preprocessing["device_vocab"])
    kwargs = {"input_contract_version": preprocessing.get("input_contract_version", 1),
              "num_categories": len(preprocessing.get("device_category_vocab", {}))}
    cls = OperatingPointDeviceEncoder if mode in OP_INPUT_MODES else DeviceEncoder
    if mode in OP_INPUT_MODES:
        kwargs["num_op_features"] = len(spec["op_names"])
    encoder = cls(ModelConfig(**model_config), num_types, len(spec["parameter_names"]),
                  len(spec["terminal_names"]), **kwargs)
    return SupervisedPerformanceModel(encoder, make_decoder(encoder.cfg,
                                      len(preprocessing["feature_spec"]["metric_names"])))


def _collate(samples, preprocessing, mode, model_config, pin_vocab):
    if mode in PIN_MODES:
        return collate_pin_samples(samples, preprocessing, pin_vocab,
                                   max_tokens=model_config["max_devices"], include_op=mode in OP_INPUT_MODES)
    return collate_samples(samples, preprocessing, include_op=mode in OP_INPUT_MODES,
                           include_performance=True, include_op_states=False)


def train_baseline(dataset, manifest, options, output, mode="zerosim_device", device="cpu"):
    """Train encoder+decoder only on labeled ID train; select using ID validation.

    Epoch count, optimizer LR, minibatches, seed, target mask and objective match
    JEPA's decoder training budget. This is *label-budget* parity, not equal FLOPs
    or parameter count. No OP pretraining is run. Pin mode defaults to no learned
    terminal edge bias; set ``options['baseline']['pin_terminal_bias']=True`` to
    opt into the explicitly recorded implementation enhancement.
    """
    _check_mode(mode)
    verify_manifest(dataset, manifest)
    _check_options(options)
    _check_dataset_options(options, manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        import json
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Refusing to overwrite a different experiment manifest")
    started = time.perf_counter()
    effective = copy.deepcopy(options)
    if mode in PIN_MODES:
        terminal_bias = effective.get("baseline", {}).get("pin_terminal_bias", False)
        if type(terminal_bias) is not bool:
            raise ValueError("baseline.pin_terminal_bias must be a boolean")
        effective["model"]["terminal_bias"] = terminal_bias
    tr = effective["training"]
    epochs = tr.get("baseline_epochs", tr["decoder_epochs"])
    if type(epochs) is not int or epochs < 1:
        raise ValueError("baseline_epochs must be a positive integer")
    seed_everything(tr["seed"], tr.get("torch_threads", 1))
    prep = fit_preprocessor(dataset, manifest, include_performance=True,
                            config=effective.get("normalization"))
    if mode in OP_INPUT_MODES and not any(prep["op"]["count"]):
        raise ValueError("Direct OP baseline requires some observed eligible ID-train DC OP")
    if not any(prep["performance"]["count"]):
        raise ValueError("No requested metric has an observed labeled-train target")
    pin_vocab = fit_pin_vocabulary(dataset, manifest) if mode in PIN_MODES else None
    model = _make_model(mode, effective["model"], prep, pin_vocab).to(device)
    initial_encoder_digest, initial_decoder_digest = state_digest(model.encoder), state_digest(model.decoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["decoder_lr"], weight_decay=tr["weight_decay"])
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError("Supervised optimizer must contain all encoder and decoder parameters")
    splits = {"train": samples_for_split(dataset, manifest, "labeled_train"),
              "val": samples_for_split(dataset, manifest, "val")}
    cost = {"labeled_train_count": len(splits["train"]), "input_scaler_train_count": len(manifest["train"]),
            "requires_op_at_inference": mode in OP_INPUT_MODES,
            "op_acquisition_cost_seconds": None, "op_acquisition_cost_status": "not_measured",
            "total_parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_parameter_count": sum(p.numel() for p in model.encoder.parameters()),
            "decoder_parameter_count": sum(p.numel() for p in model.decoder.parameters()),
            "optimizer_steps": 0, "epochs": epochs, "pretrain_epochs": 0,
            "learning_rate": tr["decoder_lr"], "batch_size": tr["batch_size"], "seed": tr["seed"]}
    checkpoint_path = output / f"{mode}_best.pt"
    history, best = [], math.inf
    for epoch in range(epochs):
        record = {"epoch": epoch + 1}
        for phase, rows in splits.items():
            model.train(phase == "train")
            total, count, steps = 0.0, 0, 0
            for samples in _batches(rows, tr["batch_size"], tr["seed"] + epoch if phase == "train" else None):
                batch = to_device(_collate(samples, prep, mode, effective["model"], pin_vocab), device)
                if mode not in OP_INPUT_MODES and any(key.startswith("op") for key in batch):
                    raise AssertionError("Supervised reference batches must not contain OP inputs")
                mask = batch["performance_mask"]
                if not mask.any():
                    continue
                with torch.set_grad_enabled(phase == "train"):
                    prediction = model(batch)
                    loss = masked_mse(prediction, batch["performance"], mask)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite supervised baseline objective")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"], error_if_nonfinite=True)
                        optimizer.step()
                        cost["optimizer_steps"] += 1
                        steps += 1
                k = int(mask.sum())
                total += float(loss.detach()) * k
                count += k
            if not count:
                raise ValueError(f"No observed performance targets in {phase}")
            record[phase] = {"normalized_mse": total / count, "observed_targets": count,
                             "optimizer_steps": steps}
        history.append(record)
        print(f"{mode} {epoch+1}/{epochs} train={record['train']['normalized_mse']:.6f} val={record['val']['normalized_mse']:.6f}", flush=True)
        if record["val"]["normalized_mse"] < best:
            best = record["val"]["normalized_mse"]
            checkpoint = {**_provenance(dataset, manifest, effective), "stage": "baseline", "mode": mode, "backend": f"cleanroom_{mode}",
                          "architecture": "clean-room-supervised-pin-reference" if mode in PIN_MODES else "clean-room-supervised-device-reference",
                          "requested_options": copy.deepcopy(options), "epoch": epoch + 1,
                          "selection": {"split": "val", "objective": "normalized_performance_mse", "value": best},
                          "preprocessing": prep, "pin_vocab": pin_vocab,
                          "pin_vocab_fitted_sample_ids": sorted(manifest["train"]) if pin_vocab is not None else None,
                          "encoder_state": model.encoder.state_dict(), "decoder_state": model.decoder.state_dict(),
                          "encoder_digest": state_digest(model.encoder), "decoder_digest": state_digest(model.decoder),
                          "initial_encoder_digest": initial_encoder_digest, "initial_decoder_digest": initial_decoder_digest,
                          "encoder_frozen": False, "freeze_verified": False, "op_inputs_used": mode in OP_INPUT_MODES,
                          "requires_op_at_inference": mode in OP_INPUT_MODES,
                          "unobserved_training_metrics": [name for name, count in zip(prep["feature_spec"]["metric_names"], prep["performance"]["count"]) if not count],
                          "training_cost": dict(cost),
                          "reference_note": "Clean-room ZeroSim-style reference; no upstream source copied; not an untouched official reproduction."}
            torch.save(checkpoint, checkpoint_path)
    # Retain the best validation weights but record the cost of the complete run,
    # including later epochs that did not improve validation performance.
    selected = load_checkpoint(checkpoint_path)
    cost["wall_time_seconds"] = time.perf_counter() - started
    cost["selected_epoch"] = selected["epoch"]
    cost["supervised_optimizer_steps"] = cost["optimizer_steps"]
    cost["pretrain_optimizer_steps"] = 0
    cost["labeled_train_examples"] = cost["labeled_train_count"]
    selected["training_cost"] = cost
    selected["timings"] = {"pretrain_seconds": 0., "supervised_seconds": cost["wall_time_seconds"]}
    selected["parameter_count"] = {"total": cost["total_parameter_count"],
                                   "trainable": cost["trainable_parameter_count"],
                                   "encoder": cost["encoder_parameter_count"],
                                   "decoder": cost["decoder_parameter_count"]}
    torch.save(selected, checkpoint_path)
    _save_json(output / f"{mode}_history.json", history)
    _save_json(output / f"preprocessing_{mode}.json", prep)
    _save_json(output / f"{mode}_training_cost.json", cost)
    if pin_vocab is not None:
        _save_json(output / "pin_vocabulary.json", {"vocabulary": pin_vocab, "fitted_sample_ids": sorted(manifest["train"])})
    _save_json(manifest_path, manifest)
    return str(checkpoint_path)


def evaluate_baseline(dataset, checkpoint_path, split="test", batch_size=32, device="cpu", output=None):
    """Reload a selected baseline and report the same physical-unit metrics as JEPA.

    An explicitly absent OOD holdout returns a zero-sample, not-configured report;
    it does not manufacture OOD results or silently substitute the ID test split.
    """
    source = load_checkpoint(checkpoint_path)
    if source.get("stage") != "baseline":
        raise ValueError("Baseline evaluation requires a supervised baseline checkpoint")
    mode = source.get("mode")
    _check_mode(mode)
    if split not in {"train", "val", "test", "ood"}:
        raise ValueError("Unknown split")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    verify_manifest(dataset, source["manifest"])
    if partition_fingerprint(source["manifest"]) != source["partition_fingerprint"]:
        raise ValueError("Baseline checkpoint partition fingerprint mismatch")
    prep = source["preprocessing"]
    if prep["fitted_sample_ids"] != sorted(source["manifest"]["train"]):
        raise ValueError("Baseline input preprocessing differs from ID training partition")
    if prep["performance_fitted_sample_ids"] != sorted(source["manifest"]["labeled_train"]):
        raise ValueError("Baseline label preprocessing differs from labeled training partition")
    pin_vocab = source.get("pin_vocab")
    if mode in PIN_MODES and (source.get("pin_vocab_fitted_sample_ids") != sorted(source["manifest"]["train"]) or
                                  pin_vocab != fit_pin_vocabulary(dataset, source["manifest"])):
        raise ValueError("Pin vocabulary differs from the ID train-only vocabulary")
    model = _make_model(mode, source["model_config"], prep, pin_vocab).to(device)
    model.encoder.load_state_dict(source["encoder_state"], strict=True)
    model.decoder.load_state_dict(source["decoder_state"], strict=True)
    model.eval()
    if state_digest(model.encoder) != source["encoder_digest"] or state_digest(model.decoder) != source["decoder_digest"]:
        raise AssertionError("Baseline checkpoint model integrity mismatch")
    rows = samples_for_split(dataset, source["manifest"], split)
    predictions, truths, masks, circuits = [], [], [], []
    normalized_predictions, normalized_targets = [], []
    with torch.no_grad():
        for samples in _batches(rows, batch_size):
            batch = to_device(_collate(samples, prep, mode, source["model_config"], pin_vocab), device)
            normalized = model(batch)
            normalized_predictions.append(normalized.cpu())
            raw_truth, normalized_truth = evaluation_targets(samples, prep)
            normalized_targets.append(normalized_truth)
            predictions.append(denormalize(normalized, prep["performance"]).cpu())
            truths.append(raw_truth)
            masks.append(batch["performance_mask"].cpu())
            circuits.extend(sample["circuit_id"] for sample in samples)
    if predictions:
        pred, target, mask = map(torch.cat, (predictions, truths, masks))
        norm_pred, norm_target = map(torch.cat, (normalized_predictions, normalized_targets))
    elif split == "ood" and not source["manifest"].get("ood", []):
        shape = (0, len(prep["feature_spec"]["metric_names"]))
        pred, target, mask = torch.empty(shape), torch.empty(shape), torch.zeros(shape, dtype=torch.bool)
        norm_pred, norm_target = torch.empty(shape), torch.empty(shape)
    else:
        raise ValueError(f"No examples in {split}")
    report = {**report_provenance(source, dataset, rows), "split": split, "synthetic": source["synthetic"], "num_samples": len(rows),
              "status": "evaluated" if rows else "not_configured", "selection_used_ood": False,
              "has_ood_holdout": bool(source["manifest"].get("ood", [])),
              "encoder_frozen": False, "encoder_frozen_verified": False, "op_inputs_used": mode in OP_INPUT_MODES,
              "requires_op_at_inference": mode in OP_INPUT_MODES,
              "unobserved_training_metrics": source.get("unobserved_training_metrics", []),
              "metric_units": "original dataset units; see feature_spec and data provider",
              "metrics": _metrics(pred, target, mask, prep, norm_pred, norm_target), "by_circuit": {},
              "training_cost": source["training_cost"], "selection": source["selection"],
              "selected_epoch": source["epoch"], "reference_note": source["reference_note"]}
    for circuit in sorted(set(circuits)):
        selected = torch.tensor([c == circuit for c in circuits])
        report["by_circuit"][circuit] = _metrics(pred[selected], target[selected], mask[selected], prep, norm_pred[selected], norm_target[selected])
    report["predictions"] = prediction_records(pred, target, mask, norm_pred, norm_target, rows, prep)
    if output:
        _save_json(output, report)
    return report
