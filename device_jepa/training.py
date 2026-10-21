"""Two disjoint optimization stages with auditable split/checkpoint provenance."""
import copy
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import torch

from .data import (dataset_fingerprint, fit_preprocessor, input_op_fingerprint,
                   partition_fingerprint, samples_for_split, verify_manifest)
from .model import DeviceEncoder, FrozenPerformanceModel, JEPAModel, ModelConfig, alignment_loss, make_decoder, masked_mse
from .tensors import collate_samples, denormalize, to_device


def seed_everything(seed, threads=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encoder_from_metadata(config, prep):
    spec = prep["feature_spec"]
    return DeviceEncoder(ModelConfig(**config), len(prep["device_vocab"]),
                         len(spec["parameter_names"]), len(spec["terminal_names"]))


def _batches(samples, batch_size, seed=None):
    order = list(range(len(samples)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [samples[i] for i in order[start:start + batch_size]]


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _check_options(options):
    tr = options["training"]
    if tr["batch_size"] < 1 or tr["pretrain_epochs"] < 1 or tr["decoder_epochs"] < 1:
        raise ValueError("batch_size and epochs must be positive")
    for k in ("pretrain_lr", "decoder_lr", "grad_clip"):
        if tr[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if options["loss"]["variance"] <= 0 or options["loss"]["covariance"] < 0:
        raise ValueError("Keep positive variance regularization to avoid collapse")


def _provenance(dataset, manifest, options):
    return {"checkpoint_version": 1, "model_config": options["model"],
            "options": copy.deepcopy(options), "manifest": copy.deepcopy(manifest),
            "input_op_fingerprint": input_op_fingerprint(dataset),
            "partition_fingerprint": partition_fingerprint(manifest),
            "synthetic": bool(dataset.get("metadata", {}).get("synthetic", dataset.get("synthetic", False))),
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__)},
            "architecture": "device-token_zero-sim-inspired_shared-op-mlp_metric-query-transformer"}


def pretrain(dataset, manifest, options, output, device="cpu"):
    """Only ID train design inputs/OP are used; validation sees OP, never labels."""
    verify_manifest(dataset, manifest)
    _check_options(options)
    tr = options["training"]
    seed_everything(tr["seed"], tr.get("torch_threads", 1))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    preprocessing = fit_preprocessor(dataset, manifest, include_performance=False)
    if "performance" in preprocessing:
        raise AssertionError("Stage 1 must not fit performance normalization")
    encoder = encoder_from_metadata(options["model"], preprocessing)
    model = JEPAModel(encoder, len(preprocessing["feature_spec"]["op_names"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["pretrain_lr"], weight_decay=tr["weight_decay"])
    train = samples_for_split(dataset, manifest, "train")
    val = samples_for_split(dataset, manifest, "val")
    # An all-OP-missing record is valid raw data but supplies no stage-1 target.
    observed = lambda s: any(v is not None for d in s["devices"] for v in d.get("op", {}).values())
    train_usable, val_usable = [s for s in train if observed(s)], [s for s in val if observed(s)]
    if not train_usable or not val_usable:
        raise ValueError("Both ID train and validation require some observed OP")
    best, history = math.inf, []
    for epoch in range(tr["pretrain_epochs"]):
        records = {}
        for phase, rows in (("train", train_usable), ("val", val_usable)):
            model.train(phase == "train")
            totals, count = {}, 0
            for samples in _batches(rows, tr["batch_size"], tr["seed"] + epoch if phase == "train" else None):
                batch = to_device(collate_samples(samples, preprocessing, include_op=True, include_performance=False), device)
                with torch.set_grad_enabled(phase == "train"):
                    a, b, valid = model(batch)
                    if int(valid.sum()) < 2:
                        continue
                    loss, detail = alignment_loss(a, b, valid, **options["loss"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite stage-1 objective")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"], error_if_nonfinite=True)
                        optimizer.step()
                weight = int(valid.sum())
                count += weight
                for key, value in {"loss": loss.detach(), **detail}.items():
                    if key != "observed_devices":
                        totals[key] = totals.get(key, 0.0) + float(value) * weight
            if not count:
                raise ValueError(f"Stage-1 {phase} has no batch with >=2 observed device OP vectors")
            records[phase] = {k: v / count for k, v in totals.items()}
            records[phase]["observed_device_tokens"] = count
        history.append({"epoch": epoch + 1, **records})
        print(f"pretrain {epoch+1}/{tr['pretrain_epochs']} train={records['train']['loss']:.6f} val={records['val']['loss']:.6f}", flush=True)
        if records["val"]["loss"] < best:
            best = records["val"]["loss"]
            checkpoint = {**_provenance(dataset, manifest, options), "stage": "pretrain",
                          "epoch": epoch + 1, "selection": {"split": "val", "objective": "op_alignment_regularized", "value": best},
                          "preprocessing": preprocessing, "model_state": model.state_dict(),
                          "encoder_state": model.encoder.state_dict(),
                          "encoder_digest": state_digest(model.encoder),
                          "skipped_all_op_missing": {"train": len(train)-len(train_usable), "val": len(val)-len(val_usable)}}
            torch.save(checkpoint, output / "pretrain_best.pt")
    _save_json(output / "pretrain_history.json", history)
    _save_json(output / "manifest.json", manifest)
    _save_json(output / "preprocessing_stage1.json", preprocessing)
    return str(output / "pretrain_best.pt")


def load_checkpoint(path):
    # Only tensors and basic Python containers. Never unpickle arbitrary objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    return checkpoint


def train_decoder(dataset, pretrain_checkpoint, options, output, device="cpu"):
    source = load_checkpoint(pretrain_checkpoint)
    if source["stage"] != "pretrain":
        raise ValueError("train_decoder requires a stage-1 checkpoint")
    # Labels may be attached after OP pretraining, but no design/OP record,
    # feature schema, or partition may silently change between the stages.
    if input_op_fingerprint(dataset) != source["input_op_fingerprint"]:
        raise ValueError("Stage-2 dataset changed immutable design/input/OP provenance; only performance values may change")
    if partition_fingerprint(source["manifest"]) != source["partition_fingerprint"]:
        raise ValueError("Stage-1 checkpoint partition fingerprint mismatch")
    if source["preprocessing"]["fitted_sample_ids"] != source["manifest"]["train"]:
        raise ValueError("Stage-1 preprocessing fitted_sample_ids differ from training partition")
    manifest = copy.deepcopy(source["manifest"])
    manifest["dataset_fingerprint"] = dataset_fingerprint(dataset)
    verify_manifest(dataset, manifest)
    _check_options(options)
    if options["model"] != source["model_config"]:
        raise ValueError("Architecture differs from pretraining checkpoint; use the same config")
    tr = options["training"]
    seed_everything(tr["seed"], tr.get("torch_threads", 1))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prep = copy.deepcopy(source["preprocessing"])
    supervised_stats = fit_preprocessor(dataset, manifest, include_performance=True)
    prep["performance"] = supervised_stats["performance"]
    if not all(c > 0 for c in prep["performance"]["count"]):
        raise ValueError("Every requested metric needs at least one ID-train observation")
    encoder = encoder_from_metadata(source["model_config"], prep)
    encoder.load_state_dict(source["encoder_state"], strict=True)
    decoder = make_decoder(encoder.cfg, len(prep["feature_spec"]["metric_names"]))
    model = FrozenPerformanceModel(encoder, decoder).to(device)
    frozen_digest = state_digest(model.encoder)
    if frozen_digest != source["encoder_digest"]:
        raise AssertionError("Encoder checkpoint integrity mismatch")
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=tr["decoder_lr"], weight_decay=tr["weight_decay"])
    encoder_parameters = {id(p) for p in model.encoder.parameters()}
    if any(id(p) in encoder_parameters for g in optimizer.param_groups for p in g["params"]):
        raise AssertionError("Encoder parameters entered the stage-2 optimizer")
    split_rows = {split: samples_for_split(dataset, manifest, split) for split in ("train", "val")}
    history, best = [], math.inf
    for epoch in range(tr["decoder_epochs"]):
        record = {"epoch": epoch + 1}
        for phase, rows in split_rows.items():
            model.train(phase == "train")
            total, count = 0.0, 0
            for samples in _batches(rows, tr["batch_size"], tr["seed"] + epoch if phase == "train" else None):
                # OP is not even loaded into stage-2 tensors.
                batch = to_device(collate_samples(samples, prep, include_op=False, include_performance=True), device)
                mask = batch["performance_mask"]
                if not mask.any():
                    continue
                with torch.set_grad_enabled(phase == "train"):
                    prediction = model(batch)
                    loss = masked_mse(prediction, batch["performance"], mask)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite stage-2 objective")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), tr["grad_clip"], error_if_nonfinite=True)
                        optimizer.step()
                k = int(mask.sum())
                total += float(loss.detach()) * k
                count += k
            if not count:
                raise ValueError(f"No observed performance targets in {phase}")
            record[phase] = {"normalized_mse": total / count, "observed_targets": count}
        # Includes all parameters AND registered buffers, not just requires_grad.
        if state_digest(model.encoder) != frozen_digest:
            raise AssertionError("Frozen encoder state changed during decoder training")
        if any(p.grad is not None for p in model.encoder.parameters()):
            raise AssertionError("Frozen encoder unexpectedly received gradients")
        history.append(record)
        print(f"decoder {epoch+1}/{tr['decoder_epochs']} train={record['train']['normalized_mse']:.6f} val={record['val']['normalized_mse']:.6f}", flush=True)
        if record["val"]["normalized_mse"] < best:
            best = record["val"]["normalized_mse"]
            checkpoint = {**_provenance(dataset, manifest, options), "stage": "decoder",
                          "epoch": epoch+1, "selection": {"split": "val", "objective": "normalized_performance_mse", "value": best},
                          "preprocessing": prep, "encoder_state": model.encoder.state_dict(),
                          "decoder_state": model.decoder.state_dict(), "encoder_digest": frozen_digest,
                          "source_pretrain_digest": source["encoder_digest"],
                          "source_pretrain_dataset_fingerprint": source["manifest"]["dataset_fingerprint"],
                          "freeze_verified": True}
            torch.save(checkpoint, output / "decoder_best.pt")
    _save_json(output / "decoder_history.json", history)
    _save_json(output / "preprocessing_stage2.json", prep)
    return str(output / "decoder_best.pt")


def evaluate(dataset, checkpoint_path, split="test", batch_size=32, device="cpu", output=None):
    source = load_checkpoint(checkpoint_path)
    if source["stage"] != "decoder":
        raise ValueError("Evaluation needs a stage-2 checkpoint")
    if split not in {"train", "val", "test", "ood"}:
        raise ValueError("Unknown split")
    verify_manifest(dataset, source["manifest"])
    prep = source["preprocessing"]
    encoder = encoder_from_metadata(source["model_config"], prep)
    encoder.load_state_dict(source["encoder_state"], strict=True)
    decoder = make_decoder(encoder.cfg, len(prep["feature_spec"]["metric_names"]))
    decoder.load_state_dict(source["decoder_state"], strict=True)
    model = FrozenPerformanceModel(encoder, decoder).to(device).eval()
    if state_digest(model.encoder) != source["encoder_digest"]:
        raise AssertionError("Encoder integrity mismatch")
    rows = samples_for_split(dataset, source["manifest"], split)
    predictions, truths, masks, circuits = [], [], [], []
    with torch.no_grad():
        for samples in _batches(rows, batch_size):
            batch = to_device(collate_samples(samples, prep, include_op=False, include_performance=True), device)
            raw = denormalize(model(batch), prep["performance"]).cpu()
            truth = denormalize(batch["performance"], prep["performance"]).cpu()
            predictions.append(raw)
            truths.append(truth)
            masks.append(batch["performance_mask"].cpu())
            circuits.extend(s["circuit_id"] for s in samples)
    if not predictions:
        raise ValueError(f"No examples in {split}")
    pred, target, mask = map(torch.cat, (predictions, truths, masks))
    report = {"split": split, "synthetic": source["synthetic"], "num_samples": len(rows),
              "selection_used_ood": False, "encoder_frozen_verified": source["freeze_verified"],
              "metric_units": "original dataset units; see feature_spec and data provider", "metrics": _metrics(pred, target, mask, prep),
              "by_circuit": {}}
    for circuit in sorted(set(circuits)):
        selected = torch.tensor([c == circuit for c in circuits])
        report["by_circuit"][circuit] = _metrics(pred[selected], target[selected], mask[selected], prep)
    report["predictions"] = [{"sample_id": s["sample_id"], "circuit_id": s["circuit_id"],
                               "prediction": {name: float(pred[i, j]) for j, name in enumerate(prep["feature_spec"]["metric_names"])}}
                              for i, s in enumerate(rows)]
    if output:
        _save_json(output, report)
    return report


def _metrics(pred, target, mask, prep):
    result = {}
    for j, name in enumerate(prep["feature_spec"]["metric_names"]):
        selected = mask[:, j]
        count = int(selected.sum())
        if not count:
            result[name] = {"count": 0, "mae": None, "rmse": None, "r2": None}
            continue
        y, p = target[selected, j].double(), pred[selected, j].double()
        errors = p - y
        denominator = ((y - y.mean()) ** 2).sum()
        result[name] = {"count": count, "mae": float(errors.abs().mean()),
                        "rmse": float(errors.square().mean().sqrt()),
                        "r2": float(1 - errors.square().sum() / denominator) if count > 1 and denominator > 0 else None,
                        "normalized_rmse": float(errors.square().mean().sqrt()) / prep["performance"]["std"][j]}
    return result
