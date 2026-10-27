"""Two disjoint optimization stages with auditable split/checkpoint provenance."""
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import random
import re
import time
import uuid
from pathlib import Path

import torch

from .data import (DEVICE_CATEGORY_VOCAB, INPUT_CONTRACT_VERSION, dataset_fingerprint, fit_preprocessor, input_op_fingerprint,
                   partition_fingerprint, samples_for_split, verify_manifest, is_op_target,
                   _preprocessing_config, _statistics)
from .model import DeviceEncoder, FrozenPerformanceModel, JEPAModel, ModelConfig, alignment_loss, make_decoder, masked_mse
from .ema_model import EMAJEPAModel, EMA_MODE, EMA_ARCHITECTURE, ema_protocol, validate_ema_decay
from .tensors import collate_samples, denormalize, to_device, evaluation_targets
from .budgets import positive_epochs
from .methods import JEPA_MODES, OP_MODES, method_spec
from .early_stopping import (EarlyStopping, PRETRAIN_OBJECTIVE, SUPERVISED_OBJECTIVE,
                             early_stopping_config, executed_epochs)
from .resident import PIPELINE, BatchProgress, ResidentDataset, require_device, deterministic_workspace


def seed_everything(seed, threads=1):
    # Must precede CUDA initialization; never weaken deterministic algorithms.
    # CPU runs are unaffected. An explicit incompatible setting is not silently
    # overwritten, because that would misrepresent the caller's execution setup.
    deterministic_workspace()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def state_digest(module):
    return _state_dict_digest(module.state_dict())


def _state_dict_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encoder_from_metadata(config, prep):
    spec = prep["feature_spec"]
    return DeviceEncoder(ModelConfig(**config), len(prep["device_vocab"]),
                         len(spec["parameter_names"]), len(spec["terminal_names"]),
                         prep.get("input_contract_version", 1), len(prep.get("device_category_vocab", {})),
                         num_models=len(prep.get("model_vocab", {})),
                         num_port_roles=len(prep.get("port_role_vocab", {})))


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
    early_stopping_config(tr)
    for key in ('batch_size', 'pretrain_epochs', 'decoder_epochs'):
        positive_epochs(tr, key)
    if 'baseline_epochs' in tr:
        positive_epochs(tr, 'baseline_epochs')
    if 'progress_every' in tr:
        positive_epochs(tr, 'progress_every')
    for k in ("pretrain_lr", "decoder_lr", "grad_clip"):
        if tr[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if options["loss"]["variance"] <= 0 or options["loss"]["covariance"] < 0:
        raise ValueError("Keep positive variance regularization to avoid collapse")
    if 'ema_decay' in tr:
        validate_ema_decay(tr['ema_decay'])


def _check_dataset_options(options, manifest):
    config = options.get("dataset")
    if config is None:
        return
    for key in ("train_circuits", "ood_circuits"):
        if sorted(config.get(key, [])) != manifest[key]:
            raise ValueError(f"Config {key} differs from persisted manifest; create an explicit new experiment")
    for key, default in (("split_ratios", [.8, .1, .1]), ("split_seed", 17), ("finetune_label_fraction", 1.)):
        if config.get(key, default) != manifest[key]:
            raise ValueError(f"Config {key} differs from persisted manifest")


def _prepare_stage_output(output, stage, dataset, manifest, source=None):
    """Reject overwrite and mixed experiments; this is never a resume API."""
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError("Training output must be a directory")
    entries = {p.name for p in output.iterdir()} if output.exists() else set()
    allowed = {"manifest.json", "data.json"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        valid_manifests = [manifest] + ([source["manifest"]] if source is not None else [])
        if existing not in valid_manifests:
            raise ValueError("Refusing to overwrite a different experiment manifest")
    data_path = output / "data.json"
    if data_path.exists():
        stored = json.loads(data_path.read_text())
        if input_op_fingerprint(stored) != input_op_fingerprint(dataset):
            raise ValueError("Training output contains data from a different experiment")
    if stage == "decoder" and (output / "pretrain_best.pt").exists():
        existing = load_checkpoint(output / "pretrain_best.pt")
        if source is None or any(existing.get(k) != source.get(k) for k in (
                "stage", "encoder_digest", "input_op_fingerprint", "partition_fingerprint", "checkpoint_run_id")):
            raise ValueError("Output pretraining checkpoint differs from the requested source")
        if _pretraining_provenance(existing) != _pretraining_provenance(source):
            raise ValueError("Output pretraining protocol differs from the requested source")
        allowed.update({"pretrain_best.pt", "pretrain_history.json", "preprocessing_stage1.json"})
    unexpected = entries - allowed
    if unexpected:
        raise ValueError("Training output is not new for this stage; refusing overwrite or stale artifacts: "
                         + ", ".join(sorted(unexpected)))
    output.mkdir(parents=True, exist_ok=True)
    return output


def _selection_metric_names(prep):
    return [name for name, count in zip(prep["feature_spec"]["metric_names"],
                                       prep["performance"]["count"]) if count > 0]


def _supervised_metric_mask(mask, prep):
    """Only metrics with training observations can choose the checkpoint."""
    trained = torch.tensor([count > 0 for count in prep["performance"]["count"]],
                           dtype=torch.bool, device=mask.device)
    return mask & trained.unsqueeze(0)


def _effective_alignment_loss(config):
    """Resolve the same defaults that the actual alignment call consumes."""
    bound = inspect.signature(alignment_loss).bind(None, None, None, **config)
    bound.apply_defaults()
    return {key: value for key, value in bound.arguments.items()
            if key not in {"predicted", "target", "valid"}}


def _fit_stage_preprocessor(dataset, manifest, device, include_performance=False, config=None):
    device = require_device(device)
    print(f"Fitting ID-train-only transforms on {device} | labels={include_performance}", flush=True)
    return fit_preprocessor(dataset, manifest, include_performance, config,
                            statistics_device=device if device.type == 'cuda' else None)


def _pretraining_provenance(source):
    """Capture executed stage-1 facts instead of stage-2 configuration wishes."""
    options, prep = source["options"], source["preprocessing"]
    tr = options["training"]
    training = {key: tr[key] for key in (
        "pretrain_epochs", "pretrain_lr", "batch_size", "weight_decay", "grad_clip")}
    training["torch_threads"] = tr.get("torch_threads", 1)
    protocol = {
        "protocol_version": 1, "optimizer": "AdamW", "training": training,
        "model_config": copy.deepcopy(source["model_config"]),
        "architecture": source["architecture"],
        "input_contract_version": prep.get("input_contract_version", 1),
        "loss": copy.deepcopy(source.get("effective_loss_config",
                                         _effective_alignment_loss(options["loss"]))),
        "op_state": copy.deepcopy(options.get("op_state", {"features": []})),
        "feature_spec": copy.deepcopy(prep["feature_spec"]),
        "normalization": copy.deepcopy(prep["transform_config"]),
        "tensor_pipeline": source.get("tensor_pipeline", "legacy_scalar"),
    }
    if 'early_stopping' in source:
        protocol['early_stopping'] = early_stopping_config(tr)
    if source.get('mode') == EMA_MODE:
        protocol.update(mode=EMA_MODE, ema=copy.deepcopy(source['ema']))
    return {"version": 1, "protocol": protocol, "protocol_digest": _json_digest(protocol),
            "seed": tr["seed"], "source_options": copy.deepcopy(options),
            "selected_epoch": source["epoch"], "selection": copy.deepcopy(source["selection"]),
            "execution": copy.deepcopy(source.get('early_stopping')),
            "encoder_digest": source["encoder_digest"],
            "input_op_fingerprint": source["input_op_fingerprint"],
            "partition_fingerprint": source["partition_fingerprint"],
            "source_dataset_fingerprint": source["manifest"]["dataset_fingerprint"],
            "checkpoint_run_id": source.get("checkpoint_run_id"),
            **({"selected_ema_updates": source['ema_updates']} if source.get('mode') == EMA_MODE else {})}


def _provenance(dataset, manifest, options, input_fingerprint=None):
    print('Preparing verified checkpoint provenance (once per training stage)', flush=True)
    return {"checkpoint_version": 2, "model_config": options["model"],
            "input_contract_version": INPUT_CONTRACT_VERSION,
            "checkpoint_run_id": uuid.uuid4().hex,
            "requires_op_at_inference": False,
            "options": copy.deepcopy(options), "manifest": copy.deepcopy(manifest),
            "input_op_fingerprint": input_fingerprint if input_fingerprint is not None else input_op_fingerprint(dataset),
            "partition_fingerprint": partition_fingerprint(manifest),
            "synthetic": bool(dataset.get("metadata", {}).get("synthetic", dataset.get("synthetic", False))),
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__)},
            "tensor_pipeline": PIPELINE,
            "architecture": "shared-structure-prefix_direct-device-op-alignment_v2",
            "shared_structure_layers": options["model"]["encoder_layers"] - options["model"]["cross_layers"]}


def _jepa_provenance(provenance, mode, ema=None):
    provenance.update(mode=mode, backend=EMA_MODE if mode == EMA_MODE else 'shared_structure_jepa_v2')
    if mode == EMA_MODE:
        provenance.update(architecture=EMA_ARCHITECTURE, shared_structure_layers=0,
                          structure_layers=provenance['model_config']['encoder_layers'] - provenance['model_config']['cross_layers'],
                          ema=copy.deepcopy(ema))
    return provenance


def pretrain(dataset, manifest, options, output, device="cpu", mode="jepa_zerosim"):
    """Only ID train design inputs/OP are used; validation sees OP, never labels."""
    started = time.perf_counter()
    optimizer_steps = 0
    if mode not in JEPA_MODES:
        raise ValueError(f'Pretraining requires a JEPA model, got {mode!r}')
    device = require_device(device)
    print(f"JEPA pretrain setup on {device}: validating dataset/manifest", flush=True)
    verify_manifest(dataset, manifest)
    _check_options(options)
    effective_loss = _effective_alignment_loss(options["loss"])
    _check_dataset_options(options, manifest)
    from .states import STATE_FEATURES, validate_op_states
    state_config = options.get("op_state", {"features": []})
    validate_op_states(dataset, state_config)
    state_features = state_config.get("features", [] if state_config.get("mode") == "preserve_only" else STATE_FEATURES)
    tr = options["training"]
    seed_everything(tr["seed"], tr.get("torch_threads", 1))
    output = _prepare_stage_output(output, "pretrain", dataset, manifest)
    manifest_path = output / "manifest.json"
    preprocessing = _fit_stage_preprocessor(dataset, manifest, device, include_performance=False, config=options.get("normalization"))
    if "performance" in preprocessing:
        raise AssertionError("Stage 1 must not fit performance normalization")
    encoder = encoder_from_metadata(options["model"], preprocessing)
    preprocessing["op_state_features"] = list(state_features)
    model_class = EMAJEPAModel if mode == EMA_MODE else JEPAModel
    extra = {'ema_decay': tr.get('ema_decay', 0.99)} if mode == EMA_MODE else {}
    model = model_class(encoder, len(preprocessing["feature_spec"]["op_names"]),
                        len(preprocessing["op_state_features"]), **extra).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=tr["pretrain_lr"], weight_decay=tr["weight_decay"])
    train = samples_for_split(dataset, manifest, "train")
    val = samples_for_split(dataset, manifest, "val")
    # An all-OP-missing record is valid raw data but supplies no stage-1 target.
    selected_op = preprocessing["feature_spec"]["op_names"]
    observed = lambda s: any(d.get("op", {}).get(name) is not None for d in s["devices"]
                             if is_op_target(d) for name in selected_op)
    train_usable, val_usable = [s for s in train if observed(s)], [s for s in val if observed(s)]
    if not train_usable or not val_usable:
        raise ValueError("Both ID train and validation require some observed OP")
    provenance = _jepa_provenance(_provenance(dataset, manifest, options), mode,
                                  ema_protocol(model.ema_decay) if mode == EMA_MODE else None)
    cached = ResidentDataset(train_usable + val_usable, preprocessing, device,
                             include_op=True, include_op_states=True,
                             max_tokens=options['model']['max_devices'], label='JEPA pretrain data')
    phase_indices = {'train': range(len(train_usable)),
                     'val': range(len(train_usable), len(train_usable)+len(val_usable))}
    best, history = math.inf, []
    stopping = EarlyStopping(tr, tr['pretrain_epochs'], PRETRAIN_OBJECTIVE)
    for epoch in range(tr["pretrain_epochs"]):
        records = {}
        for phase, rows in (("train", train_usable), ("val", val_usable)):
            model.train(phase == "train")
            totals = torch.zeros(4, dtype=torch.float64, device=device)
            count = torch.zeros((), dtype=torch.long, device=device)
            progress = BatchProgress(f"pretrain epoch {epoch+1}/{tr['pretrain_epochs']} {phase}",
                                     math.ceil(len(rows)/tr['batch_size']), tr.get('progress_every', 100))
            for step, (_, batch) in enumerate(cached.batches(tr['batch_size'], indices=phase_indices[phase],
                    seed=tr['seed']+epoch if phase == 'train' else None), 1):
                with torch.set_grad_enabled(phase == "train"):
                    a, b, valid = model(batch)
                    if int(valid.sum()) < 2:
                        progress.update(step)
                        continue
                    loss, detail = alignment_loss(a, b, valid, **effective_loss)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite stage-1 objective")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, tr["grad_clip"], error_if_nonfinite=True)
                        optimizer.step()
                        optimizer_steps += 1
                        if mode == EMA_MODE:
                            model.update_ema()
                weight = valid.sum()
                count += weight
                values = torch.stack([loss.detach(), detail['alignment_mse'], detail['variance'], detail['covariance']]).double()
                totals += values * weight
                progress.update(step, loss.detach())
            observed_count = int(count)
            if not observed_count:
                raise ValueError(f"Stage-1 {phase} has no batch with >=2 observed device OP vectors")
            records[phase] = dict(zip(('loss', 'alignment_mse', 'variance', 'covariance'),
                                       (totals / count).tolist()))
            records[phase]["observed_device_tokens"] = observed_count
        history.append({"epoch": epoch + 1, **records})
        print(f"pretrain {epoch+1}/{tr['pretrain_epochs']} train={records['train']['loss']:.6f} val={records['val']['loss']:.6f}", flush=True)
        if stopping.update(records['val']['loss'], epoch + 1):
            best = records["val"]["loss"]
            checkpoint = {**provenance, "stage": "pretrain",
                          "effective_loss_config": copy.deepcopy(effective_loss),
                          "epoch": epoch + 1, "selection": {"split": "val", "objective": "op_alignment_regularized", "value": best},
                          "preprocessing": preprocessing, "model_state": model.state_dict(),
                          "encoder_state": model.encoder.state_dict(),
                          "encoder_digest": state_digest(model.encoder),
                          "skipped_all_op_missing": {"train": len(train)-len(train_usable), "val": len(val)-len(val_usable)}}
            if mode == EMA_MODE:
                checkpoint.update(ema_updates=model.ema_updates, op_encoder_digest=state_digest(model.op_encoder))
            torch.save(checkpoint, output / "pretrain_best.pt")
        if stopping.should_stop:
            stopping.announce('JEPA pretrain')
            break
    selected = load_checkpoint(output / "pretrain_best.pt")
    selected.update(mode=mode, backend=provenance['backend'],
                    early_stopping=stopping.summary(),
                    timings={"pretrain_seconds": time.perf_counter()-started, "supervised_seconds": 0.},
                    training_cost={"pretrain_optimizer_steps": optimizer_steps, "pretrain_examples": len(train_usable),
                                   "pretrain_epochs": len(history), "supervised_epochs": 0,
                                   "total_epochs": len(history), "planned_pretrain_epochs": tr['pretrain_epochs'],
                                   "planned_supervised_epochs": 0, "planned_total_epochs": tr['pretrain_epochs'],
                                   "pretrain_device": str(device),
                                   "tensor_pipeline": PIPELINE, "resident_device": str(device),
                                   "resident_data_bytes": cached.nbytes},
                    parameter_count={"pretrain_total": sum(p.numel() for p in model.parameters()),
                                     "encoder": sum(p.numel() for p in model.encoder.parameters())})
    if mode == EMA_MODE:
        model.verify_ema_state()
        selected['training_cost']['pretrain_ema_updates'] = model.ema_updates
        selected['parameter_count'].update(pretrain_trainable=sum(p.numel() for p in trainable_parameters),
                                          pretrain_ema=sum(p.numel() for p in model.parameters() if not p.requires_grad))
    torch.save(selected, output / "pretrain_best.pt")
    _save_json(output / "pretrain_history.json", history)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to overwrite a different experiment manifest")
    _save_json(manifest_path, manifest)
    _save_json(output / "preprocessing_stage1.json", preprocessing)
    return str(output / "pretrain_best.pt")


def load_checkpoint(path):
    # PyTorch <=2.5.1 has CVE-2025-32434 even under weights_only=True.
    # Guard source-tree runs as well as the declared package dependency.
    version = str(torch.__version__).partition("+")[0]
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None or tuple(map(int, match.groups())) < (2, 6, 0):
        raise RuntimeError("Checkpoint loading requires stable PyTorch >=2.6.0 because older weights_only loaders are unsafe")
    # Only tensors and basic containers; do not load arbitrary pickled objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != 2:
        raise ValueError("Incompatible old checkpoint: shared-structure architecture and signed preprocessing require version 2; re-run pretraining")
    if checkpoint.get("preprocessing", {}).get("preprocessing_version") != 2:
        raise ValueError("Incompatible preprocessing version; do not reuse old scalers")
    prep = checkpoint["preprocessing"]
    contract = prep.get("input_contract_version", 1)
    if contract not in (1, 2, 3) or checkpoint.get("input_contract_version", contract) != contract:
        raise ValueError("Checkpoint input contract version mismatch")
    if contract >= 2 and prep.get("device_category_vocab") != DEVICE_CATEGORY_VOCAB:
        raise ValueError("Checkpoint coarse device-category schema mismatch")
    if contract == 3:
        train_ids = sorted(checkpoint["manifest"]["train"])
        for key, fit_key, reserved in (
            ("model_vocab", "model_vocabulary", {"<PAD>": 0, "<UNK>": 1, "<NONE>": 2}),
            ("port_role_vocab", "port_role_vocabulary", {"<PAD>": 0, "<UNK>": 1}),
        ):
            vocab = prep.get(key)
            if (not isinstance(vocab, dict) or any(vocab.get(k) != v for k, v in reserved.items())
                    or any(type(v) is not int for v in vocab.values())
                    or sorted(vocab.values()) != list(range(len(vocab)))
                    or any(k not in reserved and (not isinstance(k, str) or k != k.lower()) for k in vocab)):
                raise ValueError(f"Checkpoint {key} schema mismatch")
            if prep.get("fit_ids", {}).get(fit_key) != train_ids:
                raise ValueError(f"Checkpoint {key} was not fitted exclusively on the ID training partition")
    if checkpoint.get("mode") in OP_MODES:
        if not checkpoint.get("requires_op_at_inference") or not checkpoint.get("op_inputs_used"):
            raise ValueError("Direct OP checkpoint must declare its training and inference OP requirement")
        expected_design = method_spec(checkpoint['mode']).uses_design_parameters
        if checkpoint.get('uses_design_parameters', True) is not expected_design:
            raise ValueError('Checkpoint uses_design_parameters contradicts its model input contract')
    pretraining = checkpoint.get("pretrain_provenance")
    if pretraining is not None:
        if (pretraining.get("protocol_digest") != _json_digest(pretraining.get("protocol"))
                or pretraining.get("encoder_digest") != checkpoint.get("source_pretrain_digest")
                or pretraining.get("encoder_digest") != checkpoint.get("encoder_digest")
                or pretraining.get("partition_fingerprint") != checkpoint.get("partition_fingerprint")
                or pretraining.get("seed") != checkpoint.get("pretrain_seed")
                or checkpoint.get("decoder_seed") != checkpoint["options"]["training"]["seed"]):
            raise ValueError("Checkpoint pretraining provenance integrity mismatch")
    _validate_ema_checkpoint(checkpoint)
    return checkpoint


def _validate_ema_checkpoint(checkpoint):
    """Keep the new hybrid architecture distinct from legacy JEPA checkpoints."""
    is_ema = checkpoint.get('mode') == EMA_MODE
    if not is_ema:
        if (checkpoint.get('architecture') == EMA_ARCHITECTURE or 'ema' in checkpoint
                or (checkpoint.get('pretrain_provenance') or {}).get('protocol', {}).get('mode') == EMA_MODE):
            raise ValueError('EMA checkpoint identity contradicts its model mode')
        return
    if (checkpoint.get('backend') != EMA_MODE or checkpoint.get('architecture') != EMA_ARCHITECTURE
            or checkpoint.get('requires_op_at_inference') is not False
            or checkpoint.get('shared_structure_layers') != 0):
        raise ValueError('EMA checkpoint architecture or input contract mismatch')
    pretraining = checkpoint.get('pretrain_provenance')
    options = pretraining['source_options'] if pretraining else checkpoint['options']
    expected = ema_protocol(options['training'].get('ema_decay', 0.99))
    if checkpoint.get('ema') != expected:
        raise ValueError('EMA checkpoint protocol differs from executed pretraining options')
    updates = checkpoint.get('ema_updates')
    if type(updates) is not int or updates < 1:
        raise ValueError('EMA checkpoint must record its selected optimizer-step update count')
    cost = checkpoint.get('training_cost', {})
    if cost and (cost.get('pretrain_ema_updates') != cost.get('pretrain_optimizer_steps')
                 or cost.get('pretrain_ema_updates', -1) < updates):
        raise ValueError('EMA checkpoint update count contradicts executed optimizer steps')
    if checkpoint['stage'] == 'pretrain':
        state = checkpoint['model_state']
        for prefix, expected_digest in (('encoder.', checkpoint['encoder_digest']),
                                        ('op_encoder.', checkpoint.get('op_encoder_digest'))):
            partial = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if not partial or _state_dict_digest(partial) != expected_digest:
                raise ValueError('EMA checkpoint model state integrity mismatch')
        if _state_dict_digest(checkpoint['encoder_state']) != checkpoint['encoder_digest']:
            raise ValueError('EMA checkpoint exported encoder integrity mismatch')
    elif checkpoint['stage'] == 'decoder':
        if (not pretraining or pretraining['protocol'].get('mode') != EMA_MODE
                or pretraining['protocol'].get('architecture') != EMA_ARCHITECTURE
                or pretraining['protocol'].get('ema') != expected
                or pretraining.get('selected_ema_updates') != updates):
            raise ValueError('EMA decoder lost its source pretraining protocol')
    else:
        raise ValueError('Unsupported EMA checkpoint stage')


def train_decoder(dataset, pretrain_checkpoint, options, output, device="cpu"):
    started = time.perf_counter()
    optimizer_steps = 0
    device = require_device(device)
    print(f"JEPA decoder setup on {device}: validating source checkpoint", flush=True)
    source = load_checkpoint(pretrain_checkpoint)
    mode = source.get('mode', 'jepa_zerosim')
    if mode not in JEPA_MODES:
        raise ValueError('train_decoder requires a JEPA pretraining checkpoint')
    if source["stage"] != "pretrain":
        raise ValueError("train_decoder requires a stage-1 checkpoint")
    # Labels may be attached after OP pretraining, but no design/OP record,
    # feature schema, or partition may silently change between the stages.
    input_fingerprint = input_op_fingerprint(dataset)
    if input_fingerprint != source["input_op_fingerprint"]:
        raise ValueError("Stage-2 dataset changed immutable design/input/OP provenance; only performance values may change")
    if partition_fingerprint(source["manifest"]) != source["partition_fingerprint"]:
        raise ValueError("Stage-1 checkpoint partition fingerprint mismatch")
    if source["preprocessing"]["fitted_sample_ids"] != source["manifest"]["train"]:
        raise ValueError("Stage-1 preprocessing fitted_sample_ids differ from training partition")
    manifest = copy.deepcopy(source["manifest"])
    manifest["dataset_fingerprint"] = dataset_fingerprint(dataset)
    verify_manifest(dataset, manifest)
    _check_options(options)
    _check_dataset_options(options, manifest)
    if options["model"] != source["model_config"]:
        raise ValueError("Architecture differs from pretraining checkpoint; use the same config")
    tr = options["training"]
    seed_everything(tr["seed"], tr.get("torch_threads", 1))
    prep = copy.deepcopy(source["preprocessing"])
    proposed = _preprocessing_config(options.get('normalization'), dataset['feature_spec'])
    if proposed != prep["transform_config"]:
        raise ValueError("Normalization config differs from pretraining checkpoint; frozen input/OP transforms cannot change")
    if options.get("op_state", {}) != source["options"].get("op_state", {}):
        raise ValueError("OP-state config differs from pretraining checkpoint")
    print(f'Fitting labeled-train-only performance transforms on {device}; reusing frozen input/OP scales', flush=True)
    labeled = sorted(samples_for_split(dataset, manifest, 'labeled_train'), key=lambda row: row['sample_id'])
    arguments = ([(row['circuit_id'], row['performance']) for row in labeled],
                 prep['feature_spec']['metric_names'], proposed['performance'], proposed['weighting'])
    if device.type == 'cuda':
        from .resident import device_statistics
        prep['performance'] = device_statistics(*arguments, device)
    else:
        prep['performance'] = _statistics(*arguments)
    prep['performance_fitted_sample_ids'] = sorted(manifest['labeled_train'])
    prep['fit_ids']['performance'] = sorted(manifest['labeled_train'])
    if not any(c > 0 for c in prep["performance"]["count"]):
        raise ValueError("No requested metric has an observed labeled-train target")
    unobserved_metrics = [name for name, count in zip(prep["feature_spec"]["metric_names"], prep["performance"]["count"]) if not count]
    selection_metrics = _selection_metric_names(prep)
    selected_metric_mask = [count > 0 for count in prep["performance"]["count"]]
    output = _prepare_stage_output(output, "decoder", dataset, manifest, source=source)
    pretrain_provenance = _pretraining_provenance(source)
    pretrain_epochs = executed_epochs(pretrain_provenance['execution'], source['options']['training'],
                                     pretrain_provenance['protocol']['training']['pretrain_epochs'], PRETRAIN_OBJECTIVE)
    if source.get('training_cost', {}).get('pretrain_epochs', pretrain_epochs) != pretrain_epochs:
        raise ValueError('Pretraining epoch cost contradicts its executed stopping provenance')
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
    split_rows = {"train": samples_for_split(dataset, manifest, "labeled_train"),
                  "val": samples_for_split(dataset, manifest, "val")}
    provenance = _jepa_provenance(_provenance(dataset, manifest, options, input_fingerprint),
                                  mode, source.get('ema'))
    if mode == EMA_MODE:
        provenance['ema_updates'] = source['ema_updates']
    cached = ResidentDataset(split_rows['train'] + split_rows['val'], prep, device,
                             include_performance=True, performance_metric_mask=selected_metric_mask,
                             max_tokens=options['model']['max_devices'], label='JEPA decoder data')
    phase_indices = {'train': range(len(split_rows['train'])),
                     'val': range(len(split_rows['train']), len(cached.rows))}
    # Evaluation-mode frozen features are invariant across all decoder epochs.
    # Retain them on the same device; decoder dropout and gradients stay live.
    cached.cache_encoder(model.encoder, tr['batch_size'], tr.get('progress_every', 100))
    frozen_bytes = torch.cat([v.detach().contiguous().reshape(-1).view(torch.uint8)
                              for v in model.encoder.state_dict().values()]).clone()
    history, best = [], math.inf
    stopping = EarlyStopping(tr, tr['decoder_epochs'], SUPERVISED_OBJECTIVE)
    for epoch in range(tr["decoder_epochs"]):
        record = {"epoch": epoch + 1}
        for phase, rows in split_rows.items():
            model.train(phase == "train")
            total = torch.zeros((), dtype=torch.float64, device=device)
            count = torch.zeros((), dtype=torch.long, device=device)
            progress = BatchProgress(f"decoder epoch {epoch+1}/{tr['decoder_epochs']} {phase}",
                                     math.ceil(len(rows)/tr['batch_size']), tr.get('progress_every', 100))
            for step, (_, batch) in enumerate(cached.batches(tr['batch_size'], indices=phase_indices[phase],
                    seed=tr['seed']+epoch if phase == 'train' else None), 1):
                mask = batch['performance_mask']
                if not mask.any():
                    progress.update(step)
                    continue
                with torch.set_grad_enabled(phase == "train"):
                    prediction = model.decoder(batch['encoder_memory'], batch['encoder_padding'])
                    loss = masked_mse(prediction, batch["performance"], mask)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite stage-2 objective")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), tr["grad_clip"], error_if_nonfinite=True)
                        optimizer.step()
                        optimizer_steps += 1
                k = mask.sum()
                total += loss.detach().double() * k
                count += k
                progress.update(step, loss.detach())
            observed_count = int(count)
            if not observed_count:
                raise ValueError(f"No observed trained-metric performance targets in {phase}")
            record[phase] = {"normalized_mse": float(total / count), "observed_targets": observed_count}
        # Includes all parameters AND registered buffers, not just requires_grad.
        current_bytes = torch.cat([v.detach().contiguous().reshape(-1).view(torch.uint8)
                                   for v in model.encoder.state_dict().values()])
        if not torch.equal(current_bytes, frozen_bytes):
            raise AssertionError("Frozen encoder state changed during decoder training")
        if any(p.grad is not None for p in model.encoder.parameters()):
            raise AssertionError("Frozen encoder unexpectedly received gradients")
        history.append(record)
        print(f"decoder {epoch+1}/{tr['decoder_epochs']} train={record['train']['normalized_mse']:.6f} val={record['val']['normalized_mse']:.6f}", flush=True)
        if stopping.update(record['val']['normalized_mse'], epoch + 1):
            best = record["val"]["normalized_mse"]
            checkpoint = {**provenance, "stage": "decoder",
                          "input_contract_version": prep.get("input_contract_version", 1),
                          "epoch": epoch+1, "selection": {"split": "val", "objective": "normalized_performance_mse", "value": best},
                          "preprocessing": prep, "encoder_state": model.encoder.state_dict(),
                          "decoder_state": model.decoder.state_dict(), "encoder_digest": frozen_digest,
                          "source_pretrain_digest": source["encoder_digest"],
                          "source_pretrain_dataset_fingerprint": source["manifest"]["dataset_fingerprint"],
                          "unobserved_training_metrics": unobserved_metrics,
                          "selection_metric_names": selection_metrics,
                          "pretrain_provenance": copy.deepcopy(pretrain_provenance),
                          "pretrain_seed": pretrain_provenance["seed"],
                          "decoder_seed": tr["seed"],
                          "decoder_digest": state_digest(model.decoder),
                          "freeze_verified": True}
            torch.save(checkpoint, output / "decoder_best.pt")
        if stopping.should_stop:
            stopping.announce('JEPA decoder')
            break
    selected = load_checkpoint(output / "decoder_best.pt")
    selected.update(mode=mode, backend=provenance['backend'],
                    early_stopping=stopping.summary(),
                    timings={"pretrain_seconds": source["timings"]["pretrain_seconds"],
                             "supervised_seconds": time.perf_counter()-started},
                    training_cost={**source["training_cost"], "supervised_optimizer_steps": optimizer_steps,
                                   "supervised_device": str(device),
                                   "tensor_pipeline": PIPELINE, "resident_device": str(device),
                                   "resident_data_bytes": cached.nbytes,
                                   "frozen_encoder_cache_bytes": cached.encoded.numel()*cached.encoded.element_size(),
                                   "pretrain_epochs": pretrain_epochs,
                                   "supervised_epochs": len(history),
                                   "total_epochs": pretrain_epochs + len(history),
                                   "planned_pretrain_epochs": pretrain_provenance['protocol']['training']['pretrain_epochs'],
                                   "planned_supervised_epochs": tr['decoder_epochs'],
                                   "planned_total_epochs": pretrain_provenance['protocol']['training']['pretrain_epochs'] + tr['decoder_epochs'],
                                   "labeled_train_examples": len(split_rows["train"])},
                    parameter_count={**{key: value for key, value in source.get('parameter_count', {}).items()
                                        if key.startswith('pretrain_')},
                                     "total": sum(p.numel() for p in model.parameters()),
                                     "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                     "encoder": sum(p.numel() for p in model.encoder.parameters())})
    torch.save(selected, output / "decoder_best.pt")
    _save_json(output / "decoder_history.json", history)
    _save_json(output / "preprocessing_stage2.json", prep)
    _save_json(output / "manifest.json", manifest)
    return str(output / "decoder_best.pt")


def evaluate(dataset, checkpoint_path, split="test", batch_size=32, device="cpu", output=None):
    device = require_device(device)
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
    if source.get("decoder_digest") and state_digest(model.decoder) != source["decoder_digest"]:
        raise AssertionError("Decoder integrity mismatch")
    rows = samples_for_split(dataset, source["manifest"], split)
    predictions, truths, masks = [], [], []
    circuits = [s['circuit_id'] for s in rows]
    normalized_predictions, normalized_targets = [], []
    cached = ResidentDataset(rows, prep, device, include_performance=True,
                             performance_dtype=torch.float64,
                             max_tokens=source['model_config']['max_devices'], label=f'JEPA evaluate {split}')
    progress = BatchProgress(f'JEPA evaluate {split}', math.ceil(len(rows)/batch_size),
                             source['options']['training'].get('progress_every', 100), device)
    with torch.no_grad():
        for step, (_, batch) in enumerate(cached.batches(batch_size), 1):
            normalized = model(batch)
            normalized_predictions.append(normalized)
            normalized_targets.append(batch['performance'])
            raw = denormalize(normalized, prep["performance"])
            predictions.append(raw)
            truths.append(batch['physical_performance'])
            masks.append(batch["performance_mask"])
            progress.update(step)
    if not predictions:
        raise ValueError(f"No examples in {split}")
    pred, target, mask = [torch.cat(values).cpu() for values in (predictions, truths, masks)]
    normalized_predictions = [torch.cat(normalized_predictions).cpu()]
    normalized_targets = [torch.cat(normalized_targets).cpu()]
    report = {"split": split, "synthetic": source["synthetic"], "num_samples": len(rows),
              "selection_used_ood": False, "has_ood_holdout": bool(source["manifest"].get("ood", [])), "encoder_frozen_verified": source["freeze_verified"],
              "unobserved_training_metrics": source.get("unobserved_training_metrics", []),
              "metric_units": "original dataset units; see feature_spec and data provider", "metrics": _metrics(pred, target, mask, prep, torch.cat(normalized_predictions), torch.cat(normalized_targets)),
              "by_circuit": {}}
    for circuit in sorted(set(circuits)):
        selected = torch.tensor([c == circuit for c in circuits])
        report["by_circuit"][circuit] = _metrics(pred[selected], target[selected], mask[selected], prep, torch.cat(normalized_predictions)[selected], torch.cat(normalized_targets)[selected])
    report.update(report_provenance(source, dataset, rows))
    report["predictions"] = prediction_records(pred, target, mask, torch.cat(normalized_predictions),
                                               torch.cat(normalized_targets), rows, prep)
    if output:
        _save_json(output, report)
    return report


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def report_provenance(source, dataset, rows):
    """Shared report contract with explicit training/inference OP information."""
    prep, manifest = source["preprocessing"], source["manifest"]
    target_digest = _json_digest({"metric_names": prep["feature_spec"]["metric_names"],
                                  "statistics": prep["performance"]})
    performance_digest = _json_digest([{ "sample_id": row["sample_id"], "performance": row["performance"]}
                                       for row in sorted(dataset["samples"], key=lambda r: r["sample_id"])])
    mode = source.get("mode", "jepa_zerosim")
    uses_op_input = mode in OP_MODES
    uses_design = method_spec(mode).uses_design_parameters
    uses_op = uses_op_input or mode in JEPA_MODES
    op_information = {
        "training_use": "input" if uses_op_input else "pretraining_target" if uses_op else "none",
        "inference_use": "input" if uses_op_input else "none",
        "source": "dataset_dc_operating_point" if uses_op else None,
        "feature_names": prep["feature_spec"]["op_names"] if uses_op else [],
        "state_features": prep.get("op_state_features", []) if mode in JEPA_MODES else [],
        "state_config": copy.deepcopy(source["options"].get("op_state", {"features": []})) if mode in JEPA_MODES else {},
        "eligibility": "is_op_target; MOS by default; current adapter has no passive/source OP",
        "scaler_fitted_sample_ids": prep["fit_ids"]["op"] if uses_op else [],
        "acquisition_cost_seconds": None,
        "acquisition_cost_status": "not_measured",
    }
    if uses_op_input:
        names = prep["feature_spec"]["op_names"]
        devices = [d for row in rows for d in row["devices"] if is_op_target(d)]
        op_information["evaluation_coverage"] = {
            "counting_unit": "original devices, not repeated pin tokens",
            "eligible_devices": len(devices),
            "observed_devices": sum(any(d.get("op", {}).get(name) is not None for name in names) for d in devices),
            "observed_values": sum(d.get("op", {}).get(name) is not None for d in devices for name in names),
            "possible_values": len(devices) * len(names),
            "samples_without_observed_op": sum(not any(d.get("op", {}).get(name) is not None
                for d in row["devices"] if is_op_target(d) for name in names) for row in rows),
        }
    backend = source.get("backend", "shared_structure_jepa_v2")
    timings = source.get("timings", {"pretrain_seconds": 0., "supervised_seconds": 0.})
    provenance = {
        "mode": mode, "backend": backend, "seed": source["options"]["training"]["seed"],
        "requires_op_at_inference": uses_op_input, "op_information": op_information,
        "uses_design_parameters": uses_design,
        "input_contract_version": prep.get("input_contract_version", 1),
        "tensor_pipeline": source.get('tensor_pipeline', 'legacy_scalar'),
        "op_input_transform_digest": _json_digest({"op_names": prep["feature_spec"]["op_names"], "op": prep["op"]}),
        "partition_fingerprint": source["partition_fingerprint"],
        "input_op_fingerprint": source["input_op_fingerprint"],
        "performance_fingerprint": performance_digest, "target_scaler_digest": target_digest,
        "preprocessing_digest": _json_digest(prep),
        "input_transform_digest": _json_digest({"parameters": prep["parameters"],
            "device_vocab": prep["device_vocab"],
            "model_vocab": prep.get("model_vocab", {}),
            "port_role_vocab": prep.get("port_role_vocab", {}),
            "device_category_vocab": prep.get("device_category_vocab", {}),
            "input_contract_version": prep.get("input_contract_version", 1),
            "parameter_names": prep["feature_spec"]["parameter_names"],
            "terminal_names": prep["feature_spec"]["terminal_names"]}),
        "labeled_train_ids": manifest["labeled_train"],
        "train_ids": manifest["train"], "training_config": source["options"]["training"],
        "model_config": source["model_config"],
        "parameter_count": source.get("parameter_count", {}), "timing": timings,
        "training_cost": source.get("training_cost", {}),
        "early_stopping": copy.deepcopy(source.get('early_stopping')),
        "selection_metric_names": source.get("selection_metric_names"),
        "pretraining_protocol_digest": source.get("pretrain_provenance", {}).get("protocol_digest"),
        "pretrain_provenance": copy.deepcopy(source.get("pretrain_provenance")),
        "pretrain_seed": source.get("pretrain_seed"),
        "decoder_seed": source.get("decoder_seed", source["options"]["training"]["seed"]),
        "run_identity": _json_digest({"checkpoint_run_id": source.get("checkpoint_run_id"),
            "stage": source["stage"], "mode": mode, "seed": source["options"]["training"]["seed"],
            "encoder_digest": source["encoder_digest"],
            "decoder_digest": source.get("decoder_digest"),
            "input_op_fingerprint": source["input_op_fingerprint"],
            "partition_fingerprint": source["partition_fingerprint"],
            "model_config": source["model_config"],
            "pretrain_source": source.get("source_pretrain_digest")}) if source.get("checkpoint_run_id") else None,
    }
    if mode == EMA_MODE:
        provenance.update(architecture=source['architecture'], ema=copy.deepcopy(source['ema']),
                          selected_ema_updates=source['ema_updates'])
    return {"mode": mode, "backend": backend, "seed": provenance["seed"],
            "requires_op_at_inference": uses_op_input, "op_information": op_information,
            "uses_design_parameters": uses_design,
            "data_fingerprint": manifest["dataset_fingerprint"],
            "partition_fingerprint": source["partition_fingerprint"],
            "metric_preprocessing_fingerprint": target_digest,
            "labeled_train_ids": manifest["labeled_train"], "evaluation_ids": [r["sample_id"] for r in rows],
            "metric_names": prep["feature_spec"]["metric_names"], "timings": timings,
            "provenance": provenance}


def prediction_records(pred, target, mask, normalized_pred, normalized_target, rows, prep):
    names = prep["feature_spec"]["metric_names"]
    return [{"sample_id": row["sample_id"], "circuit_id": row["circuit_id"],
             "prediction": {name: float(pred[i, j]) for j, name in enumerate(names)},
             "target": {name: row["performance"].get(name) if bool(mask[i, j]) else None for j, name in enumerate(names)},
             "normalized_prediction": {name: float(normalized_pred[i, j]) for j, name in enumerate(names)},
             "normalized_target": {name: float(normalized_target[i, j]) if bool(mask[i, j]) else None for j, name in enumerate(names)}}
            for i, row in enumerate(rows)]


def _metrics(pred, target, mask, prep, normalized_pred=None, normalized_target=None):
    # Use the same numerical metric implementation as standalone comparison.
    from .comparison import regression_metrics
    from .data import transform_value
    result = {}
    for j, name in enumerate(prep["feature_spec"]["metric_names"]):
        selected = mask[:, j]
        ys = target[selected, j].double().tolist()
        ps = pred[selected, j].double().tolist()
        metric = regression_metrics(ys, ps)
        if normalized_pred is not None and normalized_target is not None:
            normalized_ys = normalized_target[selected, j].double().tolist()
            normalized_ps = normalized_pred[selected, j].double().tolist()
        else:
            normalized_ys = [transform_value(y, prep["performance"], j) for y in ys]
            normalized_ps = [transform_value(p, prep["performance"], j) for p in ps]
        metric["normalized_rmse"] = regression_metrics(normalized_ys, normalized_ps)["rmse"]
        result[name] = metric
    return result
