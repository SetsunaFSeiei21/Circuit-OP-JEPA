"""Validated device-node datasets and leakage-resistant split/preprocessing utilities.

Only the Python standard library is required. Parameter and OP units are kept as
supplied; callers must use consistent units for every named feature. ``null`` or
an omitted declared feature means missing, never a measured zero. Synthetic
labels provided here are analytic demonstrations, not circuit simulation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


TERMINAL_NAMES = ["d", "g", "s", "b", "p", "n"]
SPLITS = ("train", "val", "test", "ood")
MANIFEST_VERSION = 2
PREPROCESSING_VERSION = 2
OP_GEOMETRY_FIELDS = frozenset({"w_um", "l_um", "multiplicity"})


def _nonempty_string(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string")


def _finite_number(value: Any, location: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number or null; encode missing/failed values as JSON null, not NaN/Infinity")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError):
        finite = False
    if not finite:
        raise ValueError(f"{location} must be a finite number or null; encode missing/failed values as JSON null, not NaN/Infinity")


def _json_value(value: Any, location: str) -> None:
    """Reject non-JSON objects and nonfinite numbers, including in context."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _finite_number(value, location)
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _json_value(item, f"{location}[{i}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _json_value(item, f"{location}.{key}")
        return
    raise ValueError(f"{location} must contain only JSON-compatible values")


def _features(mapping: Any, names: list[str], location: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{location} must be an object")
    unknown = set(mapping) - set(names)
    if unknown:
        raise ValueError(f"{location} has undeclared features: {sorted(unknown)}")
    for key, value in mapping.items():
        _finite_number(value, f"{location}.{key}")


def validate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Validate the schema in place and return the original, unmodified object.

    All samples require at least one device. Device IDs are unique within each
    sample; sample IDs are unique globally. Extra JSON metadata is preserved.
    Feature names are ordered and unique, and all six canonical terminal roles
    must be declared, even if individual devices use only some of them. Context
    must be identical across the dataset: it is metadata, not a model input.
    """
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    _json_value(dataset, "dataset")
    if type(dataset.get("schema_version")) is not int or dataset["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    spec = dataset.get("feature_spec")
    if not isinstance(spec, dict):
        raise ValueError("feature_spec must be an object")
    for key in ("parameter_names", "op_names", "metric_names", "terminal_names"):
        names = spec.get(key)
        if not isinstance(names, list) or not names:
            raise ValueError(f"feature_spec.{key} must be a nonempty list")
        for name in names:
            _nonempty_string(name, f"feature_spec.{key} item")
        if len(names) != len(set(names)):
            raise ValueError(f"feature_spec.{key} must contain unique names")
    if spec["terminal_names"] != TERMINAL_NAMES:
        raise ValueError(f"terminal_names must be {TERMINAL_NAMES!r} in that order")
    samples = dataset.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a nonempty list")
    seen_samples: set[str] = set()
    for i, sample in enumerate(samples):
        loc = f"samples[{i}]"
        if not isinstance(sample, dict):
            raise ValueError(f"{loc} must be an object")
        for key in ("sample_id", "circuit_id"):
            _nonempty_string(sample.get(key), f"{loc}.{key}")
        if "design_id" not in sample and "design_key" not in sample:
            raise ValueError(f"{loc} requires a stable design_id or design_key")
        for key in ("design_id", "design_key"):
            if key in sample:
                _nonempty_string(sample[key], f"{loc}.{key}")
        if sample["sample_id"] in seen_samples:
            raise ValueError(f"duplicate sample_id: {sample['sample_id']}")
        seen_samples.add(sample["sample_id"])
        if "group_id" in sample:
            _nonempty_string(sample["group_id"], f"{loc}.group_id")
        if "context" in sample and not isinstance(sample["context"], dict):
            raise ValueError(f"{loc}.context must be an object")
        if "ports" in sample:
            if not isinstance(sample["ports"], dict):
                raise ValueError(f"{loc}.ports must be an object mapping external port names to nets")
            for port, net in sample["ports"].items():
                _nonempty_string(port, f"{loc}.ports name")
                _nonempty_string(net, f"{loc}.ports net")
        if "op_status" in sample and sample["op_status"] not in ("complete", "partial", "failed"):
            raise ValueError(f"{loc}.op_status must be complete, partial, or failed")
        devices = sample.get("devices")
        if not isinstance(devices, list) or not devices:
            raise ValueError(f"{loc}.devices must be a nonempty list")
        seen_devices: set[str] = set()
        for j, device in enumerate(devices):
            dloc = f"{loc}.devices[{j}]"
            if not isinstance(device, dict):
                raise ValueError(f"{dloc} must be an object")
            for key in ("device_id", "device_type"):
                _nonempty_string(device.get(key), f"{dloc}.{key}")
            if "op_target" in device and type(device["op_target"]) is not bool:
                raise ValueError(f"{dloc}.op_target must be a boolean")
            if device["device_type"] in ("<PAD>", "<UNK>"):
                raise ValueError(f"{dloc}.device_type uses a reserved vocabulary token")
            if device["device_id"] in seen_devices:
                raise ValueError(f"duplicate device_id in {loc}: {device['device_id']}")
            seen_devices.add(device["device_id"])
            _features(device.get("parameters"), spec["parameter_names"], f"{dloc}.parameters")
            _features(device.get("op"), spec["op_names"], f"{dloc}.op")
            terminals = device.get("terminals")
            if not isinstance(terminals, dict) or not terminals:
                raise ValueError(f"{dloc}.terminals must be a nonempty object")
            if set(terminals) - set(TERMINAL_NAMES):
                raise ValueError(f"{dloc}.terminals has undeclared terminal roles")
            for net in terminals.values():
                _nonempty_string(net, f"{dloc}.terminals net")
        _features(sample.get("performance"), spec["metric_names"], f"{loc}.performance")
    contexts = {_canonical(sample.get("context", {})) for sample in samples}
    if len(contexts) > 1:
        raise ValueError(
            "sample context must be fixed across the dataset: context is metadata, not a model input; "
            "use one fixed condition or explicitly encode varying conditions as model features in a future converter"
        )
    return dataset


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load UTF-8 JSON, validate it, and preserve context/terminal metadata."""
    with Path(path).open("r", encoding="utf-8") as handle:
        dataset = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    return validate_dataset(dataset)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def dataset_fingerprint(dataset: dict[str, Any]) -> str:
    """Hash all content, including labels, independent of sample/device order.

    This integrity digest is not a model feature and does not fit statistics or
    select splits using labels. Feature-list order remains significant.
    """
    ordered = dict(dataset)
    ordered["samples"] = []
    for sample in sorted(dataset["samples"], key=lambda item: item["sample_id"]):
        row = dict(sample)
        row["devices"] = sorted(sample["devices"], key=lambda item: item["device_id"])
        ordered["samples"].append(row)
    return hashlib.sha256(_canonical(ordered).encode("utf-8")).hexdigest()


def input_op_fingerprint(dataset: dict[str, Any]) -> str:
    """Immutable stage-one provenance, excluding only performance values.

    Schema/identities/design inputs/OP/masks/metadata remain integrity checked.
    Empty performance dictionaries may later be populated without changing this
    digest. Feature order and declared metric names remain fixed.
    """
    unlabeled = dict(dataset)
    unlabeled["samples"] = []
    for sample in dataset["samples"]:
        row = dict(sample)
        row["performance"] = {}
        unlabeled["samples"].append(row)
    return dataset_fingerprint(unlabeled)


def partition_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash experiment membership, including the supervised training subset.

    Performance observations never enter this immutable partition provenance.
    """
    content = {key: sorted(manifest[key]) for key in SPLITS}
    content["labeled_train"] = sorted(manifest.get("labeled_train", manifest["train"]))
    content["ood_circuits"] = sorted(manifest.get("ood_circuits", [manifest.get("ood_circuit")]))
    content["train_circuits"] = sorted(manifest.get("train_circuits", manifest["id_circuits"]))
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


def design_fingerprint(sample: dict[str, Any], feature_spec: dict[str, Any]) -> str:
    """Hash exact parameter/type/connectivity/context content, excluding labels.

    Sample/design/group/device IDs and device ordering are ignored. Net names
    remain significant: this is exact duplicate detection, not a graph-
    isomorphism test. OP/performance values do not influence this fingerprint.
    """
    devices = []
    for device in sample["devices"]:
        devices.append({
            "device_type": device["device_type"],
            "parameters": {
                name: None if device["parameters"].get(name) is None else float(device["parameters"][name])
                for name in feature_spec["parameter_names"]
            },
            "terminals": device["terminals"],
        })
    content = {"devices": sorted(devices, key=_canonical), "context": sample.get("context", {}),
               "ports": sample.get("ports", {})}
    return hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()


class _UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {key: key for key in ids}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while key != root:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def _design_groups(dataset: dict[str, Any]) -> dict[str, list[list[str]]]:
    samples = sorted(dataset["samples"], key=lambda item: item["sample_id"])
    uf = _UnionFind([sample["sample_id"] for sample in samples])
    seen_designs: dict[tuple[str, str, str], str] = {}
    seen_groups: dict[tuple[str, str], str] = {}
    seen_fingerprints: dict[str, tuple[str, str]] = {}
    for sample in samples:
        sid, circuit = sample["sample_id"], sample["circuit_id"]
        for field in ("design_id", "design_key"):
            if field in sample:
                design_key = (circuit, field, sample[field])
                if design_key in seen_designs:
                    uf.union(sid, seen_designs[design_key])
                else:
                    seen_designs[design_key] = sid
        if "group_id" in sample:
            key = (circuit, sample["group_id"])
            if key in seen_groups:
                uf.union(sid, seen_groups[key])
            else:
                seen_groups[key] = sid
        fingerprint = design_fingerprint(sample, dataset["feature_spec"])
        if fingerprint in seen_fingerprints:
            other_circuit, other_sid = seen_fingerprints[fingerprint]
            if circuit != other_circuit:
                raise ValueError(
                    f"exact duplicate designs cross circuit IDs ({other_sid!r}, {sid!r}); "
                    "resolve circuit identity before splitting to prevent leakage"
                )
            uf.union(sid, other_sid)
        else:
            seen_fingerprints[fingerprint] = (circuit, sid)
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[sample["circuit_id"]][uf.find(sample["sample_id"])].append(sample["sample_id"])
    return {circuit: sorted((sorted(ids) for ids in groups.values()), key=lambda ids: ids[0])
            for circuit, groups in sorted(grouped.items())}


def _positive_fraction(value: Any, location: str, allow_one: bool = False) -> float:
    upper_ok = value <= 1 if isinstance(value, (int, float)) and not isinstance(value, bool) else False
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 or not upper_ok or (value == 1 and not allow_one)):
        raise ValueError(f"{location} must be a finite positive fraction {'at most' if allow_one else 'less than'} one")
    return float(value)


def _circuit_list(value: Any, location: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{location} must be an explicit {'possibly empty ' if allow_empty else 'nonempty '}list")
    for item in value:
        _nonempty_string(item, location)
    if len(set(value)) != len(value):
        raise ValueError(f"{location} must not contain duplicates")
    return sorted(value)


def _manifest_config(dataset, ood_circuit, seed, val_fraction, test_fraction, config):
    circuits = {row["circuit_id"] for row in dataset["samples"]}
    if config is None:
        _nonempty_string(ood_circuit, "ood_circuit (or supply an explicit dataset config)")
        train_circuits, ood_circuits = sorted(circuits - {ood_circuit}), [ood_circuit]
        val = 0.1 if val_fraction is None else _positive_fraction(val_fraction, "val_fraction")
        test = 0.1 if test_fraction is None else _positive_fraction(test_fraction, "test_fraction")
        if val + test >= 1:
            raise ValueError("val_fraction + test_fraction must be less than one")
        ratios, label_fraction = [1 - val - test, val, test], 1.0
    else:
        if not isinstance(config, dict):
            raise ValueError("dataset config must be an object")
        train_circuits = _circuit_list(config.get("train_circuits"), "train_circuits")
        ood_circuits = _circuit_list(config.get("ood_circuits"), "ood_circuits", allow_empty=True)
        if ood_circuit is not None and ood_circuits != [ood_circuit]:
            raise ValueError("ood_circuit conflicts with explicit ood_circuits")
        if val_fraction is not None or test_fraction is not None:
            raise ValueError("use split_ratios, not legacy fractions, with dataset config")
        ratios = config.get("split_ratios", [0.8, 0.1, 0.1])
        if (not isinstance(ratios, (list, tuple)) or len(ratios) != 3
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or not 0 <= x <= 1 for x in ratios)
                or ratios[0] <= 0 or not math.isclose(math.fsum(ratios), 1.0, abs_tol=1e-12, rel_tol=0)):
            raise ValueError("split_ratios must be three finite nonnegative fractions summing to one, with positive train")
        ratios = [float(x) for x in ratios]
        seed = config.get("split_seed", seed)
        label_fraction = _positive_fraction(config.get("finetune_label_fraction", 1.0), "finetune_label_fraction", True)
    if type(seed) is not int:
        raise ValueError("split_seed/seed must be an integer")
    if not train_circuits or set(train_circuits) & set(ood_circuits):
        raise ValueError("train_circuits must be nonempty and disjoint from ood_circuits")
    if set(train_circuits) | set(ood_circuits) != circuits:
        raise ValueError("train_circuits and ood_circuits must exhaustively match dataset circuits")
    return train_circuits, ood_circuits, ratios, seed, label_fraction


def _group_rank(seed: int, circuit: str, ids: list[str], purpose: str) -> str:
    return hashlib.sha256(_canonical([seed, circuit, purpose, ids]).encode("utf-8")).hexdigest()


def _group_counts(n_groups: int, ratios: list[float]) -> list[int]:
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if n_groups < len(active):
        raise ValueError(f"needs at least {len(active)} independent design groups; found {n_groups}")
    # Fractions count independent groups, not repeated rows. Guarantee each
    # requested split at least one group, then give the rounding remainder to train.
    counts = [0, max(1, int(n_groups * ratios[1])) if ratios[1] else 0,
              max(1, int(n_groups * ratios[2])) if ratios[2] else 0]
    while counts[1] + counts[2] >= n_groups:
        largest = max((i for i in (1, 2) if counts[i] > 1), key=lambda i: (counts[i], -i), default=None)
        if largest is None:
            raise ValueError("not enough independent groups for nonempty splits")
        counts[largest] -= 1
    counts[0] = n_groups - counts[1] - counts[2]
    return counts


def _labeled_ids(groups, train_ids, circuits, fraction, seed):
    selected = []
    train_ids = set(train_ids)
    for circuit in circuits:
        eligible = [group for group in groups[circuit] if set(group) <= train_ids]
        ranked = sorted(eligible, key=lambda ids: (_group_rank(seed, circuit, ids, "labeled_train"), ids))
        # A positive label fraction always selects one whole design group per
        # ID circuit. Rounding is down except for that minimum, never row-wise.
        count = max(1, int(len(ranked) * fraction))
        selected.extend(sid for group in ranked[:count] for sid in group)
    return sorted(selected)


def build_manifest(
    dataset: dict[str, Any], ood_circuit: str | None = None, seed: int = 17,
    val_fraction: float | None = None, test_fraction: float | None = None,
    *, config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a NEW v2 grouped experiment partition, never update an old one.

    Prefer explicit ``config={train_circuits: [...], ood_circuits: [...],
    split_ratios: [.8,.1,.1], split_seed: 17, finetune_label_fraction: 1}``.
    Empty OOD is supported only by explicitly passing ``ood_circuits=[]``;
    this makes no OOD generalization claim. The legacy single-OOD argument
    infers the other circuits. Ratios and label fractions apply to independent
    design groups per circuit, not rows; repeated conditions stay together.

    Persist and verify this manifest for every subsequent run. Any changed or
    added dataset content fails verification. Calling this function explicitly
    creates a new experiment, and must not silently overwrite a saved split.
    """
    validate_dataset(dataset)
    train_circuits, ood_circuits, ratios, seed, fraction = _manifest_config(
        dataset, ood_circuit, seed, val_fraction, test_fraction, config)
    groups = _design_groups(dataset)
    manifest: dict[str, Any] = {split: [] for split in SPLITS}
    manifest.update({
        "schema_version": MANIFEST_VERSION,
        "train_circuits": train_circuits, "id_circuits": list(train_circuits),
        "ood_circuits": ood_circuits, "ood_circuit": ood_circuits[0] if len(ood_circuits) == 1 else None,
        "split_ratios": ratios, "split_seed": seed, "seed": seed,
        "val_fraction": ratios[1], "test_fraction": ratios[2],
        "finetune_label_fraction": fraction,
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "grouping": "circuit-scoped design_key/design_id/group_id plus exact design fingerprint",
    })
    for circuit in train_circuits:
        try:
            _, n_val, n_test = _group_counts(len(groups[circuit]), ratios)
        except ValueError as exc:
            raise ValueError(f"ID circuit {circuit!r} {exc}") from exc
        ranked = sorted(groups[circuit], key=lambda ids: (_group_rank(seed, circuit, ids, "split"), ids))
        for split, subset in (("val", ranked[:n_val]), ("test", ranked[n_val:n_val + n_test]),
                              ("train", ranked[n_val + n_test:])):
            manifest[split].extend(sid for group in subset for sid in group)
    manifest["ood"] = [sid for circuit in ood_circuits for group in groups[circuit] for sid in group]
    for split in SPLITS:
        manifest[split].sort()
    manifest["labeled_train"] = _labeled_ids(groups, manifest["train"], train_circuits, fraction, seed)
    verify_manifest(dataset, manifest)
    return manifest


def verify_manifest(dataset: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Reject stale content, invalid circuit membership, leakage or label subsets.

    Dataset hashing is strict (including labels); adding even one sample does
    not trigger a resplit. Stage two may explicitly refresh ONLY the label
    integrity digest after separately checking immutable input/OP provenance.
    """
    validate_dataset(dataset)
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != MANIFEST_VERSION):
        raise ValueError("manifest schema_version must be 2; create a new explicit experiment")
    if manifest.get("dataset_fingerprint") != dataset_fingerprint(dataset):
        raise ValueError("manifest dataset_fingerprint does not match dataset; do not silently resplit")
    required = {"train_circuits", "ood_circuits", "split_ratios", "split_seed", "finetune_label_fraction"}
    if required - set(manifest):
        raise ValueError(f"manifest is missing v2 configuration fields: {sorted(required - set(manifest))}")
    train_circuits, ood_circuits, ratios, seed, fraction = _manifest_config(
        dataset, None, 17, None, None, manifest)
    if (manifest.get("train_circuits") != train_circuits or manifest.get("ood_circuits") != ood_circuits
            or manifest.get("id_circuits") != train_circuits):
        raise ValueError("manifest circuit lists must be sorted, explicit and consistent")
    expected_ood = ood_circuits[0] if len(ood_circuits) == 1 else None
    if manifest.get("ood_circuit") != expected_ood or manifest.get("seed") != seed:
        raise ValueError("manifest legacy circuit/seed aliases disagree")
    if manifest.get("val_fraction") != ratios[1] or manifest.get("test_fraction") != ratios[2]:
        raise ValueError("manifest fraction aliases disagree with split_ratios")
    samples = {sample["sample_id"]: sample for sample in dataset["samples"]}
    assignment: dict[str, str] = {}
    for split in SPLITS:
        ids = manifest.get(split)
        if not isinstance(ids, list):
            raise ValueError(f"manifest {split} must be a list")
        for sid in ids:
            if not isinstance(sid, str) or sid not in samples:
                raise ValueError(f"manifest {split} contains an unknown sample ID: {sid!r}")
            if sid in assignment:
                raise ValueError(f"sample {sid!r} appears more than once in manifest")
            assignment[sid] = split
            is_ood = samples[sid]["circuit_id"] in ood_circuits
            if is_ood != (split == "ood"):
                raise ValueError(f"sample {sid!r} violates ID/OOD circuit separation")
        if split != "ood":
            requested = ratios[("train", "val", "test").index(split)] > 0
            if requested and {samples[sid]["circuit_id"] for sid in ids} != set(train_circuits):
                raise ValueError(f"manifest {split} must contain samples from every ID circuit")
            if not requested and ids:
                raise ValueError(f"manifest {split} must be empty for its zero split ratio")
    if set(assignment) != set(samples):
        raise ValueError("manifest must contain every dataset sample exactly once")
    groups = _design_groups(dataset)
    for circuit_groups in groups.values():
        for group in circuit_groups:
            if len({assignment[sid] for sid in group}) != 1:
                raise ValueError(f"design group is split across subsets: {group!r}")
    # Reject editing whole groups across splits as well as splitting a group:
    # a saved experiment must continue to match its explicit seed and ratios.
    for circuit in train_circuits:
        _, n_val, n_test = _group_counts(len(groups[circuit]), ratios)
        ranked = sorted(groups[circuit], key=lambda ids: (_group_rank(seed, circuit, ids, "split"), ids))
        for split, subset in (("val", ranked[:n_val]), ("test", ranked[n_val:n_val + n_test]),
                              ("train", ranked[n_val + n_test:])):
            expected_ids = {sid for group in subset for sid in group}
            actual_ids = {sid for sid in manifest[split] if samples[sid]["circuit_id"] == circuit}
            if actual_ids != expected_ids:
                raise ValueError(f"manifest {split} does not match deterministic grouped split config for {circuit!r}")
    labeled = manifest.get("labeled_train")
    if (not isinstance(labeled, list) or any(not isinstance(sid, str) for sid in labeled)
            or len(set(labeled)) != len(labeled) or not set(labeled) <= set(manifest["train"])):
        raise ValueError("labeled_train must be a unique subset of train sample IDs")
    expected = _labeled_ids(groups, manifest["train"], train_circuits, fraction, seed)
    if sorted(labeled) != expected:
        raise ValueError("labeled_train must be the deterministic grouped training-only subset")


def samples_for_split(dataset: dict[str, Any], manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Return original sample objects in manifest order after verifying safety."""
    if split not in (*SPLITS, "labeled_train"):
        raise ValueError(f"unknown split {split!r}; expected one of {(*SPLITS, 'labeled_train')}")
    verify_manifest(dataset, manifest)
    lookup = {sample["sample_id"]: sample for sample in dataset["samples"]}
    return [lookup[sid] for sid in manifest[split]]


def is_op_target(device: dict[str, Any]) -> bool:
    """Explicit eligibility wins; otherwise only case-insensitive NMOS/PMOS."""
    return device.get("op_target", device["device_type"].lower() in {"nmos", "pmos"})


def _default_kind(collection: str, name: str) -> str:
    if collection != "op":
        return "linear"
    lower = name.lower()
    if lower.endswith("_v_v") or lower.endswith("_v_inv") or "ratio" in lower:
        return "asinh"
    if lower.endswith("_v") or lower in {"vds", "vgs", "vbs", "vth", "vdsat", "vd", "vg", "vs", "vb"}:
        return "linear"
    return "asinh"


def _preprocessing_config(config, spec):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("preprocessing config must be an object")
    config = copy.deepcopy(config)
    # Accept either a focused preprocessing block or a full application config.
    if "preprocessing" in config:
        config = config["preprocessing"]
    if not isinstance(config, dict):
        raise ValueError("preprocessing config must be an object")
    normalization = config.get("normalization", config)
    if not isinstance(normalization, dict):
        raise ValueError("normalization must be an object")
    weighting = config.get("weighting", normalization.get("weighting", "equal_circuit"))
    if weighting not in ("equal_circuit", "observation"):
        raise ValueError("weighting must be equal_circuit or observation")
    fields = config.get("op_fields")
    if fields is None:
        fields = [name for name in spec["op_names"] if name.lower() not in OP_GEOMETRY_FIELDS]
    if (not isinstance(fields, list) or not fields or any(not isinstance(name, str) for name in fields)
            or len(fields) != len(set(fields)) or not set(fields) <= set(spec["op_names"])):
        raise ValueError("op_fields must be a nonempty unique subset of raw feature_spec.op_names")
    result = {"op_fields": list(fields), "weighting": weighting}
    for collection, names in (("parameters", spec["parameter_names"]), ("op", fields), ("performance", spec["metric_names"])):
        rules = normalization.get(collection, {})
        if not isinstance(rules, dict):
            raise ValueError(f"normalization.{collection} must map fields to rules")
        declared_names = spec["op_names"] if collection == "op" else names
        if set(rules) - set(declared_names):
            raise ValueError(f"normalization.{collection} has undeclared fields")
        result[collection] = {}
        for name in names:
            rule = rules.get(name, {})
            if not isinstance(rule, dict) or set(rule) - {"kind", "scale"}:
                raise ValueError(f"normalization.{collection}.{name} must contain only kind/scale")
            kind = rule.get("kind", _default_kind(collection, name))
            scale = rule.get("scale", "fit" if kind == "asinh" else 1.0)
            if kind not in ("linear", "asinh"):
                raise ValueError(f"normalization.{collection}.{name}.kind must be linear or asinh")
            if scale != "fit" and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
                raise ValueError(f"normalization.{collection}.{name}.scale must be fit or finite positive")
            result[collection][name] = {"kind": kind, "scale": scale}
    return result


def _weighted_values(rows, name, weighting, nonzero=False):
    by_circuit = defaultdict(list)
    for circuit, row in rows:
        if row.get(name) is not None and (not nonzero or row[name] != 0):
            by_circuit[circuit].append(float(row[name]))
    count = sum(len(values) for values in by_circuit.values())
    result = []
    for circuit in sorted(by_circuit):
        values = by_circuit[circuit]
        weight = 1.0 / (len(by_circuit) * len(values)) if weighting == "equal_circuit" else 1.0 / count
        result.extend((value, weight) for value in values)
    return result, len(by_circuit)


def _fit_asinh_scale(rows, name, weighting):
    values, _ = _weighted_values(rows, name, weighting, nonzero=True)
    if not values:
        return 1.0
    ordered = sorted((abs(value), weight) for value, weight in values)
    total = math.fsum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= total * 0.5 - 1e-15:
            return value
    return ordered[-1][0]


def _asinh_scaled(value: float, scale: float) -> float:
    ratio = value / scale
    if math.isfinite(ratio):
        return math.asinh(ratio)
    # Finite x and s can have an overflowing x/s. In this large-ratio regime
    # asinh(x/s) = sign(x) * (log(abs(x))-log(s)+log(2)) to float precision.
    return math.copysign(math.log(abs(value)) - math.log(scale) + math.log(2.0), value)


def _statistics(rows, names, rules, weighting):
    result = {"mean": [], "std": [], "count": [], "kind": [], "asinh_scale": [],
              "circuit_count": [], "version": PREPROCESSING_VERSION, "weighting": weighting}
    for name in names:
        rule = rules[name]
        kind = rule["kind"]
        scale = (_fit_asinh_scale(rows, name, weighting) if rule["scale"] == "fit" else float(rule["scale"])) if kind == "asinh" else 1.0
        observations, n_circuits = _weighted_values(rows, name, weighting)
        values = [(_asinh_scaled(value, scale) if kind == "asinh" else value, weight)
                  for value, weight in observations]
        mean, std = 0.0, 1.0
        magnitude = max((abs(value) for value, _ in values), default=0.0)
        if magnitude:
            total_weight = math.fsum(weight for _, weight in values)
            normalized = [(value / magnitude, weight) for value, weight in values]
            relative_mean = math.fsum(value * weight for value, weight in normalized) / total_weight
            mean = relative_mean * magnitude
            relative_std = math.sqrt(math.fsum(weight * (value - relative_mean) ** 2 for value, weight in normalized) / total_weight)
            std = relative_std * magnitude
            if std == 0.0:
                std = 1.0
        result["mean"].append(mean)
        result["std"].append(std)
        result["count"].append(len(observations))
        result["kind"].append(kind)
        result["asinh_scale"].append(scale)
        result["circuit_count"].append(n_circuits)
    return result


def transform_value(value: float, stats: dict[str, Any], feature_index: int) -> float:
    """Physics transform then z-score a finite scalar, preserving its polarity.

    Null handling/masks belong to the caller. Standard-library float arithmetic
    avoids premature tensor float32 underflow for e.g. picoamp/femtofarad data.
    """
    _finite_number(value, "transform value")
    if value is None:
        raise ValueError("transform value must not be null; use the observation mask")
    i = feature_index
    transformed = float(value)
    kind = stats.get("kind", ["linear"] * len(stats["mean"]))[i]
    if kind == "asinh":
        transformed = _asinh_scaled(transformed, stats["asinh_scale"][i])
    mean, std = stats["mean"][i], stats["std"][i]
    difference = transformed - mean
    result = difference / std if math.isfinite(difference) else transformed / std - mean / std
    if not math.isfinite(result):
        raise ValueError("non-finite transformed value; inspect units and fitted transform")
    return result


def inverse_transform_value(value: float, stats: dict[str, Any], feature_index: int) -> float:
    """Invert z-scoring and the field's signed asinh transform, without clipping."""
    _finite_number(value, "inverse transform value")
    if value is None:
        raise ValueError("inverse transform value must not be null")
    i = feature_index
    transformed = float(value) * stats["std"][i] + stats["mean"][i]
    if not math.isfinite(transformed):
        magnitude = max(abs(stats["std"][i]), abs(stats["mean"][i]))
        transformed = (float(value) * (stats["std"][i] / magnitude) + stats["mean"][i] / magnitude) * magnitude
    kind = stats.get("kind", ["linear"] * len(stats["mean"]))[i]
    if kind == "asinh":
        scale = stats["asinh_scale"][i]
        try:
            if abs(transformed) < 710:
                result = math.sinh(transformed) * scale
            else:
                result = math.copysign(math.exp(abs(transformed) + math.log(scale) - math.log(2.0)), transformed)
        except OverflowError as exc:
            raise ValueError("inverse transform exceeds finite physical range") from exc
    else:
        result = transformed
    if not math.isfinite(result):
        raise ValueError("inverse transform exceeds finite physical range")
    return result


def fit_preprocessor(
    dataset: dict[str, Any], manifest: dict[str, Any], include_performance: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit v2 field transforms and vocabulary on ID training observations only.

    Default weighting gives each circuit equal mass independently per field,
    with observed device slots (parameters/OP) or sample observations (labels)
    uniform within that circuit. The fitted asinh scale is the similarly
    circuit-balanced weighted median absolute NONZERO training value. Zeros
    remain measured values for mean/std/count; null/omitted values are missing.

    OP statistics include only ``is_op_target`` devices; graph/parameter inputs
    retain every real device. Raw OP is never rewritten. Default selected OP
    fields exclude source geometry w_um/l_um/multiplicity. Voltage-like fields
    are linear; signed currents/conductances/capacitances/ratios use asinh before
    population z-scoring. Parameters/performance default linear and each field
    can override kind and positive fixed scale (or ``fit``).

    Stage one uses every ID train row and never reads performance values for
    statistics. Stage two fits label transforms ONLY from ``labeled_train``.
    Neither stage fits anything from validation, test, or OOD observations.
    """
    train = samples_for_split(dataset, manifest, "train")
    raw_spec = dataset["feature_spec"]
    normalized_config = _preprocessing_config(config, raw_spec)
    spec = copy.deepcopy(raw_spec)
    spec["op_names"] = normalized_config["op_fields"][:]
    devices = [(sample["circuit_id"], device) for sample in train for device in sorted(sample["devices"], key=lambda d: d["device_id"])]
    types = sorted({device["device_type"] for _, device in devices})
    weighting = normalized_config["weighting"]
    result: dict[str, Any] = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "device_vocab": {"<PAD>": 0, "<UNK>": 1, **{name: index + 2 for index, name in enumerate(types)}},
        "feature_spec": spec, "raw_feature_spec": copy.deepcopy(raw_spec),
        "transform_config": normalized_config,
        "normalization_config": copy.deepcopy(normalized_config),
        "parameters": _statistics([(circuit, device["parameters"]) for circuit, device in devices], spec["parameter_names"], normalized_config["parameters"], weighting),
        "op": _statistics([(circuit, device["op"]) for circuit, device in devices if is_op_target(device)], spec["op_names"], normalized_config["op"], weighting),
        "fitted_sample_ids": sorted(manifest["train"]),
        "fit_ids": {"parameters": sorted(manifest["train"]), "op": sorted(manifest["train"]), "vocabulary": sorted(manifest["train"])},
    }
    if include_performance:
        lookup = {sample["sample_id"]: sample for sample in train}
        labeled = [lookup[sid] for sid in sorted(manifest["labeled_train"])]
        result["performance"] = _statistics([(sample["circuit_id"], sample["performance"]) for sample in labeled], spec["metric_names"], normalized_config["performance"], weighting)
        result["performance_fitted_sample_ids"] = sorted(manifest["labeled_train"])
        result["fit_ids"]["performance"] = sorted(manifest["labeled_train"])
    return result


def synthetic_dataset(n_per_circuit: int = 24, seed: int = 17) -> dict[str, Any]:
    """Create five explicitly synthetic families with 4–6 device nodes each.

    These varied toy graphs and analytic OP/performance labels are ONLY a
    reproducibility/smoke-test fixture, never transistor-accurate SPICE data or
    evidence of analog-design performance. ``synthetic_4`` is the intended OOD
    circuit. The other four are ID families; even the minimum of three samples
    per family permits nonempty train/validation/test design splits.
    """
    if type(n_per_circuit) is not int or n_per_circuit < 3:
        raise ValueError("n_per_circuit must be an integer at least 3")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    result: dict[str, Any] = {
        "schema_version": 1,
        "feature_spec": {"parameter_names": ["W", "L", "R"], "op_names": ["id", "gm", "vds"],
                         "metric_names": ["gain", "power"], "terminal_names": list(TERMINAL_NAMES)},
        "metadata": {"synthetic": True, "label_source": "analytic toy formulas; NOT circuit simulation",
                     "intended_ood_circuit": "synthetic_4", "seed": seed},
        "samples": [],
    }
    for family in range(5):
        circuit = f"synthetic_{family}"
        for index in range(n_per_circuit):
            supply = 1.2
            devices = []
            for node in range(4 + (index + family) % 3):
                resistor = node % 4 == 3
                width = (0.7 + rng.random() * 3.0 + 0.2 * family) * 1e-6
                length = (0.12 + rng.random() * 0.2) * 1e-6
                resistance = 500.0 + rng.random() * 4500.0
                if resistor:
                    parameters = {"W": None, "L": None, "R": resistance}
                    terminals = {"p": f"n{node}", "n": "0" if family % 2 else f"n{node - 1}"}
                    current, gm, vds = supply / resistance, None, supply * 0.5
                    dtype = "resistor"
                else:
                    parameters = {"W": width, "L": length, "R": None}
                    terminals = {"d": f"n{node}", "g": f"n{(node + family + 1) % 3}",
                                 "s": "0" if node % 2 == 0 else f"n{node - 1}", "b": "0"}
                    current = 1e-6 * (width / length) * (0.3 + supply * 0.2)
                    gm, vds = 2.0 * current / 0.2, supply * (0.25 + rng.random() * 0.5)
                    dtype = "pmos" if node % 2 else "nmos"
                op = {"id": current, "gm": gm, "vds": vds}
                # Deliberately missing values exercise observation masks.
                if index % 7 == 0 and node == 0:
                    op["vds"] = None
                devices.append({"device_id": f"D{node}", "device_type": dtype,
                                "parameters": parameters, "op": op, "terminals": terminals})
            total_current = sum(device["op"]["id"] for device in devices)
            total_gm = sum(device["op"]["gm"] or 0.0 for device in devices)
            gain = math.log1p(10000.0 * total_gm) * (1.0 + 0.1 * family)
            performance = {"gain": gain, "power": supply * total_current}
            if index % 11 == 0:
                performance["gain"] = None
            sid = f"{circuit}_{index:04d}"
            result["samples"].append({"sample_id": sid, "design_id": f"design_{index:04d}",
                                      "circuit_id": circuit, "context": {"supply_v": supply, "temperature_c": 27.0},
                                      "ports": {"output": "n0", "ground": "0"},
                                      "op_status": "partial" if index % 7 == 0 else "complete",
                                      "devices": devices, "performance": performance})
    return validate_dataset(result)
