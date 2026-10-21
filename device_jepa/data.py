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
        for key in ("sample_id", "design_id", "circuit_id"):
            _nonempty_string(sample.get(key), f"{loc}.{key}")
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
    """Hash sample membership and held-out circuit independently of labels."""
    content = {key: sorted(manifest[key]) for key in SPLITS}
    content["ood_circuit"] = manifest["ood_circuit"]
    content["id_circuits"] = sorted(manifest["id_circuits"])
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
    seen_designs: dict[tuple[str, str], str] = {}
    seen_groups: dict[tuple[str, str], str] = {}
    seen_fingerprints: dict[str, tuple[str, str]] = {}
    for sample in samples:
        sid, circuit = sample["sample_id"], sample["circuit_id"]
        key = (circuit, sample["design_id"])
        if key in seen_designs:
            uf.union(sid, seen_designs[key])
        else:
            seen_designs[key] = sid
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


def build_manifest(
    dataset: dict[str, Any], ood_circuit: str, seed: int = 17,
    val_fraction: float = 0.15, test_fraction: float = 0.15,
) -> dict[str, Any]:
    """Split four ID circuits by design groups, holding the fifth entirely OOD.

    Fractions apply to independent groups, not samples; repeated conditions can
    therefore change sample-level proportions. At least three independent
    groups are required in each ID circuit. Every ID circuit has nonempty
    train/validation/test splits. Hash ranking makes assignment input-order
    independent and independent of Python's randomized hash seed.
    """
    validate_dataset(dataset)
    _nonempty_string(ood_circuit, "ood_circuit")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for name, value in (("val_fraction", val_fraction), ("test_fraction", test_fraction)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be a finite number strictly between zero and one")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than one")
    groups = _design_groups(dataset)
    if len(groups) != 5:
        raise ValueError(f"exactly five distinct circuits are required; found {len(groups)}")
    if ood_circuit not in groups:
        raise ValueError(f"ood_circuit {ood_circuit!r} is not present")
    id_circuits = sorted(set(groups) - {ood_circuit})
    manifest: dict[str, Any] = {split: [] for split in SPLITS}
    manifest.update({"ood_circuit": ood_circuit, "id_circuits": id_circuits, "seed": seed,
                     "dataset_fingerprint": dataset_fingerprint(dataset),
                     "val_fraction": val_fraction, "test_fraction": test_fraction,
                     "grouping": "circuit-scoped design_id/group_id plus exact design fingerprint"})
    for circuit in id_circuits:
        circuit_groups = groups[circuit]
        n_groups = len(circuit_groups)
        if n_groups < 3:
            raise ValueError(f"ID circuit {circuit!r} needs at least 3 independent design groups; found {n_groups}")
        def rank(ids: list[str]) -> str:
            return hashlib.sha256(_canonical([seed, circuit, ids]).encode("utf-8")).hexdigest()
        ranked = sorted(circuit_groups, key=lambda ids: (rank(ids), ids))
        n_val = max(1, int(n_groups * val_fraction))
        n_test = max(1, int(n_groups * test_fraction))
        while n_val + n_test >= n_groups:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError("not enough independent groups for nonempty splits")
        for split, subset in (("val", ranked[:n_val]), ("test", ranked[n_val:n_val + n_test]),
                              ("train", ranked[n_val + n_test:])):
            manifest[split].extend(sid for group in subset for sid in group)
    manifest["ood"] = [sid for group in groups[ood_circuit] for sid in group]
    for split in SPLITS:
        manifest[split].sort()
    verify_manifest(dataset, manifest)
    return manifest


def verify_manifest(dataset: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Raise ValueError for stale manifests, omitted/duplicate IDs or leakage."""
    validate_dataset(dataset)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if manifest.get("dataset_fingerprint") != dataset_fingerprint(dataset):
        raise ValueError("manifest dataset_fingerprint does not match dataset")
    samples = {sample["sample_id"]: sample for sample in dataset["samples"]}
    circuits = {sample["circuit_id"] for sample in samples.values()}
    ood_circuit = manifest.get("ood_circuit")
    _nonempty_string(ood_circuit, "manifest.ood_circuit")
    if len(circuits) != 5 or ood_circuit not in circuits:
        raise ValueError("manifest requires exactly five circuits and one valid OOD circuit")
    id_circuits = sorted(circuits - {ood_circuit})
    if manifest.get("id_circuits") != id_circuits:
        raise ValueError("manifest id_circuits must list the four sorted non-OOD circuits")
    assignment: dict[str, str] = {}
    for split in SPLITS:
        ids = manifest.get(split)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"manifest {split} must be a nonempty list")
        for sid in ids:
            if not isinstance(sid, str) or sid not in samples:
                raise ValueError(f"manifest {split} contains an unknown sample ID: {sid!r}")
            if sid in assignment:
                raise ValueError(f"sample {sid!r} appears more than once in manifest")
            assignment[sid] = split
            is_ood = samples[sid]["circuit_id"] == ood_circuit
            if is_ood != (split == "ood"):
                raise ValueError(f"sample {sid!r} violates ID/OOD circuit separation")
        if split != "ood" and {samples[sid]["circuit_id"] for sid in ids} != set(id_circuits):
            raise ValueError(f"manifest {split} must contain samples from all four ID circuits")
    if set(assignment) != set(samples):
        raise ValueError("manifest must contain every dataset sample exactly once")
    for circuit_groups in _design_groups(dataset).values():
        for group in circuit_groups:
            if len({assignment[sid] for sid in group}) != 1:
                raise ValueError(f"design group is split across subsets: {group!r}")


def samples_for_split(dataset: dict[str, Any], manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Return original sample objects in manifest order after verifying safety."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    verify_manifest(dataset, manifest)
    lookup = {sample["sample_id"]: sample for sample in dataset["samples"]}
    return [lookup[sid] for sid in manifest[split]]


def _statistics(rows: list[dict[str, Any]], names: list[str]) -> dict[str, list[Any]]:
    means, stds, counts = [], [], []
    for name in names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        count = len(values)
        if not count:
            mean, std = 0.0, 1.0
        else:
            # Scale first to prevent overflow/underflow; do not floor variance
            # in absolute SI units (e.g. nanometer lengths, picoamp currents).
            scale = max(abs(value) for value in values)
            if not scale:
                mean, std = 0.0, 1.0
            else:
                normalized = [value / scale for value in values]
                relative_mean = math.fsum(normalized) / count
                mean = relative_mean * scale
                relative_std = math.sqrt(math.fsum((value - relative_mean) ** 2 for value in normalized) / count)
                std = relative_std * scale
                if relative_std <= 1e-12 or std == 0.0:
                    std = 1.0
        means.append(mean)
        stds.append(std)
        counts.append(count)
    return {"mean": means, "std": stds, "count": counts}


def fit_preprocessor(dataset: dict[str, Any], manifest: dict[str, Any], include_performance: bool = False) -> dict[str, Any]:
    """Fit vocabulary and population mean/std on training data only.

    The default pretraining path never fits/returns performance statistics.
    ``include_performance=True`` is exclusively for the later supervised
    decoder phase. Manifest integrity validation hashes labels, but no label
    value enters parameter/OP statistics or vocabulary. Unseen device types map
    to ``<UNK>`` in downstream tensorization. Missing observations are ignored;
    all-missing features use mean 0/std 1, constant features use std 1.
    """
    train = samples_for_split(dataset, manifest, "train")
    spec = dataset["feature_spec"]
    devices = [device for sample in train for device in sample["devices"]]
    types = sorted({device["device_type"] for device in devices})
    result: dict[str, Any] = {
        "device_vocab": {"<PAD>": 0, "<UNK>": 1, **{name: index + 2 for index, name in enumerate(types)}},
        "feature_spec": copy.deepcopy(spec),
        "parameters": _statistics([device["parameters"] for device in devices], spec["parameter_names"]),
        "op": _statistics([device["op"] for device in devices], spec["op_names"]),
        "fitted_sample_ids": list(manifest["train"]),
    }
    if include_performance:
        result["performance"] = _statistics([sample["performance"] for sample in train], spec["metric_names"])
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
