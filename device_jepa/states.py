"""Target-only operating checks, not mutually exclusive transistor regions.

The two independent categorical groups reproduce only the collector's current
and saturation-margin checks. Unknown/unchecked groups are entirely null. A
checked, missing measurement has its own observed category, unlike numeric zero.
Neither source OP values nor collector reasons/statuses are ever changed.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any


STATE_VERSION = 1
STATE_FEATURES = [
    "current_missing", "current_below", "current_pass",
    "saturation_missing", "saturation_below", "saturation_pass",
]
_GROUPS = {
    "current": ("id_a", "min_signal_current_a", "off_or_missing_current"),
    "saturation": ("saturation_margin_v", "saturation_margin_min_v", "saturation"),
}
_MODES = {"preserve_only", "collector_checked", "all_mos"}


def _finite(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{location} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


def _thresholds(value: Any, location: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    result = {}
    for _, key, _ in _GROUPS.values():
        if value.get(key) is not None:
            result[key] = _finite(value[key], f"{location}.{key}")
    if result.get("min_signal_current_a", 0) < 0:
        raise ValueError("min_signal_current_a must be nonnegative")
    return result


def encode_device_checks(op: Mapping[str, Any], checked: bool | Mapping[str, bool],
                         thresholds: Mapping[str, Any] | None = None) -> dict[str, int | None]:
    """Encode two independent missing/below/pass one-hot groups.

    ``checked`` is either a boolean for both checks or a mapping with independent
    ``current``/``saturation`` booleans (omitted groups are unchecked). Thresholds
    use the collector's exact keys. Missing checked measurements are identifiable
    without a threshold; observed values without a threshold remain unknown.
    Equality passes. Only the current comparison uses abs, as the collector does.
    """
    if not isinstance(op, Mapping):
        raise ValueError("op must be an object")
    if type(checked) is bool:
        checked = dict.fromkeys(_GROUPS, checked)
    if not isinstance(checked, Mapping) or set(checked) - set(_GROUPS):
        raise ValueError("checked must be a boolean or current/saturation mapping")
    if any(type(value) is not bool for value in checked.values()):
        raise ValueError("each checked group must be a boolean")
    limits = _thresholds(thresholds, "thresholds")
    result = dict.fromkeys(STATE_FEATURES)
    for group, (field, threshold, _) in _GROUPS.items():
        if not checked.get(group, False):
            continue
        value = op.get(field)
        if value is None:
            category = "missing"
        elif threshold not in limits:
            # Validate, but do not derive pass/below from an invented threshold.
            _finite(value, f"op.{field}")
            continue
        else:
            numeric = _finite(value, f"op.{field}")
            if group == "current":
                numeric = abs(numeric)
            category = "below" if numeric < limits[threshold] else "pass"
        for name in ("missing", "below", "pass"):
            result[f"{group}_{name}"] = int(name == category)
    return result


def _run_records(provenance: Any, path: str = "provenance", *,
                 direct_circuit_id: str | None = None, _seen_sources: dict | None = None):
    if provenance is None:
        return
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{path} must be an object")
    if _seen_sources is None:
        _seen_sources = {}
    source_id = provenance.get("source_id")
    circuit_ids = provenance.get("circuit_ids")
    if source_id is not None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{path}.source_id must be a nonempty string")
        if source_id in _seen_sources:
            raise ValueError(f"ambiguous duplicate source_id {source_id!r} in provenance")
        _seen_sources[source_id] = path
    if circuit_ids is not None:
        if (not isinstance(circuit_ids, list) or not circuit_ids
                or any(not isinstance(name, str) or not name.strip() for name in circuit_ids)
                or len(set(circuit_ids)) != len(circuit_ids)):
            raise ValueError(f"{path}.circuit_ids must be a nonempty list of unique circuit IDs")
    sources = provenance.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"{path}.sources must be a list")
    scope = {"source_id": source_id, "circuit_ids": circuit_ids,
             "direct_circuit_id": direct_circuit_id if path == "provenance" and not sources and source_id is None else None}
    runs = provenance.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"{path}.runs must be a list")
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise ValueError(f"{path}.runs[{index}] must be an object")
        yield run, f"{path}.runs[{index}]", scope
    for index, source in enumerate(sources):
        yield from _run_records(source, f"{path}.sources[{index}]", _seen_sources=_seen_sources)


def _lists(source: Mapping, path: str):
    for key in ("checked_devices", "signal_devices"):
        value = source.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
            raise ValueError(f"{path}.{key} must be a list of device names")
        names = frozenset(x.lower() for x in value)
        if len(names) != len(value):
            raise ValueError(f"{path}.{key} contains duplicate case-insensitive device names")
        yield names, f"{path}.{key}"


def _source_limits(source: Mapping, path: str):
    for key in ("thresholds", "constraints"):
        for name, value in _thresholds(source.get(key), f"{path}.{key}").items():
            yield name, value, f"{path}.{key}.{name}"


def _run_signature(run: Mapping, path: str):
    """Compare relevant run evidence, never choose the first conflicting run."""
    limits = sorted((name, value) for name, value, _ in _source_limits(run, path))
    lists = sorted(tuple(sorted(names)) for names, _ in _lists(run, path))
    return limits, lists


def _resolve_sources(sample: Mapping, records: list, config: Mapping):
    sample_id = sample.get("sample_id")
    path = f"sample[{sample_id!r}].source_observation"
    observation = sample.get("source_observation", {})
    sampling = sample.get("source_sampling", {})
    if not isinstance(observation, Mapping) or not isinstance(sampling, Mapping):
        raise ValueError(f"sample {sample_id}: source metadata must be objects")
    run_id = sampling.get("run_id")
    source_id, circuit_id = sample.get("source_id"), sample.get("circuit_id")
    if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()):
        raise ValueError(f"sample {sample_id}: source_id must be a nonempty string")
    def belongs(scope):
        if source_id is None:
            return scope["direct_circuit_id"] is not None and scope["direct_circuit_id"] == circuit_id
        if scope["source_id"] != source_id or scope["circuit_ids"] is None:
            return False
        if circuit_id not in scope["circuit_ids"]:
            raise ValueError(f"sample {sample_id}: source_id circuit scope does not contain {circuit_id!r}")
        return True
    # A missing run identity never licenses borrowing another run's settings.
    # Combined sources additionally require an exact source association. In
    # legacy unscoped combined imports even a unique run ID is not sufficient.
    runs = [(run, loc) for run, loc, scope in records
            if run_id is not None and run.get("run_id") == run_id and belongs(scope)]
    if len(runs) > 1:
        first = _run_signature(*runs[0])
        if any(_run_signature(run, loc) != first for run, loc in runs[1:]):
            raise ValueError(f"sample {sample_id}: ambiguous run {run_id!r} thresholds/checklists")
    sources = [(observation, path), *runs]
    limits, limit_sources = {}, {}
    for source, location in [*sources, (config, "config")]:
        for key, value, origin in _source_limits(source, location):
            if key in limits and limits[key] != value:
                raise ValueError(f"sample {sample_id}: conflicting threshold {key}")
            limits[key] = value
            limit_sources.setdefault(key, []).append(origin)
    lists = [record for source, location in sources for record in _lists(source, location)]
    checked_names = None
    if lists:
        checked_names = lists[0][0]
        if any(names != checked_names for names, _ in lists[1:]):
            raise ValueError(f"sample {sample_id}: conflicting checked_devices/signal_devices checklists")
    return observation, run_id, limits, limit_sources, checked_names, [loc for _, loc in lists]


def _reason_evidence(observation: Mapping, device_id: str) -> dict[str, list[str]]:
    result = {group: [] for group in _GROUPS}
    reasons = observation.get("reasons", [])
    if not isinstance(reasons, list):
        return result
    for reason in reasons:
        if not isinstance(reason, str) or ":" not in reason:
            continue
        name, label = reason.rsplit(":", 1)
        if name.lower() == device_id.lower():
            for group, (_, _, expected) in _GROUPS.items():
                if label == expected:
                    result[group].append(reason)
    return result


def apply_op_states(dataset: dict[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Copy a dataset and add auditable, target-only categorical OP checks.

    Modes: ``collector_checked`` (default), ``all_mos`` (explicit derived opt-in),
    and ``preserve_only`` (no categorical features). The optional ``features``
    list may select complete groups only. ``thresholds`` may provide absent
    collector thresholds, but conflicting evidence is rejected, not overridden.

    A listed checked device provides evidence for both groups. A raw failure
    reason provides evidence only for its named group. Absent reasons never
    establish that a check ran or passed. Explicit empty lists are meaningful.
    Direct single-circuit run settings use sample.source_sampling.run_id.
    Combined imports additionally require matching sample.source_id and the
    provenance source's circuit_ids. Unscoped legacy combined run records are
    ignored rather than borrowed from another circuit/source.
    """
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("op_state config must be an object")
    unknown_keys = set(config) - {"mode", "features", "thresholds"}
    if unknown_keys:
        raise ValueError(f"Unknown op_state config keys: {sorted(unknown_keys)}")
    mode = config.get("mode", "collector_checked")
    if mode not in _MODES:
        raise ValueError(f"op_state mode must be one of {sorted(_MODES)}")
    features = config.get("features", [] if mode == "preserve_only" else STATE_FEATURES)
    if not isinstance(features, list) or any(not isinstance(x, str) for x in features):
        raise ValueError("op_state features must be a list of names")
    if len(set(features)) != len(features) or set(features) - set(STATE_FEATURES):
        raise ValueError("op_state features must be unique names from STATE_FEATURES")
    for group in _GROUPS:
        count = sum(f"{group}_{category}" in features for category in ("missing", "below", "pass"))
        if count not in (0, 3):
            raise ValueError("op_state features must select complete categorical groups")
    if mode == "preserve_only" and features:
        raise ValueError("preserve_only mode cannot request categorical features")
    configured_limits = _thresholds(config.get("thresholds"), "config.thresholds")
    if config.get("thresholds") is not None:
        unknown_thresholds = set(config["thresholds"]) - {entry[1] for entry in _GROUPS.values()}
        if unknown_thresholds:
            raise ValueError(f"Unknown op_state thresholds: {sorted(unknown_thresholds)}")
    result = copy.deepcopy(dataset)
    circuits = {sample["circuit_id"] for sample in result["samples"]}
    direct_circuit_id = next(iter(circuits)) if len(circuits) == 1 else None
    records = list(_run_records(result.get("provenance", {}), direct_circuit_id=direct_circuit_id)) if mode != "preserve_only" else []
    scope_sources, threshold_sources = set(), set()
    for sample in result["samples"]:
        if mode == "preserve_only":
            for device in sample["devices"]:
                device["op_state"] = {}
                device["op_state_provenance"] = {"version": STATE_VERSION, "mode": mode, "target_only": True}
            continue
        observation, run_id, limits, limit_origins, checked_names, list_origins = _resolve_sources(sample, records, config)
        device_names = [device["device_id"].lower() for device in sample["devices"]]
        if len(set(device_names)) != len(device_names):
            raise ValueError(f"sample {sample['sample_id']}: ambiguous case-insensitive device identities")
        if checked_names is not None and checked_names - set(device_names):
            raise ValueError(f"sample {sample['sample_id']}: checked device absent from sample")
        scope_sources.update(list_origins)
        threshold_sources.update(loc for locations in limit_origins.values() for loc in locations)
        for device in sample["devices"]:
            mos = device["device_type"].lower() in {"nmos", "pmos"} and device.get("op_target", True)
            reasons = _reason_evidence(observation, device["device_id"])
            checked, evidence = {}, {}
            for group in _GROUPS:
                if not mos:
                    checked[group], evidence[group] = False, []
                elif mode == "all_mos":
                    checked[group], evidence[group] = True, ["config.mode=all_mos (derived_not_original_collector)"]
                else:
                    listed = checked_names is not None and device["device_id"].lower() in checked_names
                    if reasons[group] and checked_names is not None and not listed:
                        raise ValueError(f"sample {sample['sample_id']} {device['device_id']}: reason conflicts with explicit checklist")
                    checked[group] = listed or bool(reasons[group])
                    evidence[group] = list(list_origins) if listed else []
                    if reasons[group]:
                        evidence[group].append("source_observation.reasons:" + group)
                scope_sources.update(evidence[group])
            states = encode_device_checks(device.get("op", {}), checked, limits)
            if mode == "collector_checked" and mos:
                for group in _GROUPS:
                    if reasons[group] and states[f"{group}_pass"] == 1:
                        raise ValueError(f"sample {sample['sample_id']} {device['device_id']}: {group} failure reason conflicts with OP/threshold")
            device["op_state"] = {name: states[name] for name in features}
            device["op_state_provenance"] = {
                "version": STATE_VERSION, "mode": mode, "target_only": True,
                "derived_not_original_collector": mode == "all_mos",
                "source_sample_id": sample.get("source_sample_id", sample.get("sample_id")),
                "source_run_id": run_id, "checked": checked,
                "checked_evidence": evidence, "thresholds": dict(limits),
                "threshold_sources": copy.deepcopy(limit_origins), "raw_reason_labels": reasons,
                "encoding_status": {
                    group: "unchecked" if not checked[group] else "unknown_threshold"
                    if states[f"{group}_missing"] is None else "observed"
                    for group in _GROUPS
                },
            }
            if sample.get("source_id") is not None:
                device["op_state_provenance"]["source_id"] = sample["source_id"]
    metadata = result.setdefault("metadata", {})
    metadata["state_encoding"] = {
        "version": STATE_VERSION, "mode": mode, "feature_names": list(features),
        "target_only": True, "derived_not_original_collector": mode == "all_mos",
        "groups": {group: [f"{group}_{category}" for category in ("missing", "below", "pass")]
                   for group in _GROUPS},
        "semantics": "Two independent collector checks, not transistor operating regions; null groups are unobserved",
        "scope_sources": sorted(scope_sources), "threshold_sources": sorted(threshold_sources),
        "explicit_thresholds": configured_limits,
        "per_device_provenance": "device.op_state_provenance",
    }
    return result


def _same_encoding(actual: Any, expected: Any) -> bool:
    """Exact recursive equality; bool True is not a valid replacement for 1."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_encoding(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_encoding(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def validate_op_states(dataset: dict[str, Any], config: Mapping[str, Any] | None = None) -> None:
    """Require persisted states to match the requested encoding and raw evidence.

    Call before state-enabled training. Re-encoding uses a deep copy; this
    validator never repairs or mutates the supplied dataset. Matching feature
    names alone are insufficient: dataset metadata, every categorical value,
    and every device's scope/threshold provenance must match a fresh encoding.
    Raw or partially encoded datasets are rejected even if the expected groups
    would all be unknown. Import them with the requested config first.

    An explicitly empty ``features`` list deliberately disables state use and
    bypasses validation, as does ``preserve_only`` with omitted features. Thus
    state-disabled experiments can consume otherwise state-encoded datasets.
    """
    instruction = "Re-import the dataset with the matching op_state config before training."
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError(f"OP-state validation: config must be an object. {instruction}")
    features = config.get("features", [] if config.get("mode") == "preserve_only" else STATE_FEATURES)
    if isinstance(features, list) and not features:
        return
    try:
        expected = apply_op_states(dataset, config)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"OP-state validation could not reproduce the requested encoding: {exc}. {instruction}") from exc
    metadata = dataset.get("metadata")
    if not isinstance(metadata, Mapping) or not _same_encoding(
        metadata.get("state_encoding"), expected["metadata"]["state_encoding"]
    ):
        raise ValueError(f"OP-state metadata.state_encoding is missing or does not match the requested encoding. {instruction}")
    for sample, expected_sample in zip(dataset["samples"], expected["samples"]):
        for device, expected_device in zip(sample["devices"], expected_sample["devices"]):
            for key in ("op_state", "op_state_provenance"):
                if not _same_encoding(device.get(key), expected_device[key]):
                    raise ValueError(
                        f"OP-state sample {sample.get('sample_id')!r} device {device.get('device_id')!r}: "
                        f"{key} is missing or inconsistent with the requested encoding/raw evidence. {instruction}"
                    )
