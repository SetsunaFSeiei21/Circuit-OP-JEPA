#!/usr/bin/env python3
"""Predict from a final checkpoint without fitting or loading any teacher."""
import argparse
import copy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def inference_dataset(data, preprocessing, *, measured_op=False):
    """Remove unused outcomes BEFORE validation, permitting outcome-free inputs.

    Design models, including student_predicted_op, accept ordinary canonical datasets with omitted/empty OP and
    performance dictionaries. Supplied OP values, missingness, state or failure
    metadata cannot enter their inference path, even if those outcomes are NaN.
    """
    from device_jepa.data import validate_dataset

    data = copy.deepcopy(data)
    if not isinstance(data, dict):
        raise ValueError("Inference data must be a canonical dataset object")
    expected = preprocessing["feature_spec"]
    spec = data.get("feature_spec")
    if not isinstance(spec, dict):
        raise ValueError("Inference data requires feature_spec")
    for field in ("parameter_names", "terminal_names"):
        if spec.get(field) != expected[field]:
            raise ValueError(f"Inference {field} must match the saved checkpoint schema")
    spec["metric_names"] = expected["metric_names"][:]
    # Only checkpoint-selected OP fields are meaningful for the reference. For
    # design-input models this is validation schema, not an input requirement.
    spec["op_names"] = expected["op_names"][:]
    supported = {name for name, count in zip(expected["op_names"], preprocessing["op"]["count"]) if count > 0}
    for sample in data.get("samples", []):
        if not isinstance(sample, dict):
            continue  # The canonical validator will produce the precise error.
        sample["performance"] = {}
        sample.pop("op_status", None)
        for device in sample.get("devices", []):
            if not isinstance(device, dict):
                continue
            device.pop("op_state", None)
            if measured_op:
                raw = device.get("op", {})
                if not isinstance(raw, dict):
                    raise ValueError("The OP reference requires an OP dictionary per device")
                device["op"] = {key: value for key, value in raw.items() if key in supported}
            else:
                device["op"] = {}
                device.pop("op_target", None)
    return validate_dataset(data)


def predict(checkpoint, data_path, output, *, device="cpu", batch_size=None,
            include_predicted_op=False):
    """Write normalized and physical predictions; never replace an existing file."""
    output = Path(output).expanduser().resolve()
    checkpoint, data_path = (Path(p).expanduser().resolve() for p in (checkpoint, data_path))
    if output.exists() or output in {checkpoint, data_path}:
        raise ValueError("Prediction output must be a new file; refusing to overwrite any input/output")
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    import torch
    from device_jepa.data import _reject_duplicate_keys
    from device_jepa.resident import BatchProgress, ResidentDataset, require_device
    from device_jepa.tensors import denormalize
    from experiments.op_transfer.engine import attach_identity, load_model
    from experiments.op_transfer.models import OPPredictor, PredictedOPCascade

    device = require_device(device)
    model, saved = load_model(checkpoint, device)
    standalone_op = isinstance(model, OPPredictor)
    if include_predicted_op and not isinstance(model, (OPPredictor, PredictedOPCascade)):
        raise ValueError("--include-predicted-op requires an OP predictor or cascade checkpoint")
    include_predicted_op = include_predicted_op or standalone_op
    measured_op = bool(saved["inference_requires_op"])
    raw = json.loads(data_path.read_text(), object_pairs_hook=_reject_duplicate_keys)
    data = inference_dataset(raw, saved["preprocessing"], measured_op=measured_op)
    batch_size = batch_size or saved["options"]["batch_size"]
    cache = ResidentDataset(data["samples"], saved["preprocessing"], device,
                            include_op=measured_op, include_performance=False,
                            include_op_states=False, max_tokens=saved["model_config"]["max_devices"],
                            label="op-transfer inference")
    progress = BatchProgress("inference", (len(data["samples"]) + batch_size - 1) // batch_size,
                             saved["options"].get("progress_every", 100), device)
    names = saved["preprocessing"]["feature_spec"]["metric_names"]
    supported_metrics = [count > 0 for count in saved["preprocessing"]["performance"]["count"]]
    records = []
    with torch.no_grad():
        for step, (indices, batch) in enumerate(cache.batches(batch_size), 1):
            rows = [data["samples"][i] for i in indices.tolist()]
            attach_identity(batch, rows)
            normalized = physical = None
            if not standalone_op:
                normalized = model(batch)
                if not bool(torch.isfinite(normalized).all()):
                    raise FloatingPointError("Non-finite normalized performance prediction")
                active = torch.tensor(supported_metrics, device=normalized.device)
                normalized = normalized.masked_fill(~active, 0.)
                physical = denormalize(normalized, saved["preprocessing"]["performance"])
            op_normalized = op_physical = op_mask = None
            if include_predicted_op:
                op_normalized, op_mask = (model if standalone_op else model.predictor)(batch)
                if not bool(torch.isfinite(op_normalized).all()):
                    raise FloatingPointError("Non-finite normalized OP prediction")
                op_physical = denormalize(op_normalized, saved["preprocessing"]["op"])
                op_normalized, op_physical, op_mask = (value.cpu().tolist() for value in
                                                      (op_normalized, op_physical, op_mask))
            normal_values = normalized.cpu().tolist() if normalized is not None else None
            physical_values = physical.cpu().tolist() if physical is not None else None
            for index, row in enumerate(rows):
                record = {"sample_id": row["sample_id"], "circuit_id": row["circuit_id"]}
                if not standalone_op:
                    record.update(normalized_performance={name: value if active else None for name, value, active in
                                                          zip(names, normal_values[index], supported_metrics)},
                                  performance={name: value if active else None for name, value, active in
                                               zip(names, physical_values[index], supported_metrics)})
                if include_predicted_op:
                    op_names = saved["preprocessing"]["feature_spec"]["op_names"]
                    record["predicted_op"] = []
                    for j, dev in enumerate(sorted(row["devices"], key=lambda d: d["device_id"])):
                        support = op_mask[index][j]
                        record["predicted_op"].append({"device_id": dev["device_id"],
                            "normalized": {name: value if active else None for name, value, active in
                                           zip(op_names, op_normalized[index][j], support)},
                            "physical": {name: value if active else None for name, value, active in
                                         zip(op_names, op_physical[index][j], support)},
                            "supported": dict(zip(op_names, support))})
                records.append(record)
            progress.update(step)
    result = {"format": "op_transfer_predictions_v1", "checkpoint": str(checkpoint),
              "mode": saved["mode"], "requires_measured_op": measured_op,
              "normalization": "saved train-only checkpoint transforms; no inference fitting",
              "physical_units": "canonical schema units for the named fields",
              "unsupported_metrics": [name for name, active in zip(names, supported_metrics) if not active],
              "predictions": records}
    encoded = json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(encoded)
    print(f"Saved {len(records)} predictions to {output}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="canonical inputs; performance may be omitted; op_oracle/design_op_oracle require measured OP")
    parser.add_argument("--output", required=True, help="new prediction JSON file")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--include-predicted-op", action="store_true", help="also export cascade OP predictions in physical and normalized units")
    args = parser.parse_args(argv)
    try:
        predict(args.checkpoint, args.data, args.output, device=args.device,
                batch_size=args.batch_size, include_predicted_op=args.include_predicted_op)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
