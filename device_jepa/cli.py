"""Explicit data import, immutable experiment splits, and two-stage training."""
import argparse
from importlib.resources import files
import json
from pathlib import Path

from .data import build_manifest, load_dataset, synthetic_dataset, verify_manifest


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read(path):
    return json.loads(Path(path).read_text())


def _max_table_rows(value):
    if value.lower() == "none":
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return result


def _new_manifest(dataset, options, path):
    path = Path(path)
    if path.exists():
        manifest = _read(path)
        verify_manifest(dataset, manifest)
        from .training import _check_dataset_options
        _check_dataset_options(options, manifest)
        return manifest
    manifest = build_manifest(dataset, config=options["dataset"])
    _write(path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared-structure device/OP alignment and frozen-encoder metric decoding")
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import_analog_agent", help="Read supplied netlist/SQLite safely; no simulation or feasibility filtering")
    imp.add_argument("--input", nargs="+", required=True, help="One or more per-circuit dataset directories")
    imp.add_argument("--output", required=True)
    imp.add_argument("--config", help="Optional OP-state configuration; default collector_checked")
    imp.add_argument("--max-table-rows", type=_max_table_rows, default=1_000_000,
                     help="Per-source-table safety limit (default: 1000000); raise explicitly or use 'none'. Canonical data is retained in memory.")
    synth = commands.add_parser("make_synthetic", help="Generate toy fixtures, not analog simulation data")
    synth.add_argument("--output", required=True)
    synth.add_argument("--per-circuit", type=int, default=24)
    synth.add_argument("--seed", type=int, default=17)
    split = commands.add_parser("split", help="Create or verify a persisted grouped experiment manifest")
    split.add_argument("--data", required=True)
    split.add_argument("--config", help="Explicit dataset.train_circuits/ood_circuits and split settings")
    split.add_argument("--ood-circuit", help="Legacy single-OOD shorthand; prefer --config")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=17)
    split.add_argument("--val-fraction", type=float)
    split.add_argument("--test-fraction", type=float)
    for name in ("pretrain", "train_decoder"):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--config", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="cpu")
        if name == "pretrain":
            command.add_argument("--manifest", required=True)
        else:
            command.add_argument("--pretrained", required=True)
    baseline = commands.add_parser("train_baseline", help="Clean-room supervised pin/device ZeroSim-style baseline")
    baseline.add_argument("--mode", choices=["zerosim_pin", "zerosim_device", "zerosim_pin_op", "zerosim_device_op"], required=True)
    baseline.add_argument("--data", required=True)
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--device", default="cpu")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["train", "val", "test", "ood"], default="test")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--device", default="cpu")
    for name in ("smoke", "smoke_real"):
        command = commands.add_parser(name, help="Tiny mechanics-only validation, never an accuracy claim")
        command.add_argument("--output", default=f"runs/{name}")
        command.add_argument("--config", default=None)
        if name == "smoke_real":
            command.add_argument("--input", nargs="+", required=True)
            command.add_argument("--max-table-rows", type=_max_table_rows, default=1_000_000,
                                 help="Per-source-table safety limit, or 'none' for no row limit")
    args = parser.parse_args(argv)
    if args.command == "import_analog_agent":
        from .adapter import adapt_analog_agent, combine_datasets
        datasets = [adapt_analog_agent(path, max_table_rows=args.max_table_rows) for path in args.input]
        dataset = datasets[0] if len(datasets) == 1 else combine_datasets(datasets)
        del datasets
        from .states import apply_op_states
        dataset = apply_op_states(dataset, _read(args.config).get("op_state") if args.config else None)
        _write(args.output, dataset)
        print(json.dumps({"samples": len(dataset["samples"]), "circuits": sorted({s["circuit_id"] for s in dataset["samples"]}),
                          "raw_op_fields": len(dataset["feature_spec"]["op_names"]), "metrics": dataset["feature_spec"]["metric_names"],
                          "feasibility_filtered": False}, indent=2))
        return
    if args.command == "make_synthetic":
        from .states import apply_op_states
        _write(args.output, apply_op_states(synthetic_dataset(args.per_circuit, args.seed)))
        print(f"Wrote SYNTHETIC fixture: {args.output}")
        return
    if args.command == "split":
        dataset = load_dataset(args.data)
        if Path(args.output).exists():
            manifest = _read(args.output)
            verify_manifest(dataset, manifest)
            if args.config:
                # Pure data command: do not import PyTorch just to check config.
                expected = build_manifest(dataset, config=_read(args.config)["dataset"])
                if expected != manifest:
                    raise ValueError("Existing manifest/config differ; choose a new experiment path explicitly")
            elif args.ood_circuit and manifest.get("ood_circuit") != args.ood_circuit:
                raise ValueError("Existing manifest has a different OOD circuit")
            print("Existing manifest verified; no resplit performed")
        else:
            if args.config:
                manifest = build_manifest(dataset, config=_read(args.config)["dataset"])
            else:
                manifest = build_manifest(dataset, args.ood_circuit, args.seed, args.val_fraction, args.test_fraction)
            _write(args.output, manifest)
        print(json.dumps({key: len(manifest[key]) for key in ("train", "val", "test", "ood", "labeled_train")}))
        return
    from .training import evaluate, pretrain, seed_everything, train_decoder
    if args.command in {"smoke", "smoke_real"}:
        output = Path(args.output)
        # Smoke owns the whole experiment directory. Reject existing artifacts
        # before creating a manifest or replacing data; the stage-level guard
        # runs later and cannot undo those earlier writes.
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Smoke output must be a new or empty directory; refusing to overwrite existing artifacts")
        output.mkdir(parents=True, exist_ok=True)
        default = "smoke.json" if args.command == "smoke" else "opamp2_smoke.json"
        config = _read(args.config) if args.config else json.loads(files("device_jepa").joinpath("defaults", default).read_text())
        if args.command == "smoke":
            dataset = synthetic_dataset(24, config["training"]["seed"])
            label = "SYNTHETIC"
        else:
            from .adapter import adapt_analog_agent, combine_datasets
            sources = [adapt_analog_agent(path, max_table_rows=args.max_table_rows) for path in args.input]
            dataset = sources[0] if len(sources) == 1 else combine_datasets(sources)
            del sources
            # Never turn a smoke request into a long experiment via custom config.
            from .budgets import smoke_epochs
            smoke_epochs(config)
            label = "REAL-DATA PIPELINE CHECK (NOT ACCURACY EVIDENCE)"
        from .states import apply_op_states
        dataset = apply_op_states(dataset, config.get("op_state"))
        manifest = _new_manifest(dataset, config, output / "manifest.json")
        _write(output / "data.json", dataset)
        p = pretrain(dataset, manifest, config, output)
        d = train_decoder(dataset, p, config, output)
        for subset in ("test", "ood"):
            if manifest[subset]:
                report = evaluate(dataset, d, split=subset, output=output / f"{subset}_report.json")
                print(f"{label} {subset}: {json.dumps(report['metrics'])}")
        if not manifest["ood"]:
            print("NO OOD HOLDOUT: this dataset/config cannot establish cross-circuit OOD generalization.")
        print("Smoke complete: mechanics only. No real performance claim.")
        return
    dataset = load_dataset(args.data)
    if args.command == "pretrain":
        print(pretrain(dataset, _read(args.manifest), _read(args.config), args.output, args.device))
    elif args.command == "train_baseline":
        from .baselines import train_baseline
        print(train_baseline(dataset, _read(args.manifest), _read(args.config), args.output, mode=args.mode, device=args.device))
    elif args.command == "train_decoder":
        print(train_decoder(dataset, args.pretrained, _read(args.config), args.output, args.device))
    else:
        seed_everything(17)
        from .training import load_checkpoint
        evaluator = evaluate
        if load_checkpoint(args.checkpoint)["stage"] == "baseline":
            from .baselines import evaluate_baseline
            evaluator = evaluate_baseline
        report = evaluator(dataset, args.checkpoint, args.split, args.batch_size, args.device, args.output)
        print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
