"""Command-line entrypoints; data-only commands do not require PyTorch."""
import argparse
import json
from pathlib import Path
from importlib.resources import files

from .data import build_manifest, load_dataset, synthetic_dataset


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Device-token OP alignment followed by frozen-encoder metric decoding")
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("make_synthetic", help="Generate synthetic fixtures, NOT analog simulation data")
    synth.add_argument("--output", required=True)
    synth.add_argument("--per-circuit", type=int, default=24)
    synth.add_argument("--seed", type=int, default=17)
    split = commands.add_parser("split", help="Four ID circuits with grouped splits; fifth completely held out")
    split.add_argument("--data", required=True)
    split.add_argument("--ood-circuit", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=17)
    split.add_argument("--val-fraction", type=float, default=.15)
    split.add_argument("--test-fraction", type=float, default=.15)
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
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["train", "val", "test", "ood"], default="test")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--device", default="cpu")
    smoke = commands.add_parser("smoke", help="Reproducible SYNTHETIC end-to-end pipeline")
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if args.command == "make_synthetic":
        _write(args.output, synthetic_dataset(args.per_circuit, args.seed))
        print(f"Wrote SYNTHETIC fixture: {args.output}")
        return
    if args.command == "split":
        dataset = load_dataset(args.data)
        manifest = build_manifest(dataset, args.ood_circuit, args.seed, args.val_fraction, args.test_fraction)
        _write(args.output, manifest)
        print(json.dumps({key: len(manifest[key]) for key in ("train", "val", "test", "ood")}))
        return
    from .training import evaluate, pretrain, seed_everything, train_decoder
    if args.command == "smoke":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        config = _read(args.config) if args.config else json.loads(files("device_jepa").joinpath("defaults/smoke.json").read_text())
        dataset = synthetic_dataset(24, config["training"]["seed"])
        _write(output / "synthetic.json", dataset)
        dataset = load_dataset(output / "synthetic.json")
        manifest = build_manifest(dataset, "synthetic_4", config["training"]["seed"])
        _write(output / "manifest.json", manifest)
        p = pretrain(dataset, manifest, config, output)
        d = train_decoder(dataset, p, config, output)
        for split in ("test", "ood"):
            report = evaluate(dataset, d, split=split, output=output / f"{split}_report.json")
            print(f"SYNTHETIC {split}: {json.dumps(report['metrics'])}")
        print("Smoke complete: mechanics only; synthetic metrics are NOT evidence of analog-circuit generalization.")
        return
    dataset = load_dataset(args.data)
    if args.command == "pretrain":
        print(pretrain(dataset, _read(args.manifest), _read(args.config), args.output, args.device))
    elif args.command == "train_decoder":
        print(train_decoder(dataset, args.pretrained, _read(args.config), args.output, args.device))
    else:
        seed_everything(17)
        report = evaluate(dataset, args.checkpoint, args.split, args.batch_size, args.device, args.output)
        print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
