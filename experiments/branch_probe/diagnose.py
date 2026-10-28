#!/usr/bin/env python3
"""Diagnose fixed-device representation variation and fit simple frozen readouts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.branch_probe.run import result_directory


def sample_limit(value):
    if value.lower() == 'none':
        return None
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError('Use >=2 or none')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Existing canonical JSON; read only')
    parser.add_argument('--pretrained', required=True, help='Existing full pretrain_best.pt; read only')
    parser.add_argument('--output', help='New directory inside experiments/branch_probe/results')
    parser.add_argument('--circuit', help='Required for multi-circuit checkpoints; ID circuit only')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-train-samples', type=sample_limit, default=4096)
    parser.add_argument('--max-val-samples', type=sample_limit, default=1024)
    parser.add_argument('--encoder-batch-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256, help='MLP batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum MLP epochs')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--ridge-alpha', type=float, default=1.)
    parser.add_argument('--std-floor', type=float, default=1e-4)
    parser.add_argument('--stats-only', action='store_true', help='Skip ridge/MLP; inspect frozen features only')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    try:
        output = result_directory(args.output)
        data, checkpoint = Path(args.data).expanduser().resolve(), Path(args.pretrained).expanduser().resolve()
        if not data.is_file() or not checkpoint.is_file():
            raise ValueError('--data and --pretrained must name existing files')
        # Keep --help usable on login nodes without torch.
        from device_jepa.data import load_dataset
        from device_jepa.resident import require_device
        from experiments.branch_probe.probe import prepare_probe
        from experiments.branch_probe.diagnostics import run_diagnostics
        device = require_device(args.device)
        print(f'Loading canonical data (CPU JSON/validation): {data}', flush=True)
        dataset = load_dataset(data)
        print('Validating checkpoint identity and original train/val membership', flush=True)
        context = prepare_probe(dataset, checkpoint, device)
        run_diagnostics(context, output, circuit=args.circuit, seed=args.seed,
                        max_train=args.max_train_samples, max_val=args.max_val_samples,
                        encoder_batch_size=args.encoder_batch_size, batch_size=args.batch_size,
                        epochs=args.epochs, hidden=args.hidden, lr=args.lr, patience=args.patience,
                        ridge_alpha=args.ridge_alpha, std_floor=args.std_floor,
                        stats_only=args.stats_only, device=device)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
