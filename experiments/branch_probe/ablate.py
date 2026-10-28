#!/usr/bin/env python3
"""Compare attention/MLP x raw/standardized frozen features, on both JEPA branches."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.branch_probe.diagnose import sample_limit
from experiments.branch_probe.run import result_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Existing canonical JSON; read only')
    parser.add_argument('--pretrained', required=True, help='Full pretrain_best.pt; read only')
    parser.add_argument('--output', help='New directory inside experiments/branch_probe/results')
    parser.add_argument('--circuit', help='ID circuit only; required for multi-circuit checkpoints')
    parser.add_argument('--seeds', type=int, nargs='+', default=[17], help='Decoder seeds, not pretraining seeds')
    parser.add_argument('--sample-seed', type=int, default=17, help='Fixed subset independent of decoder seeds')
    parser.add_argument('--max-train-samples', type=sample_limit, default=4096)
    parser.add_argument('--max-val-samples', type=sample_limit, default=1024)
    parser.add_argument('--encoder-batch-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256, help='Shared by all readouts')
    parser.add_argument('--epochs', type=int, default=50, help='All readouts run this many epochs by default')
    parser.add_argument('--hidden', type=int, default=128, help='MLP width; attention uses checkpoint model config')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--patience', type=int, default=0, help='0: fixed epochs; >0: shared early-stopping rule')
    parser.add_argument('--std-floor', type=float, default=1e-4)
    parser.add_argument('--progress-every', type=int, default=25)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    try:
        output = result_directory(args.output)
        data, checkpoint = Path(args.data).expanduser().resolve(), Path(args.pretrained).expanduser().resolve()
        if not data.is_file() or not checkpoint.is_file():
            raise ValueError('--data and --pretrained must name existing files')
        # --help remains usable on login nodes without torch.
        from device_jepa.data import load_dataset
        from device_jepa.resident import require_device
        from experiments.branch_probe.ablation import run_ablation, validate_options
        from experiments.branch_probe.probe import prepare_probe
        settings = {name: getattr(args, name) for name in ('epochs', 'batch_size', 'encoder_batch_size',
            'hidden', 'lr', 'weight_decay', 'grad_clip', 'patience', 'std_floor', 'progress_every')}
        validate_options(seeds=args.seeds, sample_seed=args.sample_seed, **settings)
        device = require_device(args.device)
        print(f'Loading canonical data (CPU JSON/validation): {data}', flush=True)
        dataset = load_dataset(data)
        print('Validating source checkpoint and original split membership', flush=True)
        context = prepare_probe(dataset, checkpoint, device)
        run_ablation(context, output, circuit=args.circuit, seeds=args.seeds, sample_seed=args.sample_seed,
                     max_train=args.max_train_samples, max_val=args.max_val_samples, device=device, **settings)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
