#!/usr/bin/env python3
"""Read out both frozen branches of ONE pretrained JEPA checkpoint."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def result_directory(value=None):
    """Keep experimental output inside this folder, including symlink targets."""
    base = EXPERIMENT_ROOT / 'results'
    if base.resolve() != base:
        raise ValueError('branch_probe/results must not redirect outside the experiment folder')
    path = Path(value).expanduser().resolve() if value else base / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    if path == base or base not in path.parents:
        raise ValueError(f'--output must be a run subdirectory of {base}')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output must be new or empty; refusing to overwrite an experiment')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Existing canonical data.json, read only')
    parser.add_argument('--pretrained', required=True, help='Full pretrain_best.pt, not decoder_best.pt')
    parser.add_argument('--output', help='New run directory inside experiments/branch_probe/results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17], help='Decoder seeds; pretraining stays fixed')
    parser.add_argument('--epochs', type=int, help='Override source training.decoder_epochs for BOTH probes')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float, help='Override source training.decoder_lr for BOTH probes')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--splits', nargs='+', choices=('val', 'test', 'ood'), default=['val'],
                        help='Default val only; test/OOD require explicit opt-in and never choose checkpoints')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('--seeds must be unique integers in [0, 2**32)')
    if len(set(args.splits)) != len(args.splits):
        parser.error('--splits must be unique')
    try:
        output = result_directory(args.output)
        data_path, checkpoint_path = Path(args.data).expanduser().resolve(), Path(args.pretrained).expanduser().resolve()
        if not data_path.is_file() or not checkpoint_path.is_file():
            raise ValueError('--data and --pretrained must name existing files')
        # Lazy imports make --help usable on login nodes without torch.
        from device_jepa.data import load_dataset
        from device_jepa.resident import require_device
        from experiments.branch_probe.probe import prepare_probe, run_pair, probe_training, fit_probe_preprocessor
        from experiments.branch_probe.summarize import summarize_reports
        device = require_device(args.device)
        overrides = {key: value for key, value in (
            ('decoder_epochs', args.epochs), ('batch_size', args.batch_size), ('decoder_lr', args.lr)) if value is not None}
        print(f'Loading source data: {data_path}', flush=True)
        dataset = load_dataset(data_path)
        context = prepare_probe(dataset, checkpoint_path, device=device)
        probe_training(context, args.seeds[0], overrides)  # validate before ANY output write
        prep = fit_probe_preprocessor(context, device)
        active = [name for name, count in zip(prep['feature_spec']['metric_names'],
                                             prep['performance']['count']) if count > 0]
        if not any(row['performance'].get(name) is not None for row in context['split_rows']['val']
                   for name in active):
            raise ValueError('Validation has no observed targets for trained metrics')
        for split in args.splits:
            if not context['split_rows'][split]:
                raise ValueError(f'Requested evaluation split {split!r} is empty')
    except ValueError as exc:
        parser.error(str(exc))
    plan = dict(experiment_kind='frozen_branch_probe', source_data=str(data_path),
                source_checkpoint=str(checkpoint_path), source_checkpoint_sha256=context['source_sha256'],
                source_mode=context['source'].get('mode', 'jepa_zerosim'),
                pretrain_seed=context['source']['options']['training']['seed'],
                decoder_seeds=args.seeds, evaluation_splits=args.splits, training_overrides=overrides,
                device=str(device), complete=False,
                seed_interpretation='decoder initializations conditional on one fixed pretrained checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'experiment.json'
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Paired frozen-branch experiment: {output}', flush=True)
    print('Same source checkpoint; identical decoder initialization per pair; OP required only for the OP probe.', flush=True)
    reports = []
    for seed in args.seeds:
        reports.extend(run_pair(context, output / f'seed_{seed}', seed, overrides,
                                device=device, eval_splits=tuple(args.splits)))
    summarize_reports(reports, output / 'comparison')
    plan['complete'] = True
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Complete. Summary and Excel: {output / "comparison"}', flush=True)


if __name__ == '__main__':
    main()
