#!/usr/bin/env python3
"""Run five matched methods; one immutable split and train-label budget."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from device_jepa.comparison import MODES
from device_jepa.budgets import comparison_epoch_budget, smoke_epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES),
                        help='Methods to run; defaults to all five. List the original three for a legacy experiment.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true', help='JEPA 1+1 epochs, baselines 2; mechanics only')
    args = parser.parse_args()
    data = Path(args.data).resolve()
    config = json.loads(Path(args.config).read_text())
    # Fail before producing a partial experiment when dependencies are absent.
    import torch  # noqa: F401
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be new/empty; use a new experiment directory, never overwrite results')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    if len(set(args.modes)) != len(args.modes):
        parser.error('Modes must be unique')
    if args.smoke:
        smoke_epochs(config)
    from device_jepa.training import _check_options
    _check_options(config)
    budgets = comparison_epoch_budget(config, args.modes)
    from device_jepa.resident import require_device
    require_device(args.device)
    print('Epoch budgets (equal total): ' + json.dumps(budgets, sort_keys=True), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    def run(*parts):
        command = [sys.executable, '-m', 'device_jepa', *map(str, parts)]
        print('+ ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    base_config = out/'config.json'
    base_config.write_text(json.dumps(config, indent=2)+'\n')
    manifest = out/'manifest.json'
    run('split', '--data', data, '--config', base_config, '--output', manifest)
    split = json.loads(manifest.read_text())
    evaluations = [name for name in ('test', 'ood') if split[name]]
    if 'test' not in evaluations:
        parser.error('Comparison requires a nonempty held-out test split')
    if 'ood' not in evaluations:
        print('NO OOD HOLDOUT: this run cannot support cross-circuit generalization claims.', flush=True)
    reports = []
    for seed in args.seeds:
        options = copy.deepcopy(config)
        options['training']['seed'] = seed
        cfg = out/f'config_seed_{seed}.json'
        cfg.write_text(json.dumps(options, indent=2)+'\n')
        for mode in args.modes:
            dest = out/mode/f'seed_{seed}'
            common = ['--data', data, '--config', cfg, '--output', dest, '--device', args.device]
            if mode == 'jepa_zerosim':
                run('pretrain', *common, '--manifest', manifest)
                run('train_decoder', *common, '--pretrained', dest/'pretrain_best.pt')
                checkpoint = dest/'decoder_best.pt'
            else:
                run('train_baseline', '--mode', mode, *common, '--manifest', manifest)
                checkpoint = dest/f'{mode}_best.pt'
            for subset in evaluations:
                report = dest/f'{subset}_report.json'
                run('evaluate', '--data', data, '--checkpoint', checkpoint, '--split', subset,
                    '--device', args.device, '--output', report)
                reports.append(str(report))
    from device_jepa.comparison import main as compare
    compare(['--reports', *reports, '--output', str(out/'comparison'), '--modes', *args.modes])
    print(f"Comparison files: {out/'comparison/summary.csv'} | {out/'comparison/per_seed.csv'} | {out/'comparison/costs.csv'} | {out/'comparison/evaluation_costs.csv'} | {out/'comparison/comparison.json'}", flush=True)
    if (out/'comparison/paired_summary.csv').exists():
        print(f"OP/no-OP paired deltas: {out/'comparison/paired_summary.csv'} | {out/'comparison/paired_per_seed.csv'}", flush=True)
    print(f'Completed {len(args.modes)}-mode smoke: mechanics only.' if args.smoke else f'Completed matched {len(args.modes)}-mode experiment.')


if __name__ == '__main__':
    main()
