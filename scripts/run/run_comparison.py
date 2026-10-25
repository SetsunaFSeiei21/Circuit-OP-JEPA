#!/usr/bin/env python3
"""Run three matched methods; one immutable split and train-label budget."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true', help='Force exactly one epoch per phase; mechanics only')
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
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        for key in ('pretrain_epochs', 'decoder_epochs', 'baseline_epochs'):
            config['training'][key] = 1
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
        for mode in ('zerosim_pin', 'zerosim_device', 'jepa_zerosim'):
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
    compare(['--reports', *reports, '--output', str(out/'comparison')])
    print(f"Comparison files: {out/'comparison/summary.csv'} | {out/'comparison/per_seed.csv'} | {out/'comparison/costs.csv'} | {out/'comparison/comparison.json'}", flush=True)
    print('Completed three-mode smoke: mechanics only.' if args.smoke else 'Completed matched three-mode experiment.')


if __name__ == '__main__':
    main()
