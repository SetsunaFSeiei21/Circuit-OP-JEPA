#!/usr/bin/env python3
"""Run selected OP-transfer experiments on one shared canonical dataset/split."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAMES = (
    'design_supervised', 'op_oracle', 'jepa_standardized', 'jepa_grouped',
    'jepa_residual', 'jepa_op_aux', 'op_distill', 'op_distill_finetune',
    'predicted_op', 'predicted_op_joint', 'dual_distill', 'dual_predicted_op',
    'dual_design_control', 'design_op_oracle', 'student_predicted_op',
)
DEFAULT_MODELS = (
    'design_supervised', 'op_oracle', 'op_distill', 'predicted_op',
    'dual_distill', 'dual_design_control',
)


def positive_int(value):
    try:
        number = int(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('Expected a positive integer') from exc
    if number < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return number


def result_directory(value=None):
    """Validate without writing; symlinks cannot escape the experiment folder."""
    base = EXPERIMENT_ROOT / 'results'
    if base.resolve() != base:
        raise ValueError('op_transfer/results must not redirect outside the experiment folder')
    path = Path(value).expanduser().resolve() if value else base / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    if path == base or base not in path.parents:
        raise ValueError(f'--output must be a run subdirectory of {base}')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output must be new or empty; refusing to overwrite an experiment')
    return path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-models', action='store_true', help='List experiment names and exit')
    parser.add_argument('--data', help='Existing canonical data.json; read only')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs/default.json'),
                        help='Model/training/dataset configuration; use the actual circuit config')
    parser.add_argument('--manifest', help='Existing split manifest; recommended for matched comparisons')
    parser.add_argument('--output', help='New directory inside experiments/op_transfer/results')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(DEFAULT_MODELS),
                        help='Only these outcomes and their required prerequisite stages run')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17], help='Unique training seeds')
    parser.add_argument('--device', default='cuda', help='cuda by default; cpu for mechanism tests')
    parser.add_argument('--max-train', type=positive_int, help='Optional deterministic train subset limit')
    parser.add_argument('--max-val', type=positive_int, help='Optional deterministic validation subset limit')
    parser.add_argument('--max-test', type=positive_int, help='Optional evaluation subset limit')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print('\n'.join(MODEL_NAMES))
        return
    if not args.data:
        parser.error('--data is required unless --list-models is used')
    if len(args.models) != len(set(args.models)):
        parser.error('--models must not contain duplicates')
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('--seeds must be unique integers in [0, 2**32)')
    try:
        output = result_directory(args.output)
        data = Path(args.data).expanduser().resolve()
        config = Path(args.config).expanduser().resolve()
        manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
        for label, path in (('--data', data), ('--config', config), ('--manifest', manifest)):
            if path is not None and not path.is_file():
                raise ValueError(f'{label} must name an existing file: {path}')
        # Import torch only after --help/list/path validation, for login-node use.
        from experiments.op_transfer.engine import run_experiment
        run_experiment(data_path=data, config_path=config, output=output,
                       models=args.models, seeds=args.seeds, device=args.device,
                       manifest_path=manifest, max_train=args.max_train,
                       max_val=args.max_val, max_test=args.max_test)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
