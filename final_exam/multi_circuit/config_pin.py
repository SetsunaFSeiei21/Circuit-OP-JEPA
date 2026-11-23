"""Deterministic Cartesian grids and independent model defaults (stdlib only)."""
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
from final_exam.run_pin import MODEL_NAMES_PIN

BASELINES_PIN = tuple(MODEL_NAMES_PIN[:8])
ROOT = Path(__file__).resolve().parent


def read_pin(path):
    return json.loads(Path(path).read_text())


def digest_pin(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_pin(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def set_path_pin(config, path, value):
    target = config
    keys = path.split('.')
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise ValueError(f'Unknown grid path: {path}')
        target = target[key]
    if keys[-1] not in target:
        raise ValueError(f'Unknown grid path: {path}')
    target[keys[-1]] = copy.deepcopy(value)


def allowed_grid_pin(model):
    if model == 'xgboost_pin':
        return {'final_exam_pin.xgboost.' + k for k in ('num_boost_round', 'eta', 'max_depth',
            'min_child_weight', 'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha', 'max_bin')}
    result = {'op_transfer.' + k for k in ('lr', 'weight_decay', 'batch_size', 'grad_clip')}
    arch = {'dropout', 'encoder_layers'}
    if model == 'mlp_pin':
        arch |= {'ffn_dim'}
    else:
        arch |= {'dim'}
        if model not in ('deepgen_supervised_pin', 'deepgen_op_pretrained_pin'):
            arch |= {'heads'}
        if model not in ('gat_pin', 'gtn_pin', 'accformer_pin', 'deepgen_supervised_pin', 'deepgen_op_pretrained_pin'):
            arch |= {'ffn_dim'}
        if model in ('accformer_pin', 'deepgen_supervised_pin', 'deepgen_op_pretrained_pin'):
            arch |= {'decoder_layers'}
        if model not in BASELINES_PIN or model == 'zerosim_supervised_pin':
            arch |= {'cross_layers', 'decoder_layers', 'decoder_ffn_dim', 'terminal_bias'}
    result |= {'model.' + k for k in arch}
    if model.startswith('width_decoder'):
        result |= {'width_decoder_pin.' + k for k in ('n_experts', 'hidden_dim', 'dropout')}
    if model.startswith('lasd_'):
        result |= {'lasd_moe_pin.' + k for k in ('hidden_dim', 'dropout')}
    return result


def model_config_pin(model, config_dir, protocol):
    if model not in MODEL_NAMES_PIN or '_shuffle_' in model:
        raise ValueError(f'Unsupported joint model: {model}')
    source = Path(config_dir)/'models'/f'{model}.json'
    if not source.is_file():
        raise ValueError(f'Missing model config {source}; run --init-model {model} first')
    config = read_pin(source)
    config['dataset'] = copy.deepcopy(protocol['dataset'])
    config['normalization'] = copy.deepcopy(protocol['normalization'])
    config['final_exam_pin']['models'] = [model]
    if config['final_exam_pin'].get('model_overrides'):
        raise ValueError('Multi-circuit model files must set model directly, without hidden model_overrides')
    return config


def expand_grid_pin(model, config, space):
    if set(space) != {'grid'} or not isinstance(space['grid'], dict):
        raise ValueError('Search file must contain exactly a grid object')
    grid = space['grid']
    unknown = set(grid) - allowed_grid_pin(model)
    if unknown:
        raise ValueError(f'Inactive/unapproved search dimensions for {model}: {sorted(unknown)}')
    for key, values in grid.items():
        if not isinstance(values, list) or not values or len({digest_pin(v) for v in values}) != len(values):
            raise ValueError(f'Grid {key} needs distinct nonempty choices')
    trials, skipped = [], []
    for values in itertools.product(*grid.values()):
        cfg = copy.deepcopy(config)
        params = dict(zip(grid, values))
        for path, value in params.items():
            set_path_pin(cfg, path, value)
        m = cfg['model']
        if any(type(m[k]) is not int or m[k] < 1 for k in ('dim', 'heads', 'cross_layers', 'encoder_layers')):
            raise ValueError('Dimensions, heads and depths must be positive integers')
        if m['dim'] < 2 or m['dim'] % m['heads'] or not 0 < m['cross_layers'] < m['encoder_layers']:
            skipped.append({'parameters': params, 'reason': 'head divisibility or PE/encoder depth'})
            continue
        trials.append({'id': 'trial_' + digest_pin(params)[:12] + '_pin', 'parameters': params, 'config': cfg})
    if not trials:
        raise ValueError(f'No valid grid combinations for {model}')
    return trials, skipped


def init_model_pin(model, config_dir):
    if model not in MODEL_NAMES_PIN or '_shuffle_' in model:
        raise ValueError('Choose a registered, non-shuffle pin model')
    cfg = read_pin(ROOT/'configs_pin'/'template_pin.json')
    cfg['final_exam_pin']['models'] = [model]
    path = Path(config_dir)/'models'/f'{model}.json'
    gridpath = Path(config_dir)/'search_spaces'/f'{model}.json'
    if path.exists() or gridpath.exists():
        raise ValueError('Refusing to overwrite existing model/search configuration')
    atomic_pin(path, cfg)
    atomic_pin(gridpath, {'grid': {'op_transfer.lr': [0.0001, 0.0003]}} if model != 'xgboost_pin' else
               {'grid': {'final_exam_pin.xgboost.eta': [0.03, 0.1]}})
    return path, gridpath
