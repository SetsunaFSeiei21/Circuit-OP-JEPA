"""Deterministic Cartesian grids and independent model defaults (stdlib only)."""
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
from final_exam.run_pin import MODEL_NAMES_PIN, CASCADE_MODELS_PIN, ALL_GATE_MODELS_PIN, WIDTH_MODELS_PIN
from final_exam.lasd_moe_registry_pin import GRAPH_LAYER_POWER_MODELS_PIN, LASD_MOE_MODELS_PIN

BASELINES_PIN = tuple(MODEL_NAMES_PIN[:8])
ROOT = Path(__file__).resolve().parent
OP_MODELS_PIN = frozenset(CASCADE_MODELS_PIN) | (frozenset(ALL_GATE_MODELS_PIN) -
    {'device_op_gate_only_base_pin'}) | frozenset(WIDTH_MODELS_PIN) | frozenset(LASD_MOE_MODELS_PIN) | {
        'deepgen_op_pretrained_pin'}
OP_TRAINING_DEFAULTS_PIN = dict(epochs=30, lr=1e-4, batch_size=32, weight_decay=1e-4,
    grad_clip=1., patience=10, progress_every=100, torch_threads=1)


def materialize_op_predictor_pin(config, model):
    """Freeze current defaults in new templates; remain usable without torch."""
    if model not in OP_MODELS_PIN:
        return config
    existing = config.setdefault('op_predictor', {})
    existing['model'] = {**copy.deepcopy(config['model']), **existing.get('model', {})}
    tr, transfer = config.get('training', {}), config.get('op_transfer', {})
    training = dict(OP_TRAINING_DEFAULTS_PIN)
    training.update({key: tr[key] for key in ('batch_size', 'weight_decay', 'grad_clip', 'torch_threads') if key in tr})
    training['lr'] = tr.get('decoder_lr', training['lr'])
    training.update({key: transfer[key] for key in training if key != 'epochs' and key in transfer})
    training['epochs'] = transfer.get('epochs', {}).get('op_predictor', training['epochs'])
    existing['training'] = {**training, **existing.get('training', {})}
    if model == 'width_decoder_op_pin':
        width = dict(n_experts=4, hidden_dim=config['model']['dim'], dropout=config['model']['dropout'])
        width.update(config.get('width_decoder_pin', {}))
        existing['width_decoder_pin'] = {**width, **existing.get('width_decoder_pin', {})}
    return config


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
    if model in GRAPH_LAYER_POWER_MODELS_PIN:
        result.add('lasd_moe_pin.aggregation_distance')
    if model in OP_MODELS_PIN:
        op_arch = {'dim', 'encoder_layers', 'dropout'}
        if model != 'deepgen_op_pretrained_pin':
            op_arch |= {'heads', 'cross_layers', 'ffn_dim', 'terminal_bias'}
        if model == 'width_decoder_op_pin':
            op_arch |= {'decoder_layers', 'decoder_ffn_dim'}
            result |= {'op_predictor.width_decoder_pin.' + key
                       for key in ('n_experts', 'hidden_dim', 'dropout')}
        result |= {'op_predictor.model.' + key for key in op_arch}
        result |= {'op_predictor.training.' + key for key in
                   ('epochs', 'lr', 'batch_size', 'weight_decay', 'grad_clip', 'patience')}
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
    # Older files can opt into the new grid namespace without being edited in
    # place. Materialize once, before applying any downstream trial overrides.
    config = copy.deepcopy(config)
    if any(path.startswith('op_predictor.') for path in grid):
        materialize_op_predictor_pin(config, model)
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
        if model in OP_MODELS_PIN:
            op_model = {**m, **cfg.get('op_predictor', {}).get('model', {})}
            if any(type(op_model[k]) is not int or op_model[k] < 1
                   for k in ('dim', 'heads', 'cross_layers', 'encoder_layers')):
                raise ValueError('OP dimensions, heads and depths must be positive integers')
            if (op_model['dim'] < 2 or op_model['dim'] % op_model['heads'] or
                    not 0 < op_model['cross_layers'] < op_model['encoder_layers']):
                skipped.append({'parameters': params, 'reason': 'OP head divisibility or PE/encoder depth'})
                continue
            if model == 'deepgen_op_pretrained_pin' and any(
                    op_model[k] != m[k] for k in ('dim', 'encoder_layers', 'dropout', 'max_devices')):
                skipped.append({'parameters': params, 'reason':
                    'DeepGEN transfers its encoder: OP and downstream encoder settings must match'})
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
    materialize_op_predictor_pin(cfg, model)
    if model in GRAPH_LAYER_POWER_MODELS_PIN:
        cfg['lasd_moe_pin']['aggregation_distance'] = 2
    path = Path(config_dir)/'models'/f'{model}.json'
    gridpath = Path(config_dir)/'search_spaces'/f'{model}.json'
    if path.exists() or gridpath.exists():
        raise ValueError('Refusing to overwrite existing model/search configuration')
    atomic_pin(path, cfg)
    atomic_pin(gridpath, {'grid': {'op_transfer.lr': [0.0001, 0.0003]}} if model != 'xgboost_pin' else
               {'grid': {'final_exam_pin.xgboost.eta': [0.03, 0.1]}})
    return path, gridpath
