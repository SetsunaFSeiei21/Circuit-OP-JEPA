"""Independent OP pretraining configuration with backwards-compatible inheritance.

Importing this module is safe on a login node; runtime validators are imported
only when resolving a configuration. The caller supplies its already-effective
legacy model configuration (including any baseline architecture override).
"""
import copy


OP_TRAINING_KEYS = frozenset({
    'epochs', 'lr', 'batch_size', 'weight_decay', 'grad_clip', 'patience',
    'progress_every', 'torch_threads',
})


def predictor_config(config, smoke=False):
    """Return the effective OP-only config without mutating downstream settings.

``op_predictor.model`` overrides the inherited encoder architecture;
``op_predictor.training.epochs`` overrides only the OP stage budget. Remaining
training fields override their corresponding ``op_transfer`` options. Absent
fields retain the historical shared values, so old run/checkpoint configs work.
    """
    from device_jepa.model import ModelConfig
    from experiments.op_transfer.engine import settings

    value = copy.deepcopy(config)
    section = value.get('op_predictor', {})
    if not isinstance(section, dict) or set(section) - {'model', 'training', 'width_decoder_pin'}:
        raise ValueError('op_predictor accepts model, training and width_decoder_pin only')
    model = section.get('model', {})
    training = section.get('training', {})
    if not isinstance(model, dict):
        raise ValueError('op_predictor.model must be an object')
    if not isinstance(training, dict) or set(training) - OP_TRAINING_KEYS:
        raise ValueError('Unknown op_predictor.training settings; accepted keys: '
                         + ', '.join(sorted(OP_TRAINING_KEYS)))
    # Historical resumable checkpoints hash the literal stage config. Do not
    # materialize defaults in old configs: preserving behavior alone would still
    # change their identity and prevent an otherwise safe recovery.
    if not section:
        try:
            ModelConfig(**value['model'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid op_predictor.model: {exc}') from exc
        settings(value)
        if smoke:
            value.setdefault('op_transfer', {}).setdefault('epochs', {})['op_predictor'] = 1
            value['op_transfer']['patience'] = 0
        return value
    try:
        value['model'] = ModelConfig(**{**value['model'], **model}).to_dict()
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid op_predictor.model: {exc}') from exc
    options = settings(value)
    for key, item in training.items():
        if key == 'epochs':
            options['epochs']['op_predictor'] = item
        else:
            options[key] = item
    # Validate explicit values even for smoke jobs before applying the short run.
    value['op_transfer'] = options
    value['op_transfer'] = settings(value)
    if smoke:
        value['op_transfer']['epochs']['op_predictor'] = 1
        value['op_transfer']['patience'] = 0
        if 'op_predictor' in value:
            value['op_predictor'].setdefault('training', {}).update(epochs=1, patience=0)
    if 'width_decoder_pin' in section or 'width_decoder_pin' in value:
        from final_exam.width_registry_pin import width_settings_pin
        width = section.get('width_decoder_pin', {})
        if not isinstance(width, dict):
            raise ValueError('op_predictor.width_decoder_pin must be an object')
        value['width_decoder_pin'] = width_settings_pin(
            {**value.get('width_decoder_pin', {}), **width}, value['model'])
    return value


def validate_encoder_transfer(config, resolved):
    """DeepGEN fine-tunes the same encoder; its architecture must stay compatible."""
    from device_jepa.model import ModelConfig
    downstream_model = ModelConfig(**config['model']).to_dict()
    op_model = ModelConfig(**resolved['model']).to_dict()
    keys = ('dim', 'encoder_layers', 'dropout', 'max_devices')
    different = [key for key in keys if downstream_model[key] != op_model[key]]
    if different:
        raise ValueError('DeepGEN OP pretraining transfers the encoder directly; '
                         'op_predictor.model must match downstream model for: '
                         + ', '.join(different)
                         + '. OP training hyperparameters may be independent.')
