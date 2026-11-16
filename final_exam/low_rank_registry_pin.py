"""Torch-free contracts for independent low-rank performance decoder factors."""

LOW_RANK_MODELS_PIN = (
    'width_decoder_lowrank_attention_pin',
    'width_decoder_lowrank_ffn_pin',
    'width_decoder_lowrank_both_pin',
)


def low_rank_settings_pin(value=None):
    result = dict(attention_rank=32, ffn_rank=64)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise ValueError('low_rank_decoder_pin accepts attention_rank and ffn_rank only')
        result.update(value)
    for key, rank in result.items():
        if type(rank) is not int or rank < 1:
            raise ValueError(f'low_rank_decoder_pin.{key} must be a positive integer')
    return result


def low_rank_spec_pin(mode, value=None, model_config=None):
    if mode not in LOW_RANK_MODELS_PIN:
        return None
    settings = low_rank_settings_pin(value)
    cfg = model_config or {}
    dim, hidden = cfg.get('dim', 128), cfg.get('decoder_ffn_dim', 2048)
    if any(type(size) is not int or size < 1 for size in (dim, hidden)):
        raise ValueError('Low-rank decoder dimensions must be positive integers')
    attention = mode != 'width_decoder_lowrank_ffn_pin'
    ffn = mode != 'width_decoder_lowrank_attention_pin'
    if attention and settings['attention_rank'] * (2 * dim) >= dim * dim:
        raise ValueError('attention_rank must reduce projection parameters: 2 * rank < dim')
    if ffn and settings['ffn_rank'] * (dim + hidden) >= dim * hidden:
        raise ValueError('ffn_rank must reduce parameters: rank * (dim + decoder_ffn_dim) '
                         '< dim * decoder_ffn_dim')
    return dict(version=1, scope='performance_decoder_only',
                parameterization='independent_weight_factors', initialization='from_scratch',
                attention=attention, ffn=ffn,
                attention_rank=settings['attention_rank'] if attention else None,
                ffn_rank=settings['ffn_rank'] if ffn else None,
                dim=dim, decoder_ffn_dim=hidden)


def validate_low_rank_contract_pin(mode, spec, feature_dim):
    if mode not in LOW_RANK_MODELS_PIN:
        if spec is not None:
            raise ValueError('Dense wide decoder must not claim low-rank factors')
        return
    if not isinstance(spec, dict):
        raise ValueError('Low-rank decoder requires its saved low_rank contract')
    active = {key: spec[key] for key in ('attention_rank', 'ffn_rank')
              if spec.get(key) is not None}
    expected = low_rank_spec_pin(mode, active,
        dict(dim=feature_dim, decoder_ffn_dim=spec.get('decoder_ffn_dim')))
    if (set(spec) != set(expected)
            or any(type(spec[key]) is not type(value) or spec[key] != value
                   for key, value in expected.items())):
        raise ValueError('Low-rank decoder contract is inconsistent with model ID or dimensions')
