"""Torch-free contracts for shared input factors across PE decoder branches."""

SHARED_DECODER_MODELS_PIN = (
    'width_decoder_shared_kv_pin',
    'width_decoder_shared_attention_pin',
    'width_decoder_shared_ffn_pin',
    'width_decoder_shared_both_pin',
)


def shared_decoder_settings_pin(value=None):
    result = dict(attention_rank=32, ffn_rank=64)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise ValueError('shared_decoder_pin accepts attention_rank and ffn_rank only')
        result.update(value)
    for key, rank in result.items():
        if type(rank) is not int or rank < 1:
            raise ValueError(f'shared_decoder_pin.{key} must be a positive integer')
    return result


def shared_decoder_spec_pin(mode, value=None, model_config=None):
    if mode not in SHARED_DECODER_MODELS_PIN:
        return None
    settings = shared_decoder_settings_pin(value)
    cfg = model_config or {}
    dim, hidden = cfg.get('dim', 128), cfg.get('decoder_ffn_dim', 2048)
    branches, layers = cfg.get('cross_layers', 3), cfg.get('decoder_layers', 2)
    if any(type(v) is not int or v < 1 for v in (dim, hidden, branches, layers)):
        raise ValueError('Shared decoder dimensions and layer counts must be positive integers')
    if branches < 2:
        raise ValueError('Shared decoder requires at least two PE branches (cross_layers >= 2)')
    attention = mode != 'width_decoder_shared_ffn_pin'
    ffn = mode in ('width_decoder_shared_ffn_pin', 'width_decoder_shared_both_pin')
    attention_scope = ('cross_kv' if mode == 'width_decoder_shared_kv_pin'
                       else 'all' if attention else 'none')
    roles = (['multihead_attn.k', 'multihead_attn.v'] if attention_scope == 'cross_kv'
             else [f'{name}.{part}' for name in ('self_attn', 'multihead_attn')
                   for part in ('q', 'k', 'v', 'out')] if attention else [])
    if ffn:
        roles += ['linear1', 'linear2']
    dense, down, up = 0, 0, 0
    for role in roles:
        ins, outs = ((dim, hidden) if role == 'linear1' else (hidden, dim)
                     if role == 'linear2' else (dim, dim))
        rank = settings['ffn_rank' if role.startswith('linear') else 'attention_rank']
        if rank > min(ins, outs):
            raise ValueError(f'Shared decoder rank exceeds matrix dimensions for {role}')
        original, compressed = branches * ins * outs, rank * ins + branches * outs * rank
        if compressed >= original:
            raise ValueError(f'Shared decoder rank must reduce total group parameters for {role}')
        dense += layers * original
        down += layers * rank * ins
        up += layers * branches * outs * rank
    return dict(version=1, scope='performance_decoder_only',
                parameterization='shared_down_independent_up', initialization='from_scratch',
                attention=attention, ffn=ffn, attention_scope=attention_scope,
                attention_rank=settings['attention_rank'] if attention else None,
                ffn_rank=settings['ffn_rank'] if ffn else None,
                dim=dim, decoder_ffn_dim=hidden, branch_count=branches, decoder_layers=layers,
                shared_roles=roles, shared_group_count=len(roles) * layers,
                dense_weight_parameters=dense, shared_down_parameters=down,
                independent_up_parameters=up, shared_weight_parameters=down + up,
                parameter_saving_fraction=1 - (down + up) / dense,
                bias_included=False, factors_trainable=True)


def validate_shared_decoder_contract_pin(mode, spec, feature_dim, branch_count, decoder_layers):
    if mode not in SHARED_DECODER_MODELS_PIN:
        if spec is not None:
            raise ValueError('Non-shared decoder must not claim a shared down basis')
        return
    if not isinstance(spec, dict):
        raise ValueError('Shared decoder requires its saved shared_down_basis contract')
    active = {key: spec[key] for key in ('attention_rank', 'ffn_rank')
              if spec.get(key) is not None}
    expected = shared_decoder_spec_pin(mode, active, dict(dim=feature_dim,
        decoder_ffn_dim=spec.get('decoder_ffn_dim'), cross_layers=branch_count,
        decoder_layers=decoder_layers))
    if (set(spec) != set(expected)
            or any(type(spec[key]) is not type(value) or spec[key] != value
                   for key, value in expected.items())):
        raise ValueError('Shared decoder contract is inconsistent with model ID or dimensions')
