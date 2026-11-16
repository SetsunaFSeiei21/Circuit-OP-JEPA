"""Torch-free contracts for the user-selected node-memory LASD experiment."""
import math

LASD_MODELS_PIN = ('width_decoder_score_pin', 'width_decoder_lasd_pin')


def lasd_settings_pin(value=None):
    result = dict(enabled=True, hidden_dim=128, dropout=0., score_chunk_size=128,
                  checkpoint_scores=True, diagnostic_samples=8)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise ValueError('Unknown lasd_pin configuration keys')
        result.update(value)
    for key in ('enabled', 'checkpoint_scores'):
        if type(result[key]) is not bool:
            raise ValueError(f'lasd_pin.{key} must be boolean')
    for key in ('hidden_dim', 'score_chunk_size', 'diagnostic_samples'):
        if type(result[key]) is not int or result[key] < (0 if key == 'diagnostic_samples' else 1):
            raise ValueError(f'Invalid lasd_pin.{key}')
    if (type(result['dropout']) not in (int, float) or not math.isfinite(result['dropout'])
            or not 0 <= result['dropout'] < 1):
        raise ValueError('lasd_pin.dropout must be finite in [0, 1)')
    return result


def lasd_spec_pin(mode, value, model_config):
    if mode not in LASD_MODELS_PIN:
        return None
    options = lasd_settings_pin(value)
    dim, heads = model_config['dim'], model_config['heads']
    layers = model_config['cross_layers']
    offset = model_config['encoder_layers'] - layers
    dynamics = mode == 'width_decoder_lasd_pin'
    features = (4 if dynamics else 2) * heads
    hidden = options['hidden_dim']
    return dict(version=1, enabled=options['enabled'], dynamics=dynamics,
        attention_scope='pe_self_attention_only', heads=heads, pe_layers=layers,
        head_reduction='none', difference='same_head_index' if dynamics else 'none',
        first_delta='zero_boundary_fill_not_observed',
        score_order='R0,T0,...,R(H-1),T(H-1); then dR0,dT0,... when dynamics enabled',
        attention_weights='softmax_after_original_mask_before_attention_dropout',
        diagonal='excluded_after_softmax_without_renormalization',
        graph_token='included_and_separately_identified', padding='exclude_both_query_and_key',
        node_order='unchanged_graph_token_then_native_pin_port_order',
        layer_masks=['local' if (offset+i) % 2 == 0 else 'global' for i in range(layers)],
        self_connections=True, score_dim=features, adapter_input_dim=dim+features,
        adapter_hidden_dim=hidden, adapter_output_dim=dim, adapter_dropout=options['dropout'],
        adapter='independent_per_pe_linear_relu_dropout_linear',
        fusion='concat_projected_pe_memory_and_scores_before_existing_decoder',
        encoder_feedback=False, detach_scores=False, decoder_queries='independent_per_pe',
        final_readout='concat_all_pe_decoder_features_then_existing_moe',
        graph_pooling='none', normalization='none_raw_scores',
        adapter_parameters=(layers*((dim+features)*hidden+hidden+hidden*dim+dim)
                            if options['enabled'] else 0),
        score_chunk_size=options['score_chunk_size'], checkpoint_scores=options['checkpoint_scores'])


def validate_lasd_contract_pin(mode, spec, feature_dim, branch_count):
    if mode not in LASD_MODELS_PIN:
        if spec is not None:
            raise ValueError('A non-LASD model cannot claim LASD score fusion')
        return
    if not isinstance(spec, dict):
        raise ValueError('LASD model requires its saved score contract')
    masks = spec.get('layer_masks', [])
    if len(masks) != branch_count or not masks or masks[0] not in ('local', 'global'):
        raise ValueError('Invalid LASD layer mask metadata')
    # The full effective configuration is checked separately at export/rebuild.
    offset = 2 if masks[0] == 'local' else 1
    cfg = dict(dim=feature_dim, heads=spec.get('heads'), cross_layers=branch_count,
               encoder_layers=branch_count+offset)
    if type(cfg['heads']) is not int or cfg['heads'] < 1 or feature_dim % cfg['heads']:
        raise ValueError('Invalid LASD head metadata')
    options = dict(enabled=spec.get('enabled'), hidden_dim=spec.get('adapter_hidden_dim'),
                   dropout=spec.get('adapter_dropout'), score_chunk_size=spec.get('score_chunk_size'),
                   checkpoint_scores=spec.get('checkpoint_scores'))
    if spec != lasd_spec_pin(mode, options, cfg):
        raise ValueError('LASD architecture contract is inconsistent')
