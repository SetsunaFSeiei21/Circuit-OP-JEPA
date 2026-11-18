"""Torch-free configuration and exact contracts for node-wise LASD-MoE fusion."""
import math

HEADWISE_LASD_MOE_MODELS_PIN = ('lasd_moe_headwise_pin', 'lasd_moe_headwise_detach_pin')
LASD_MOE_MODELS_PIN = ('lasd_moe_pin', 'lasd_moe_detach_pin', *HEADWISE_LASD_MOE_MODELS_PIN)


def lasd_moe_settings_pin(value=None):
    result = dict(n_experts=4, top_k=2, hidden_dim=128, dropout=0.,
                  score_chunk_size=128, checkpoint_scores=True, diagnostic_samples=8)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise ValueError('Unknown lasd_moe_pin configuration keys')
        result.update(value)
    for key in ('n_experts', 'top_k', 'hidden_dim', 'score_chunk_size', 'diagnostic_samples'):
        if type(result[key]) is not int or result[key] < (0 if key == 'diagnostic_samples' else 1):
            raise ValueError(f'Invalid lasd_moe_pin.{key}')
    if result['n_experts'] != 4 or result['top_k'] != 2:
        raise ValueError('LASD-MoE requires four experts and top-2 selection')
    if type(result['checkpoint_scores']) is not bool:
        raise ValueError('lasd_moe_pin.checkpoint_scores must be boolean')
    if (type(result['dropout']) not in (int, float) or not math.isfinite(result['dropout'])
            or not 0 <= result['dropout'] < 1):
        raise ValueError('lasd_moe_pin.dropout must be finite in [0, 1)')
    return result


def lasd_moe_spec_pin(mode, value, model_config):
    if mode not in LASD_MOE_MODELS_PIN:
        return None
    options = lasd_moe_settings_pin(value)
    layers, heads = model_config['cross_layers'], model_config['heads']
    offset = model_config['encoder_layers'] - layers
    spec = dict(version=1, mode=mode, attention_scope='pe_self_attention_only',
        base_layers=layers, encoder_layers=model_config['encoder_layers'], structure_layers=offset,
        pe_order=list(range(1, layers+1)),
        fusion_layers=model_config['cross_layers'], heads=heads, feature_dim=model_config['dim'],
        input_dim=4*layers*heads, score_axes='B,N,L,H,2', head_reduction='none',
        difference='same_head_index_no_semantic_alignment',
        first_delta='zero_boundary_fill_not_observed',
        feature_order='layer_then_head_then_R_T_deltaR_deltaT',
        attention_weights='softmax_after_original_mask_before_attention_dropout',
        diagonal='excluded_after_softmax_without_renormalization', self_connections=True,
        layer_masks=['local' if (offset+i) % 2 == 0 else 'global' for i in range(layers)],
        graph_token='included_as_original_index_zero',
        padding='exclude_query_key_routing_and_fusion',
        node_order='unchanged_graph_token_then_native_pin_port_order',
        layer_memory='shared_existing_output_norm_and_projection_per_pe_block',
        normalization='none_raw_scores', score_detach=mode in ('lasd_moe_detach_pin', 'lasd_moe_headwise_detach_pin'),
        hidden_state_detach=False, routing='selected_top2_logits_softmax',
        expert='independent_linear_gelu_dropout_linear',
        experts=options['n_experts'], top_k=options['top_k'], hidden_dim=options['hidden_dim'],
        dropout=options['dropout'], auxiliary_loss=False,
        layer_fusion='node_wise_softmax_weighted_sum',
        final_encoder_layers=1, final_encoder='native_encoder_block_without_cross_attention',
        final_attention_mask='original_local_topology_self_graph_and_terminal_role_bias',
        decoder='one_existing_metric_query_decoder', decoder_layers=model_config['decoder_layers'],
        op_predictor='ordinary_frozen_device_op_predictor', requires_measured_op=False,
        score_chunk_size=options['score_chunk_size'], checkpoint_scores=options['checkpoint_scores'])
    if mode in HEADWISE_LASD_MOE_MODELS_PIN:
        if model_config['dim'] % heads:
            raise ValueError('Headwise PE fusion requires feature width divisible by head count')
        spec.update(version=2, layer_fusion='node_wise_headwise_softmax_weighted_sum',
                    layer_weight_axes='B,N,H,L', expert_output_dim=heads * layers,
                    channels_per_head=model_config['dim'] // heads,
                    fusion_channels='contiguous_groups_of_projected_pe_memory',
                    expert_selection_scope='one_top2_pair_per_node')
    return spec


def validate_lasd_moe_contract_pin(mode, spec, model_config=None, value=None):
    if mode not in LASD_MOE_MODELS_PIN:
        raise ValueError('Unknown LASD-MoE model')
    if not isinstance(spec, dict):
        raise ValueError('LASD-MoE requires an architecture contract')
    if model_config is None:
        model_config = dict(encoder_layers=spec.get('encoder_layers'),
            cross_layers=spec.get('fusion_layers'), heads=spec.get('heads'),
            dim=spec.get('feature_dim'), decoder_layers=spec.get('decoder_layers'))
        if any(type(v) is not int or v < 1 for v in model_config.values()):
            raise ValueError('Invalid LASD-MoE architecture dimensions')
        if (model_config['encoder_layers'] <= model_config['cross_layers']
                or model_config['dim'] % model_config['heads']):
            raise ValueError('Invalid LASD-MoE encoder layout')
        if value is None:
            value = dict(n_experts=spec.get('experts'), top_k=spec.get('top_k'),
                hidden_dim=spec.get('hidden_dim'), dropout=spec.get('dropout'),
                score_chunk_size=spec.get('score_chunk_size'),
                checkpoint_scores=spec.get('checkpoint_scores'))
    if spec != lasd_moe_spec_pin(mode, value, model_config):
        raise ValueError('LASD-MoE architecture contract disagrees with saved configuration')
