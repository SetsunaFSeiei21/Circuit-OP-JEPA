"""Torch-free configuration and exact contracts for node-wise LASD-MoE fusion."""
import math

HEADWISE_LASD_MOE_MODELS_PIN = ('lasd_moe_headwise_pin', 'lasd_moe_headwise_detach_pin')
GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN = (
    'lasd_moe_headwise_global_pin', 'lasd_moe_headwise_detach_global_pin')
RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN = ('lasd_moe_headwise_pin_v1', 'lasd_moe_headwise_detach_pin_v1')
HYPER_LASD_MOE_MODEL_PIN = 'lasd_moe_headwise_detach_hc_pin'
# Keep the singular name as the original detached model for existing callers.
HYPER_LASD_MOE_MODELS_PIN = ('lasd_moe_headwise_hc_pin', HYPER_LASD_MOE_MODEL_PIN)
GRAPH_LAYER_LASD_MODEL_PIN = 'lasd_graph_layer_headwise_pin'
GRAPH_LAYER_LASD_DETACH_MODEL_PIN = 'lasd_graph_layer_headwise_detach_pin'
GRAPH_LAYER_FUSION_DETACH_MODELS_PIN = (
    'lasd_graph_layer_headwise_detach_pin_v1',
    'lasd_graph_layer_headwise_detach_pin_v2',
    'lasd_graph_layer_headwise_detach_pin_v3',
)
GRAPH_LAYER_V3_EXTENSION_MODELS_PIN = (
    'lasd_graph_layer_headwise_detach_pin_v5',
    'lasd_graph_layer_headwise_detach_pin_v6',
)
GRAPH_LAYER_POWER_MODEL_PIN = 'lasd_graph_layer_headwise_detach_pin_v7'
GRAPH_LAYER_POWER_HC_MODEL_PIN = 'lasd_graph_layer_headwise_detach_pin_v8'
GRAPH_LAYER_POWER_HC_NORM_MODEL_PIN = 'lasd_graph_layer_headwise_detach_pin_v9'
GRAPH_LAYER_POWER_HC_MODELS_PIN = (GRAPH_LAYER_POWER_HC_MODEL_PIN, GRAPH_LAYER_POWER_HC_NORM_MODEL_PIN)
GRAPH_LAYER_POWER_MODELS_PIN = (GRAPH_LAYER_POWER_MODEL_PIN, *GRAPH_LAYER_POWER_HC_MODELS_PIN)
PE_HYPER_LASD_MODELS_PIN = (*HYPER_LASD_MOE_MODELS_PIN, GRAPH_LAYER_V3_EXTENSION_MODELS_PIN[0],
                          *GRAPH_LAYER_POWER_HC_MODELS_PIN)
GRAPH_LAYER_SCORE_DETACH_MODELS_PIN = (
    GRAPH_LAYER_LASD_DETACH_MODEL_PIN, *GRAPH_LAYER_FUSION_DETACH_MODELS_PIN,
    *GRAPH_LAYER_V3_EXTENSION_MODELS_PIN, *GRAPH_LAYER_POWER_MODELS_PIN)
# v4 retains the historical model-name prefix, but its score path is live.
# Keep older model contracts and the original v1/v2/v3 tuple unchanged.
GRAPH_LAYER_SHALLOW_DETACH_MODEL_PIN = 'lasd_graph_layer_headwise_detach_pin_v4'
GRAPH_LAYER_LASD_MODELS_PIN = (GRAPH_LAYER_LASD_MODEL_PIN, *GRAPH_LAYER_SCORE_DETACH_MODELS_PIN,
                             GRAPH_LAYER_SHALLOW_DETACH_MODEL_PIN)
LASD_MOE_MODELS_PIN = ('lasd_moe_pin', 'lasd_moe_detach_pin', *HEADWISE_LASD_MOE_MODELS_PIN,
                       *HYPER_LASD_MOE_MODELS_PIN, *RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN,
                       *GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN, *GRAPH_LAYER_LASD_MODELS_PIN)


def lasd_moe_settings_pin(value=None):
    result = dict(n_experts=4, top_k=2, hidden_dim=128, dropout=0.,
                  score_chunk_size=128, checkpoint_scores=True, diagnostic_samples=8)
    if value is not None:
        if not isinstance(value, dict) or set(value) - (set(result) | {
                'hyper_connection_streams', 'global_std_epsilon', 'aggregation_distance'}):
            raise ValueError('Unknown lasd_moe_pin configuration keys')
        result.update(value)
    # Keep older saved settings intact for exact stage resume and checkpoint replay.
    if 'aggregation_distance' in result:
        if type(result['aggregation_distance']) is not int or result['aggregation_distance'] < 1:
            raise ValueError('lasd_moe_pin.aggregation_distance must be a positive integer')
    if 'global_std_epsilon' in result:
        epsilon = result['global_std_epsilon']
        if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('lasd_moe_pin.global_std_epsilon must be finite and positive')
    if 'hyper_connection_streams' in result:
        streams = result['hyper_connection_streams']
        if type(streams) is not int or streams < 1:
            raise ValueError('lasd_moe_pin.hyper_connection_streams must be a positive integer')
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
        normalization='none_raw_scores',
        score_detach=mode in ('lasd_moe_detach_pin', 'lasd_moe_headwise_detach_pin', HYPER_LASD_MOE_MODEL_PIN),
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
    if mode in (*HEADWISE_LASD_MOE_MODELS_PIN, *HYPER_LASD_MOE_MODELS_PIN,
                *RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN, *GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN,
                *GRAPH_LAYER_LASD_MODELS_PIN):
        if model_config['dim'] % heads:
            raise ValueError('Headwise PE fusion requires feature width divisible by head count')
        spec.update(version=2, layer_fusion='node_wise_headwise_softmax_weighted_sum',
                    layer_weight_axes='B,N,H,L', expert_output_dim=heads * layers,
                    channels_per_head=model_config['dim'] // heads,
                    fusion_channels='contiguous_groups_of_projected_pe_memory',
                    expert_selection_scope='one_top2_pair_per_node')
    if mode in PE_HYPER_LASD_MODELS_PIN:
        spec.update(version=3, layer_memory='sum_residual_streams_then_shared_output_norm_and_projection',
                    hyper_connection=dict(streams=options.get('hyper_connection_streams', 4),
                        scope='performance_pe_only', sublayers=['self_attention', 'parameter_op_cross_attention', 'ffn'],
                        independent_per_sublayer=True, dynamic='static_plus_scaled_tanh_of_layernorm_projection',
                        initial_read='pe_index_mod_streams', initial_carry='identity', initial_write=1.,
                        initial_dynamic_projection=0., initial_dynamic_scale=0.01,
                        normalization='native_post_norm_per_stream', readout='sum_each_pe_for_lasd'))
    if mode in RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN:
        detached = mode == 'lasd_moe_headwise_detach_pin_v1'
        spec.update(version=4, residual_connection='last_pe_memory_add',
                    score_detach=detached, hidden_state_detach=detached,
                    hidden_state_detach_scope='fusion_inputs_only', final_pe_residual_detach=False)
    if mode in (*GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN, *GRAPH_LAYER_LASD_MODELS_PIN):
        spec.update(version=5, input_dim=10*layers*heads,
                    score_detach=mode in ('lasd_moe_headwise_detach_global_pin', *GRAPH_LAYER_SCORE_DETACH_MODELS_PIN),
                    feature_order='node_R_T_deltas_flat_then_global_statistics_deltas_flat',
                    score_detach_scope='node_and_global_features_before_moe',
                    global_context=dict(
                        statistics=['mean_R', 'std_R', 'std_T'], axes='B,L,H,3',
                        reduction='valid_nodes_within_each_sample_including_graph_token',
                        variance='population_divisor_N', std='sqrt_variance_plus_epsilon',
                        epsilon=options.get('global_std_epsilon', 1e-8),
                        difference='adjacent_pe_same_head_index',
                        first_delta='zero_boundary_fill_not_observed',
                        feature_order='layer_then_head_then_meanR_stdR_stdT_deltaMeanR_deltaStdR_deltaStdT',
                        input_dim=6*layers*heads, broadcast='same_graph_valid_nodes_only',
                        distribution_assumption='none', running_statistics=False))
    if mode in GRAPH_LAYER_LASD_MODELS_PIN:
        # This routing branch uses two encoder axes and a shared MLP, not experts.
        # Keep all older contracts byte-for-byte unchanged for checkpoint replay.
        ffn_dim, block_dropout = model_config.get('ffn_dim', 512), model_config.get('dropout', .1)
        if type(ffn_dim) is not int or ffn_dim < 1:
            raise ValueError('LASD graph/layer context requires a positive integer model.ffn_dim')
        if (type(block_dropout) not in (int, float) or not math.isfinite(block_dropout)
                or not 0 <= block_dropout < 1):
            raise ValueError('LASD graph/layer context requires model.dropout in [0, 1)')
        spec.pop('expert_output_dim')
        spec.pop('expert_selection_scope')
        spec.update(version=6, experts=0, top_k=0, expert=None,
                    routing='graph_then_nodewise_layer_encoder', routing_output_dim=layers*heads,
                    score_detach_scope=('node_and_global_features_before_graph_layer_router'
                                        if mode in GRAPH_LAYER_SCORE_DETACH_MODELS_PIN else 'none'),
                    graph_layer_context=dict(
                        input='entire_node_and_global_score_trajectory',
                        input_projection='shared_linear_gelu_dropout_linear_to_model_dim',
                        graph_encoder_layers=layers, graph_encoder_parameters='independent_per_block',
                        graph_attention_mask='original_local_topology_self_graph_and_terminal_role_bias',
                        retained_outputs='all_graph_context_blocks_in_depth_order',
                        layer_encoder_layers=1, layer_tokens=layers,
                        layer_attention_axes='B*N,H,L,L', layer_attention_mask='full_bidirectional',
                        layer_encoder_shared='across_nodes_and_samples',
                        layer_position_embedding='none',
                        block='native_encoder_block_without_cross_attention',
                        ffn_dim=ffn_dim, block_dropout=block_dropout,
                        output_projection='shared_linear_gelu_dropout_linear_D_to_H_per_token',
                        logits_axes='B,N,L,H_then_B,N,H,L', softmax_axis='PE_depth_L',
                        memory_source='original_PE_outputs_in_unchanged_order'))
    if mode in (*GRAPH_LAYER_FUSION_DETACH_MODELS_PIN, *GRAPH_LAYER_V3_EXTENSION_MODELS_PIN,
                *GRAPH_LAYER_POWER_MODELS_PIN):
        # Only these new contracts change the fusion taps and optional post-fusion
        # block. Existing checkpoint contracts and PE chains remain unchanged.
        has_final_encoder = mode == 'lasd_graph_layer_headwise_detach_pin_v1'
        has_residual = mode in ('lasd_graph_layer_headwise_detach_pin_v3',
                               *GRAPH_LAYER_V3_EXTENSION_MODELS_PIN, *GRAPH_LAYER_POWER_MODELS_PIN)
        spec.update(version=7, hidden_state_detach=True,
                    hidden_state_detach_scope='fusion_inputs_only', encoder_chain_detach=False,
                    final_encoder_layers=int(has_final_encoder),
                    final_encoder=('native_encoder_block_without_cross_attention'
                                   if has_final_encoder else 'none'),
                    final_attention_mask=(spec['final_attention_mask'] if has_final_encoder else 'none'),
                    residual_connection='last_pe_memory_add' if has_residual else 'none',
                    residual_position='metric_decoder_input' if has_residual else 'none',
                    final_pe_residual_detach=False if has_residual else None)
    if mode in GRAPH_LAYER_V3_EXTENSION_MODELS_PIN:
        spec.update(version=9, decoder_merge_inputs=['live_last_pe_memory', 'score_weighted_fusion'],
                    decoder_input_normalization='none')
        if mode == GRAPH_LAYER_V3_EXTENSION_MODELS_PIN[1]:
            spec.update(decoder_input_normalization='layer_norm_after_residual_sum',
                        decoder_input_norm=dict(normalized_shape=[model_config['dim']],
                                                epsilon=1e-5, elementwise_affine=True))
    if mode == GRAPH_LAYER_SHALLOW_DETACH_MODEL_PIN:
        # Stop only direct shallow-memory fusion gradients. Score/global features,
        # the final fusion memory and the sequential PE chain all remain live.
        spec.update(version=8, score_detach=False, score_detach_scope='none',
                    hidden_state_detach=True, hidden_state_detach_scope='shallow_fusion_inputs_only',
                    fusion_memory_detach=[True] * (layers - 1) + [False],
                    last_fusion_memory_detach=False, encoder_chain_detach=False,
                    final_encoder_layers=1,
                    residual_connection='none', residual_position='none', final_pe_residual_detach=None)
    if mode in GRAPH_LAYER_POWER_MODELS_PIN:
        distance = options.get('aggregation_distance', 2)
        spec.update(version=10, aggregation_distance=distance,
                    decoder_merge_inputs=['live_last_pe_memory', 'score_weighted_fusion'],
                    decoder_input_normalization='none',
                    score_detach_scope='all_orders_local_and_global_before_order_projections',
                    attention_powers=dict(
                        orders=list(range(1, distance+1)), operation='same_pe_same_head_matrix_power',
                        self_connections='retained_in_every_multiplication',
                        diagonal='excluded_only_when_reading_each_power_R_T',
                        topology_mask='original_mask_before_first_softmax_only_no_remask_of_powers',
                        normalization='no_second_softmax_or_row_renormalization',
                        difference='adjacent_pe_same_order_same_head_index',
                        global_context='independently_per_order_same_v3_statistics',
                        score_axes='B,N,K,L,H,2', feature_axes='B,N,K,10LH',
                        graph_token='retained_in_paths_and_statistics',
                        extraction='no_grad_one_graph_row_chunks',
                        compute_dtype='float32_or_float64',
                        independent_input_projections=distance,
                        merge='sum_projected_orders_before_shared_graph_and_layer_encoders',
                        projected_dim=model_config['dim']))
        spec['graph_layer_context'].update(
            input='per_order_entire_node_and_global_score_trajectory',
            input_projection='independent_per_order_linear_gelu_dropout_linear_then_sum')
    if mode in GRAPH_LAYER_POWER_HC_MODELS_PIN:
        # Compose v7 routing with the exact v5 performance-PE HC contract.
        # The observer sees the HC read vector actually used by self-attention.
        spec.update(version=11)
        spec['attention_powers']['attention_input'] = 'hyper_connection_self_attention_read'
    if mode == GRAPH_LAYER_POWER_HC_NORM_MODEL_PIN:
        # Exactly v6's post-sum, per-token feature LayerNorm, now on v8 memories.
        spec.update(version=12, decoder_input_normalization='layer_norm_after_residual_sum',
                    decoder_input_norm=dict(normalized_shape=[model_config['dim']],
                                            epsilon=1e-5, elementwise_affine=True))
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
            graph_layer = mode in GRAPH_LAYER_LASD_MODELS_PIN
            value = dict(n_experts=4 if graph_layer else spec.get('experts'),
                top_k=2 if graph_layer else spec.get('top_k'),
                hidden_dim=spec.get('hidden_dim'), dropout=spec.get('dropout'),
                score_chunk_size=spec.get('score_chunk_size'),
                checkpoint_scores=spec.get('checkpoint_scores'))
            if mode in PE_HYPER_LASD_MODELS_PIN:
                hyper = spec.get('hyper_connection')
                if not isinstance(hyper, dict):
                    raise ValueError('Missing LASD-MoE hyper-connection architecture')
                value['hyper_connection_streams'] = hyper.get('streams')
            if mode in GRAPH_LAYER_POWER_MODELS_PIN:
                value['aggregation_distance'] = spec.get('aggregation_distance')
            if mode in (*GLOBAL_HEADWISE_LASD_MOE_MODELS_PIN, *GRAPH_LAYER_LASD_MODELS_PIN):
                context = spec.get('global_context')
                if not isinstance(context, dict):
                    raise ValueError('Missing LASD-MoE global context architecture')
                value['global_std_epsilon'] = context.get('epsilon')
        if mode in GRAPH_LAYER_LASD_MODELS_PIN:
            context = spec.get('graph_layer_context')
            if not isinstance(context, dict):
                raise ValueError('Missing LASD graph/layer context architecture')
            model_config.update(ffn_dim=context.get('ffn_dim'), dropout=context.get('block_dropout'))
    if spec != lasd_moe_spec_pin(mode, value, model_config):
        raise ValueError('LASD-MoE architecture contract disagrees with saved configuration')
