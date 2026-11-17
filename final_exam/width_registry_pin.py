"""Torch-free model IDs and configuration checks for multi-PE readout studies."""
import math
from final_exam.low_rank_registry_pin import LOW_RANK_MODELS_PIN, validate_low_rank_contract_pin
from final_exam.shared_decoder_registry_pin import SHARED_DECODER_MODELS_PIN, validate_shared_decoder_contract_pin
from final_exam.lasd_registry_pin import LASD_MODELS_PIN, validate_lasd_contract_pin

WIDTH_MODELS_PIN = ('width_decoder_pin', 'width_decoder_op_pin')
WIDTH_DETACH_MODELS_PIN = ('width_decoder_detach_pin',)
ALL_WIDTH_MODELS_PIN = (WIDTH_MODELS_PIN + LOW_RANK_MODELS_PIN + SHARED_DECODER_MODELS_PIN
                       + LASD_MODELS_PIN + WIDTH_DETACH_MODELS_PIN)
WIDTH_PREDICTOR_PIN = 'width_device_op_predictor_pin'


def detach_memory_spec_pin(branch_count):
    """Only shallow performance-decoder inputs are detached; PE depth is 1-based."""
    return dict(version=1, policy='detach_shallow_decoder_memory',
                detached_decoder_inputs=list(range(1, branch_count)),
                attached_decoder_inputs=[branch_count],
                location='after_encoder_output_norm_and_projection_before_decoder',
                encoder_chain_detached=False, decoders_frozen=False)


def width_settings_pin(value=None, model_config=None):
    cfg = model_config or {}
    result = dict(n_experts=4, hidden_dim=cfg.get('dim', 128),
                  dropout=cfg.get('dropout', .1))
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise ValueError('width_decoder_pin accepts n_experts, hidden_dim and dropout only')
        result.update(value)
    for key in ('n_experts', 'hidden_dim'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'width_decoder_pin.{key} must be a positive integer')
    if (type(result['dropout']) not in (int, float)
            or not math.isfinite(result['dropout']) or not 0 <= result['dropout'] < 1):
        raise ValueError('width_decoder_pin.dropout must be finite in [0, 1)')
    return result


def validate_width_contract_pin(mode, contract):
    """Keep Excel/report metadata honest without importing the training runtime."""
    if mode not in ALL_WIDTH_MODELS_PIN:
        return
    if not isinstance(contract, dict):
        raise ValueError('Wide decoder report requires its saved architecture contract')
    for key, value in dict(mode=mode, gate=False, requires_measured_op=False,
                           predictor_granularity='device', performance_granularity='pin',
                           predictor_frozen=True).items():
        if contract.get(key) != value:
            raise ValueError(f'Wide decoder report has inconsistent {key}')
    branches = [contract.get('performance')]
    if mode == 'width_decoder_op_pin':
        branches.append(contract.get('op_predictor'))
    elif contract.get('op_predictor') != dict(readout='last_pe_device_mlp', query_kind='none'):
        raise ValueError('width_decoder_pin must retain the original OP predictor')
    for index, branch in enumerate(branches):
        if not isinstance(branch, dict):
            raise ValueError('Wide decoder branch metadata is missing')
        for key in ('branch_count', 'feature_dim', 'decoder_layers', 'output_dim'):
            if type(branch.get(key)) is not int or branch[key] < 1:
                raise ValueError(f'Invalid wide decoder {key}')
        if (branch.get('readout') != 'per_pe_decoder_then_dense_moe_mlp'
                or branch.get('independent_decoders') is not (mode not in SHARED_DECODER_MODELS_PIN)
                or branch.get('pe_order') != list(range(1, branch['branch_count'] + 1))
                or branch.get('concatenated_dim') != branch['branch_count'] * branch['feature_dim']
                or branch.get('query_kind') != ('metric' if index == 0 else 'physical_device_memory')
                or branch.get('output_projection') != 'shared_encoder_norm_and_projection_per_pe'):
            raise ValueError('Wide decoder PE/query ordering contract is inconsistent')
        if index == 0 and (type(branch.get('metric_count')) is not int or branch['metric_count'] < 1
                           or branch['output_dim'] != 1):
            raise ValueError('Performance readout requires one scalar per metric')
        if index == 1 and branch.get('metric_count') is not None:
            raise ValueError('OP readout uses physical-device queries, not performance queries')
        validate_low_rank_contract_pin(mode if index == 0 else 'width_decoder_pin',
                                       branch.get('low_rank'), branch['feature_dim'])
        validate_shared_decoder_contract_pin(mode if index == 0 else 'width_decoder_pin',
            branch.get('shared_down_basis'), branch['feature_dim'], branch['branch_count'],
            branch['decoder_layers'])
        validate_lasd_contract_pin(mode if index == 0 else 'width_decoder_pin',
                                  branch.get('lasd'), branch['feature_dim'], branch['branch_count'])
        expected_gradients = (detach_memory_spec_pin(branch['branch_count'])
                              if index == 0 and mode in WIDTH_DETACH_MODELS_PIN else None)
        if branch.get('memory_gradients') != expected_gradients:
            raise ValueError('Wide decoder memory gradient policy disagrees with model ID')
        mixture = branch.get('mixture')
        if not isinstance(mixture, dict):
            raise ValueError('Wide readout mixture metadata is missing')
        width_settings_pin({k: mixture.get(k) for k in ('n_experts', 'hidden_dim', 'dropout')})
        if (mixture.get('expert') != 'linear_gelu_dropout_linear'
                or mixture.get('routing') != 'per_token_softmax_dense'
                or mixture.get('auxiliary_loss') is not False):
            raise ValueError('Wide readout mixture contract is inconsistent')
