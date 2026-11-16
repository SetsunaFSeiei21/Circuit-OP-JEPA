"""Low-rank identities, rank boundaries and saved report/configuration fidelity."""
import json

import pytest

from final_exam.low_rank_registry_pin import (
    LOW_RANK_MODELS_PIN, low_rank_settings_pin, low_rank_spec_pin,
)
from final_exam.width_registry_pin import (
    WIDTH_MODELS_PIN, ALL_WIDTH_MODELS_PIN, validate_width_contract_pin,
)
from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_summary_pin import record, write_run, workbook, text_values


def contract(mode):
    performance = dict(readout='per_pe_decoder_then_dense_moe_mlp',
        branch_count=3, decoder_layers=1, feature_dim=128, concatenated_dim=384,
        independent_decoders=True, pe_order=[1, 2, 3], query_kind='metric',
        metric_count=9, output_dim=1,
        output_projection='shared_encoder_norm_and_projection_per_pe',
        mixture=dict(n_experts=4, hidden_dim=128, dropout=.1,
                     expert='linear_gelu_dropout_linear', routing='per_token_softmax_dense',
                     auxiliary_loss=False))
    spec = low_rank_spec_pin(mode)
    if spec is not None:
        performance['low_rank'] = spec
    return dict(mode=mode, gate=False, requires_measured_op=False,
                predictor_granularity='device', performance_granularity='pin',
                predictor_frozen=True, performance=performance,
                op_predictor=dict(readout='last_pe_device_mlp', query_kind='none'))


def result(mode):
    r = record(mode)
    r['report']['width_decoder_pin'] = contract(mode)
    return r


def source(tmp_path):
    records = [result(mode) for mode in ('width_decoder_pin', *LOW_RANK_MODELS_PIN)]
    directory = write_run(tmp_path, records)
    cfg = dict(model=dict(dim=128, decoder_ffn_dim=2048),
               low_rank_decoder_pin=low_rank_settings_pin(),
               dataset=dict(train_circuits=['Opamp1']))
    (directory / 'config_pin.json').write_text(json.dumps(cfg))
    for item in records:
        (directory / f'seed_17_pin/{item["mode"]}/config_pin.json').write_text(json.dumps(cfg))
    return directory


@pytest.mark.parametrize('mode', LOW_RANK_MODELS_PIN)
def test_modes_keep_original_width_models_and_validate_contract(mode):
    assert len(WIDTH_MODELS_PIN) == 2
    assert mode in ALL_WIDTH_MODELS_PIN
    validate_width_contract_pin(mode, contract(mode))
    assert len(summary_rows_pin([result(mode)])) == 1


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '32', None])
def test_invalid_ranks(value):
    with pytest.raises(ValueError, match='positive integer'):
        low_rank_settings_pin({'attention_rank': value})


def test_rank_validation_only_constrains_active_components():
    with pytest.raises(ValueError, match='projection parameters'):
        low_rank_spec_pin(LOW_RANK_MODELS_PIN[0], {'attention_rank': 64})
    with pytest.raises(ValueError, match='ffn_rank must reduce'):
        low_rank_spec_pin(LOW_RANK_MODELS_PIN[1], {'ffn_rank': 121})
    # Unused, but syntactically valid rank settings do not affect this variant.
    assert low_rank_spec_pin(LOW_RANK_MODELS_PIN[1], {'attention_rank': 1000})['attention_rank'] is None
    assert low_rank_spec_pin('width_decoder_pin') is None
    with pytest.raises(ValueError, match='accepts'):
        low_rank_settings_pin({'typo': 4})


@pytest.mark.parametrize('key,value', [('attention', False), ('ffn', True),
    ('scope', 'all_model'), ('initialization', 'svd'), ('version', True),
    ('dim', 64), ('attention_rank', None), ('ffn_rank', 64)])
def test_tampered_architecture_rejected(key, value):
    mode = LOW_RANK_MODELS_PIN[0]
    c = contract(mode)
    c['performance']['low_rank'][key] = value
    with pytest.raises(ValueError):
        validate_width_contract_pin(mode, c)


def test_missing_and_dense_false_lowrank_metadata_rejected():
    mode = LOW_RANK_MODELS_PIN[0]
    c = contract(mode)
    del c['performance']['low_rank']
    with pytest.raises(ValueError, match='saved low_rank contract'):
        validate_width_contract_pin(mode, c)
    dense = contract('width_decoder_pin')
    dense['performance']['low_rank'] = low_rank_spec_pin(mode)
    with pytest.raises(ValueError, match='must not claim'):
        validate_width_contract_pin('width_decoder_pin', dense)


def test_four_group_export_and_saved_rank_audit(tmp_path):
    directory = source(tmp_path)
    sheets = workbook(export_results_pin(directory))
    assert 'WidthDecoder' in sheets and 'LowRankDecoder' in sheets
    texts = text_values(sheets['LowRankDecoder'])
    assert all(mode in texts for mode in ('width_decoder_pin', *LOW_RANK_MODELS_PIN))
    assert 'Attention rank' in texts and 'FFN rank' in texts


def test_configuration_mismatch_fails_before_export(tmp_path):
    directory = source(tmp_path)
    mode = LOW_RANK_MODELS_PIN[0]
    path = directory / f'seed_17_pin/{mode}/config_pin.json'
    cfg = json.loads(path.read_text())
    cfg['low_rank_decoder_pin']['attention_rank'] = 16
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='saved configuration'):
        export_results_pin(directory)
    assert not (directory / 'summary_pin.xlsx').exists()


def benchmark():
    return dict(protocol='resident_first_test_batch_full_cascade_v1',
                includes_op_predictor=True, uses_cached_op=False, input_transfer_included=False,
                model_parameters=12, batch_size=8, measured_iterations=10, warmup_iterations=3,
                mean_latency_ms=2., latency_std_ms=.2, samples_per_second=4000.,
                peak_allocated_bytes=None, baseline_allocated_bytes=None,
                incremental_peak_allocated_bytes=None)


def test_observed_benchmark_export_includes_dense_reference(tmp_path):
    directory = source(tmp_path)
    path = directory / 'results_pin.json'
    results = json.loads(path.read_text())
    for r in results:
        r['report']['low_rank_benchmark_pin'] = benchmark()
    path.write_text(json.dumps(results))
    sheets = workbook(export_results_pin(directory))
    texts = text_values(sheets['LowRankDecoder'])
    assert texts.count('resident_first_test_batch_full_cascade_v1') == 4
    assert 'CUDA process peak MiB' in texts
    assert 'Full-cascade latency ms' in texts


@pytest.mark.parametrize('key,value', [('uses_cached_op', True), ('model_parameters', 13),
                                       ('mean_latency_ms', float('nan'))])
def test_misleading_benchmark_fails_export(tmp_path, key, value):
    directory = source(tmp_path)
    path = directory / 'results_pin.json'
    results = json.loads(path.read_text())
    b = benchmark()
    b[key] = value
    results[0]['report']['low_rank_benchmark_pin'] = b
    path.write_text(json.dumps(results))
    with pytest.raises(ValueError, match='benchmark'):
        export_results_pin(directory)
    assert not (directory / 'summary_pin.xlsx').exists()
