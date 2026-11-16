"""Shared decoder metadata, rank fidelity and unique-parameter reporting."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from final_exam.low_rank_registry_pin import LOW_RANK_MODELS_PIN
from final_exam.shared_decoder_registry_pin import (
    SHARED_DECODER_MODELS_PIN, shared_decoder_settings_pin, shared_decoder_spec_pin,
)
from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_low_rank_registry_summary_pin import contract as base_contract, benchmark
from final_exam.tests.test_summary_pin import NS, record, write_run, workbook, text_values
from final_exam.width_registry_pin import validate_width_contract_pin


MODEL_CONFIG = dict(dim=128, decoder_ffn_dim=2048, cross_layers=3, decoder_layers=1)


def contract(mode):
    value = base_contract(mode)
    if mode in SHARED_DECODER_MODELS_PIN:
        value['performance']['independent_decoders'] = False
        value['performance']['shared_down_basis'] = shared_decoder_spec_pin(mode, model_config=MODEL_CONFIG)
    return value


def source(tmp_path, *, include_independent=True):
    modes = ('width_decoder_pin', *SHARED_DECODER_MODELS_PIN)
    if include_independent:
        modes += LOW_RANK_MODELS_PIN
    records = []
    for mode in modes:
        item = record(mode)
        item['report']['width_decoder_pin'] = contract(mode)
        item['report']['low_rank_benchmark_pin'] = benchmark()
        records.append(item)
    directory = write_run(tmp_path, records)
    cfg = dict(model=MODEL_CONFIG, shared_decoder_pin=shared_decoder_settings_pin(),
               dataset=dict(train_circuits=['Opamp1']))
    (directory / 'config_pin.json').write_text(json.dumps(cfg))
    for item in records:
        (directory / f'seed_17_pin/{item["mode"]}/config_pin.json').write_text(json.dumps(cfg))
    return directory, records


@pytest.mark.parametrize('mode', SHARED_DECODER_MODELS_PIN)
def test_shared_modes_validate_and_keep_unique_factor_contract(mode):
    value = contract(mode)
    validate_width_contract_pin(mode, value)
    spec = value['performance']['shared_down_basis']
    assert spec['parameterization'] == 'shared_down_independent_up'
    assert spec['shared_weight_parameters'] == spec['shared_down_parameters'] + spec['independent_up_parameters']
    item = record(mode)
    item['report']['width_decoder_pin'] = value
    assert len(summary_rows_pin([item])) == 1


def test_export_displays_shared_dense_and_independent_selected_models(tmp_path):
    directory, records = source(tmp_path)
    before = deepcopy(records)
    sheets = workbook(export_results_pin(directory))
    assert 'WidthDecoder' in sheets and 'SharedDecoder' in sheets and 'LowRankDecoder' in sheets
    texts = text_values(sheets['SharedDecoder'])
    assert all(item['mode'] in texts for item in records)
    assert 'shared_down_independent_up' in texts
    assert 'independent_weight_factors' in texts
    assert 'dense_weights' in texts
    assert 'cross_kv' in texts
    assert 'Total unique model parameters' in texts
    assert texts.count('resident_first_test_batch_full_cascade_v1') == len(records)
    assert records == before
    # Numeric shared down count is stored once per sharing group, not multiplied by PE branches.
    xml_rows = sheets['SharedDecoder'].findall('x:sheetData/x:row', NS)[4:]
    rows = {row.find('x:c/x:is/x:t', NS).text: row for row in xml_rows}
    mode = SHARED_DECODER_MODELS_PIN[0]
    cells = {c.attrib['r'].rstrip('0123456789'): c for c in rows[mode].findall('x:c', NS)}
    spec = contract(mode)['performance']['shared_down_basis']
    assert int(cells['K'].find('x:v', NS).text) == spec['shared_down_parameters']
    assert int(cells['L'].find('x:v', NS).text) == spec['independent_up_parameters']
    assert int(cells['N'].find('x:v', NS).text) == spec['shared_weight_parameters']


@pytest.mark.parametrize('field,new', [('attention_rank', 16), ('ffn_rank', 32)])
def test_saved_shared_rank_mismatch_fails_before_export(tmp_path, field, new):
    directory, _ = source(tmp_path, include_independent=False)
    mode = 'width_decoder_shared_both_pin'
    path = directory / f'seed_17_pin/{mode}/config_pin.json'
    cfg = json.loads(path.read_text())
    cfg['shared_decoder_pin'][field] = new
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='saved configuration'):
        export_results_pin(directory)
    assert not (directory / 'summary_pin.xlsx').exists()


@pytest.mark.parametrize('key,value', [
    ('shared_down_parameters', 1), ('shared_group_count', 99),
    ('initialization', 'svd'), ('factors_trainable', False),
    ('shared_roles', ['multihead_attn.q']),
])
def test_tampered_shared_spec_is_rejected(key, value):
    mode = 'width_decoder_shared_kv_pin'
    c = contract(mode)
    c['performance']['shared_down_basis'][key] = value
    with pytest.raises(ValueError):
        validate_width_contract_pin(mode, c)


def test_missing_or_false_shared_contract_is_rejected():
    mode = SHARED_DECODER_MODELS_PIN[0]
    c = contract(mode)
    del c['performance']['shared_down_basis']
    with pytest.raises(ValueError):
        validate_width_contract_pin(mode, c)
    dense = contract('width_decoder_pin')
    dense['performance']['shared_down_basis'] = shared_decoder_spec_pin(mode, model_config=MODEL_CONFIG)
    with pytest.raises(ValueError):
        validate_width_contract_pin('width_decoder_pin', dense)


def test_benchmark_parameter_mismatch_rejected_for_shared_model(tmp_path):
    directory, records = source(tmp_path, include_independent=False)
    records[1]['report']['low_rank_benchmark_pin']['model_parameters'] = 13
    (directory / 'results_pin.json').write_text(json.dumps(records))
    with pytest.raises(ValueError, match='parameters disagree'):
        export_results_pin(directory)


def test_reporting_imports_without_torch():
    code = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('Torch must not be imported for reporting')
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import final_exam.summarize_pin
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
