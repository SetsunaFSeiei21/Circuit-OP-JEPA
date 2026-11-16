"""Saved LASD configuration and score semantics remain auditable in Excel."""
from copy import deepcopy
import json

import pytest

from final_exam.lasd_registry_pin import LASD_MODELS_PIN, lasd_settings_pin, lasd_spec_pin
from final_exam.summarize_pin import export_results_pin
from final_exam.tests.test_low_rank_registry_summary_pin import contract as base_contract, benchmark
from final_exam.tests.test_summary_pin import record, write_run, workbook, text_values
from final_exam.width_registry_pin import validate_width_contract_pin

CFG = dict(dim=128, heads=8, encoder_layers=6, cross_layers=3, decoder_layers=1, decoder_ffn_dim=2048)


def contract(mode):
    c = base_contract(mode)
    if mode in LASD_MODELS_PIN:
        c['performance']['lasd'] = lasd_spec_pin(mode, None, CFG)
    return c


def source(tmp_path):
    records = []
    for mode in ('width_decoder_pin', *LASD_MODELS_PIN):
        item = record(mode)
        item['report']['width_decoder_pin'] = contract(mode)
        item['report']['low_rank_benchmark_pin'] = benchmark()
        records.append(item)
    directory = write_run(tmp_path, records)
    config = dict(model=CFG, lasd_pin=lasd_settings_pin(), dataset=dict(train_circuits=['Opamp1']))
    (directory / 'config_pin.json').write_text(json.dumps(config))
    for item in records:
        (directory / f'seed_17_pin/{item["mode"]}/config_pin.json').write_text(json.dumps(config))
    return directory, records


def test_lasd_workbook_reports_three_models_and_added_capacity(tmp_path):
    directory, records = source(tmp_path)
    sheets = workbook(export_results_pin(directory))
    assert {'LASD', 'WidthDecoder', 'MAE', 'RMSE', 'R2'} <= set(sheets)
    texts = text_values(sheets['LASD'])
    assert all(r['mode'] in texts for r in records)
    assert 'Added adapter parameters' in texts and 'Readout seconds' in texts
    assert any('same_head_index' in text for text in texts if text)


@pytest.mark.parametrize('field,value', [('head_reduction', 'mean'), ('diagonal', 'include'),
    ('detach_scores', True), ('decoder_queries', 'recurrent'), ('adapter_parameters', 1),
    ('layer_masks', ['global']*3), ('first_delta', 'observed')])
def test_lasd_metadata_tampering_is_rejected(field, value):
    mode = 'width_decoder_lasd_pin'
    c = contract(mode)
    c['performance']['lasd'][field] = value
    with pytest.raises(ValueError):
        validate_width_contract_pin(mode, c)


def test_lasd_effective_config_mismatch_is_rejected_before_export(tmp_path):
    directory, _ = source(tmp_path)
    path = directory / 'seed_17_pin/width_decoder_lasd_pin/config_pin.json'
    cfg = json.loads(path.read_text())
    cfg['lasd_pin']['hidden_dim'] = 64
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='saved configuration'):
        export_results_pin(directory)
    assert not (directory / 'summary_pin.xlsx').exists()


def test_original_model_cannot_claim_lasd_scores():
    c = contract('width_decoder_pin')
    c['performance']['lasd'] = deepcopy(contract('width_decoder_score_pin')['performance']['lasd'])
    with pytest.raises(ValueError, match='non-LASD'):
        validate_width_contract_pin('width_decoder_pin', c)
