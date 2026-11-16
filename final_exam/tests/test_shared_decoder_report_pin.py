"""Independent checkpoint identities and portable shared-decoder reports."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from final_exam.shared_decoder_report_pin import write_shared_decoder_report_pin, TABLE_NAMES_PIN
from final_exam.tests.test_summary_pin import workbook, text_values


def fixture():
    tables = {name: [] for name in TABLE_NAMES_PIN}
    checkpoints = []
    for circuit, energy in [('Opamp1', .7), ('two_stage_folded_opamp', .9)]:
        identity = dict(circuit=circuit, model='width_decoder_pin', seed=17,
                        group='performance/layer_0/self_attn.q')
        checkpoints.append(dict(circuit=circuit, model='width_decoder_pin', seed=17,
                                path=f'/tmp/{circuit}/checkpoint_pin.pt', config={'rank': [1, 2]}))
        tables['Groups'].append(dict(**identity, branches=3, input_dim=2, output_dim=2,
                                     dense_weight_parameters=12, rank_99=2))
        tables['Budgets'].append(dict(**identity, requested_rank=1, shared_rank=1,
            independent_rank=1, dense_parameters=12, independent_parameters=12,
            shared_parameters=8, shared_saving_fraction=1/3))
        tables['Errors'].append(dict(**identity, branch=1, requested_rank=1,
            shared_rank=1, independent_rank=1, shared_weight_error=.2,
            independent_weight_error=.1, shared_output_error=None,
            independent_output_error=None, activation_note='not captured'))
        tables['Spectra'].extend([dict(**identity, index=1, singular_value=2., cumulative_energy=energy),
                                 dict(**identity, index=2, singular_value=1., cumulative_energy=1.)])
        tables['Validation'].append(dict(**identity, scope='all', strategy='shared', requested_rank=1,
            split='val', metric='DC_Gain_dB', count=3, mae=.4, rmse=.5, r2=.8,
            normalized_mse=.25, status='complete'))
        tables['DecoderFeatures'].append(dict(**identity, scope='all', strategy='shared',
            requested_rank=1, branch=1, relative_error=.2))
    return tables, dict(checkpoints=checkpoints, seeds=[17], ranks=[1, 2])


def test_portable_export_preserves_every_independent_record_and_no_plot_dependency(tmp_path):
    tables, record = fixture()
    with patch('final_exam.shared_decoder_report_pin.require_plotting_pin',
               side_effect=AssertionError('No plotting should be imported')):
        path = write_shared_decoder_report_pin(tmp_path, tables, record, max_plots=0)
    assert path.name == 'shared_decoder_summary_pin.xlsx'
    sheets = workbook(path)
    assert set(sheets) == {'ReadMe', 'Checkpoints', *TABLE_NAMES_PIN} - {'Spectra'}
    values = text_values(sheets['Groups'])
    assert values.count('Opamp1') == values.count('two_stage_folded_opamp') == 1
    assert 'DecoderFeatures' in sheets and 'Validation' in sheets
    with (tmp_path/'Errors_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
        errors = list(csv.DictReader(handle))
    assert len(errors) == 2 and all(row['shared_output_error'] == '' for row in errors)
    with (tmp_path/'Spectra_pin.csv').open(encoding='utf-8-sig', newline='') as handle:
        spectra = list(csv.DictReader(handle))
    assert len(spectra) == 4
    assert [float(row['cumulative_energy']) for row in spectra if row['index'] == '1'] == [.7, .9]
    metadata = json.loads((tmp_path/'report_metadata_pin.json').read_text())
    assert metadata['record'] == record
    assert metadata['sharing_scope'] == 'corresponding_decoder_branches_within_one_checkpoint'
    assert metadata['plots'] == []
    assert metadata['tables']['Validation'] == 2
    assert not (tmp_path/'plots_pin').exists()


def test_html_escapes_checkpoint_values_and_json_csv_preserve_nested_configuration(tmp_path):
    tables, record = fixture()
    record['checkpoints'][0].update(circuit='<script>alert(1)</script>',
                                    path='a"<b>&', config={'quote': '<x>'})
    write_shared_decoder_report_pin(tmp_path, tables, record, max_plots=0)
    html = (tmp_path/'report_pin.html').read_text()
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert 'a&quot;&lt;b&gt;&amp;' in html
    metadata = json.loads((tmp_path/'report_metadata_pin.json').read_text())
    assert metadata['record']['checkpoints'][0]['config'] == {'quote': '<x>'}


def test_reports_reject_existing_outputs_and_nonfinite_records_before_writes(tmp_path):
    tables, record = fixture()
    tables['Budgets'][0]['shared_saving_fraction'] = float('nan')
    with pytest.raises(ValueError):
        write_shared_decoder_report_pin(tmp_path/'invalid', tables, record, max_plots=0)
    assert not (tmp_path/'invalid').exists()
    tables, record = fixture()
    output = tmp_path/'existing'
    output.mkdir()
    (output/'report_pin.html').write_text('preserve')
    with pytest.raises(FileExistsError):
        write_shared_decoder_report_pin(output, tables, record, max_plots=0)
    assert (output/'report_pin.html').read_text() == 'preserve'
    assert list(output.iterdir()) == [output/'report_pin.html']


def test_plot_identity_keeps_identical_projection_names_in_separate_circuits(tmp_path):
    pytest.importorskip('matplotlib')
    tables, record = fixture()
    write_shared_decoder_report_pin(tmp_path, tables, record, max_plots=10)
    manifest = json.loads((tmp_path/'plots_pin/manifest_pin.json').read_text())
    assert len(manifest) == 2
    assert {row['circuit'] for row in manifest} == {'Opamp1', 'two_stage_folded_opamp'}
    assert len({row['path'] for row in manifest}) == 2
    assert len({row['group'] for row in manifest}) == 1
    for row in manifest:
        assert (tmp_path/row['path']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_plot_limit_is_global_and_no_validation_data_is_invented(tmp_path):
    pytest.importorskip('matplotlib')
    tables, record = fixture()
    tables['Validation'] = []
    tables['DecoderFeatures'] = []
    write_shared_decoder_report_pin(tmp_path, tables, record, max_plots=1)
    metadata = json.loads((tmp_path/'report_metadata_pin.json').read_text())
    assert len(metadata['plots']) == 1
    assert metadata['tables']['Validation'] == 0
    assert metadata['tables']['DecoderFeatures'] == 0
