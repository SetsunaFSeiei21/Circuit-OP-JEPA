"""Original IDs, physical OP inversion, read-only source joins and saved traces."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import torch

from final_exam.error_case_data_pin import (TrainReferencePin, raw_evidence_pin,
                                          select_cases_pin, waveform_evidence_pin)
from final_exam.error_case_probe_pin import run_error_case_probe_pin, infer_op_cases_pin
from final_exam.gate_probe_pin import file_digest

BASE = 'device_op_cascade_pin'
V3 = 'lasd_graph_layer_headwise_detach_pin_v3'


@pytest.fixture(scope='module')
def saved_cases(tmp_path_factory):
    from final_exam.tests.test_run_pin import tiny_inputs
    from final_exam.run_pin import run_experiment_pin
    root = tmp_path_factory.mktemp('case_probe')
    source = root / 'experiment'
    source.mkdir()
    data, cfg, _, _ = tiny_inputs(source)
    for i, row in enumerate(data['samples']):
        row['source_sample_id'] = 1000 + i * 7
        row['raw_design_parameters'] = dict(width_override=2. + i, constant=1.)
        row['design_key'] = row['design_id']
        row['devices'].reverse()  # Tensor axis must follow sorted device_id, not this input order.
    cfg['model'].update(encoder_layers=4, cross_layers=3)
    cfg['op_transfer']['epochs'] = dict(op_predictor=1, readout=2, student_predicted_op=2,
                                      predicted_op_distill_encoder=1, predicted_op_distill_decoder=1)
    cfg['lasd_moe_pin'] = dict(hidden_dim=8, dropout=0., score_chunk_size=8)
    (source / 'data_pin.json').write_text(json.dumps(data))
    (source / 'input_pin.json').write_text(json.dumps(cfg))
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment_pin(source/'data_pin.json', source/'input_pin.json', source/'training_pin',
                           [BASE, V3], [29], 'cpu', log_dir=root/'train_logs_pin')
    manifest = json.loads((source/'training_pin/manifest_pin.json').read_text())
    rows = [r for r in data['samples'] if r['sample_id'] in manifest['test']][:2][::-1]
    raw = root / 'raw'
    raw.mkdir()
    with sqlite3.connect(raw/'sampling_history.sqlite3') as con:
        con.execute('CREATE TABLE dataset_schema(id INTEGER,circuit_name TEXT,parameter_names_json TEXT,metric_names_json TEXT)')
        con.execute('INSERT INTO dataset_schema VALUES (1,?,?,?)', ('synthetic_0',
            json.dumps(['width_override','constant']), json.dumps(data['feature_spec']['metric_names'])))
        con.execute('CREATE TABLE samples(sample_id INTEGER,design_key TEXT,design_values_json TEXT,metric_values_json TEXT,observation_json TEXT)')
        fields = data['feature_spec']['op_names']
        con.execute('CREATE TABLE dc_device_operating_points(sample_id INTEGER,device_name TEXT,' +
                    ','.join('"'+n+'" REAL' for n in fields) + ')')
        for row in rows:
            con.execute('INSERT INTO samples VALUES (?,?,?,?,?)', (row['source_sample_id'], row['design_key'],
                json.dumps(row['raw_design_parameters']), json.dumps(row['performance']),
                json.dumps(dict(waveform_path='not_retained.csv'))))
            for d in row['devices']:
                values = [row['source_sample_id'], d['device_id'].lower()] + [d.get('op', {}).get(k) for k in fields]
                con.execute('INSERT INTO dc_device_operating_points VALUES ('+','.join('?' for _ in values)+')', values)
    return root, source, raw, data, rows


def test_replay_op_identity_training_references_and_source_immutability(saved_cases, tmp_path, monkeypatch):
    from final_exam.models.op_gate_pin import DeviceOPPredictorPin
    from final_exam.run_pin import load_checkpoint_pin
    from device_jepa.data import transform_value
    root, source, raw, data, rows = saved_cases
    before = {p: file_digest(p) for folder in (source, raw) for p in folder.rglob('*') if p.is_file()}
    original, seen = DeviceOPPredictorPin.forward, []
    def checked(self, batch):
        assert not self.training and not torch.is_grad_enabled()
        assert not {'op', 'op_mask', 'owner_op', 'owner_op_mask', 'performance', 'physical_performance'} & batch.keys()
        seen.append(True)
        return original(self, batch)
    monkeypatch.setattr(DeviceOPPredictorPin, 'forward', checked)
    with contextlib.redirect_stdout(io.StringIO()):
        output = run_error_case_probe_pin(source_run=source, circuit='synthetic_0',
            sample_ids=[r['source_sample_id'] for r in rows], models=[BASE,V3], seeds=[29], raw_dir=raw,
            device='cpu', reference_samples=3, output=tmp_path/'out', log_dir=tmp_path/'logs', plots=True)
    assert seen and all(file_digest(p) == digest for p, digest in before.items())
    record = json.loads((output/'probe_pin.json').read_text())
    stats = json.loads((output/'case_statistics_pin.json').read_text())
    assert record['status'] == 'complete' and record['raw_source']['case_values_match']
    assert all(r['status'] == 'matched' for r in stats['RawChecks'])
    assert {r['status'] for r in stats['Waveforms']} == {'not_supplied'}
    assert stats['PredictorChecks'][1]['same_predictor_as'] == BASE
    train_ids = json.loads((source/'training_pin/manifest_pin.json').read_text())['train']
    for chk in record['checkpoints']:
        assert len(chk['reference_sample_ids']) == 3 and set(chk['reference_sample_ids']) <= set(train_ids)
    constant = [r for r in stats['Parameters'] if r['field'] == 'constant']
    assert constant and all(r['train_percentile'] == 50. and r['train_count'] == len(train_ids) for r in constant)
    model, saved = load_checkpoint_pin(source/f'training_pin/seed_29_pin/{BASE}/checkpoint_pin.pt')
    names, predictions = infer_op_cases_pin(model, saved, rows, 1, torch.device('cpu'))
    for report in (r for r in stats['OPComparison'] if r['model'] == BASE and r['supported']):
        i = [r['source_sample_id'] for r in rows].index(report['source_sample_id'])
        j = names.index(report['field'])
        d = next(d for d in rows[i]['devices'] if d['device_id'] == report['device'])
        assert report['true_op'] == d['op'].get(report['field'])
        assert report['predicted_op'] == pytest.approx(predictions[i][report['device']]['physical'][j], rel=3e-5, abs=1e-10)
        if report['true_op'] is not None:
            norm = transform_value(report['true_op'], saved['preprocessing']['device_preprocessing']['op'], j)
            assert report['normalized_residual'] == pytest.approx(report['normalized_predicted'] - norm)
    assert list((output/'plots_pin').glob('*.png'))
    assert (output/'case_review_pin.html').is_file()
    import openpyxl
    book = openpyxl.load_workbook(output/'case_summary_pin.xlsx', read_only=True, data_only=True)
    assert {'OPComparison','RawChecks','Waveforms','Parameters','Performance'} <= set(book.sheetnames)
    book.close()


def test_raw_mismatch_missing_source_and_ids_are_not_indices(saved_cases, tmp_path):
    _, _, raw, data, rows = saved_cases
    ids = [r['source_sample_id'] for r in rows]
    selected = select_cases_pin(data, 'synthetic_0', ids)
    assert [r['source_sample_id'] for r in selected] == ids
    with pytest.raises(ValueError, match='absent'):
        select_cases_pin(data, 'synthetic_0', [0,1])
    info, details, checks, candidates = raw_evidence_pin(raw, 'synthetic_0', rows)
    assert info['case_values_match'] and candidates and not candidates[0]['exists']
    changed = json.loads(json.dumps(rows))
    changed[0]['raw_design_parameters']['width_override'] += 1.
    info, _, checks, _ = raw_evidence_pin(raw, 'synthetic_0', changed)
    assert not info['case_values_match'] and any(r['status'] == 'mismatch' for r in checks)
    assert raw_evidence_pin(tmp_path/'absent', 'synthetic_0', rows)[0]['status'] == 'missing_database'


def test_reject_wrong_split_before_inference(saved_cases, tmp_path):
    _, source, raw, _, rows = saved_cases
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match='not in saved train'):
        run_error_case_probe_pin(source_run=source, circuit='synthetic_0', sample_ids=[rows[0]['source_sample_id']],
            models=[BASE], seeds=[29], split='train', device='cpu', raw_dir=raw, plots=False,
            output=tmp_path/'out', log_dir=tmp_path/'logs')


def test_empirical_percentiles_not_gaussian_and_identity():
    data = [dict(raw_design_parameters={'x':v}, devices=[dict(device_id='M1',device_type='PMOS',model_name='p',op={'id':-v})])
            for v in (1.,1.,1.,100.)]
    reference = TrainReferencePin(data)
    desc = reference.describe('design', None, 'x', 1.)
    assert desc['train_percentile'] == 37.5
    assert not desc['outside_train_range']
    assert reference.describe('design', None, 'x', 101.)['outside_train_range']
    assert reference.describe('op', dict(device_id='M1',device_type='NMOS',model_name='n'), 'id', -1.)['train_count'] == 0


def test_waveforms_explicit_identity_units_duplicate_time_and_no_fake_sr(tmp_path):
    out = tmp_path/'out'
    out.mkdir()
    (out/'plots_pin').mkdir()
    source = tmp_path/'trace.csv'
    source.write_text('t_us,out_mV\n0,0\n1,1000\n1,1000\n2,3000\n')
    manifest = tmp_path/'manifest.json'
    content = dict(version=1, circuit='C', waveforms=[dict(source_sample_id=41074, path='trace.csv',
        time_column='t_us', voltage_columns=['out_mV'], time_scale_to_s=1e-6, voltage_scale_to_v=1e-3,
        provenance='original_saved')])
    manifest.write_text(json.dumps(content))
    before = file_digest(source)
    rows, figures = waveform_evidence_pin(manifest, 'C', [41074,38711], out, plots=True)
    assert rows[0]['status'] == 'loaded' and rows[1]['status'] == 'not_supplied'
    assert rows[0]['duplicate_time_intervals'] == 1
    assert rows[0]['diagnostic_slope_max_v_us'] == pytest.approx(2.)
    assert rows[0]['voltage_max_v'] == pytest.approx(3.)
    assert file_digest(source) == before and figures
    assert not any('Slew_' in key for key in rows[0])
    with pytest.raises(ValueError, match='identity'):
        waveform_evidence_pin(manifest, 'C', [38711], out, plots=False)
    content['waveforms'][0]['path'] = 'missing.csv'
    manifest.write_text(json.dumps(content))
    rows, _ = waveform_evidence_pin(manifest, 'C', [41074], out, plots=False)
    assert rows[0]['status'] == 'unavailable_or_invalid'
