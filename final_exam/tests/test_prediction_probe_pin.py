"""Replay real saved model families, exact rows, missing labels, and source safety."""
import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from final_exam.prediction_probe_pin import inspect_sources_pin, run_prediction_probe_pin, infer_split_pin
from final_exam.gate_probe_pin import file_digest

BASE = 'device_op_cascade_pin'
V3 = 'lasd_graph_layer_headwise_detach_pin_v3'


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    from final_exam.tests.test_run_pin import tiny_inputs
    from final_exam.run_pin import run_experiment_pin
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('prediction_replay')
    for run_name, models, seeds in [('first', [BASE, V3, 'mlp_pin', 'device_op_gate_oracle_pin'], [17]),
                                     ('second', [BASE, V3], [29])]:
        source = root / run_name
        source.mkdir()
        data, cfg, _, _ = tiny_inputs(source)
        cfg['model'].update(encoder_layers=4, cross_layers=3)
        cfg['op_transfer']['epochs'] = dict(op_predictor=1, readout=2, student_predicted_op=2,
            predicted_op_distill_encoder=1, predicted_op_distill_decoder=1)
        cfg['op_transfer']['patience'] = 0
        cfg['lasd_moe_pin'] = dict(hidden_dim=8, dropout=0., score_chunk_size=8, diagnostic_samples=1)
        # A valid absent label tests effective counts and observed CSV markers.
        for i, row in enumerate(data['samples']):
            row['source_sample_id'] = i + 1
            if i % 3 == 0:
                row['performance'][data['feature_spec']['metric_names'][0]] = None
        (source / 'data_pin.json').write_text(json.dumps(data))
        (source / 'input_pin.json').write_text(json.dumps(cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            run_experiment_pin(source/'data_pin.json', source/'input_pin.json', source/'training_pin',
                               models, seeds, 'cpu', log_dir=root/f'{run_name}_logs_pin')
    return root


def run(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_prediction_probe_pin(**kwargs)


def test_real_checkpoint_replay_all_sources_models_seeds_and_sources_unchanged(saved, tmp_path, monkeypatch):
    from final_exam.models.lasd_moe_pin import LASDMoEModelPin
    original = LASDMoEModelPin.forward
    seen = []
    def checked(self, batch, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        assert not {'op', 'owner_op', 'performance', 'performance_mask', 'physical_performance'} & batch.keys()
        seen.append(True)
        return original(self, batch, **kwargs)
    monkeypatch.setattr(LASDMoEModelPin, 'forward', checked)
    sources = [saved/'first', saved/'second/training_pin']
    before = {str(p):file_digest(p) for source in (saved/'first', saved/'second') for p in source.rglob('*') if p.is_file()}
    out = run(source_runs=sources, device='cpu', output=tmp_path/'out', log_dir=tmp_path/'logs', bootstrap=50, top_errors=2, plots=False)
    result = json.loads((out/'statistics_pin.json').read_text())
    record = json.loads((out/'probe_pin.json').read_text())
    assert record['status'] == 'complete' and len(record['checkpoints']) == 6 and seen
    assert all(file_digest(p) == sha for p, sha in before.items())
    for row in result['Aggregate']:
        if row['split'] == 'test':
            assert abs(row['replay_mse_difference']) < 2e-6
            assert row['replay_status'] == 'matched_tolerance'
    summary = next(r for r in result['Summary'] if r['model'] == V3 and r['split'] == 'test')
    assert summary['num_seeds'] == 2 and summary['seeds'] == [17,29]
    assert result['PairedSeeds'] and result['PairedSummary'] and result['ValTestGap']
    with gzip.open(out/'predictions_pin.csv.gz','rt') as f:
        rows = list(csv.DictReader(f))
    assert rows and any(r['observed'] == 'False' and r['target'] == '' for r in rows)
    assert {r['split'] for r in rows} == {'val', 'test'}
    for r in rows:
        manifest = json.loads((saved/'first/training_pin/manifest_pin.json').read_text())
        assert r['sample_id'] in manifest[r['split']]
    import openpyxl
    book = openpyxl.load_workbook(out/'prediction_summary_pin.xlsx', read_only=True)
    assert {'Metrics','Summary','PairedSummary','Strata','ReadMe'} <= set(book.sheetnames)
    book.close()


def test_plots_all_metrics_and_saved_normalization_no_refitting(saved, tmp_path, monkeypatch):
    import device_jepa.data as data
    def no_fit(*args, **kwargs):
        raise AssertionError('Must never fit normalization during diagnostics')
    monkeypatch.setattr(data, 'fit_preprocessor', no_fit)
    out = run(source_runs=[saved/'first'], models=[V3], seeds=[17], device='cpu',
        output=tmp_path/'plots', log_dir=tmp_path/'logs', bootstrap=0, top_errors=1, max_plot_points=2)
    index = json.loads((out/'plots_pin/index_pin.json').read_text())
    assert len(index) == 2 * len(json.loads((saved/'first/data_pin.json').read_text())['feature_spec']['metric_names'])
    assert all((out/r['path']).is_file() and r['plotted_scatter_samples'] <= 2 for r in index)
    assert 'prediction_summary_pin.xlsx' in (out/'prediction_viewer_pin.html').read_text()


def test_preflight_stdlib_unfinished_run_and_missing_model_pair(saved, tmp_path):
    script = Path(__file__).resolve().parents[1]/'prediction_probe_pin.py'
    result = subprocess.run([sys.executable,'-S',str(script),'--source-runs',str(saved/'first'),
                             '--inspect-source'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    record_path = saved/'first/training_pin/run_pin.json'
    original = record_path.read_bytes()
    try:
        record = json.loads(original)
        record['status'] = 'running'
        record['seeds'].append(43)
        record_path.write_text(json.dumps(record))
        sources, inventory = inspect_sources_pin([saved/'first'])
        assert len(sources[0]['checkpoints']) == 4
        assert sum(r['status']=='skipped_incomplete' for r in inventory) == 4
    finally:
        record_path.write_bytes(original)
    with pytest.raises(ValueError,match='Duplicate source'):
        inspect_sources_pin([saved/'first', saved/'first/training_pin'])
    with pytest.raises(ValueError,match='separate from source'):
        run(source_runs=[saved/'first'], device='cpu', output=saved/'first/new', log_dir=tmp_path/'logs', plots=False)


def test_duplicate_checkpoint_not_cherry_picked(saved, tmp_path):
    import shutil
    clone=tmp_path/'copy'
    shutil.copytree(saved/'first',clone)
    with pytest.raises(ValueError,match='Multiple completed checkpoints'):
        run(source_runs=[saved/'first',clone],models=[BASE],device='cpu',
            output=tmp_path/'out',log_dir=tmp_path/'logs',plots=False)


def test_changed_data_rejected(saved, tmp_path):
    import shutil
    clone=tmp_path/'changed'
    shutil.copytree(saved/'first',clone)
    path=clone/'data_pin.json'
    data=json.loads(path.read_text())
    name=data['feature_spec']['metric_names'][1]
    data['samples'][0]['performance'][name] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='fingerprint'):
        run(source_runs=[clone],models=[BASE],device='cpu',output=tmp_path/'out',log_dir=tmp_path/'logs',plots=False)


def test_resume_data_pointer_and_immutable_saved_preprocessing(saved, tmp_path):
    import shutil
    clone = tmp_path / 'resume'
    shutil.copytree(saved/'first/training_pin', clone/'training_pin')
    manifest = json.loads((clone/'training_pin/manifest_pin.json').read_text())
    (clone/'data_source_pin.json').write_text(json.dumps(dict(
        data_path=str(saved/'first/data_pin.json'), dataset_fingerprint=manifest['dataset_fingerprint'])))
    out = run(source_runs=[clone/'training_pin'], models=[BASE], device='cpu',
              output=tmp_path/'out', log_dir=tmp_path/'logs', plots=False)
    report = json.loads((out/'statistics_pin.json').read_text())
    assert report['Aggregate'][-1]['replay_status'] == 'matched_tolerance'


def test_same_named_model_with_different_settings_not_pooled(saved, tmp_path):
    import shutil
    clone = tmp_path / 'changed_config'
    shutil.copytree(saved/'second', clone)
    path = clone/'training_pin/seed_29_pin'/BASE/'checkpoint_pin.pt'
    checkpoint = torch.load(path, weights_only=True)
    checkpoint['config']['op_transfer']['lr'] = .01
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='different saved training settings'):
        run(source_runs=[saved/'first', clone], models=[BASE], device='cpu',
            output=tmp_path/'out', log_dir=tmp_path/'logs', plots=False)


def test_native_tree_and_exact_sampled_ids_from_larger_canonical_source(saved, tmp_path):
    pytest.importorskip('xgboost')
    from final_exam.run_pin import run_experiment_pin
    source = tmp_path/'tree_source'
    source.mkdir()
    data = (saved/'first/data_pin.json').read_text()
    config = json.loads((saved/'first/input_pin.json').read_text())
    config['dataset'].update(max_samples_per_circuit=16, sampling_seed=17)
    config['final_exam_pin']['xgboost'] = dict(num_boost_round=2, max_depth=2)
    (source/'data_pin.json').write_text(data)
    (source/'input_pin.json').write_text(json.dumps(config))
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment_pin(source/'data_pin.json', source/'input_pin.json', source/'training_pin',
                           ['xgboost_pin'], [17], 'cpu', log_dir=tmp_path/'tree_logs_pin')
    before = file_digest(source/'data_pin.json')
    out = run(source_runs=[source], device='cpu', output=tmp_path/'out', log_dir=tmp_path/'logs', plots=False)
    assert file_digest(source/'data_pin.json') == before
    report = json.loads((out/'statistics_pin.json').read_text())
    assert report['Aggregate'][-1]['replay_status'] == 'matched_tolerance'
    manifest = json.loads((source/'training_pin/manifest_pin.json').read_text())
    with (out/'sample_membership_pin.csv').open(encoding='utf-8-sig') as f:
        members = list(csv.DictReader(f))
    expected = set(manifest['train'] + manifest['val'] + manifest['test'])
    assert len(members) == 16 and {m['sample_id'] for m in members} == expected
    assert len(json.loads(data)['samples']) > len(members)
    assert all(row['status'] == 'unpaired' for row in report['PairStatus'])
