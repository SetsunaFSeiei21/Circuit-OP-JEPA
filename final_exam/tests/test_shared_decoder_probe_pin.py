"""Independent multi-circuit checkpoint diagnostics with held-out validation."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from final_exam.shared_decoder_probe_pin import inspect_sources_pin, run_shared_decoder_probe_pin
from final_exam.spectral_probe_pin import sha256
from final_exam.tests.test_spectral_probe_pin import saved_runs


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def read_table(output, name):
    with (output / (name + '_pin.csv')).open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def run(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_shared_decoder_probe_pin(**kwargs)


def test_multi_circuit_independent_fits_validation_features_and_sources_unchanged(saved_runs, tmp_path, monkeypatch):
    root, sources, _ = saved_runs
    before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
    from final_exam.models.width_decoder_pin import WidthDecoderModelPin
    original_forward = WidthDecoderModelPin.forward
    seen = []
    def checked_forward(self, batch):
        assert not {'op', 'owner_op', 'performance', 'physical_performance', 'performance_mask'} & set(batch)
        seen.append(self.mode)
        return original_forward(self, batch)
    monkeypatch.setattr(WidthDecoderModelPin, 'forward', checked_forward)
    output = run(sources=sources, models=['width_decoder_pin', 'width_decoder_op_pin'], seeds=[17],
        ranks=[1, 16], num_samples=3, num_val_samples=2, max_tokens=12, batch_size=2,
        component='both', scope='all', validation_scope='all', device='cpu',
        output=tmp_path/'out', log_dir=tmp_path/'logs', max_plots=1)
    record = json.loads((output/'probe_pin.json').read_text())
    assert record['status'] == 'complete' and len(record['checkpoints']) == 4 and seen
    assert record['cross_circuit_fit'] is False and record['checkpoint_modified'] is False
    assert record['version']==2 and record['vector_limit']==64
    for circuit, selected in record['selected_sample_ids'].items():
        manifest = json.loads((root/circuit/'training_pin/manifest_pin.json').read_text())
        assert set(selected['train']) <= set(manifest['train'])
        assert set(selected['val']) <= set(manifest['val'])
        assert set(selected['train'] + selected['val']).isdisjoint(manifest['test'])
    groups = read_table(output, 'Groups')
    assert len(groups) == 60  # two circuits x (10 performance + 20 performance/OP)
    assert all(int(g['branches']) == 2 for g in groups)
    assert len({(g['circuit'],g['model'],g['seed'],g['group']) for g in groups}) == len(groups)
    pairs=read_table(output,'SubspaceOverlap')
    assert len(pairs)==120  # one decoder pair x two ranks x 60 independent groups
    assert all((r['branch_a'],r['branch_b'])==('1','2') for r in pairs)
    assert len(read_table(output,'CrossBasis'))==240
    assert read_table(output,'PrincipalAngles') and read_table(output,'VectorCosines')
    errors = read_table(output, 'Errors')
    qkv = [r for r in errors if 'in_proj_weight::' in r['matrix']]
    assert qkv and all(0 < int(r['activation_tokens']) <= 12 for r in qkv)
    functional_o = [r for r in errors if '.out_proj.weight' in r['matrix']]
    assert functional_o and all(r['shared_output_error'] == '' for r in functional_o)
    validation = read_table(output, 'Validation')
    assert {r['strategy'] for r in validation} == {'original', 'independent', 'shared'}
    assert all(r['split'] == 'val' and r['status'] == 'ok' for r in validation)
    full = [r for r in validation if r['requested_rank'] == '16']
    assert full and all(float(r['relative_prediction_change']) < 2e-5 for r in full)
    feature_rows = read_table(output, 'DecoderFeatures')
    assert feature_rows and any(r['branch'].startswith('predictor.head') for r in feature_rows)
    assert all(float(r['relative_feature_error']) < 2e-5 for r in feature_rows if r['requested_rank'] == '16')
    for entry in record['checkpoints']:
        assert (output/entry['output']/'shared_decoder_summary_pin.xlsx').is_file()
    import openpyxl
    wb = openpyxl.load_workbook(output/'shared_decoder_summary_pin.xlsx', read_only=True)
    assert {'Budgets','Validation','DecoderFeatures','Errors','ReadMe',
            'SubspaceOverlap','PrincipalAngles','CrossBasis','VectorCosines'} <= set(wb.sheetnames)
    wb.close()
    assert all(sha256(path) == digest for path, digest in before.items())
    # A second circuit cannot change the first checkpoint's fitted basis/errors.
    solo = run(sources=sources[:1], models=['width_decoder_pin'], seeds=[17], ranks=[1,16],
        num_samples=0, num_val_samples=0, scope='all', component='performance', device='cpu',
        output=tmp_path/'solo', log_dir=tmp_path/'solo_logs', max_plots=0)
    first_budget = [r for r in read_table(output,'Budgets')
                    if r['circuit']=='synthetic_0' and r['model']=='width_decoder_pin']
    assert read_table(solo,'Budgets') == first_budget
    first_pairs=[r for r in pairs if r['circuit']=='synthetic_0' and r['model']=='width_decoder_pin']
    assert read_table(solo,'SubspaceOverlap')==first_pairs


def test_weights_only_no_data_load_preflight_stdlib_and_group_validation(saved_runs, tmp_path):
    root, sources, _ = saved_runs
    data = root/'synthetic_0/data_pin.json'
    original = data.read_bytes()
    data.unlink()
    try:
        script = Path(__file__).resolve().parents[1]/'shared_decoder_probe_pin.py'
        result = subprocess.run([sys.executable,'-S',str(script),'--source',sources[0],
            '--num-samples','0','--num-val-samples','0','--inspect-source'],capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
        output = run(sources=sources[:1],models=['width_decoder_pin'],seeds=[17],ranks=[2],
            num_samples=0,num_val_samples=0,scope='ffn',device='cpu',
            output=tmp_path/'weight',log_dir=tmp_path/'weight_logs',max_plots=0)
        assert all(r['shared_output_error']=='' for r in read_table(output,'Errors'))
        assert not read_table(output,'Validation')
        assert read_table(output,'SubspaceOverlap') and read_table(output,'CrossBasis')
        assert all(r['cross_output_error']=='' for r in read_table(output,'CrossBasis'))
    finally:
        data.write_bytes(original)
    output = run(sources=sources[:1],models=['width_decoder_pin'],seeds=[17],ranks=[2],
        num_samples=0,num_val_samples=2,scope='ffn',validation_scope='group',device='cpu',
        output=tmp_path/'groups',log_dir=tmp_path/'groups_logs',max_plots=0)
    records = read_table(output,'Validation')
    assert {r['group'] for r in records} == {'original','performance.layer_1.linear1','performance.layer_1.linear2'}


def test_invalid_sources_arguments_and_destination_rejected_before_writes(saved_runs, tmp_path):
    root, sources, _ = saved_runs
    with pytest.raises(ValueError,match='unique circuit'):
        inspect_sources_pin(sources[:1]*2,['width_decoder_pin'],[17],0,0)
    with pytest.raises(ValueError,match='absent'):
        inspect_sources_pin(sources,['width_decoder_pin'],[999],0,0)
    with pytest.raises(ValueError,match='Only width_decoder_op_pin'):
        inspect_sources_pin(sources,['width_decoder_pin'],[17],0,0,'op_predictor')
    output = tmp_path/'out'
    output.mkdir()
    (output/'sentinel').write_text('unchanged')
    with pytest.raises(ValueError,match='new output'):
        run(sources=sources[:1],models=['width_decoder_pin'],seeds=[17],num_samples=0,num_val_samples=0,
            device='cpu',output=output,log_dir=tmp_path/'logs',max_plots=0)
    assert (output/'sentinel').read_text() == 'unchanged' and not (tmp_path/'logs').exists()
    with pytest.raises(ValueError,match='Ranks'):
        run(sources=sources[:1],models=['width_decoder_pin'],seeds=[17],ranks=[0],
            num_samples=0,num_val_samples=0,device='cpu',output=tmp_path/'bad',max_plots=0)
    assert not (tmp_path/'bad').exists()


def test_existing_factorized_checkpoints_and_seeds_analyzed_independently(tmp_path):
    from final_exam.tests.test_run_pin import tiny_inputs
    from final_exam.run_pin import run_experiment_pin
    source = tmp_path/'source'
    source.mkdir()
    data, config, _, _ = tiny_inputs(source)
    config['model'].update(encoder_layers=3, cross_layers=2)
    config['width_decoder_pin'] = dict(n_experts=2, hidden_dim=8, dropout=0.)
    dataset = source/'data_pin.json'
    cfg_path = source/'input_pin.json'
    dataset.write_text(json.dumps(data))
    cfg_path.write_text(json.dumps(config))
    mode = 'width_decoder_lowrank_both_pin'
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment_pin(dataset,cfg_path,source/'training_pin',[mode],[17,29],'cpu',smoke=True)
    paths = list((source/'training_pin').rglob('checkpoint_pin.pt'))
    hashes = {str(p):sha256(p) for p in paths}
    output = run(sources=[f'synthetic_0={source}'],models=[mode],seeds=[17,29],ranks=[16],
        num_samples=3,num_val_samples=2,max_tokens=10,batch_size=2,device='cpu',
        output=tmp_path/'out',log_dir=tmp_path/'logs',max_plots=0)
    groups = read_table(output,'Groups')
    assert len(groups)==20 and {g['seed'] for g in groups}=={'17','29'}
    assert all(g['source_parameterization']=='effective_weights' for g in groups)
    assert all(int(r['activation_tokens'])>0 for r in read_table(output,'Errors'))
    assert all(float(r['relative_prediction_change'])<2e-5 for r in read_table(output,'Validation'))
    assert all(sha256(path)==digest for path,digest in hashes.items())
