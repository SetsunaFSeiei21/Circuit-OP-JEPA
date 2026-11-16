"""Numerical ground truth, real checkpoint calibration and source protection."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from final_exam.spectral_analysis_pin import (MatrixPin, ActivationCapturePin,
    analyze_matrix_pin, matrices_pin, shared_basis_pin, subspace_overlap_pin)
from final_exam.spectral_probe_pin import inspect_sources_pin, run_probe_pin, sha256


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def item_for(weight, ranks=(1, 2, 4), x=None):
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False).to(weight)
    layer.weight.data.copy_(weight)
    return analyze_matrix_pin(MatrixPin('layer.weight', layer.weight, layer, 'layer', 'linear'), ranks, x)


def test_known_spectrum_rank_budget_and_actual_output_error():
    weight = torch.diag(torch.tensor([4., 2., 0., 0.], dtype=torch.float64))
    x = torch.tensor([[1., 0.], [0., 10.], [0., 0.], [0., 0.]], dtype=torch.float64)
    summary, rows, item = item_for(weight, x=x)
    assert summary['rank_99'] == summary['numerical_rank'] == 2
    assert summary['stable_rank'] == pytest.approx(1.25)
    assert rows[0]['relative_weight_error'] == pytest.approx((4/20)**.5)
    assert rows[0]['relative_layer_output_error'] == pytest.approx((400/416)**.5)
    assert rows[0]['optimal_calibration_output_error'] == pytest.approx((16/416)**.5)
    assert rows[1]['relative_weight_error'] == 0
    assert rows[1]['relative_layer_output_error'] == 0
    assert rows[-1]['parameter_saving_fraction'] < 0  # rank 4 factorization doubles 4x4 weights


def test_common_subspace_recovers_multiple_core_matrices():
    torch.manual_seed(1)
    u = torch.linalg.qr(torch.randn(7, 2, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(5, 2, dtype=torch.float64))[0]
    a = item_for(u @ torch.diag(torch.tensor([3., 1.], dtype=torch.float64)) @ v.T)[2]
    b = item_for(u @ torch.tensor([[1., 2.], [3., 1.]], dtype=torch.float64) @ v.T)[2]
    overlap = subspace_overlap_pin(a, b, 2, 'cpu')
    assert overlap['u_overlap'] == pytest.approx(1)
    assert overlap['v_overlap'] == pytest.approx(1)
    groups, errors = shared_basis_pin([a, b], [2], 'cpu')
    assert groups[0]['shared_parameters'] == 2*(7+5)+2*2*2
    assert all(r['relative_weight_error'] < 1e-12 for r in errors)


def test_zero_matrices_and_orthogonal_subspaces_have_honest_overlap():
    a = item_for(torch.diag(torch.tensor([2., 0., 0., 0.])))[2]
    b = item_for(torch.diag(torch.tensor([0., 3., 0., 0.])))[2]
    assert subspace_overlap_pin(a, b, 1, 'cpu')['u_overlap'] == 0
    summary, rows, z = item_for(torch.zeros(4, 4))
    assert summary['rank_99'] == 0 and rows[0]['relative_weight_error'] == 0
    assert subspace_overlap_pin(a, z, 2, 'cpu')['u_overlap'] is None
    _, errors = shared_basis_pin([z, z], [1], 'cpu')
    assert all(r['relative_weight_error'] == 0 for r in errors)


def test_captures_exclude_padding_and_respect_token_budget_and_batch_quotas():
    layer = torch.nn.Linear(3, 4)
    spec = MatrixPin('encoder.ffn.0.weight', layer.weight, layer, 'encoder.ffn.0', 'linear')
    capture = ActivationCapturePin([spec], max_tokens=4)
    mask = torch.tensor([[True, True, False]])
    with capture.install(), torch.inference_mode():
        for batch in range(2):
            capture.set_batch({'device_mask':mask}, batch, 2)
            # Graph + two valid pins + padding with a sentinel.
            x = torch.ones(1, 4, 3) * (batch + 1)
            x[:, -1] = 999
            layer(x)
    x = capture.get(spec.name)
    assert x.shape == (3, 4)
    assert set(x.flatten().tolist()) == {1., 2.}
    assert not layer._forward_pre_hooks


def test_moe_orientation_and_weight_only_mha_are_not_silently_omitted():
    from final_exam.models.width_decoder_pin import DenseMoEMLPPin
    model = torch.nn.ModuleDict({'decoder': DenseMoEMLPPin(6, 2, dict(n_experts=2,hidden_dim=4,dropout=0.)),
                                'attention': torch.nn.MultiheadAttention(8, 2, batch_first=True)})
    specs = matrices_pin(model, 'all')
    a = next(s for s in specs if s.name == 'decoder.weight1::expert_0')
    b = next(s for s in specs if s.name == 'decoder.weight2::expert_0')
    assert a.weight.shape == (4, 6) and b.weight.shape == (2, 4)
    assert {s.kind for s in specs} >= {'mha_q', 'mha_k', 'mha_v', 'moe_1', 'moe_2'}
    assert sum(s.weight.numel() for s in specs) == sum(p.numel() for n,p in model.named_parameters() if 'bias' not in n)


@pytest.fixture(scope='module')
def saved_runs(tmp_path_factory):
    from final_exam.tests.test_run_pin import tiny_inputs
    from final_exam.run_pin import run_experiment_pin
    root = tmp_path_factory.mktemp('spectral_real')
    sources = []
    modes = ['device_op_cascade_pin','width_decoder_pin','width_decoder_op_pin']
    for circuit in ('synthetic_0', 'synthetic_other'):
        source = root / circuit
        source.mkdir()
        data, cfg, _, _ = tiny_inputs(source)
        for row in data['samples']:
            row['circuit_id'] = circuit
        cfg['dataset']['train_circuits'] = [circuit]
        cfg['model'].update(encoder_layers=3,cross_layers=2)
        cfg['width_decoder_pin'] = dict(n_experts=2,hidden_dim=8,dropout=0.)
        cfg['op_transfer']['epochs'] = {k:1 for k in ('teacher','op_predictor','readout')}
        cfg['op_transfer']['patience'] = 0
        dataset = source / 'data_pin.json'
        config = source / 'input_pin.json'
        dataset.write_text(json.dumps(data))
        config.write_text(json.dumps(cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            run_experiment_pin(dataset,config,source/'training_pin',modes,[17],'cpu',smoke=True)
        sources.append(f'{circuit}={source}')
    return root, sources, modes


def test_real_models_calibration_excel_plots_and_immutable_sources(saved_runs):
    root, sources, modes = saved_runs
    before = {str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
    with contextlib.redirect_stdout(io.StringIO()):
        output = run_probe_pin(sources=sources,models=modes,seeds=[17],ranks=[1,2,4],num_samples=3,
            max_tokens=12,batch_size=2,device='cpu',output=root/'probe',log_dir=root/'logs',max_plots=2)
    record = json.loads((output/'probe_pin.json').read_text())
    assert record['status'] == 'complete'
    assert len(record['checkpoints']) == 6
    assert record['retrained'] is False and record['compressed_model_evaluated'] is False
    for circuit, ids in record['selected_sample_ids'].items():
        manifest = json.loads((root/circuit/'training_pin/manifest_pin.json').read_text())
        assert set(ids) <= set(manifest['train'])
        assert set(ids).isdisjoint(manifest['test'])
    def rows(name):
        with (output/(name+'_pin.csv')).open(encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    matrices = rows('Matrices')
    assert all(0 < int(r['activation_tokens']) <= 12 for r in matrices)
    assert rows('SharedBudget') and rows('SharedErrors') and rows('SubspaceOverlap')
    import openpyxl
    wb = openpyxl.load_workbook(output/'spectral_summary_pin.xlsx',read_only=True,data_only=True)
    assert {'Checkpoints','Parameters','Matrices','SharedBudget','ReadMe'} <= set(wb.sheetnames)
    assert len(list((output/'plots_pin').glob('*.png'))) == 2
    assert 'spectral_summary_pin.xlsx' in (output/'report_pin.html').read_text()
    assert all(sha256(path) == value for path,value in before.items())


def test_weights_only_needs_no_raw_data_and_one_circuit_is_supported(saved_runs, tmp_path):
    root, sources, modes = saved_runs
    # Weight-only preflight does not even read the canonical file.
    source = sources[0]
    data = root/'synthetic_0/data_pin.json'
    contents = data.read_bytes()
    data.unlink()
    try:
        assert inspect_sources_pin([source],[modes[0]],[17],0)[0]['data'] is None
        with contextlib.redirect_stdout(io.StringIO()):
            out = run_probe_pin(sources=[source],models=[modes[0]],seeds=[17],ranks=[2],num_samples=0,
                device='cpu',output=tmp_path/'out',log_dir=tmp_path/'logs',max_plots=0)
        with (out/'IndependentLowRank_pin.csv').open(encoding='utf-8-sig') as f:
            assert all(r['relative_layer_output_error']=='' for r in csv.DictReader(f))
        assert not (out/'plots_pin').exists()
    finally:
        data.write_bytes(contents)


def test_all_scope_wide_op_captures_experts_and_leaves_functional_mha_blank(saved_runs, tmp_path):
    _, sources, _ = saved_runs
    with contextlib.redirect_stdout(io.StringIO()):
        output = run_probe_pin(sources=[sources[0]],models=['width_decoder_op_pin'],seeds=[17],ranks=[2],
            num_samples=3,max_tokens=12,batch_size=2,scope='all',device='cpu',
            output=tmp_path/'all',log_dir=tmp_path/'logs',max_plots=0)
    with (output/'Matrices_pin.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert all(int(r['activation_tokens']) > 0 for r in rows if r['kind'].startswith('moe_'))
    assert all(int(r['activation_tokens']) == 0 for r in rows if r['kind'].startswith('mha_'))
    assert any(r['kind']=='moe_1' and r['matrix'].startswith('predictor.') for r in rows)


def test_preflight_does_not_import_torch_and_bad_identity_fails(saved_runs):
    root, sources, modes = saved_runs
    script = Path(__file__).resolve().parents[1]/'spectral_probe_pin.py'
    run = subprocess.run([sys.executable,'-S',str(script),'--source',sources[0],
        '--models',modes[0],'--seeds','17','--num-samples','0','--inspect-source'],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    with pytest.raises(ValueError,match='mismatch'):
        inspect_sources_pin(['wrong='+str(root/'synthetic_0')],[modes[0]],[17],0)
    with pytest.raises(ValueError,match='absent'):
        inspect_sources_pin(sources,[modes[0]],[999],0)


def test_existing_output_rejected_without_any_overwrite(saved_runs, tmp_path):
    _, sources, modes = saved_runs
    output = tmp_path/'out'
    output.mkdir()
    (output/'sentinel').write_text('original')
    with pytest.raises(ValueError,match='new output'):
        run_probe_pin(sources=sources,models=[modes[0]],seeds=[17],num_samples=0,
            device='cpu',output=output,log_dir=tmp_path/'logs',max_plots=0)
    assert (output/'sentinel').read_text() == 'original'
    assert not (tmp_path/'logs').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware required')
def test_cuda_svd_matches_cpu():
    weight = torch.randn(24,16)
    a = item_for(weight)[1]
    b = item_for(weight.cuda())[1]
    for x,y in zip(a,b):
        assert x['relative_weight_error'] == pytest.approx(y['relative_weight_error'],abs=1e-8)
