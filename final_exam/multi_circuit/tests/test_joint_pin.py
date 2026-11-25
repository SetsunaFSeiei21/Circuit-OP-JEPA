"""Joint training, physical pooled metrics, strict searches and atomic resume."""
import contextlib
import copy
import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from device_jepa.data import synthetic_dataset, build_manifest, dataset_fingerprint, _canonical
from final_exam.run_pin import run_experiment_pin, load_checkpoint_pin, validate_request_pin
from final_exam.tests.test_run_pin import tiny_inputs
from final_exam.multi_circuit.context_pin import MultiCircuitContextPin
from final_exam.multi_circuit.metrics_pin import summarize_arrays_pin
from final_exam.multi_circuit.config_pin import expand_grid_pin
from final_exam.multi_circuit.summarize_pin import aggregate_pin, paired_pin
from device_jepa.training import state_digest


@pytest.fixture
def joint(tmp_path):
    _, cfg, path, config_path = tiny_inputs(tmp_path)
    data = synthetic_dataset(24)
    data['samples'] = [r for r in data['samples'] if r['circuit_id'] in ('synthetic_0', 'synthetic_1')]
    for row in data['samples']:
        row['devices'] = row['devices'][:3 if row['circuit_id'] == 'synthetic_0' else 4]
    cfg['dataset']['train_circuits'] = ['synthetic_0', 'synthetic_1']
    cfg['model'].update(encoder_layers=4,cross_layers=2)
    cfg['final_exam_pin']['model_overrides'] = {}
    cfg['lasd_moe_pin'] = dict(n_experts=4,top_k=2,hidden_dim=8,dropout=0.,diagnostic_samples=0)
    cfg['width_decoder_pin'] = dict(n_experts=2,hidden_dim=8,dropout=0.)
    path.write_text(json.dumps(data))
    config_path.write_text(json.dumps(cfg))
    torch.set_num_threads(1)
    def run(models, name='run', split='test', resume=None):
        output = tmp_path/(name+'_pin')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(path,config_path,output,list(models),[17],'cpu',smoke=True,
                joint=True,evaluation_split=split,resume_from=resume)
        return output,result
    return data,cfg,tmp_path,run


MODELS = ['mlp_pin','xgboost_pin','gat_pin','deepgen_supervised_pin','deepgen_op_pretrained_pin',
          'gtn_pin','zerosim_supervised_pin','accformer_pin','device_op_cascade_pin','width_decoder_pin',
          'width_decoder_op_pin','predicted_op_pin','student_predicted_op_pin','student_predicted_op_distill_pin',
          'op_oracle_pin','design_op_oracle_pin','lasd_graph_layer_headwise_detach_pin',
          *[f'lasd_graph_layer_headwise_detach_pin_v{i}' for i in (1,2,3,4,5,6,7,8,9)]]


@pytest.mark.parametrize('name', MODELS)
def test_joint_model_replay_two_topologies_and_search_boundary(joint, name):
    data,cfg,root,run=joint
    # A search is structurally forbidden from requesting test feature batches.
    output,result=run([name],split='val')
    report=result['results'][0]['report']
    assert set(report['per_circuit']) == {'synthetic_0','synthetic_1'}
    assert not list(output.rglob('test_pin.json'))
    path=output/f'seed_17_pin/{name}'
    model,metadata=load_checkpoint_pin(path/('checkpoint_pin.json' if name=='xgboost_pin' else 'checkpoint_pin.pt'))
    manifest=json.loads((output/'manifest_pin.json').read_text())
    prep=json.loads((output/'preprocessing_pin.json').read_text())
    replay=root/'replay_pin';replay.mkdir()
    context=MultiCircuitContextPin(data,manifest,cfg,'cpu',{},replay,preprocessing=prep)
    assert context.cache('train').device.type=='cpu'
    with pytest.raises(RuntimeError,match='Search cannot'):
        context.cache('test')
    predictions=[]
    with torch.no_grad():
        for batch in context.batches('val',8,op=name in ('op_oracle_pin','design_op_oracle_pin')):
            predictions.append(model(batch).double().numpy())
    with np.load(path/'val_predictions_pin.npz',allow_pickle=False) as saved:
        np.testing.assert_allclose(np.concatenate(predictions),saved['normalized_prediction'],rtol=1e-5,atol=1e-6)
        assert len(set(saved['circuit_id']))==2
    if name!='xgboost_pin':
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('retrained complete stage')):
            resumed,_=run([name],name='resume',split='val',resume=output)
        other,_=load_checkpoint_pin(resumed/f'seed_17_pin/{name}/checkpoint_pin.pt')
        assert state_digest(model)==state_digest(other)


def test_metrics_pooled_is_not_average_of_circuit_metrics():
    prep={'feature_spec':{'metric_names':['y']},'performance':{}}
    y=torch.tensor([[0.],[1.],[100.],[102.]],dtype=torch.float64)
    pred=y+torch.tensor([[1.],[1.],[3.],[3.]])
    mask=torch.ones_like(y,dtype=torch.bool)
    r=summarize_arrays_pin(pred,y,mask,['a','a','b','b'],prep,pred,y)
    assert r['pooled']['y']['rmse']==pytest.approx(np.sqrt(5))
    assert r['circuit_macro']['y']['rmse']==pytest.approx(2)
    assert r['macro_r2']==pytest.approx((-3-8)/2)
    assert r['pooled']['y']['r2']>0.99-0.01
    y[:2]=0
    r=summarize_arrays_pin(pred,y,mask,['a','a','b','b'],prep,pred,y)
    assert r['macro_r2'] is None # no eligible R2 in a; fail selection, don't average b alone


def test_seed_std_and_five_pair_inference():
    rows=[dict(model='m',circuit='a',metric='y',seed=s,count=9,mae=v,rmse=v,r2=v,normalized_rmse=v)
          for s,v in [(17,1.),(29,3.)]]
    assert aggregate_pin(rows)[0]['mae_std']==pytest.approx(2**.5)
    details=[]
    for seed in range(5):
        for model,r2 in [('base',.1),('new',.2)]:
            details.append(dict(view='x',circuit='a',metric='y',model=model,seed=seed,mae=1-r2,rmse=1-r2,r2=r2))
    assert all(r['exact_sign_flip_p']==.0625 for r in paired_pin(details,'base'))


def test_fingerprint_streaming_preserves_existing_digest(joint):
    import hashlib
    data,_,_,_=joint
    ordered=dict(data)
    ordered['samples']=[dict(row,devices=sorted(row['devices'],key=lambda x:x['device_id'])) for row in sorted(data['samples'],key=lambda x:x['sample_id'])]
    assert dataset_fingerprint(data)==hashlib.sha256(_canonical(ordered).encode()).hexdigest()


def test_grid_is_deterministic_and_rejects_inactive_keys(joint):
    _,cfg,_,_=joint
    space={'grid':{'op_transfer.lr':[.0001,.0003],'model.dim':[8,16]}}
    a,_=expand_grid_pin('zerosim_supervised_pin',cfg,space)
    b,_=expand_grid_pin('zerosim_supervised_pin',cfg,space)
    assert a==b and len(a)==4
    with pytest.raises(ValueError,match='Inactive'):
        expand_grid_pin('mlp_pin',cfg,{'grid':{'model.heads':[2,4]}})
    with pytest.raises(ValueError,match='Inactive'):
        expand_grid_pin('accformer_pin',cfg,{'grid':{'model.decoder_ffn_dim':[256]}})
    with pytest.raises(ValueError):
        validate_request_pin(cfg,['mlp_pin'],[17],'cpu') # single-circuit boundary unchanged
