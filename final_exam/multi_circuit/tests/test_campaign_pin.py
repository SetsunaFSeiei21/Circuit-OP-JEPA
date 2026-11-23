import copy
import json
import subprocess
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from openpyxl import load_workbook
from final_exam.multi_circuit.tests.test_joint_pin import joint
from final_exam.multi_circuit.config_pin import atomic_pin, read_pin, digest_pin
from final_exam.multi_circuit.worker_pin import prepare_pin, model_worker_pin
from final_exam.multi_circuit.summarize_pin import export_pin
from final_exam.multi_circuit.launch_pin import submit_pin, snapshot_pin
from experiments.op_transfer import engine
from final_exam.run_pin import load_checkpoint_pin
from device_jepa.training import state_digest


def campaign_fixture(joint):
    data,cfg,root,_=joint
    campaign=root/'campaign_pin';campaign.mkdir()
    model='mlp_pin'
    config=copy.deepcopy(cfg)
    config['final_exam_pin']['models']=[model]
    trials=[]
    for width in (8,16):
        c=copy.deepcopy(config);c['model']['ffn_dim']=width
        trials.append(dict(id=f'trial_{width}_pin',parameters={'model.ffn_dim':width},config=c))
    plan={model:dict(config=config,trials=trials)}
    protocol=dict(dataset=config['dataset'],normalization=config['normalization'],search_seed=11,
                  seeds=[17,29,43,59,71],selection='circuit_macro_val_r2',baseline='mlp_pin')
    record=dict(protocol=protocol,models=[model],source_files={},source_digest=digest_pin({}),plan_digest=digest_pin(plan),
        canonical_data=str(root/'data.json'),code_commit='fixture',smoke=True,fixed_config=False,device='cpu',
        conda_home='',conda_env='',raw_root='',log_dir=str(root/'logs_pin'),hpc_dir=str(root/'hpc_logs_pin'))
    atomic_pin(campaign/'campaign_pin.json',record);atomic_pin(campaign/'plan_pin.json',plan)
    return campaign,record


def test_complete_search_five_fresh_seeds_excel_and_idempotence(joint):
    campaign,record=campaign_fixture(joint)
    prepare_pin(campaign)
    model_worker_pin(campaign,0,'search')
    assert not (campaign/'training_pin').exists()
    selected=read_pin(campaign/'selected_pin/mlp_pin.json')
    trials=read_pin(campaign/'search_pin/mlp_pin/trials_pin.json')
    assert selected['validation_macro_r2']==max(t['macro_r2'] for t in trials)
    model_worker_pin(campaign,0,'train')
    output=export_pin(campaign)
    assert all(row['completed_seeds']==5 for row in output['PooledTest'])
    report=load_workbook(campaign/'reports_pin/summary_pin.xlsx',read_only=True)
    assert {'PerCircuit','PooledTest','CircuitMacro','SeedDetails','SearchTrials','BestConfigs','TrainingCost','RunStatus','PairedComparison'}<=set(report.sheetnames)
    for path in (campaign/'search_pin').rglob('training_pin'):
        assert not list(path.rglob('test_predictions_pin.npz'))
    with patch('final_exam.multi_circuit.worker_pin.subprocess.run',side_effect=AssertionError('completed jobs were relaunched')):
        prepare_pin(campaign)
        model_worker_pin(campaign,0,'all')
    plan=read_pin(campaign/'plan_pin.json');plan['mlp_pin']['config']['model']['dim']=16
    atomic_pin(campaign/'plan_pin.json',plan)
    with pytest.raises(ValueError,match='plan changed'):
        model_worker_pin(campaign,0,'train')


def test_partial_epoch_resume_matches_reference(joint):
    data,cfg,root,run=joint
    reference,_=run(['mlp_pin'],name='reference')
    original=engine.fit_stage
    def interrupted(ctx,model,*args,**kwargs):
        def before(_):
            before.count+=1
            if before.count==2:raise RuntimeError('simulated outage')
        before.count=0
        return original(ctx,model,*args,**kwargs,before_epoch=before)
    with patch.object(engine,'fit_stage',side_effect=interrupted):
        with pytest.raises(RuntimeError,match='simulated outage'):
            run(['mlp_pin'],name='failed')
    resumed,result=run(['mlp_pin'],name='recovered',resume=root/'failed_pin')
    assert result['stages'][0]['resume']['optimizer_state_restored']
    assert result['stages'][0]['resume']['restored_epoch']==1
    a,_=load_checkpoint_pin(reference/'seed_17_pin/mlp_pin/checkpoint_pin.pt')
    b,_=load_checkpoint_pin(resumed/'seed_17_pin/mlp_pin/checkpoint_pin.pt')
    assert state_digest(a)==state_digest(b)


def test_slurm_dag_array_resources_and_absolute_snapshot(joint):
    campaign,record=campaign_fixture(joint)
    record['models']=['mlp_pin','zerosim_supervised_pin'];record['device']='cuda'
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,0,stdout=str(100+len(calls))+'\n',stderr='')
    with patch('final_exam.multi_circuit.launch_pin.subprocess.run',side_effect=fake):
        submit_pin(campaign,record,'all',False,False,2,dict(cpus=8,mem='80G',time='01:00:00',account='b_phzhwu',partition='ex01A800'))
    assert '--dependency=afterok:101' in calls[1]
    assert '--array=0-1%2' in calls[1]
    assert '--gres=gpu:1' in calls[1] and '--gres=gpu:1' not in calls[0]
    assert '--dependency=afterany:102' in calls[2]
    assert str(campaign/'source_pin/final_exam/multi_circuit/run_multi_pin.sh') in calls[1]


def test_snapshot_is_independent_of_future_checkout_edits(tmp_path):
    target=tmp_path/'snapshot_pin'
    files,commit=snapshot_pin(target)
    assert len(commit)==40
    assert 'final_exam/multi_circuit/worker_pin.py' in files
    assert not any('/results_pin/' in p or '/logs_pin/' in p for p in files)
    assert (target/'final_exam/multi_circuit/run_multi_pin.sh').stat().st_mode & 0o111
    assert not (target/'final_exam/multi_circuit/worker_pin.py').stat().st_mode & 0o222
