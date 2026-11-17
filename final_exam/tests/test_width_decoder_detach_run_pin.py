"""Shared OP dependency, exact interrupted resume, checkpoint policy and Excel."""
import json
import zipfile
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.run_pin import load_checkpoint_pin
from final_exam.tests.test_width_decoder_run_pin import setup
from final_exam.width_registry_pin import validate_width_contract_pin

MODE = 'width_decoder_detach_pin'


def test_paired_experiment_costs_metadata_excel_and_completed_resume(setup):
    root, cfg, data, config, run = setup
    modes = ['width_decoder_pin', MODE]
    out, result = run('detach_complete', modes)
    assert len(result['stages']) == 4  # one OP predictor, one cache, two performance stages
    assert len(result['results']) == 2
    audits = []
    for mode in modes:
        directory = out / f'seed_17_pin/{mode}'
        model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
        assert saved['recipe']['architecture'] == model.architecture_contract
        assert (directory / 'config_pin.json').is_file()
        report = json.loads((directory / 'test_pin.json').read_text())
        validate_width_contract_pin(mode, report['width_decoder_pin'])
        assert not report['requires_measured_op']
        audits.append(json.loads((directory / 'width_audit_pin.json').read_text()))
        stage = next(s for s in result['stages'] if s['stage_id'].endswith('/' + mode))
        assert stage['epochs'] == 2 and stage['seconds'] > 0
    assert audits[0]['initial_encoder_digest'] == audits[1]['initial_encoder_digest']
    assert audits[0]['predictor_digest'] == audits[1]['predictor_digest']
    assert audits[0]['decoder_parameters'] == audits[1]['decoder_parameters']
    spec = audits[1]['performance']['memory_gradients']
    assert spec['detached_decoder_inputs'] == [1] and spec['attached_decoder_inputs'] == [2]
    assert not spec['encoder_chain_detached'] and not spec['decoders_frozen']
    with zipfile.ZipFile(out / 'summary_pin.xlsx') as book:
        xml = ''.join(book.read(p).decode() for p in book.namelist() if p.endswith('.xml'))
    for label in ('WidthDecoder', 'MAE', 'RMSE', 'R2', 'TrainingCost', MODE,
                  'detach_shallow_decoder_memory', 'Detached decoder PE inputs', 'Attached decoder PE inputs'):
        assert label in xml
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('unexpected retraining')):
        resumed, replay = run('detach_recovered', modes, out)
    for stage in replay['stages']:
        if stage['epochs']:
            assert stage['resume']['new_optimizer_steps'] == 0
    for mode in modes:
        a, _ = load_checkpoint_pin(out / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(a) == state_digest(b)
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('unexpected retraining')):
        _, subset = run('detach_subset', [MODE], out)
    assert len(subset['results']) == 1 and len(subset['stages']) == 3


def test_detach_interrupted_epoch_resume_matches_continuous_training(setup):
    root, cfg, data, config, run = setup
    reference, _ = run('detach_reference', [MODE])
    original = engine.fit_stage
    def interrupt(ctx, model, name, *args, **kwargs):
        if name == MODE:
            calls = 0
            def before(_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated detach epoch interruption')
            kwargs['before_epoch'] = before
        return original(ctx, model, name, *args, **kwargs)
    with patch.object(engine, 'fit_stage', side_effect=interrupt):
        with pytest.raises(RuntimeError, match='simulated detach epoch'):
            run('detach_failed', [MODE])
    resumed, result = run('detach_resumed', [MODE], root / 'detach_failed_pin')
    stage = next(s for s in result['stages'] if s['stage_id'].endswith('/' + MODE))
    assert stage['resume']['optimizer_state_restored']
    assert stage['resume']['restored_epoch'] == 1
    a, _ = load_checkpoint_pin(reference / f'seed_17_pin/{MODE}/checkpoint_pin.pt')
    b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{MODE}/checkpoint_pin.pt')
    assert state_digest(a) == state_digest(b)
    assert b.decoder.detach_shallow_memory
