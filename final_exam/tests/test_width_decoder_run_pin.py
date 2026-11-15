"""Training, mixed-family caches, checkpoint deployment, Excel and exact resume."""
import contextlib
import copy
import io
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import pytest
import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.run_pin import run_experiment_pin, load_checkpoint_pin, validate_request_pin
from final_exam.tests.test_run_pin import tiny_inputs
from final_exam.width_registry_pin import WIDTH_MODELS_PIN


@pytest.fixture
def setup(tmp_path):
    _, cfg, data, config = tiny_inputs(tmp_path)
    torch.set_num_threads(1)
    cfg['model'].update(encoder_layers=4, cross_layers=2, dropout=.1)
    cfg['width_decoder_pin'] = dict(n_experts=2, hidden_dim=8, dropout=.1)
    options = engine.settings(cfg)
    options['epochs'] = dict.fromkeys(options['epochs'], 1)
    options['epochs'].update(readout=2, student_predicted_op=2, op_predictor=2)
    options.update(patience=0, batch_size=8)
    cfg['op_transfer'] = options
    config.write_text(json.dumps(cfg))
    def run(name, models, source=None):
        output = tmp_path / (name + '_pin')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_experiment_pin(data, config, output, list(models), [17], 'cpu', resume_from=source)
        return output, result
    return tmp_path, cfg, data, config, run


def test_three_models_export_replay_dependencies_and_completed_resume(setup):
    root, cfg, data, config, run = setup
    # Wide predictor first, original predictor second, then reuse original.
    models = ('width_decoder_op_pin', 'device_op_cascade_pin', 'width_decoder_pin')
    output, result = run('complete', models)
    stages = {s['stage_id'].split('/')[-1]: s for s in result['stages']}
    assert len(stages) == len(result['stages']) == 7
    assert {'device_op_predictor_pin', 'width_device_op_predictor_pin',
            'frozen_device_op_cache_pin', 'frozen_width_device_op_cache_pin'} <= set(stages)
    for mode in WIDTH_MODELS_PIN:
        model, saved = load_checkpoint_pin(output / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert saved['input_contract_pin'] == model.architecture_contract
        assert all(not p.requires_grad for p in model.predictor.parameters())
        assert 'width_decoder_pin' in saved['config']
        report = json.loads((output / f'seed_17_pin/{mode}/test_pin.json').read_text())
        assert report['width_decoder_pin'] == model.architecture_contract
        assert not report['requires_measured_op']
    with zipfile.ZipFile(output / 'summary_pin.xlsx') as z:
        xml = ''.join(z.read(p).decode() for p in z.namelist() if p.endswith('.xml'))
        assert 'WidthDecoder' in xml and all(mode in xml for mode in models)
        assert 'physical_device_memory' in xml and 'last_pe_device_mlp' in xml
    # Complete stages must replay without any optimizer update.
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('unexpected retraining')):
        resumed, replay = run('recovered', models, output)
    for stage in replay['stages']:
        if stage['epochs']:
            assert stage['resume']['new_optimizer_steps'] == 0
    for mode in models:
        a, _ = load_checkpoint_pin(output / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(a) == state_digest(b)
    for mode in WIDTH_MODELS_PIN:
        _, single = run('subset_' + mode, [mode], output)
        assert len(single['results']) == 1
        assert len(single['stages']) == 3


@pytest.mark.parametrize('interrupted_stage', ['width_device_op_predictor_pin', 'width_decoder_op_pin'])
def test_epoch_resume_matches_uninterrupted_weights(setup, interrupted_stage):
    root, cfg, data, config, run = setup
    models = ['width_decoder_op_pin']
    reference, _ = run('reference', models)
    original = engine.fit_stage
    def interrupt(ctx, model, name, *args, **kwargs):
        if name == interrupted_stage:
            calls = 0
            def before(_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated epoch interruption')
            kwargs['before_epoch'] = before
        return original(ctx, model, name, *args, **kwargs)
    with patch.object(engine, 'fit_stage', side_effect=interrupt):
        with pytest.raises(RuntimeError, match='simulated epoch'):
            run('failed', models)
    resumed, result = run('resumed', models, root / 'failed_pin')
    stage = next(s for s in result['stages'] if s['stage_id'].endswith('/' + interrupted_stage))
    assert stage['resume']['optimizer_state_restored']
    assert stage['resume']['restored_epoch'] == 1
    a, _ = load_checkpoint_pin(reference / 'seed_17_pin/width_decoder_op_pin/checkpoint_pin.pt')
    b, _ = load_checkpoint_pin(resumed / 'seed_17_pin/width_decoder_op_pin/checkpoint_pin.pt')
    assert state_digest(a) == state_digest(b)


def test_switch_back_to_original_cached_staged_base(setup):
    *_, run = setup
    models = ['device_op_supervised_gate_pin', 'width_decoder_op_pin',
              'device_op_supervised_gate_fusion_pin', 'width_decoder_pin']
    _, result = run('cache_switch', models)
    assert [r['mode'] for r in result['results']] == models


def test_model_subsets_and_settings_validation(setup):
    root, cfg, data, config, run = setup
    for mode in WIDTH_MODELS_PIN:
        output, result = run('only_' + mode, [mode])
        assert len(result['stages']) == 3  # predictor, its cache, and downstream
        assert len(result['results']) == 1
        changed = copy.deepcopy(cfg)
        changed['width_decoder_pin']['n_experts'] += 1
        config.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='configuration must match'):
            run('bad_resume_' + mode, [mode], output)
        assert not (root / ('bad_resume_' + mode + '_pin')).exists()
        config.write_text(json.dumps(cfg))
    invalid = copy.deepcopy(cfg)
    invalid['width_decoder_pin']['dropout'] = -1
    with pytest.raises(ValueError, match='dropout'):
        validate_request_pin(invalid, ['width_decoder_pin'], [17], 'cpu')
