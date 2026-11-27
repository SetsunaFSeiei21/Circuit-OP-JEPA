"""Independent OP architecture/optimizer settings through the actual pin runner."""
import contextlib
import copy
import io
import json
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.data_pin import PinDataContext
from final_exam.run_pin import load_checkpoint_pin, run_experiment_pin, validate_request_pin
from final_exam.tests.test_run_pin import tiny_inputs
from final_exam.multi_circuit.tests.test_joint_pin import joint


@pytest.fixture
def separate_op(tmp_path):
    data, cfg, data_path, config_path = tiny_inputs(tmp_path)
    torch.set_num_threads(1)
    cfg['model'].update(encoder_layers=3, cross_layers=2)
    cfg['op_transfer']['epochs'] = dict.fromkeys(engine.settings(cfg)['epochs'], 1)
    cfg['op_transfer']['epochs'].update(readout=2, student_predicted_op=2)
    cfg['op_transfer'].update(lr=1e-4, batch_size=8, patience=0)
    cfg['width_decoder_pin'] = dict(n_experts=2, hidden_dim=8, dropout=0.)
    cfg['lasd_moe_pin'] = dict(hidden_dim=8, diagnostic_samples=0,
                                score_chunk_size=8, checkpoint_scores=False)
    cfg['op_predictor'] = dict(
        model=dict(dim=12, heads=3, encoder_layers=2, cross_layers=1,
                   ffn_dim=24, decoder_layers=2, decoder_ffn_dim=24, dropout=0.),
        training=dict(epochs=3, lr=3e-3, batch_size=4, patience=0,
                      weight_decay=2e-4, grad_clip=.5, torch_threads=1),
        width_decoder_pin=dict(n_experts=3, hidden_dim=6, dropout=0.))
    config_path.write_text(json.dumps(cfg))

    def run(name, models, *, source=None, smoke=False):
        output = tmp_path / (name + '_pin')
        observed = {}
        fit = engine.fit_stage

        def observe(ctx, model, stage, epochs, options, *args, **kwargs):
            observed[stage] = dict(epochs=epochs, options=copy.deepcopy(options),
                                   config=copy.deepcopy(ctx.config))
            return fit(ctx, model, stage, epochs, options, *args, **kwargs)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch.object(engine, 'fit_stage', side_effect=observe):
                result = run_experiment_pin(data_path, config_path, output, models,
                                            [17], 'cpu', resume_from=source, smoke=smoke)
        return output, result, observed

    return data, cfg, config_path, run


def test_cascade_lasd_separate_op_training_replay_and_completed_resume(separate_op, tmp_path):
    data, cfg, _, run = separate_op
    modes = ['device_op_cascade_pin', 'lasd_moe_headwise_detach_pin']
    output, result, seen = run('independent', modes)
    op = seen['device_op_predictor_pin']
    assert op['epochs'] == 3
    for field in ('lr', 'batch_size', 'weight_decay', 'grad_clip', 'patience', 'torch_threads'):
        assert op['options'][field] == cfg['op_predictor']['training'][field]
    assert op['config']['model']['dim'] == 12
    assert op['config']['model']['cross_layers'] == 1
    for mode in modes:
        assert seen[mode]['epochs'] == 2
        assert seen[mode]['options']['lr'] == 1e-4
        assert seen[mode]['options']['batch_size'] == 8
        assert seen[mode]['config']['model']['dim'] == 8
        assert seen[mode]['config']['model']['cross_layers'] == 2
    assert sum(s['stage_id'].endswith('/device_op_predictor_pin')
               for s in result['stages']) == 1

    predictor, checkpoint = load_checkpoint_pin(
        output / 'seed_17_pin/shared_pin/device_op_predictor_pin/checkpoint_pin.pt')
    assert predictor.model_config['dim'] == 12
    assert checkpoint['options']['lr'] == 3e-3
    assert checkpoint['options']['batch_size'] == 4
    assert checkpoint['model_config']['encoder_layers'] == 2
    saved_config = json.loads((output / 'config_pin.json').read_text())
    manifest = json.loads((output / 'manifest_pin.json').read_text())
    replay_dir = tmp_path / 'replay'
    replay_dir.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = PinDataContext(data, manifest, saved_config, torch.device('cpu'), {}, replay_dir)
        batch = next(ctx.batches('test', 8))
    for mode in modes:
        path = output / f'seed_17_pin/{mode}/checkpoint_pin.pt'
        first, _ = load_checkpoint_pin(path)
        second, _ = load_checkpoint_pin(path)
        assert first.model_config['dim'] == 8
        assert first.predictor.model_config['dim'] == 12
        assert all(not p.requires_grad for p in first.predictor.parameters())
        with torch.no_grad():
            torch.testing.assert_close(first(batch), second(batch), atol=0, rtol=0)
            assert torch.isfinite(first(batch)).all()

    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retrained completed stage')):
        resumed, _, _ = run('resumed', modes, source=output)
    for mode in modes:
        first, _ = load_checkpoint_pin(output / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        second, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(first) == state_digest(second)


@pytest.mark.parametrize('mode,predictor_name', [
    ('predicted_op_pin', 'op_predictor_pin'),
    ('width_decoder_op_pin', 'width_device_op_predictor_pin'),
])
def test_pin_and_wide_predictors_keep_separate_shapes(separate_op, mode, predictor_name):
    _, _, _, run = separate_op
    output, _, seen = run(mode, [mode])
    assert seen[predictor_name]['epochs'] == 3
    assert seen[predictor_name]['options']['lr'] == 3e-3
    model, _ = load_checkpoint_pin(output / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    assert model.predictor.encoder.cfg.dim == 12
    if mode == 'width_decoder_op_pin':
        assert model.model_config['dim'] == 8
        assert model.width_config['n_experts'] == 2
        assert model.predictor.width_config['n_experts'] == 3
        assert model.predictor.width_config['hidden_dim'] == 6
        assert model.predictor.model_config['decoder_layers'] == 2
        assert model.model_config['decoder_layers'] == 1
    else:
        assert model.downstream.encoder.cfg.dim == 8


def test_smoke_caps_explicit_op_epochs_without_changing_user_config(separate_op):
    _, cfg, _, run = separate_op
    before = copy.deepcopy(cfg)
    _, _, seen = run('smoke', ['device_op_cascade_pin'], smoke=True)
    assert seen['device_op_predictor_pin']['epochs'] == 1
    assert seen['device_op_predictor_pin']['options']['patience'] == 0
    assert seen['device_op_cascade_pin']['epochs'] == 2
    assert cfg == before


def test_legacy_config_still_inherits_downstream_settings(separate_op):
    _, cfg, path, run = separate_op
    cfg.pop('op_predictor')
    cfg['op_transfer']['epochs']['op_predictor'] = 2
    path.write_text(json.dumps(cfg))
    _, _, seen = run('legacy', ['device_op_cascade_pin'])
    op = seen['device_op_predictor_pin']
    assert op['epochs'] == 2
    assert op['config']['model'] == seen['device_op_cascade_pin']['config']['model']
    for key in ('lr', 'batch_size', 'weight_decay', 'grad_clip', 'patience'):
        assert op['options'][key] == seen['device_op_cascade_pin']['options'][key]


def test_invalid_independent_op_architecture_rejected_before_training(separate_op):
    _, cfg, _, _ = separate_op
    cfg['op_predictor']['model']['heads'] = 5
    with pytest.raises(ValueError, match='divisible'):
        validate_request_pin(cfg, ['device_op_cascade_pin'], [17], 'cpu')


def test_device_op_capacity_is_independent_of_downstream_pin_capacity(separate_op):
    data, cfg, path, run = separate_op
    devices = max(len(row['devices']) for row in data['samples'])
    cfg['op_predictor']['model']['max_devices'] = devices
    assert devices < cfg['model']['max_devices']
    path.write_text(json.dumps(cfg))
    output, _, _ = run('device_capacity', ['device_op_cascade_pin'])
    model, _ = load_checkpoint_pin(output / 'seed_17_pin/device_op_cascade_pin/checkpoint_pin.pt')
    assert model.predictor.model_config['max_devices'] == devices
    assert model.model_config['max_devices'] == 32


def test_downstream_search_changes_do_not_change_explicit_op_training(separate_op):
    _, cfg, path, run = separate_op
    first, _, _ = run('reference_op', ['device_op_cascade_pin'])
    cfg['model'].update(dim=16, heads=4, ffn_dim=32, dropout=.2)
    cfg['op_transfer'].update(lr=7e-5, batch_size=6, weight_decay=1e-2, grad_clip=2.)
    path.write_text(json.dumps(cfg))
    second, _, _ = run('changed_downstream', ['device_op_cascade_pin'])
    relative = 'seed_17_pin/shared_pin/device_op_predictor_pin/checkpoint_pin.pt'
    before, _ = load_checkpoint_pin(first / relative)
    after, _ = load_checkpoint_pin(second / relative)
    assert state_digest(before) == state_digest(after)


def test_joint_search_independent_op_shapes_options_and_completed_resume(joint):
    _, cfg, root, run = joint
    cfg['op_predictor'] = dict(
        model=dict(dim=12, heads=3, encoder_layers=3, cross_layers=1,
                   ffn_dim=24, dropout=0., max_devices=4),
        training=dict(epochs=20, lr=3e-3, batch_size=2, patience=0))
    (root / 'input.json').write_text(json.dumps(cfg))
    modes = ['device_op_cascade_pin', 'lasd_graph_layer_headwise_detach_pin_v3']
    seen = {}
    original = engine.fit_stage

    def observe(ctx, model, stage, epochs, options, *args, **kwargs):
        seen[stage] = dict(epochs=epochs, options=copy.deepcopy(options),
                           config=copy.deepcopy(ctx.config))
        return original(ctx, model, stage, epochs, options, *args, **kwargs)

    with patch.object(engine, 'fit_stage', side_effect=observe):
        output, result = run(modes, name='independent_joint', split='val')
    assert seen['device_op_predictor_pin']['epochs'] == 1  # Explicit OP budget obeys smoke.
    assert seen['device_op_predictor_pin']['options']['lr'] == 3e-3
    assert seen['device_op_predictor_pin']['options']['batch_size'] == 2
    assert not list(output.rglob('test_pin.json'))
    assert not list(output.rglob('test_predictions_pin.npz'))
    for row in result['results']:
        assert row['report']['split'] == 'val'
        assert set(row['report']['per_circuit']) == {'synthetic_0', 'synthetic_1'}
    for mode in modes:
        assert seen[mode]['options']['lr'] == cfg['op_transfer']['lr']
        assert seen[mode]['options']['batch_size'] == 8
        model, _ = load_checkpoint_pin(output / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert model.model_config['dim'] == 8
        assert model.model_config['cross_layers'] == 2
        assert model.predictor.model_config['dim'] == 12
        assert model.predictor.model_config['encoder_layers'] == 3
        assert model.predictor.model_config['cross_layers'] == 1
        assert model.predictor.model_config['max_devices'] == 4
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retrained completed joint stage')):
        resumed, replay = run(modes, name='joint_resume', split='val', resume=output)
    assert len(replay['results']) == len(modes)
    for mode in modes:
        before, _ = load_checkpoint_pin(output / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        after, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(before) == state_digest(after)
