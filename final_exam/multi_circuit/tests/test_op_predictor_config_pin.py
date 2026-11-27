"""OP grids vary independently without importing the training runtime."""
import copy
from pathlib import Path

import pytest

from final_exam.multi_circuit.config_pin import (
    ROOT, OP_MODELS_PIN, allowed_grid_pin, expand_grid_pin, init_model_pin,
    materialize_op_predictor_pin, read_pin,
)


def template():
    return read_pin(ROOT / 'configs_pin' / 'template_pin.json')


def test_new_op_templates_materialize_defaults_without_changing_downstream(tmp_path):
    for model in ('device_op_cascade_pin', 'width_decoder_op_pin', 'lasd_moe_headwise_detach_pin'):
        path, _ = init_model_pin(model, tmp_path)
        cfg = read_pin(path)
        assert cfg['op_predictor']['model'] == cfg['model']
        assert cfg['op_predictor']['training']['lr'] == cfg['op_transfer']['lr']
        assert cfg['op_predictor']['training']['epochs'] == cfg['op_transfer']['epochs']['op_predictor']
        cfg['model']['dim'] *= 2
        cfg['op_transfer']['lr'] *= 2
        assert cfg['op_predictor']['model']['dim'] != cfg['model']['dim']
        assert cfg['op_predictor']['training']['lr'] != cfg['op_transfer']['lr']
        if model == 'width_decoder_op_pin':
            assert cfg['op_predictor']['width_decoder_pin'] == cfg['width_decoder_pin']
    path, _ = init_model_pin('mlp_pin', tmp_path)
    assert 'op_predictor' not in read_pin(path)


def test_downstream_and_op_search_dimensions_are_independent():
    cfg = materialize_op_predictor_pin(template(), 'device_op_cascade_pin')
    original = copy.deepcopy(cfg)
    trials, skipped = expand_grid_pin('device_op_cascade_pin', cfg, {'grid': {
        'model.dim': [64, 128], 'op_transfer.lr': [1e-4, 3e-4],
        'op_predictor.model.cross_layers': [1, 3], 'op_predictor.training.lr': [2e-4],
        'op_predictor.training.epochs': [7],
    }})
    assert len(trials) == 8 and not skipped
    assert cfg == original
    for trial in trials:
        c = trial['config']
        assert c['op_predictor']['model']['dim'] == 128
        assert c['op_predictor']['training']['lr'] == 2e-4
        assert c['op_predictor']['training']['epochs'] == 7
        assert c['model']['cross_layers'] == original['model']['cross_layers']


def test_op_grid_skips_invalid_op_dimensions_without_changing_main_network():
    cfg = template()
    original = copy.deepcopy(cfg)
    trials, skipped = expand_grid_pin('device_op_cascade_pin', cfg, {'grid': {
        'op_predictor.model.dim': [64, 65], 'op_predictor.model.cross_layers': [1, 6],
    }})
    assert len(trials) == 1 and len(skipped) == 3
    assert trials[0]['config']['model'] == original['model']
    assert 'op_predictor' not in cfg
    assert all('OP ' in row['reason'] for row in skipped)


def test_inactive_op_search_dimensions_are_rejected():
    for model in ('mlp_pin', 'gat_pin', 'xgboost_pin', 'device_op_gate_only_base_pin'):
        assert not any(key.startswith('op_predictor.') for key in allowed_grid_pin(model))
        with pytest.raises(ValueError, match='Inactive'):
            expand_grid_pin(model, template(), {'grid': {'op_predictor.training.lr': [1e-4]}})
    assert 'op_predictor.model.decoder_layers' not in allowed_grid_pin('device_op_cascade_pin')
    assert 'op_predictor.model.decoder_layers' in allowed_grid_pin('width_decoder_op_pin')
    assert 'op_predictor.width_decoder_pin.n_experts' in allowed_grid_pin('width_decoder_op_pin')
    assert 'op_predictor.width_decoder_pin.n_experts' not in allowed_grid_pin('width_decoder_pin')


def test_deepgen_grid_preserves_transfer_compatibility():
    cfg = read_pin(ROOT / 'configs_pin/models/deepgen_op_pretrained_pin.json')
    space = read_pin(ROOT / 'configs_pin/search_spaces/deepgen_op_pretrained_pin.json')
    trials, skipped = expand_grid_pin('deepgen_op_pretrained_pin', cfg, space)
    assert len(trials) == 4 and len(skipped) == 4
    assert all(t['config']['model']['dim'] == t['config']['op_predictor']['model']['dim'] for t in trials)
    assert all('transfers its encoder' in t['reason'] for t in skipped)


def test_all_shipped_op_configs_have_explicit_independent_defaults():
    for path in (ROOT / 'configs_pin/models').glob('*.json'):
        cfg = read_pin(path)
        if path.stem in OP_MODELS_PIN:
            assert cfg['op_predictor']['model'] == cfg['model']
            assert cfg['op_predictor']['training']['lr'] == cfg['op_transfer']['lr']
        else:
            assert 'op_predictor' not in cfg
    for path in (ROOT.parent / 'configs').glob('*.json'):
        assert 'op_predictor' in read_pin(path)
