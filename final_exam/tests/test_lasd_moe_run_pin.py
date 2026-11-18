"""LASD-MoE pipeline: paired budgets, frozen OP, standalone replay and exports."""
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
from final_exam.run_pin import MODEL_NAMES_PIN, load_checkpoint_pin
from final_exam.lasd_moe_registry_pin import (
    HEADWISE_LASD_MOE_MODELS_PIN, HYPER_LASD_MOE_MODEL_PIN, LASD_MOE_MODELS_PIN, lasd_moe_spec_pin,
)
from final_exam.summarize_pin import export_results_pin, summary_rows_pin
from final_exam.tests.test_summary_pin import workbook, text_values, record
from final_exam.tests.test_width_decoder_run_pin import setup

MODES = ('lasd_moe_pin', 'lasd_moe_detach_pin')


def configure(cfg, path):
    cfg['lasd_moe_pin'] = dict(hidden_dim=12, score_chunk_size=3, diagnostic_samples=2)
    path.write_text(json.dumps(cfg))


@pytest.mark.parametrize('pair', [MODES, HEADWISE_LASD_MOE_MODELS_PIN,
                                  ('lasd_moe_headwise_detach_pin', HYPER_LASD_MOE_MODEL_PIN)])
def test_paired_models_end_to_end_replay_diagnostics_and_completed_resume(setup, pair):
    root, cfg, data_path, config, run = setup
    configure(cfg, config)
    has_hyper = HYPER_LASD_MOE_MODEL_PIN in pair
    if has_hyper:
        cfg['lasd_moe_pin']['hyper_connection_streams'] = 3
        config.write_text(json.dumps(cfg))
    modes = ('device_op_cascade_pin', *pair)
    assert set(pair) <= set(MODEL_NAMES_PIN)
    out, result = run('lasd_moe_complete', modes)
    assert len(result['results']) == 3 and len(result['stages']) == 5
    assert json.loads((out / 'run_pin.json').read_text())['status'] == 'complete'
    saved_config = json.loads((out / 'config_pin.json').read_text())
    manifest = json.loads((out / 'manifest_pin.json').read_text())
    dataset = json.loads(data_path.read_text())
    (root / 'replay').mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = PinDataContext(dataset, manifest, saved_config, torch.device('cpu'), {}, root / 'replay')
        batch = next(ctx.batches('test', 8, op=True))
    poisoned = copy.deepcopy(batch)
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        poisoned[key].fill_(float('nan'))
    poisoned['op_mask'] = ~poisoned['op_mask']
    poisoned['owner_op_mask'] = ~poisoned['owner_op_mask']
    audits, costs = [], []
    for mode in pair:
        directory = out / f'seed_17_pin/{mode}'
        audits.append(json.loads((directory / 'lasd_moe_audit_pin.json').read_text()))
        model, checkpoint = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
        assert checkpoint['recipe']['architecture'] == model.architecture_contract
        assert checkpoint['input_contract_pin'] == model.architecture_contract
        assert not model.requires_measured_op
        assert all(not parameter.requires_grad for parameter in model.predictor.parameters())
        with torch.no_grad():
            prediction = model(batch)
            assert torch.equal(prediction, model(poisoned))
        report = json.loads((directory / 'test_pin.json').read_text())
        assert report['lasd_moe_pin'] == model.architecture_contract
        assert report['lasd_moe_pin']['base_layers'] == cfg['model']['cross_layers']
        assert report['lasd_moe_pin']['input_dim'] == 4 * cfg['model']['cross_layers'] * cfg['model']['heads']
        diagnostic = report['lasd_moe_diagnostics_pin']
        assert diagnostic['split'] == 'val' and diagnostic['samples'] > 0
        for key in ('scores', 'weights', 'experts', 'metadata'):
            assert (directory / diagnostic[key]).is_file()
        stage = next(s for s in result['stages'] if s['stage_id'].endswith('/' + mode))
        costs.append(stage)
        assert stage['epochs'] == 2 and stage['seconds'] > 0
        history = json.loads((out / f'seed_17_pin/stages_pin/{mode}/history_pin.json').read_text())
        assert history and all('encoder_gradients' in row for row in history)
    if has_hyper:
        assert audits[1]['architecture']['hyper_connection']['streams'] == 3
        assert costs[1]['trainable_parameters'] > costs[0]['trainable_parameters']
    else:
        assert len({a['initial_encoder_digest'] for a in audits}) == 1
        assert len({a['initial_model_digest'] for a in audits}) == 1
        assert costs[0]['parameters'] == costs[1]['parameters']
        assert costs[0]['trainable_parameters'] == costs[1]['trainable_parameters']
    assert len({a['predictor_digest'] for a in audits}) == 1
    sheets = workbook(out / 'summary_pin.xlsx')
    assert {'LASD_MoE', 'MAE', 'RMSE', 'R2', 'TrainingCost', 'Configuration'} <= set(sheets)
    texts = text_values(sheets['LASD_MoE'])
    assert all(mode in texts for mode in modes)
    assert any('lasd_moe_scores_pin.csv' in text for text in texts if text)
    if has_hyper or pair == HEADWISE_LASD_MOE_MODELS_PIN:
        assert 'B,N,H,L' in texts
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retrained completed stage')):
        resumed, _ = run('lasd_moe_completed_resume', modes, out)
        _, subset = run('lasd_moe_subset_resume', [pair[1]], out)
    assert len(subset['results']) == 1 and len(subset['stages']) == 3
    for mode in modes:
        a, _ = load_checkpoint_pin(out / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(a) == state_digest(b)


def test_saved_lasd_moe_configuration_tampering_rejects_export(setup):
    root, cfg, data_path, config, run = setup
    configure(cfg, config)
    cfg['lasd_moe_pin']['diagnostic_samples'] = 0
    config.write_text(json.dumps(cfg))
    out, _ = run('lasd_moe_bad_config', [MODES[0]])
    path = out / f'seed_17_pin/{MODES[0]}/config_pin.json'
    saved = json.loads(path.read_text())
    saved['lasd_moe_pin']['hidden_dim'] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='saved configuration'):
        export_results_pin(out, out / 'reexport_pin.xlsx')
    assert not (out / 'reexport_pin.xlsx').exists()


@pytest.mark.parametrize('mode', LASD_MOE_MODELS_PIN)
def test_interrupted_epoch_recovers_exact_weights_and_gradient_history(setup, mode):
    root, cfg, data_path, config, run = setup
    configure(cfg, config)
    cfg['lasd_moe_pin']['diagnostic_samples'] = 0
    config.write_text(json.dumps(cfg))
    reference, _ = run('lasd_moe_reference', [mode])
    original = engine.fit_stage
    def interrupt(ctx, model, name, *args, **kwargs):
        if name == mode:
            count = 0
            def before(_):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError('simulated LASD-MoE interruption')
            kwargs['before_epoch'] = before
        return original(ctx, model, name, *args, **kwargs)
    with patch.object(engine, 'fit_stage', side_effect=interrupt):
        with pytest.raises(RuntimeError, match='simulated LASD-MoE'):
            run('lasd_moe_interrupted', [mode])
    resumed, result = run('lasd_moe_recovered', [mode], root / 'lasd_moe_interrupted_pin')
    stage = next(s for s in result['stages'] if s['stage_id'].endswith('/'+mode))
    assert stage['resume']['restored_epoch'] == 1
    assert stage['resume']['optimizer_state_restored']
    a, _ = load_checkpoint_pin(reference / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    assert state_digest(a) == state_digest(b)
    old_history = json.loads((reference / f'seed_17_pin/stages_pin/{mode}/history_pin.json').read_text())
    new_history = json.loads((resumed / f'seed_17_pin/stages_pin/{mode}/history_pin.json').read_text())
    assert [r['encoder_gradients'] for r in old_history] == [r['encoder_gradients'] for r in new_history]


@pytest.mark.parametrize('field,value', [
    ('attention_scope', 'all_base_encoder_self_attention'), ('head_reduction', 'mean'),
    ('diagonal', 'included'), ('hidden_state_detach', True), ('score_detach', True),
    ('base_layers', 6), ('input_dim', 192), ('first_delta', 'observed'),
])
def test_malformed_architecture_rejected_before_summary(field, value):
    cfg = dict(dim=128, heads=8, encoder_layers=6, cross_layers=3, decoder_layers=2)
    item = record(MODES[0])
    item['report']['lasd_moe_pin'] = lasd_moe_spec_pin(MODES[0], None, cfg)
    item['report']['lasd_moe_pin'][field] = value
    with pytest.raises(ValueError):
        summary_rows_pin([item])


def test_original_baseline_cannot_claim_lasd_moe_metadata():
    cfg = dict(dim=128, heads=8, encoder_layers=6, cross_layers=3, decoder_layers=2)
    item = record('mlp_pin')
    item['report']['lasd_moe_pin'] = lasd_moe_spec_pin(MODES[0], None, cfg)
    with pytest.raises(ValueError, match='non-LASD-MoE'):
        summary_rows_pin([item])
