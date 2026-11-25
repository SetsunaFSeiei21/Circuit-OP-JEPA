"""v7/v8/v9 training, all-order exports, deployment reconstruction and safe resume."""
import csv
import json
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.lasd_moe_registry_pin import GRAPH_LAYER_POWER_MODELS_PIN
from final_exam.run_pin import MODEL_NAMES_PIN, load_checkpoint_pin
from final_exam.tests.test_summary_pin import workbook, text_values
from final_exam.tests.test_width_decoder_run_pin import setup


@pytest.mark.parametrize('mode', GRAPH_LAYER_POWER_MODELS_PIN)
def test_train_replay_all_order_diagnostics_excel_and_completed_resume(setup, mode):
    root, cfg, _, config, run = setup
    cfg['lasd_moe_pin'] = dict(aggregation_distance=3, hidden_dim=12,
        dropout=.1, score_chunk_size=3, diagnostic_samples=2, hyper_connection_streams=3)
    config.write_text(json.dumps(cfg))
    assert mode in MODEL_NAMES_PIN
    out, result = run('powers_complete', [mode])
    assert len(result['results']) == 1 and len(result['stages']) == 3
    directory = out / f'seed_17_pin/{mode}'
    model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
    spec = model.architecture_contract
    assert saved['recipe']['architecture'] == saved['input_contract_pin'] == spec
    assert spec['attention_powers']['orders'] == [1, 2, 3]
    if mode.endswith(('_v8', '_v9')):
        assert spec['hyper_connection']['streams'] == 3
        assert spec['hyper_connection']['scope'] == 'performance_pe_only'
    report = json.loads((directory / 'test_pin.json').read_text())
    assert report['lasd_moe_pin'] == spec and not report['requires_measured_op']
    assert all({'mae', 'rmse', 'r2'} <= set(m) for m in report['metrics'].values())
    assert json.loads((directory / 'config_pin.json').read_text())['lasd_moe_pin']['aggregation_distance'] == 3
    files = report['lasd_moe_diagnostics_pin']
    def rows(name):
        with (directory / files[name]).open() as stream:
            return list(csv.DictReader(stream))
    ordinary, powers, globals_ = rows('scores'), rows('power_scores'), rows('power_global_scores')
    assert len(powers) == 3 * len(ordinary)
    assert {int(r['attention_order']) for r in powers} == {1, 2, 3}
    assert len(globals_) == 3 * len(rows('global_scores'))
    key = lambda r: (r['sample_id'], r['token_index'], r['pe_layer'], r['head'])
    first = {key(r): r for r in powers if r['attention_order'] == '1'}
    for r in ordinary:
        for field in ('receive', 'transmit', 'delta_receive', 'delta_transmit'):
            assert r[field] == first[key(r)][field]
    for r in powers:
        if r['pe_layer'] == '1':
            assert float(r['delta_receive']) == float(r['delta_transmit']) == 0
            assert r['delta_is_boundary_fill'] == 'True'
    # Recompute exported global statistics from the actual per-node order scores.
    for row in globals_:
        rs = [r for r in powers if all(r[k] == row[k] for k in
              ('sample_id', 'attention_order', 'pe_layer', 'head'))]
        values = torch.tensor([[float(r['receive']), float(r['transmit'])] for r in rs])
        std = (values.var(0, unbiased=False) + float(row['std_epsilon'])).sqrt()
        assert float(row['mean_receive']) == pytest.approx(values[:, 0].mean().item(), rel=1e-5)
        assert float(row['std_receive']) == pytest.approx(std[0].item(), rel=1e-5)
        assert float(row['std_transmit']) == pytest.approx(std[1].item(), rel=1e-5)
    metadata = json.loads((directory / files['metadata']).read_text())
    assert len(metadata['power_statistics']) == 3 * len(metadata['statistics'])
    assert max(r['max_conservation_error'] for r in metadata['power_statistics']) < 1e-4
    history = json.loads((out / f'seed_17_pin/stages_pin/{mode}/history_pin.json').read_text())
    for record in history:
        groups = {g['layer']: g for g in record['encoder_gradients']['layers']}
        assert 'FINAL1' not in groups
        assert ('DECODER_INPUT_NORM1' in groups) == mode.endswith('_v9')
        if mode.endswith('_v9'):
            assert groups['DECODER_INPUT_NORM1']['mean_l2'] > 0
        for name in ('SCORE_ORDER1', 'SCORE_ORDER2', 'SCORE_ORDER3', 'SCORE_GRAPH1', 'SCORE_LAYER1'):
            assert groups[name]['mean_l2'] > 0
    sheets = workbook(out / 'summary_pin.xlsx')
    assert {'MAE', 'RMSE', 'R2', 'Configuration', 'TrainingCost', 'LASD_MoE'} <= set(sheets)
    assert {'Aggregation distance', 'Order projection MLPs', 'Order merge', mode} <= set(text_values(sheets['LASD_MoE']))
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retraining completed stage')):
        resumed, _ = run('powers_completed_resume', [mode], out)
    restored, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    assert state_digest(restored) == state_digest(model)


@pytest.mark.parametrize('mode', GRAPH_LAYER_POWER_MODELS_PIN)
def test_interrupted_readout_restores_optimizer_rng_and_order_projections(setup, mode):
    root, cfg, _, config, run = setup
    cfg['lasd_moe_pin'] = dict(aggregation_distance=2, hidden_dim=12,
        dropout=.2, score_chunk_size=3, diagnostic_samples=0, hyper_connection_streams=3)
    config.write_text(json.dumps(cfg))
    reference, _ = run('powers_reference', [mode])
    original = engine.fit_stage
    def interrupted(ctx, model, name, *args, **kwargs):
        if name == mode:
            calls = 0
            def before(_):
                nonlocal calls
                calls += 1
                if calls == 2: raise RuntimeError('simulated powers interruption')
            kwargs['before_epoch'] = before
        return original(ctx, model, name, *args, **kwargs)
    with patch.object(engine, 'fit_stage', side_effect=interrupted):
        with pytest.raises(RuntimeError, match='simulated powers interruption'):
            run('powers_interrupted', [mode])
    resumed, result = run('powers_resumed', [mode], root / 'powers_interrupted_pin')
    stage = next(s for s in result['stages'] if s['stage_id'].endswith('/'+mode))
    assert stage['resume']['restored_epoch'] == 1 and stage['resume']['optimizer_state_restored']
    a, _ = load_checkpoint_pin(reference / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    assert state_digest(a) == state_digest(b)
    cfg['lasd_moe_pin']['aggregation_distance'] = 3
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        run('powers_changed_distance_resume', [mode], root / 'powers_interrupted_pin')

    if mode.endswith(('_v8', '_v9')):
        cfg['lasd_moe_pin'].update(aggregation_distance=2, hyper_connection_streams=4)
        config.write_text(json.dumps(cfg))
        with pytest.raises(ValueError):
            run('powers_changed_streams_resume', [mode], root / 'powers_interrupted_pin')
