"""Paired score detach experiments, diagnostics, Excel and exact stage recovery."""
import copy
import csv
import json
import zipfile
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.lasd_registry_pin import ALL_LASD_MODELS_PIN, LASD_DETACH_MODELS_PIN
from final_exam.run_pin import load_checkpoint_pin
from final_exam.tests.test_width_decoder_run_pin import setup
from final_exam.width_registry_pin import validate_width_contract_pin


def configure(cfg, path):
    cfg['lasd_pin'] = dict(hidden_dim=12, score_chunk_size=3, diagnostic_samples=2)
    path.write_text(json.dumps(cfg))


def test_paired_training_shared_predictor_exports_and_completed_resume(setup):
    root, cfg, data, config, run = setup
    configure(cfg, config)
    out, result = run('score_detach', ALL_LASD_MODELS_PIN)
    assert len(result['results']) == 4 and len(result['stages']) == 6
    audits = {}
    for mode in ALL_LASD_MODELS_PIN:
        directory = out / f'seed_17_pin/{mode}'
        model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
        validate_width_contract_pin(mode, model.architecture_contract)
        assert saved['recipe']['architecture'] == model.architecture_contract
        assert (directory / 'config_pin.json').is_file()
        assert (directory / 'benchmark_pin.json').is_file()
        stage = next(s for s in result['stages'] if s['stage_id'].endswith('/'+mode))
        assert stage['epochs'] == 2 and stage['seconds'] > 0
        audits[mode] = json.loads((directory / 'width_audit_pin.json').read_text())
        diag = json.loads((directory / 'lasd_diagnostics_pin.json').read_text())
        assert len(diag['sample_ids']) == 2
        assert all(s['max_conservation_error'] < 1e-4 for s in diag['statistics'])
        with (directory / 'lasd_scores_pin.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert any(r['token_id'] == 'GRAPH' for r in rows)
        assert all(float(r['delta_receive']) == float(r['delta_transmit']) == 0
                   for r in rows if r['pe_layer'] == '1')
        assert not model.architecture_contract['requires_measured_op']
    assert len({a['initial_encoder_digest'] for a in audits.values()}) == 1
    assert len({a['predictor_digest'] for a in audits.values()}) == 1
    for mode in LASD_DETACH_MODELS_PIN:
        assert audits[mode]['decoder_parameters'] == audits[mode.replace('_detach_pin', '_pin')]['decoder_parameters']
    with zipfile.ZipFile(out / 'summary_pin.xlsx') as book:
        xml = ''.join(book.read(n).decode() for n in book.namelist() if n.endswith('.xml'))
    assert all(m in xml for m in ALL_LASD_MODELS_PIN)
    assert all(x in xml for x in ('LASD', 'WidthDecoder', 'TrainingCost', 'MAE', 'RMSE', 'R2',
        'Readout gradient policy', 'Previous score in delta', 'stop_gradient',
        'detach_shallow_memory_and_score_inputs'))
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('finished stage retrained')):
        resumed, _ = run('score_detach_complete_resume', ALL_LASD_MODELS_PIN, out)
        _, subset = run('score_detach_subset', [LASD_DETACH_MODELS_PIN[1]], out)
    assert len(subset['results']) == 1 and len(subset['stages']) == 3
    for mode in ALL_LASD_MODELS_PIN:
        a, _ = load_checkpoint_pin(out / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(a) == state_digest(b)
    bad = copy.deepcopy(cfg)
    bad['lasd_pin']['enabled'] = False
    config.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='configuration must match'):
        run('bad_score_detach_resume', LASD_DETACH_MODELS_PIN, out)
    assert not (root / 'bad_score_detach_resume_pin').exists()


@pytest.mark.parametrize('mode', LASD_DETACH_MODELS_PIN)
def test_interrupted_epoch_exact_resume(setup, mode):
    root, cfg, data, config, run = setup
    configure(cfg, config)
    reference, _ = run('score_detach_reference', [mode])
    original = engine.fit_stage
    def interrupt(ctx, model, name, *args, **kwargs):
        if name == mode:
            count = 0
            def before(_):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError('simulated score detach interruption')
            kwargs['before_epoch'] = before
        return original(ctx, model, name, *args, **kwargs)
    with patch.object(engine, 'fit_stage', side_effect=interrupt):
        with pytest.raises(RuntimeError, match='simulated score detach'):
            run('score_detach_failed', [mode])
    resumed, result = run('score_detach_recovered', [mode], root / 'score_detach_failed_pin')
    stage = next(s for s in result['stages'] if s['stage_id'].endswith('/'+mode))
    assert stage['resume']['restored_epoch'] == 1 and stage['resume']['optimizer_state_restored']
    a, _ = load_checkpoint_pin(reference / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    assert state_digest(a) == state_digest(b)
    assert b.decoder.detach_shallow_memory
