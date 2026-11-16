"""Same cohorts/OP predictor, all three ablations, full checkpoints and resume."""
import copy
import csv
import json
import zipfile
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.run_pin import load_checkpoint_pin
from final_exam.tests.test_width_decoder_run_pin import setup

MODES = ('width_decoder_pin', 'width_decoder_score_pin', 'width_decoder_lasd_pin')


def configure(cfg, path):
    cfg['lasd_pin'] = dict(hidden_dim=12, score_chunk_size=3, diagnostic_samples=2)
    path.write_text(json.dumps(cfg))


def test_three_ablations_export_diagnostics_and_complete_resume(setup):
    root, cfg, data, config, run = setup
    configure(cfg, config)
    out, result = run('lasd_complete', MODES)
    assert len(result['results']) == 3 and len(result['stages']) == 5
    audits = []
    for mode in MODES:
        directory = out / f'seed_17_pin/{mode}'
        audits.append(json.loads((directory / 'width_audit_pin.json').read_text()))
        model, saved = load_checkpoint_pin(directory / 'checkpoint_pin.pt')
        assert saved['recipe']['architecture'] == model.architecture_contract
        assert (directory / 'benchmark_pin.json').is_file()
        if mode != MODES[0]:
            diag = json.loads((directory / 'lasd_diagnostics_pin.json').read_text())
            assert len(diag['sample_ids']) == 2
            assert all(s['max_conservation_error'] < 1e-4 for s in diag['statistics'])
            rows = list(csv.DictReader((directory / 'lasd_scores_pin.csv').open()))
            assert any(r['token_id'] == 'GRAPH' for r in rows)
            assert all(float(r['delta_receive']) == float(r['delta_transmit']) == 0
                       for r in rows if r['pe_layer'] == '1')
            assert all(r['token_id'] for r in rows)
    assert len({a['initial_encoder_digest'] for a in audits}) == 1
    assert len({a['predictor_digest'] for a in audits}) == 1
    with zipfile.ZipFile(out / 'summary_pin.xlsx') as book:
        xml = ''.join(book.read(n).decode() for n in book.namelist() if n.endswith('.xml'))
        assert 'LASD' in xml and all(m in xml for m in MODES)
        assert 'MAE' in xml and 'RMSE' in xml and 'R2' in xml
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retrained finished stage')):
        resumed, _ = run('lasd_complete_resume', MODES, out)
    for mode in MODES:
        a, _ = load_checkpoint_pin(out / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(a) == state_digest(b)
    changed = copy.deepcopy(cfg)
    changed['lasd_pin']['hidden_dim'] += 1
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='configuration must match'):
        run('lasd_bad_resume', MODES, out)
    assert not (root / 'lasd_bad_resume_pin').exists()


def test_lasd_epoch_resume_matches_continuous_training(setup):
    root, cfg, data, config, run = setup
    configure(cfg, config)
    mode = 'width_decoder_lasd_pin'
    reference, _ = run('lasd_reference', [mode])
    original = engine.fit_stage
    def interrupt(ctx, model, name, *args, **kwargs):
        if name == mode:
            count = 0
            def before(_):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError('simulated LASD interruption')
            kwargs['before_epoch'] = before
        return original(ctx, model, name, *args, **kwargs)
    with patch.object(engine, 'fit_stage', side_effect=interrupt):
        with pytest.raises(RuntimeError, match='simulated LASD'):
            run('lasd_failed', [mode])
    resumed, result = run('lasd_resumed', [mode], root / 'lasd_failed_pin')
    stage = next(s for s in result['stages'] if s['stage_id'].endswith('/'+mode))
    assert stage['resume']['restored_epoch'] == 1 and stage['resume']['optimizer_state_restored']
    a, _ = load_checkpoint_pin(reference / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    assert state_digest(a) == state_digest(b)
