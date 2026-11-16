"""Four-way training, complete/epoch recovery, contracts and clean deployment."""
import copy
import json
import zipfile
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.low_rank_registry_pin import LOW_RANK_MODELS_PIN
from final_exam.run_pin import load_checkpoint_pin, validate_request_pin
from final_exam.tests.test_width_decoder_run_pin import setup


FOUR = ('width_decoder_pin',) + LOW_RANK_MODELS_PIN


def test_four_models_shared_predictor_export_and_completed_resume(setup):
    root, cfg, data, config, run = setup
    out, result = run('lowrank', FOUR)
    assert len(result['results']) == 4
    assert len(result['stages']) == 6  # four readouts; one predictor + cache
    audits, counts = [], {}
    for mode in FOUR:
        path = out / 'seed_17_pin' / mode
        model, payload = load_checkpoint_pin(path / 'checkpoint_pin.pt')
        assert payload['recipe']['architecture'] == model.architecture_contract
        audits.append(json.loads((path / 'width_audit_pin.json').read_text()))
        counts[mode] = sum(p.numel() for p in model.parameters())
        assert all(not p.requires_grad for p in model.predictor.parameters())
        report = json.loads((path / 'test_pin.json').read_text())
        assert report['low_rank_benchmark_pin']['batch_size'] > 0
        assert (path / 'benchmark_pin.json').is_file()
    assert len({a['predictor_digest'] for a in audits}) == 1
    assert len({a['initial_encoder_digest'] for a in audits}) == 1
    base = counts['width_decoder_pin']
    assert all(counts[m] < base for m in LOW_RANK_MODELS_PIN)
    assert counts[LOW_RANK_MODELS_PIN[2]] == counts[LOW_RANK_MODELS_PIN[0]] + counts[LOW_RANK_MODELS_PIN[1]] - base
    with zipfile.ZipFile(out / 'summary_pin.xlsx') as z:
        text = ''.join(z.read(n).decode() for n in z.namelist() if n.endswith('.xml'))
        assert 'LowRankDecoder' in text and all(m in text for m in FOUR)
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retrained complete stage')):
        resumed, _ = run('lowrank_resumed', FOUR, out)
    for mode in FOUR:
        a, _ = load_checkpoint_pin(out / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(a) == state_digest(b)
    bad = copy.deepcopy(cfg)
    bad['low_rank_decoder_pin']['attention_rank'] = 1
    config.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='configuration must match'):
        run('wrong_rank_resume', FOUR, out)
    assert not (root / 'wrong_rank_resume_pin').exists()


def test_lowrank_epoch_resume_matches_uninterrupted(setup):
    root, cfg, data, config, run = setup
    mode = 'width_decoder_lowrank_both_pin'
    reference, _ = run('lowrank_reference', [mode])
    original = engine.fit_stage
    def interrupt(ctx, model, name, *args, **kwargs):
        if name == mode:
            calls = 0
            def before(_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated lowrank interruption')
            kwargs['before_epoch'] = before
        return original(ctx, model, name, *args, **kwargs)
    with patch.object(engine, 'fit_stage', side_effect=interrupt):
        with pytest.raises(RuntimeError, match='simulated lowrank'):
            run('lowrank_failed', [mode])
    resumed, result = run('lowrank_recovered', [mode], root / 'lowrank_failed_pin')
    stage = next(s for s in result['stages'] if s['stage_id'].endswith('/' + mode))
    assert stage['resume']['optimizer_state_restored']
    assert stage['resume']['restored_epoch'] == 1
    a, _ = load_checkpoint_pin(reference / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    assert state_digest(a) == state_digest(b)


def test_rank_preflight_respects_effective_override(setup):
    _, cfg, *_ = setup
    cfg = copy.deepcopy(cfg)
    cfg['final_exam_pin']['model_overrides']['zerosim_supervised_pin'] = {'dim': 4}
    with pytest.raises(ValueError):
        validate_request_pin(cfg, ['width_decoder_lowrank_attention_pin'], [17], 'cpu')


def test_baseline_subset_resume_resolves_lowrank_defaults_from_source(tmp_path):
    from final_exam.stage_resume_pin import validate_source_pin
    from final_exam.width_registry_pin import width_settings_pin
    from final_exam.low_rank_registry_pin import low_rank_settings_pin
    source = tmp_path / 'default_study_pin'
    source.mkdir()
    raw = {'model': {'dim': 128, 'dropout': .1}}
    saved = copy.deepcopy(raw)
    saved['width_decoder_pin'] = width_settings_pin(None, raw['model'])
    saved['low_rank_decoder_pin'] = low_rank_settings_pin()
    files = {
        'run_pin.json': {'granularity': 'pin', 'models': list(FOUR),
                         'seeds': [17], 'dataset_fingerprint': 'fixture', 'smoke': False},
        'config_pin.json': saved,
        'preprocessing_pin.json': {'granularity': 'pin'},
        'cohort_pin.json': {},
        'manifest_pin.json': {},
    }
    for name, contents in files.items():
        (source / name).write_text(json.dumps(contents))
    config = copy.deepcopy(raw)
    result = validate_source_pin(source, config, ['width_decoder_pin'], [17],
        output=tmp_path / 'recovered_pin', logs=tmp_path / 'logs_pin',
        smoke=False, allowed_models=FOUR)
    assert result[0] == source
    assert config == saved
    changed = copy.deepcopy(raw)
    changed['low_rank_decoder_pin'] = {'attention_rank': 16, 'ffn_rank': 64}
    with pytest.raises(ValueError, match='configuration must match'):
        validate_source_pin(source, changed, ['width_decoder_pin'], [17],
            output=tmp_path / 'changed_pin', logs=tmp_path / 'changed_logs_pin',
            smoke=False, allowed_models=FOUR)
