"""Shared factors survive training, deployment and interrupted-stage recovery."""
import copy
import json
import zipfile
from unittest.mock import patch

import pytest
import torch

from device_jepa.training import state_digest
from experiments.op_transfer import engine
from final_exam.run_pin import load_checkpoint_pin, validate_request_pin
from final_exam.shared_decoder_registry_pin import SHARED_DECODER_MODELS_PIN
from final_exam.tests.test_width_decoder_run_pin import setup


FIVE = ('width_decoder_pin',) + SHARED_DECODER_MODELS_PIN


def test_shared_models_train_export_replay_and_completed_resume(setup):
    root, cfg, data, config, run = setup
    out, result = run('shared', FIVE)
    assert len(result['results']) == 5
    assert len(result['stages']) == 7  # five readouts plus one OP predictor and cache
    audits, counts = [], {}
    for mode in FIVE:
        path = out / 'seed_17_pin' / mode
        model, payload = load_checkpoint_pin(path / 'checkpoint_pin.pt')
        assert payload['recipe']['architecture'] == model.architecture_contract
        audits.append(json.loads((path / 'width_audit_pin.json').read_text()))
        counts[mode] = sum(p.numel() for p in model.parameters())
        assert not any(p.requires_grad for p in model.predictor.parameters())
        report = json.loads((path / 'test_pin.json').read_text())
        assert report['low_rank_benchmark_pin']['model_parameters'] == counts[mode]
        assert (path / 'benchmark_pin.json').is_file()
        if mode in SHARED_DECODER_MODELS_PIN:
            assert model.architecture_contract['performance']['shared_down_basis']
            # Serialization must not repeat any aliased/shared down matrix.
            assert sum(t.numel() for t in model.decoder.state_dict().values()) == sum(
                p.numel() for p in model.decoder.parameters())
    assert len({a['predictor_digest'] for a in audits}) == 1
    assert len({a['initial_encoder_digest'] for a in audits}) == 1
    base = counts['width_decoder_pin']
    assert all(counts[m] < base for m in SHARED_DECODER_MODELS_PIN)
    assert counts['width_decoder_shared_both_pin'] == (
        counts['width_decoder_shared_attention_pin'] + counts['width_decoder_shared_ffn_pin'] - base)
    with zipfile.ZipFile(out / 'summary_pin.xlsx') as book:
        xml = ''.join(book.read(n).decode() for n in book.namelist() if n.endswith('.xml'))
        assert 'SharedDecoder' in xml and all(m in xml for m in FIVE)
    with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('retrained complete stage')):
        resumed, _ = run('shared_recovered', FIVE, out)
    for mode in FIVE:
        a, _ = load_checkpoint_pin(out / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
        assert state_digest(a) == state_digest(b)
    bad = copy.deepcopy(cfg)
    bad['shared_decoder_pin']['attention_rank'] = 1
    config.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='configuration must match'):
        run('wrong_shared_rank_resume', FIVE, out)
    assert not (root / 'wrong_shared_rank_resume_pin').exists()


def test_shared_epoch_resume_matches_uninterrupted(setup):
    root, cfg, data, config, run = setup
    mode = 'width_decoder_shared_both_pin'
    reference, _ = run('shared_reference', [mode])
    original = engine.fit_stage

    def interrupt(ctx, model, name, *args, **kwargs):
        if name == mode:
            calls = 0

            def before(_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated shared interruption')
            kwargs['before_epoch'] = before
        return original(ctx, model, name, *args, **kwargs)

    with patch.object(engine, 'fit_stage', side_effect=interrupt):
        with pytest.raises(RuntimeError, match='simulated shared'):
            run('shared_failed', [mode])
    resumed, result = run('shared_resumed', [mode], root / 'shared_failed_pin')
    stage = next(s for s in result['stages'] if s['stage_id'].endswith('/' + mode))
    assert stage['resume']['optimizer_state_restored']
    assert stage['resume']['restored_epoch'] == 1
    a, _ = load_checkpoint_pin(reference / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    b, _ = load_checkpoint_pin(resumed / f'seed_17_pin/{mode}/checkpoint_pin.pt')
    assert state_digest(a) == state_digest(b)


def test_shared_preflight_uses_effective_architecture(setup):
    _, cfg, *_ = setup
    cfg = copy.deepcopy(cfg)
    cfg['shared_decoder_pin']['attention_rank'] = 4
    # Two branches at dim8 reduce parameters even though per-matrix rank4 does not.
    validate_request_pin(cfg, ['width_decoder_shared_kv_pin'], [17], 'cpu')
    cfg['final_exam_pin']['model_overrides']['zerosim_supervised_pin'] = {'dim': 4}
    with pytest.raises(ValueError):
        validate_request_pin(cfg, ['width_decoder_shared_kv_pin'], [17], 'cpu')


def test_baseline_subset_resume_resolves_shared_defaults(tmp_path):
    from final_exam.stage_resume_pin import validate_source_pin
    from final_exam.width_registry_pin import width_settings_pin
    from final_exam.shared_decoder_registry_pin import shared_decoder_settings_pin
    source = tmp_path / 'source_pin'
    source.mkdir()
    raw = {'model': {'dim': 128, 'dropout': .1}}
    saved = copy.deepcopy(raw)
    saved['width_decoder_pin'] = width_settings_pin(None, raw['model'])
    saved['shared_decoder_pin'] = shared_decoder_settings_pin()
    for name, value in {
        'run_pin.json': dict(granularity='pin', models=list(FIVE), seeds=[17],
                             dataset_fingerprint='fixture', smoke=False),
        'config_pin.json': saved, 'preprocessing_pin.json': {'granularity': 'pin'},
        'cohort_pin.json': {}, 'manifest_pin.json': {},
    }.items():
        (source / name).write_text(json.dumps(value))
    resolved = copy.deepcopy(raw)
    validate_source_pin(source, resolved, ['width_decoder_pin'], [17],
        output=tmp_path / 'recovered_pin', logs=tmp_path / 'logs_pin',
        smoke=False, allowed_models=FIVE)
    assert resolved == saved
