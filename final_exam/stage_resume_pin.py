"""Validate a read-only source run before recovering pin training into a new run."""
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def training_directory_pin(path):
    path = Path(path).expanduser().resolve()
    if (path / 'training_pin' / 'run_pin.json').is_file():
        path = path / 'training_pin'
    if not (path / 'run_pin.json').is_file():
        raise ValueError('Resume source must contain training_pin/run_pin.json or be a training directory')
    return path


def validate_source_pin(path, config, models, seeds, *, output, logs, smoke, allowed_models):
    """No writes; data/manifest/cohort identity is checked separately after loading."""
    source = training_directory_pin(path)
    protected = source.parent if source.name == 'training_pin' else source
    for destination in (Path(output).resolve(), Path(logs).resolve()):
        if destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError('Resume destination must be separate from the source run and its artifacts')
    record = read_json(source / 'run_pin.json')
    if record.get('granularity') != 'pin':
        raise ValueError('Resume source is not a pin training run')
    if not set(models) <= set(allowed_models):
        raise ValueError('Resume supports registered staged gates, versioned joint gates and their cascade baseline')
    if not set(models) <= set(record.get('models', [])) or not set(seeds) <= set(record.get('seeds', [])):
        raise ValueError('Resume models and seeds must be selected from the source run')
    # A cascade-only subset still belongs to its original versioned study. Resolve
    # its defaults even when the requested subset no longer names that gate;
    # explicit values must still match exactly (never copy source settings).
    from final_exam.gate_version_registry_pin import gate_version_pin
    versions = {gate_version_pin(mode) for mode in record.get('models', [])}
    version = next((v for v in ('v5', 'v4', 'v3', 'v2') if v in versions), None)
    if version:
        from final_exam.models.op_gate_pin import validate_gate_config_pin
        config['op_gate_pin'] = validate_gate_config_pin(config.get('op_gate_pin'), version=version)
    if read_json(source / 'config_pin.json') != config:
        raise ValueError('Resume configuration must match the saved source configuration exactly')
    if bool(record.get('smoke', False)) != bool(smoke):
        raise ValueError('Resume smoke setting must match the source run')
    if not record.get('dataset_fingerprint'):
        raise ValueError('Resume source lacks a dataset fingerprint')
    # These files are required even for old checkpoints: re-fitting statistics
    # would change the inputs/objective of a supposedly resumed model.
    preprocessing = read_json(source / 'preprocessing_pin.json')
    cohort = read_json(source / 'cohort_pin.json')
    manifest = read_json(source / 'manifest_pin.json')
    if preprocessing.get('granularity') != 'pin':
        raise ValueError('Resume source lacks valid saved pin preprocessing')
    return source, record, preprocessing, cohort, manifest


def validate_data_pin(record, source_manifest, manifest, fingerprint, cohort):
    if record['dataset_fingerprint'] != fingerprint:
        raise ValueError('Resume dataset fingerprint differs from the source run')
    if source_manifest != manifest:
        raise ValueError('Resume manifest differs from the source run')
    expected = {key: sorted(manifest[key]) for key in
                ('train', 'val', 'test', 'ood', 'labeled_train')}
    if cohort != expected:
        raise ValueError('Resume cohort differs from the saved split; subset runs are not interchangeable')


def scaler_digest_pin(state):
    """Include non-module teacher scale state in the stage continuation identity."""
    import torch

    def plain(value):
        if torch.is_tensor(value):
            return {'dtype': str(value.dtype), 'shape': list(value.shape),
                    'values': value.detach().cpu().tolist()}
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    encoded = json.dumps(plain(state), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def recover_cache_cost_pin(source, seed, cost):
    """Retain known prior cache work while reporting this attempt separately."""
    if source is None:
        return cost
    path = source / f'seed_{seed}_pin/shared_pin/device_op_predictor_pin/prediction_cache_pin.json'
    audit = dict(policy='rebuilt_cache', source_stage=str(path), restored_epoch=0,
                 optimizer_state_restored=False, new_epochs=0, new_optimizer_steps=0,
                 new_seconds=cost['seconds'], new_examples_seen=cost['examples_seen'])
    if path.is_file():
        previous = read_json(path)
        if (previous.get('stage_id') != cost['stage_id'] or previous.get('optimizer_steps') != 0
                or type(previous.get('seconds')) not in (int, float)
                or not math.isfinite(previous['seconds']) or previous['seconds'] < 0
                or type(previous.get('examples_seen')) is not int or previous['examples_seen'] < 0):
            raise ValueError('Saved prediction cache cost is invalid')
        cost['seconds'] += previous['seconds']
        cost['examples_seen'] += previous['examples_seen']
        if previous.get('resume'):
            audit['source_recovery'] = previous['resume']
    cost['resume'] = audit
    return cost


def restore_scaler_pin(source, seed, base_name, ctx, cfg, options):
    """Restore a prior objective's frozen scale rather than recomputing moments."""
    import torch
    from experiments.op_transfer.losses import LatentScaler
    from experiments.op_transfer.resume import source_stage_metadata

    if source is None:
        return None
    root = source / f'seed_{seed}_pin'
    path = root / 'shared_pin' / base_name / 'alignment_scales_pin.pt'
    if path.is_file():
        state = torch.load(path, map_location='cpu', weights_only=True)
    else:
        metadata = source_stage_metadata(root / 'stages_pin' / 'device_op_distill_encoder_pin', suffix='_pin')
        state = (metadata or {}).get('teacher_scaler')
    if state is None:
        return None
    scaler = LatentScaler.from_state_dict(state)
    expected_keys = {key for row in ctx.keys['train'] for key in row}
    if (scaler.hidden_dim != cfg['model']['dim'] or scaler.floor != options['latent_floor']
            or scaler.fitted_samples != len(ctx.rows['train']) or set(scaler.groups) != expected_keys):
        raise ValueError('Saved teacher latent scaler does not match the resumed training cohort/configuration')
    print(f'Resume: restored train-fitted teacher latent scales for seed={seed}', flush=True)
    return scaler
