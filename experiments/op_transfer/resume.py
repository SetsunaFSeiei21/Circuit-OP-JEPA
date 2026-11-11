"""Atomic, opt-in stage checkpoints for recovery into a fresh experiment.

Only tensors and primitive containers are saved, so loading remains compatible
with ``torch.load(weights_only=True)``. Source directories are read-only.
"""
import copy
import errno
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from device_jepa.training import _state_dict_digest, state_digest


FORMAT = 'op_transfer_stage_resume_v1'


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp',
                                            dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist the rename as well as the file contents. Some shared file
        # systems do not support directory fsync; do not mask real I/O errors.
        if hasattr(os, 'O_DIRECTORY'):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                try:
                    os.fsync(directory)
                except OSError as error:
                    if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                        raise
            finally:
                os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_torch_save(path, value):
    _atomic(path, lambda stream: torch.save(value, stream))


def atomic_json(path, value):
    contents = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    _atomic(path, lambda stream: stream.write(contents))


def same_value(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_value(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    return type(left) == type(right) and left == right


def stage_identity(ctx, model, stage_id, epochs, options, seed, extra):
    return dict(version=1, stage_id=stage_id, seed=seed, epochs=epochs,
                config=copy.deepcopy(ctx.config), preprocessing=copy.deepcopy(ctx.prep),
                options=copy.deepcopy(options), initial_state_digest=state_digest(model),
                trainable_parameters=[name for name, value in model.named_parameters() if value.requires_grad],
                extra=copy.deepcopy(extra))


def capture_rng():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=[numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                       int(numpy_state[3]), float(numpy_state[4])],
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python'])
    name, keys, position, gaussian, cached = value['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda']:
        if not torch.cuda.is_available() or len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('Epoch resume requires the saved CUDA RNG device count')
        torch.cuda.set_rng_state_all([state.cpu() for state in value['cuda']])


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _load(path):
    try:
        result = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError(f'Cannot read recovery checkpoint {path}: {error}') from error
    if not isinstance(result, dict):
        raise ValueError(f'Invalid recovery checkpoint {path}')
    return result


def _history(source, suffix, payload):
    if isinstance(payload.get('history'), list):
        return copy.deepcopy(payload['history'])
    path = source / f'history{suffix}.json'
    if not path.is_file():
        raise ValueError(f'Missing recovery stage history: {path}')
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'Invalid recovery stage history: {path}') from error
    if not isinstance(result, list):
        raise ValueError(f'Invalid recovery stage history: {path}')
    return result


def _validate_state(model, state, *, frozen=True):
    expected = model.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError('Recovery checkpoint state keys do not match the stage model')
    for name, value in state.items():
        current = expected[name]
        if not torch.is_tensor(value) or value.shape != current.shape or value.dtype != current.dtype:
            raise ValueError(f'Recovery checkpoint tensor differs for {name}')
    if frozen:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad and not torch.equal(parameter.detach().cpu(), state[name].cpu()):
                raise ValueError(f'Recovery checkpoint changes frozen dependency {name}')


def _validate_history(history, count):
    if len(history) != count:
        raise ValueError('Recovery history length differs from completed epoch count')
    for epoch, record in enumerate(history, 1):
        if record.get('epoch') != epoch:
            raise ValueError('Recovery history must contain consecutive completed epochs')
        for phase in ('train', 'val'):
            value = record.get(phase, {}).get('loss')
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Recovery history contains a non-finite/missing objective')


def _validate_counters(value):
    for key in ('optimizer_steps', 'examples_seen'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f'Invalid recovery counter: {key}')
    seconds = value.get('seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid recovery elapsed time')


def load_source(ctx, model, stage_id, epochs, options, seed, suffix, source, identity):
    """Validate recovery input before any destination writes or feature caching."""
    if source is None:
        return dict(policy='fresh', source_stage=None)
    source = Path(source).resolve()
    audit = dict(source_stage=str(source))
    if not source.exists():
        return dict(policy='fresh', **audit)
    if not source.is_dir():
        raise ValueError('Recovery source stage must be a directory')
    best_path, last_path = source / f'best{suffix}.pt', source / f'last{suffix}.pt'
    best = _load(best_path) if best_path.is_file() else None
    if best is not None and type(best.get('complete')) is not bool:
        raise ValueError('Recovery best checkpoint lacks a valid completion marker')
    # A finalized checkpoint is authoritative; last may precede final selection.
    if best is not None and best['complete']:
        payload, policy = best, 'completed_stage'
    elif last_path.is_file():
        payload, policy = _load(last_path), 'epoch_resume'
    elif best is not None:
        payload, policy = best, 'legacy_weights_restart'
    else:
        # The stage directory is created before the first optimizer step. An
        # empty directory can therefore represent a mid-first-epoch failure,
        # whose work was never checkpointed and must not be reported as zero.
        return dict(policy='missing_checkpoint_restart', **audit)

    for key, expected in (('config', ctx.config), ('preprocessing', ctx.prep)):
        if not same_value(payload.get(key), expected):
            raise ValueError(f'Recovery checkpoint {key} differs from the current stage')
    saved_identity = payload.get('resume_identity')
    if saved_identity is not None and not same_value(saved_identity, identity):
        raise ValueError('Recovery stage identity differs (data/configuration/dependencies/budget/seed)')
    _validate_state(model, payload.get('state_dict'))

    if policy == 'completed_stage':
        cost = payload.get('cost')
        if (not isinstance(cost, dict) or cost.get('stage_id') != stage_id
                or cost.get('planned_epochs') != epochs
                or cost.get('trainable_parameters') != sum(p.numel() for p in model.parameters() if p.requires_grad)):
            raise ValueError('Completed recovery stage cost/budget does not match')
        _validate_counters(cost)
        if cost.get('selected_state_digest') != _state_dict_digest(payload['state_dict']):
            raise ValueError('Completed recovery checkpoint digest differs from selected state')
        history = _history(source, suffix, payload)
        _validate_history(history, cost['epochs'])
        return dict(policy=policy, payload=payload, history=history, **audit)

    if policy == 'epoch_resume':
        if payload.get('format') != FORMAT or saved_identity is None:
            raise ValueError('Unsupported epoch recovery checkpoint format')
        count = payload.get('epoch')
        if type(count) is not int or not 1 <= count <= epochs:
            raise ValueError('Invalid recovery checkpoint epoch')
        history = payload.get('history')
        if not isinstance(history, list):
            raise ValueError('Missing recovery history')
        _validate_history(history, count)
        for key in ('optimizer_state_dict', 'rng_state', 'early_stopping', 'best_state_dict',
                    'best_stage_metadata', 'optimizer_steps', 'examples_seen', 'seconds'):
            if key not in payload:
                raise ValueError(f'Missing epoch recovery field: {key}')
        _validate_counters(payload)
        _validate_state(model, payload['best_state_dict'])
        if payload['early_stopping'].get('epochs_ran') != count:
            raise ValueError('Recovery early stopping differs from completed epochs')
        if payload['rng_state'].get('cuda'):
            if not torch.cuda.is_available() or len(payload['rng_state']['cuda']) != torch.cuda.device_count():
                raise ValueError('Epoch resume requires the saved CUDA RNG device count')
        return dict(policy=policy, payload=payload, history=copy.deepcopy(history), **audit)

    # Historical best-only snapshots have no optimizer or random state. Do not
    # invent that state or pretend this is an exact continuation.
    selection = payload.get('selection', {})
    count = selection.get('epoch')
    if type(count) is not int or not 1 <= count <= epochs:
        raise ValueError('Legacy recovery best checkpoint has an invalid selected epoch')
    return dict(policy=policy, payload=payload, restored_epoch=count,
                partial_prior_cost_unknown=True, **audit)


def source_stage_metadata(source, suffix='_pin'):
    """Read non-module state (e.g. a latent scaler) before reconstructing a stage."""
    if source is None:
        return None
    source = Path(source)
    best_path, last_path = source / f'best{suffix}.pt', source / f'last{suffix}.pt'
    best = _load(best_path) if best_path.is_file() else None
    if best is not None and best.get('complete') is True:
        return copy.deepcopy(best.get('stage_metadata', {}))
    if last_path.is_file():
        return copy.deepcopy(_load(last_path).get('stage_metadata', {}))
    return copy.deepcopy(best.get('stage_metadata', {})) if best is not None else None
