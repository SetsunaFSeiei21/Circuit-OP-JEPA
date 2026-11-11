"""Stable, label-blind per-circuit sampling before experiment partitions.

The saved selection is part of dataset identity. It is never silently redrawn
when loading a sampled dataset or resuming a training stage.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import copy
from collections import defaultdict


ALGORITHM = 'sha256_rank_v1'


def validate_sampling_options(max_samples_per_circuit=None, sampling_seed=17):
    if max_samples_per_circuit is not None and (
            type(max_samples_per_circuit) is not int or max_samples_per_circuit < 1):
        raise ValueError('dataset.max_samples_per_circuit must be a positive integer or null')
    if type(sampling_seed) is not int or sampling_seed < 0:
        raise ValueError('dataset.sampling_seed must be a nonnegative integer')
    return max_samples_per_circuit, sampling_seed


def sampling_options(config=None):
    if config is not None and not isinstance(config, dict):
        raise ValueError('dataset sampling configuration must be an object')
    config = config or {}
    return validate_sampling_options(config.get('max_samples_per_circuit'), config.get('sampling_seed', 17))


def select_sample_ids(ids, circuit, cap=None, seed=17):
    """Uniform hash-ranked subset, independent of input order and other circuits.

    IDs are canonical strings (raw adapters use ``circuit:source_sample_id``).
    The same seed produces nested subsets as the cap increases. Labels, OP,
    feasibility, and training RNG state never participate in the selection.
    """
    validate_sampling_options(cap, seed)
    ids = list(ids)
    if any(not isinstance(sid, str) or not sid for sid in ids):
        raise ValueError('Sampling requires nonempty canonical string sample IDs')
    if len(set(ids)) != len(ids):
        raise ValueError('Sampling requires unique sample IDs')
    if cap is None or cap >= len(ids):
        return sorted(ids)
    def rank(sid):
        key = json.dumps([seed, circuit, sid], ensure_ascii=False, separators=(',', ':'))
        return hashlib.sha256(key.encode('utf-8')).digest(), sid
    return sorted(heapq.nsmallest(cap, ids, key=rank))


def build_selection_record(circuit, source_ids, selected_ids, cap=None, seed=17):
    validate_sampling_options(cap, seed)
    source, selected = list(source_ids), sorted(selected_ids)
    if len(set(source)) != len(source) or len(set(selected)) != len(selected):
        raise ValueError(f'Duplicate sampling IDs for circuit {circuit}')
    if not set(selected) <= set(source) or len(selected) != min(cap or len(source), len(source)):
        raise ValueError(f'Invalid sample selection for circuit {circuit}')
    return dict(source_count=len(source), selected_count=len(selected), selected_ids=selected)


def make_sampling_metadata(records, cap, seed):
    validate_sampling_options(cap, seed)
    return dict(schema_version=1, algorithm=ALGORITHM, max_samples_per_circuit=cap,
                sampling_seed=seed, circuits=dict(sorted(records.items())))


def validate_sampling_metadata(dataset):
    """Check that persisted IDs/counts describe exactly this dataset."""
    saved = dataset.get('sampling')
    if saved is None:
        return
    if not isinstance(saved, dict) or type(saved.get('schema_version')) is not int or saved['schema_version'] != 1:
        raise ValueError('Invalid dataset sampling schema')
    if saved.get('algorithm') != ALGORITHM:
        raise ValueError('Unsupported dataset sampling algorithm')
    if not {'max_samples_per_circuit', 'sampling_seed', 'circuits'} <= saved.keys():
        raise ValueError('Saved sampling metadata is missing required fields')
    cap, _ = sampling_options(saved)
    if cap is None:
        raise ValueError('Saved sampling metadata requires a positive max_samples_per_circuit')
    circuits = saved.get('circuits')
    actual = defaultdict(list)
    for sample in dataset['samples']:
        actual[sample['circuit_id']].append(sample['sample_id'])
    if not isinstance(circuits, dict) or set(circuits) != set(actual):
        raise ValueError('Saved sampling circuits do not match dataset')
    for circuit, ids in actual.items():
        record = circuits[circuit]
        if (not isinstance(record, dict) or type(record.get('source_count')) is not int
                or type(record.get('selected_count')) is not int or record['source_count'] < 1
                or record['selected_count'] != min(cap, record['source_count'])
                or record['selected_count'] != len(ids) or record.get('selected_ids') != sorted(ids)
                or len(set(ids)) != len(ids)):
            raise ValueError(f'Saved sampling IDs/counts do not match circuit {circuit}')


def check_sampling_config(config, saved):
    """Omitted settings inherit persisted selection; explicit changes fail."""
    cap, seed = sampling_options(config)
    if saved is None:
        if cap is not None:
            raise ValueError('Apply per-circuit sampling before creating/using the manifest')
        return
    config = config or {}
    for key, value in (('max_samples_per_circuit', cap), ('sampling_seed', seed)):
        if key in config and value != saved[key]:
            raise ValueError(f'Config {key} differs from saved sampling; use original data and a new experiment')


def apply_dataset_sampling(dataset, config=None):
    """Return a shallow dataset copy with selected rows; never mutate the input.

    Already-sampled inputs are verified and reused. Expanding/replacing their
    selection needs the original full source and an explicit new experiment.
    No cap preserves legacy data and fingerprints exactly.
    """
    cap, seed = sampling_options(config)
    if not isinstance(dataset, dict):
        raise ValueError('dataset must be an object')
    saved = dataset.get('sampling')
    replay = config if isinstance(config, dict) and 'algorithm' in config else None
    if replay is not None and cap is None:
        raise ValueError('Saved sampling metadata requires a positive max_samples_per_circuit')
    if saved is not None:
        validate_sampling_metadata(dataset)
        check_sampling_config(config, saved)
        if replay is not None and replay != saved:
            raise ValueError('Saved sampling selection differs from dataset')
        return dataset
    if cap is None:
        return dataset
    groups = defaultdict(list)
    all_ids = set()
    for sample in dataset['samples']:
        sid, circuit = sample.get('sample_id'), sample.get('circuit_id')
        if not isinstance(sid, str) or not sid or not isinstance(circuit, str) or not circuit:
            raise ValueError('Sampling requires nonempty sample_id and circuit_id')
        if sid in all_ids:
            raise ValueError(f'duplicate sample_id: {sid}')
        all_ids.add(sid)
        groups[circuit].append(sid)
    records, selected = {}, set()
    if replay is not None:
        # A checkpoint/manifest owns the selection. Reuse those IDs directly,
        # including when its original input was an unsampled canonical JSON.
        records = replay.get('circuits')
        if not isinstance(records, dict) or set(records) != set(groups):
            raise ValueError('Saved sampling circuits differ from source')
        for circuit, ids in groups.items():
            record = records[circuit]
            if (not isinstance(record, dict) or record.get('source_count') != len(ids)
                    or not isinstance(record.get('selected_ids'), list)
                    or any(not isinstance(sid, str) for sid in record['selected_ids'])
                    or not set(record['selected_ids']) <= set(ids)):
                raise ValueError(f'Saved sampling source count/IDs differ for circuit {circuit}')
            selected.update(record['selected_ids'])
    else:
        for circuit, ids in sorted(groups.items()):
            chosen = select_sample_ids(ids, circuit, cap, seed)
            selected.update(chosen)
            records[circuit] = build_selection_record(circuit, ids, chosen, cap, seed)
    result = dict(dataset)
    result['samples'] = [sample for sample in dataset['samples'] if sample['sample_id'] in selected]
    result['sampling'] = copy.deepcopy(replay) if replay is not None else make_sampling_metadata(records, cap, seed)
    validate_sampling_metadata(result)
    return result
