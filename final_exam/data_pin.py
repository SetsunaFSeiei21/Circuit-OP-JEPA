"""Pin-token resident data with one train-fitted OP target per physical device.

Each observed physical terminal and each named external port is a token. Numeric
statistics are fitted on the canonical device rows exactly once, before pin
expansion. Device values remain stored once; GPU gathers produce the token view.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from device_jepa.baselines import _pin_tokens
from device_jepa.data import canonical_model_name, fit_preprocessor, samples_for_split
from device_jepa.resident import ResidentDataset, _graph, require_device
from experiments.op_transfer.engine import write_json


PIN_CONTRACT_VERSION = 1


def _topology_prefix_pin(sample):
    devices = sorted(sample['devices'], key=lambda d: d['device_id'])
    topology = [sample.get('ports', {}), [(d['device_id'], d['device_type'],
        canonical_model_name(d), d['terminals']) for d in devices]]
    digest = hashlib.sha256(json.dumps(topology, sort_keys=True).encode()).hexdigest()
    return json.dumps([sample['circuit_id'], digest])


def group_keys_pin(sample, terminal_names, input_contract_version=3):
    """Identity of every terminal/port token, independent of sizing and targets."""
    prefix = _topology_prefix_pin(sample)
    tokens = _pin_tokens(sample, terminal_names, input_contract_version)
    # Hex keeps the complete identity in one ':'-free suffix, including unusual
    # device IDs that themselves contain colons or non-ASCII characters.
    return [prefix + ':' + token['token_id'].encode('utf-8').hex() for token in tokens]


def make_preprocessing_pin(device_preprocessing, train_rows):
    """Reuse device statistics; fit only token vocabulary on ID train rows."""
    rows = list(train_rows)
    if not rows or {row['sample_id'] for row in rows} != set(device_preprocessing['fitted_sample_ids']):
        raise ValueError('Pin vocabulary must use exactly the fitted ID training rows')
    spec = device_preprocessing['feature_spec']
    version = device_preprocessing.get('input_contract_version', 1)
    keys = {token['key'] for row in rows for token in _pin_tokens(row, spec['terminal_names'], version)}
    vocab = {'<PAD>': 0, '<UNK>': 1, **{key: i+2 for i, key in enumerate(sorted(keys))}}
    prep = copy.deepcopy(device_preprocessing)
    prep.update(granularity='pin', pin_contract_version=PIN_CONTRACT_VERSION,
                device_preprocessing=copy.deepcopy(device_preprocessing),
                device_vocab=vocab, pin_vocab=copy.deepcopy(vocab),
                pin_vocab_fitted_sample_ids=sorted(row['sample_id'] for row in rows))
    return prep


class PinResidentDataset(ResidentDataset):
    """Canonical numeric tables plus static pin and original-owner topology.

    The standard batch keys have a pin axis P. ``owner_*`` keys have the original
    device axis D. ``owner_index`` (also ``pin_to_device``) maps P to D, with -1
    for external ports and padding. OP prediction pools P -> D before its loss;
    measured OP is broadcast D -> P only for the explicit OP-input models.
    """
    def __init__(self, samples, preprocessing, device, **kwargs):
        if preprocessing.get('granularity') != 'pin' or 'device_preprocessing' not in preprocessing:
            raise ValueError('PinResidentDataset requires saved pin preprocessing')
        if 'pin_vocab' in kwargs:
            raise ValueError('Pin vocabulary comes from saved pin preprocessing')
        self.pin_preprocessing = preprocessing
        device_prep = preprocessing['device_preprocessing']
        super().__init__(samples, device_prep, device, pin_vocab=preprocessing['device_vocab'], **kwargs)
        self.prep = preprocessing
        self.device_preprocessing = device_prep
        representatives = {}
        for index, graph_id in enumerate(self.host_graph_ids):
            representatives.setdefault(graph_id, samples[index])
        terminal_names = preprocessing['feature_spec']['terminal_names']
        version = preprocessing.get('input_contract_version', 1)
        owners, terminal_indices, net_indices = [], [], []
        for graph_id in range(len(self.graphs)):
            row = representatives[graph_id]
            devices = sorted(row['devices'], key=lambda d: d['device_id'])
            owners.append(_graph(row, devices, device_prep, self.device, None))
            tokens = _pin_tokens(row, terminal_names, version)
            nets = sorted({token['net'] for token in tokens})
            net_ids = {net: i for i, net in enumerate(nets)}
            terminal_indices.append(torch.tensor([
                terminal_names.index(token['terminal']) if token['terminal'] is not None else -1
                for token in tokens], dtype=torch.long, device=self.device))
            net_indices.append(torch.tensor([net_ids[token['net']] for token in tokens],
                                            dtype=torch.long, device=self.device))
        self.max_owners = max(len(graph['device_ids']) for graph in owners)
        self.owner_topologies = [_topology_prefix_pin(representatives[i])
                                 for i in range(len(self.graphs))]
        self._owner_topology_by_sample_id = {
            row['sample_id']: self.owner_topologies[self.host_graph_ids[i]]
            for i, row in enumerate(samples)}
        owner_names = {'device_mask': 'owner_mask', 'device_types': 'owner_device_types',
                       'device_categories': 'owner_device_categories',
                       'device_models': 'owner_device_models', 'edge_roles': 'owner_edge_roles',
                       'adjacency': 'owner_adjacency', 'port_roles': 'owner_port_roles',
                       'named_port_roles': 'owner_named_port_roles'}
        self._owner_static_names = []
        for source, target in owner_names.items():
            if source not in owners[0]:
                continue
            padded_values = []
            for graph in owners:
                value = graph[source]
                shape = list(value.shape)
                shape[0] = self.max_owners
                if source in ('adjacency', 'edge_roles'):
                    shape[1] = self.max_owners
                padded = value.new_zeros(shape)
                padded[tuple(slice(0, size) for size in value.shape)] = value
                padded_values.append(padded)
            self.static[target] = torch.stack(padded_values)
            self._owner_static_names.append(target)
        for name, values in (('pin_terminal_index', terminal_indices), ('pin_net_index', net_indices)):
            table = torch.full((len(values), self.max_nodes), -1, dtype=torch.long, device=self.device)
            for i, value in enumerate(values):
                table[i, :len(value)] = value
            self.static[name] = table
        self.static['owner_index'] = torch.where(self.static['device_mask'],
            self.static['node_parent'], torch.full_like(self.static['node_parent'], -1))
        self.static['pin_to_device'] = self.static['owner_index']

    def batches(self, batch_size, *, indices=None, seed=None, metadata=False):
        # Ask the resident base for its already-known host IDs rather than
        # copying CUDA indices back to the host solely for topology metadata.
        for idx, batch in super().batches(batch_size, indices=indices, seed=seed, metadata=True):
            topology = self.graph_ids.index_select(0, idx)
            # The base resident class slices node-like fields by the pin width.
            # Owners may have a different padded width in a heterogeneous store;
            # explicitly gather their complete D axis here.
            for name in self._owner_static_names:
                batch[name] = self.static[name].index_select(0, topology)
            for source in ('parameters', 'parameter_mask', 'parameter_applicability',
                           'op', 'op_mask', 'op_target_mask'):
                if source in self.tables:
                    batch['owner_' + source] = self.tables[source].index_select(0, idx)
            batch['_owner_topology_pin'] = [self._owner_topology_by_sample_id[sid]
                                            for sid in batch['sample_ids']]
            if not metadata:
                batch.pop('sample_ids')
                batch.pop('device_ids')
            yield idx, batch


class PinDataContext:
    """Same training interface as op_transfer.DataContext, with pin artifacts."""
    def __init__(self, dataset, manifest, config, device, limits, output):
        self.device, self.config, self.output = require_device(device), config, Path(output)
        self.full_manifest = manifest
        self.rows, selected = {}, copy.deepcopy(manifest)
        for split in ('train', 'val', 'test', 'ood'):
            rows = samples_for_split(dataset, manifest, split)
            limit = limits.get('test' if split == 'ood' else split)
            ranked = sorted(rows, key=lambda r: hashlib.sha256(
                f"{manifest['split_seed']}:{r['sample_id']}".encode()).hexdigest())
            self.rows[split] = ranked if limit is None else ranked[:limit]
            selected[split] = sorted(row['sample_id'] for row in self.rows[split])
        selected['labeled_train'] = sorted(set(selected['train']) & set(manifest['labeled_train']))
        if not self.rows['train'] or not self.rows['val'] or not selected['labeled_train']:
            raise ValueError('Nonempty train, validation and labeled train cohorts required')
        self.selection = selected
        print('Fitting device statistics on ID train only; expanding to pins after normalization', flush=True)
        device_prep = fit_preprocessor(dataset, selected, include_performance=True,
            config=config.get('normalization'), statistics_device=self.device)
        device_prep['op_state_features'] = []
        self.device_preprocessing = device_prep
        self.prep = make_preprocessing_pin(device_prep, self.rows['train'])
        self.pin_vocab = self.prep['device_vocab']
        self.metric_mask = [count > 0 for count in self.prep['performance']['count']]
        if not any(self.metric_mask):
            raise ValueError('No training performance metric is observed')
        self.labeled = set(selected['labeled_train'])
        self.caches = {}
        # Only frozen predictor outputs may be cached. Each seed/model digest
        # replaces the preceding cache; pin models must verify this identity.
        self.device_prediction_digest_pin = None
        self.device_predictions_pin = {}
        terminal_names = self.prep['feature_spec']['terminal_names']
        version = self.prep.get('input_contract_version', 1)
        self.keys = {split: [group_keys_pin(row, terminal_names, version) for row in rows]
                     for split, rows in self.rows.items()}
        self.groups = {}
        for split, rows in self.rows.items():
            groups = defaultdict(list)
            for index, row in enumerate(rows):
                groups[_topology_prefix_pin(row)].append(index)
            self.groups[split] = dict(groups)
        write_json(self.output/'cohort_pin.json', {key: selected[key] for key in
            ('train', 'val', 'test', 'ood', 'labeled_train')})
        write_json(self.output/'preprocessing_pin.json', self.prep)

    def cache(self, split, op=False):
        key = (split, op)
        if key not in self.caches:
            rows = self.rows[split]
            if not rows:
                return None
            if op:
                supported = {name for name, count in zip(self.prep['feature_spec']['op_names'],
                                                         self.prep['op']['count']) if count > 0}
                if len(supported) != len(self.prep['feature_spec']['op_names']):
                    # Exclude untrained fields BEFORE arithmetic or float32.
                    rows = [{**row, 'devices': [{**dev, 'op': {name: value for name, value in
                        dev.get('op', {}).items() if name in supported}} for dev in row['devices']]}
                        for row in rows]
            self.caches[key] = PinResidentDataset(rows, self.prep, self.device, include_op=op,
                include_performance=True, performance_metric_mask=self.metric_mask,
                performance_dtype=torch.float64, max_tokens=self.config['model']['max_devices'],
                label=f'final_exam {split}_pin ({"OP supervision" if op else "design only"})')
        return self.caches[key]

    def batches(self, split, batch_size, seed=None, op=False, repeated=False):
        cache = self.cache(split, op)
        if cache is None:
            return
        chunks, rng = [], random.Random(seed)
        for source_indices in self.groups[split].values():
            indices = source_indices[:]
            if seed is not None:
                rng.shuffle(indices)
            parts = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
            if repeated and len(parts) > 1 and len(parts[-1]) == 1:
                parts[-2].extend(parts.pop())
            chunks.extend(parts)
        if seed is not None:
            rng.shuffle(chunks)
        for indices in chunks:
            batch_indices, batch = next(cache.batches(len(indices), indices=indices))
            rows = [cache.rows[i] for i in indices]
            batch['_group_keys'] = [self.keys[split][i] for i in indices]
            batch['_indices'] = batch_indices
            batch['_sample_ids'] = [row['sample_id'] for row in rows]
            if split in self.device_predictions_pin:
                predicted, supported = self.device_predictions_pin[split]
                batch['_predicted_device_op'] = predicted.index_select(0, batch_indices)
                batch['_predicted_device_op_mask'] = supported.index_select(0, batch_indices)
                batch['_predicted_device_op_digest'] = self.device_prediction_digest_pin
            if split == 'train':
                labeled = torch.tensor([row['sample_id'] in self.labeled for row in rows], device=self.device)
                batch['performance_mask'] &= labeled[:, None]
            yield batch
