"""Device inputs and train-only controls for OP-conditioned pin attention.

The two graph granularities share canonical physical device tables; device
inputs are never recovered by pooling or copying expanded pin features.
"""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from device_jepa.resident import _reserve
from device_jepa.training import state_digest


_OWNER_INPUTS = {
    'owner_mask': 'device_mask',
    'owner_device_types': 'device_types',
    'owner_device_categories': 'device_categories',
    'owner_device_models': 'device_models',
    'owner_adjacency': 'adjacency',
    'owner_edge_roles': 'edge_roles',
    'owner_port_roles': 'port_roles',
    'owner_named_port_roles': 'named_port_roles',
    'owner_parameters': 'parameters',
    'owner_parameter_mask': 'parameter_mask',
    'owner_parameter_applicability': 'parameter_applicability',
}


def device_input_pin(batch):
    """Return measured-data-free inputs on the original physical-device axis."""
    required = {'owner_mask', 'owner_device_types', 'owner_adjacency',
                'owner_edge_roles', 'owner_port_roles', 'owner_parameters',
                'owner_parameter_mask', 'owner_parameter_applicability'}
    missing = required - batch.keys()
    if missing:
        raise ValueError(f'Canonical device inputs missing: {sorted(missing)}')
    mask = batch['owner_mask']
    if not torch.is_tensor(mask) or mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError('owner_mask must be a [batch, devices] boolean tensor')
    shape = mask.shape
    result = {}
    for source, target in _OWNER_INPUTS.items():
        if source not in batch:
            continue
        value = batch[source]
        if not torch.is_tensor(value) or value.shape[:2] != shape or value.device != mask.device:
            raise ValueError(f'{source} must use the canonical device axis and device')
        if target in ('adjacency', 'edge_roles') and (value.ndim < 3 or value.shape[2] != shape[1]):
            raise ValueError(f'{source} must have two physical device axes')
        if target in ('device_types', 'device_categories', 'device_models'):
            if value.shape != shape or value.dtype != torch.long:
                raise ValueError(f'{source} must contain one integer identity per device')
        result[target] = value
    params = result['parameters']
    if params.ndim != 3 or not params.is_floating_point():
        raise ValueError('owner_parameters must be a floating [batch, devices, fields] tensor')
    for name in ('parameter_mask', 'parameter_applicability'):
        if result[name].shape != params.shape or result[name].dtype != torch.bool:
            raise ValueError(f'Canonical {name} does not match design values')
    if result['adjacency'].dtype != torch.bool:
        raise ValueError('Canonical adjacency must be boolean')
    return result


def _sample_key(sample_id):
    # Preserve the distinction between e.g. integer 1 and string "1".
    if not isinstance(sample_id, (str, int)) or isinstance(sample_id, bool):
        raise ValueError('Sample IDs must be strings or integers')
    return json.dumps(sample_id, ensure_ascii=False, separators=(',', ':'))


class ShuffleBankPin(nn.Module):
    """Frozen training predictions with deterministic, batch-independent donors.

Each target sample receives a different design's complete device OP array.
Field/device ordering is unchanged. For a training sample its own bank row is
excluded. Held-out samples query this same training-only bank.
"""
    def __init__(self, values, support, sample_ids, seed, topology=None):
        super().__init__()
        if (not torch.is_tensor(values) or values.ndim != 3 or not values.is_floating_point()
                or not torch.is_tensor(support) or support.shape != values.shape
                or support.dtype != torch.bool or support.device != values.device):
            raise ValueError('Shuffle bank requires matching floating OP and boolean support [samples, devices, fields]')
        if values.shape[0] < 2 or values.shape[1] < 1 or values.shape[2] < 1:
            raise ValueError('Shuffle control requires at least two training samples and nonempty device OP')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Shuffle seed must be an integer in [0, 2**32)')
        ids = list(sample_ids)
        keys = [_sample_key(sid) for sid in ids]
        if len(ids) != values.shape[0] or len(set(keys)) != len(keys):
            raise ValueError('Shuffle bank sample IDs must be unique and match its rows')
        if topology is not None and not isinstance(topology, str):
            raise ValueError('Shuffle topology identity must be a string')
        if not torch.isfinite(values[support]).all():
            raise ValueError('Supported shuffle OP must be finite')
        self.register_buffer('values', torch.where(support, values.detach(), 0.).clone())
        self.register_buffer('support', support.detach().clone())
        self.sample_ids, self.seed, self.topology = ids, seed, topology
        self._positions = {key: index for index, key in enumerate(keys)}
        self._donors = {}

    def donor_indices(self, sample_ids):
        donors = []
        size = len(self.sample_ids)
        for sid in sample_ids:
            key = _sample_key(sid)
            if key not in self._donors:
                digest = hashlib.sha256(f'{self.seed}:{key}'.encode('utf-8')).digest()
                value = int.from_bytes(digest[:8], 'big')
                own = self._positions.get(key)
                if own is None:
                    donor = value % size
                else:
                    donor = value % (size - 1)
                    donor += donor >= own
                self._donors[key] = donor
            donors.append(self._donors[key])
        return donors

    def forward(self, batch):
        ids = batch.get('_sample_ids', batch.get('sample_ids'))
        owners = batch.get('owner_mask')
        if not isinstance(ids, (list, tuple)) or owners is None or len(ids) != owners.shape[0]:
            raise ValueError('Shuffle control requires stable sample IDs for every batch row')
        if owners.ndim != 2 or owners.shape[1] != self.values.shape[1]:
            raise ValueError('Shuffle bank device ordering/width differs from the target batch')
        if owners.device != self.values.device:
            raise ValueError('Shuffle bank and target batch must be on the same device')
        if self.topology is not None:
            topologies = batch.get('_owner_topology_pin')
            if not isinstance(topologies, (tuple, list)) or len(topologies) != len(ids) or any(
                    value != self.topology for value in topologies):
                raise ValueError('Shuffle control requires the saved circuit/device topology identity')
        selected = torch.tensor(self.donor_indices(ids), dtype=torch.long, device=self.values.device)
        supported = self.support.index_select(0, selected) & owners.unsqueeze(-1)
        return torch.where(supported, self.values.index_select(0, selected), 0.), supported


def describe_shuffle_bank_pin(bank):
    if not isinstance(bank, ShuffleBankPin):
        raise TypeError('Expected ShuffleBankPin')
    return dict(version=1, shape=list(bank.values.shape), sample_ids=list(bank.sample_ids),
                seed=bank.seed, topology=bank.topology, dtype=str(bank.values.dtype).split('.')[-1],
                donor_source='train_predicted_op_only', self_donor='excluded')


def rebuild_shuffle_bank_pin(recipe):
    if recipe.get('version') != 1 or recipe.get('donor_source') != 'train_predicted_op_only':
        raise ValueError('Unsupported shuffle bank recipe')
    shape = recipe.get('shape')
    if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError('Invalid shuffle bank shape')
    dtype = {'float32': torch.float32, 'float64': torch.float64,
             'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(recipe.get('dtype'))
    if dtype is None or recipe.get('self_donor') != 'excluded':
        raise ValueError('Invalid shuffle dtype/donor policy')
    # The ordinary strict state_dict restore supplies the actual frozen values.
    return ShuffleBankPin(torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=torch.bool),
                          recipe['sample_ids'], recipe['seed'], recipe.get('topology'))


def prepare_device_predictions_pin(ctx, predictor, options, splits=('train', 'val', 'test')):
    """Cache only frozen predicted physical OP, keyed by the complete model state."""
    if predictor.training or any(parameter.requires_grad for parameter in predictor.parameters()):
        raise ValueError('Device OP caching requires a frozen eval predictor')
    batch_size = options['batch_size']
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Prediction cache batch_size must be positive')
    digest = state_digest(predictor)
    if getattr(ctx, 'device_prediction_digest_pin', None) != digest:
        ctx.device_predictions_pin = {}
        ctx.device_prediction_digest_pin = digest
    for split in splits:
        if split not in ctx.rows:
            raise ValueError(f'Unknown prediction split {split}')
        if not ctx.rows[split] or split in ctx.device_predictions_pin:
            continue
        values = support = None
        print(f'Caching frozen device OP: {split} | {len(ctx.rows[split])} samples', flush=True)
        with torch.no_grad():
            for step, batch in enumerate(ctx.batches(split, batch_size, op=False), 1):
                clean = {key: value for key, value in batch.items()
                         if not key.startswith('_predicted_device_op')}
                predicted, supported = predictor.predict_devices(clean)
                if (predicted.ndim != 3 or predicted.shape != supported.shape
                        or supported.dtype != torch.bool or predicted.shape[:2] != batch['owner_mask'].shape
                        or not torch.isfinite(predicted[supported]).all()):
                    raise ValueError('Device OP predictor returned invalid cache tensors')
                if values is None:
                    shape = (len(ctx.rows[split]), *predicted.shape[1:])
                    _reserve(ctx.device, shape[0] * shape[1] * shape[2] * (predicted.element_size() + 1))
                    values = predicted.new_zeros(shape)
                    support = supported.new_zeros(shape)
                if predicted.shape[1:] != values.shape[1:]:
                    raise ValueError('Device OP cache requires a fixed physical device ordering')
                values.index_copy_(0, batch['_indices'], torch.where(supported, predicted, 0.))
                support.index_copy_(0, batch['_indices'], supported)
                every = options.get('progress_every', 100)
                if every and step % every == 0:
                    print(f'Caching frozen device OP: {split} batch {step}', flush=True)
        ctx.device_predictions_pin[split] = (values, support)
    return digest


def build_shuffle_bank_pin(ctx, predictor, batch_size, seed):
    """Build the diagnostic donor bank exclusively from selected training rows."""
    if len(ctx.rows['train']) < 2:
        raise ValueError('Shuffle control requires at least two training samples')
    if len(ctx.groups['train']) != 1:
        raise ValueError('Shuffle control requires one fixed training circuit/device topology')
    prepare_device_predictions_pin(ctx, predictor, {'batch_size': batch_size}, splits=('train',))
    values, support = ctx.device_predictions_pin['train']
    return ShuffleBankPin(values, support, [row['sample_id'] for row in ctx.rows['train']], seed,
                          topology=next(iter(ctx.groups['train'])))


__all__ = ['device_input_pin', 'ShuffleBankPin', 'describe_shuffle_bank_pin',
           'rebuild_shuffle_bank_pin', 'prepare_device_predictions_pin', 'build_shuffle_bank_pin']
