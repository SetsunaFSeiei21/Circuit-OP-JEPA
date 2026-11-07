"""Pin-node input adapters; numeric kernels are shared with device baselines.

A token is an observed terminal or a named external port. Repeated owner design
values do not change the train-only, device-based normalization statistics.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .common import DeviceFeatures, model_config

_PIN_TOPOLOGY = frozenset({'device_types', 'device_categories', 'device_models',
    'device_mask', 'adjacency', 'edge_roles', 'port_roles', 'named_port_roles',
    'pin_to_device', 'owner_index', 'owner_mask', 'owner_device_types',
    'owner_device_models', 'owner_edge_roles', 'pin_terminal_index', 'pin_net_index',
    'gtn_loop_membership_by_pin'})
_DESIGN = frozenset({'parameters', 'parameter_mask', 'parameter_applicability'})


def design_input_pin(batch):
    """Exclude OP, OP availability, labels and simulation state from baselines."""
    return {key: value for key, value in batch.items() if key in _PIN_TOPOLOGY | _DESIGN}


def require_pin_batch(batch):
    """Reject passing device graphs to a model merely renamed with _pin."""
    shape = batch['device_mask'].shape
    for name in ('pin_to_device', 'pin_terminal_index', 'pin_net_index'):
        if name not in batch or batch[name].shape != shape:
            raise ValueError(f'Pin models require {name} with shape [batch, pins]')
    return batch


class PinFeatures(DeviceFeatures):
    """Named pin type/role, parent model identity and parent design features."""
    def forward(self, batch):
        return super().forward(require_pin_batch(batch))


class PinEmbedding(nn.Module):
    def __init__(self, config, prep):
        super().__init__()
        self.cfg = model_config(config)
        self.features = PinFeatures(prep)
        self.projection = nn.Linear(self.features.out_dim, self.cfg.dim)

    def forward(self, batch):
        if batch['device_mask'].shape[1] > self.cfg.max_devices:
            raise ValueError('Pin/port count exceeds max_devices token capacity; no truncation')
        return self.projection(self.features(batch)).masked_fill(
            ~batch['device_mask'].unsqueeze(-1), 0.)


class MLPPin(nn.Module):
    """Fixed canonical pin ordering for the single-circuit tabular baseline."""
    def __init__(self, config, prep):
        super().__init__()
        self.cfg = model_config(config)
        self.features = PinFeatures(prep)
        layers = [nn.Linear(self.features.out_dim*self.cfg.max_devices, self.cfg.ffn_dim),
                  nn.ReLU(), nn.Dropout(self.cfg.dropout)]
        for _ in range(self.cfg.encoder_layers-1):
            layers.extend((nn.Linear(self.cfg.ffn_dim,self.cfg.ffn_dim),nn.ReLU(),nn.Dropout(self.cfg.dropout)))
        layers.append(nn.Linear(self.cfg.ffn_dim,len(prep['feature_spec']['metric_names'])))
        self.network = nn.Sequential(*layers)

    def forward(self, batch):
        values = self.features(design_input_pin(batch))
        remaining = self.cfg.max_devices-values.shape[1]
        if remaining < 0:
            raise ValueError('Pin/port count exceeds max_devices token capacity; no truncation')
        return self.network(F.pad(values,(0,0,0,remaining)).flatten(1))
