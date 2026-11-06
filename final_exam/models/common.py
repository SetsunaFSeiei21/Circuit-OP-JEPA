"""Device-granularity inputs shared by the final experiment adapters.

Only topology and design parameters are read here. OP values, OP missingness,
simulation status and performance labels never enter a baseline's features.
"""
import torch
from torch import nn
from torch.nn import functional as F

from device_jepa.model import ModelConfig


def model_config(value):
    return value if isinstance(value, ModelConfig) else ModelConfig(**value)


def masked_mean(values, valid, dim=1):
    masked = torch.where(valid.unsqueeze(-1), values, 0.)
    count = valid.sum(dim=dim, keepdim=True).clamp_min(1)
    return masked.sum(dim=dim) / count


def graph_memory(nodes, valid):
    graph = masked_mean(nodes, valid).unsqueeze(1)
    memory = torch.cat((graph, nodes.masked_fill(~valid.unsqueeze(-1), 0.)), dim=1)
    graph_valid = torch.ones_like(valid[:, :1])
    return memory, ~torch.cat((graph_valid, valid), dim=1)


class DeviceFeatures(nn.Module):
    """Named parameter values/masks and categorical identity per whole device."""
    def __init__(self, prep):
        super().__init__()
        spec = prep['feature_spec']
        self.parameter_count = len(spec['parameter_names'])
        self.terminal_count = len(spec['terminal_names'])
        self.type_count = len(prep['device_vocab'])
        self.category_count = len(prep.get('device_category_vocab', {}))
        self.model_count = len(prep.get('model_vocab', {}))
        self.port_count = len(prep.get('port_role_vocab', {}))
        self.out_dim = (3 * self.parameter_count + self.type_count + self.category_count
                        + self.model_count + self.terminal_count
                        + self.terminal_count * self.port_count)

    def forward(self, batch):
        valid = batch['device_mask']
        params = batch['parameters']
        observed = batch['parameter_mask'] & valid.unsqueeze(-1)
        applicable = batch['parameter_applicability'] & valid.unsqueeze(-1)
        if params.shape != (*valid.shape, self.parameter_count):
            raise ValueError('Parameter fields disagree with saved feature schema')
        if not bool(valid.any(dim=1).all()):
            raise ValueError('Every circuit must contain at least one real device')
        pieces = [torch.where(observed, params, 0.), observed.to(params.dtype),
                  applicable.to(params.dtype),
                  F.one_hot(batch['device_types'], self.type_count).to(params.dtype)]
        for key, count in (('device_categories', self.category_count),
                           ('device_models', self.model_count)):
            if count:
                pieces.append(F.one_hot(batch[key], count).to(params.dtype))
        pieces.append(batch['port_roles'])
        if self.port_count:
            pieces.append(batch['named_port_roles'])
        return torch.cat(pieces, dim=-1).masked_fill(~valid.unsqueeze(-1), 0.)


class DeviceEmbedding(nn.Module):
    def __init__(self, config, prep):
        super().__init__()
        self.cfg = model_config(config)
        self.features = DeviceFeatures(prep)
        self.projection = nn.Linear(self.features.out_dim, self.cfg.dim)

    def forward(self, batch):
        if batch['device_mask'].shape[1] > self.cfg.max_devices:
            raise ValueError('Device count exceeds max_devices; truncation is forbidden')
        return self.projection(self.features(batch)).masked_fill(
            ~batch['device_mask'].unsqueeze(-1), 0.)


def tabular_features(batch, prep, max_devices):
    """Fixed canonical device order, used only for single-circuit tabular models."""
    features = DeviceFeatures(prep)(batch)
    missing = max_devices - features.shape[1]
    if missing < 0:
        raise ValueError('Device count exceeds fixed tabular width')
    return F.pad(features, (0, 0, 0, missing)).flatten(1)


class MaskedBatchNorm(nn.BatchNorm1d):
    """Upstream BatchNorm semantics excluding padded device rows."""
    def forward(self, values, valid=None):
        if valid is None:
            valid = torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
        selected = values[valid]
        if selected.shape[0] > 1 or not self.training:
            normalized = super().forward(selected)
        else:
            # A single-token minibatch has no sample variance. Use the already
            # saved running statistics instead of changing the graph or labels.
            normalized = F.batch_norm(selected, self.running_mean, self.running_var,
                                      self.weight, self.bias, training=False,
                                      momentum=self.momentum, eps=self.eps)
        output = torch.zeros_like(values)
        output[valid] = normalized
        return output


class DeviceMLP(nn.Module):
    """Whole-device features flattened in canonical order for one topology."""
    def __init__(self, config, prep):
        super().__init__()
        self.cfg = model_config(config)
        self.features = DeviceFeatures(prep)
        layers = [nn.Linear(self.features.out_dim * self.cfg.max_devices, self.cfg.ffn_dim),
                  nn.ReLU(), nn.Dropout(self.cfg.dropout)]
        for _ in range(self.cfg.encoder_layers - 1):
            layers.extend((nn.Linear(self.cfg.ffn_dim, self.cfg.ffn_dim),
                           nn.ReLU(), nn.Dropout(self.cfg.dropout)))
        layers.append(nn.Linear(self.cfg.ffn_dim, len(prep['feature_spec']['metric_names'])))
        self.network = nn.Sequential(*layers)

    def forward(self, batch):
        values = self.features(batch)
        missing = self.cfg.max_devices - values.shape[1]
        if missing < 0:
            raise ValueError('Device count exceeds max_devices; truncation is forbidden')
        return self.network(F.pad(values, (0, 0, 0, missing)).flatten(1))
