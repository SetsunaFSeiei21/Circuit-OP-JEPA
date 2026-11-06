"""Dense device adaptations of circuit-fewshot-code GAT and DeepGEN.

Architectural source: kouroshHakha/circuit-fewshot-code, commit
8d210af3e2ffbe2a5b5b6566dd2d80cab1ac2039, cgl/models/gnn.py.
See SOURCES.json and LICENSE_circuit_fewshot for attribution and changes.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .common import DeviceEmbedding, MaskedBatchNorm, graph_memory, model_config


def allowed_edges(batch):
    valid = batch['device_mask']
    n = valid.shape[1]
    adjacency = batch['adjacency'] | torch.eye(n, dtype=torch.bool, device=valid.device)[None]
    # Give padded queries a real key so softmax is finite, then zero their output.
    return (adjacency | ~valid.unsqueeze(-1)) & valid.unsqueeze(1)


class GATLayer(nn.Module):
    """GATConv with mean heads and terminal-pair edge attention features."""
    def __init__(self, cfg, edge_dim):
        super().__init__()
        self.heads, self.dim = cfg.heads, cfg.dim
        self.projection = nn.Linear(cfg.dim, cfg.heads * cfg.dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(cfg.heads, cfg.dim))
        self.attention_target = nn.Parameter(torch.empty(cfg.heads, cfg.dim))
        self.edge_attention = nn.Linear(edge_dim, cfg.heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.dim))
        self.dropout = cfg.dropout
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(self, nodes, allowed, edge_roles, valid):
        projected = self.projection(nodes).reshape(*nodes.shape[:2], self.heads, self.dim)
        source = (projected * self.attention_source).sum(-1)
        target = (projected * self.attention_target).sum(-1)
        scores = source.unsqueeze(2) + target.unsqueeze(1) + self.edge_attention(edge_roles)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        weights = scores.masked_fill(~allowed.unsqueeze(-1), -torch.inf).softmax(dim=2)
        weights = F.dropout(weights, self.dropout, self.training)
        update = torch.einsum('bijh,bjhd->bihd', weights, projected).mean(2)
        return F.relu(update + self.bias).masked_fill(~valid.unsqueeze(-1), 0.)


class GATEncoder(nn.Module):
    def __init__(self, config, prep):
        super().__init__()
        self.cfg = model_config(config)
        self.embedding = DeviceEmbedding(self.cfg, prep)
        edge_dim = len(prep['feature_spec']['terminal_names']) ** 2
        self.layers = nn.ModuleList(GATLayer(self.cfg, edge_dim)
                                    for _ in range(self.cfg.encoder_layers))

    def forward(self, batch):
        nodes = self.embedding(batch)
        valid, allowed = batch['device_mask'], allowed_edges(batch)
        for layer in self.layers:
            nodes = layer(nodes, allowed, batch['edge_roles'], valid)
        return graph_memory(nodes, valid)


class MeanReadout(nn.Module):
    """Native pooled GNN regression head; no shared Transformer decoder."""
    def __init__(self, cfg, num_metrics):
        super().__init__()
        self.output = nn.Linear(cfg.dim, num_metrics)

    def forward(self, memory, padding):
        return self.output(memory[:, 0])


class GENLayer(nn.Module):
    """GENConv channel-wise softmax aggregation in a preactivation res+ block."""
    def __init__(self, cfg, edge_dim):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.dim)
        self.temperature = nn.Parameter(torch.ones(1))
        self.edge_projection = nn.Linear(edge_dim, cfg.dim, bias=False)
        self.first = nn.Linear(cfg.dim, cfg.dim * 2)
        self.mlp_norm = MaskedBatchNorm(cfg.dim * 2)
        self.second = nn.Linear(cfg.dim * 2, cfg.dim)
        self.dropout = cfg.dropout

    def forward(self, nodes, allowed, edge_roles, valid):
        normalized = F.dropout(F.relu(self.norm(nodes)), self.dropout, self.training)
        messages = F.relu(normalized.unsqueeze(1) + self.edge_projection(edge_roles)) + 1e-7
        logits = (messages * self.temperature).masked_fill(~allowed.unsqueeze(-1), -torch.inf)
        aggregate = (logits.softmax(dim=2) * messages).sum(dim=2)
        hidden = self.first(normalized + aggregate)
        update = self.second(F.relu(self.mlp_norm(hidden, valid)))
        return (nodes + update).masked_fill(~valid.unsqueeze(-1), 0.)


class DeepGENEncoder(nn.Module):
    def __init__(self, config, prep):
        super().__init__()
        self.cfg = model_config(config)
        self.embedding = DeviceEmbedding(self.cfg, prep)
        edge_dim = len(prep['feature_spec']['terminal_names']) ** 2
        self.layers = nn.ModuleList(GENLayer(self.cfg, edge_dim)
                                    for _ in range(self.cfg.encoder_layers))

    def forward(self, batch):
        nodes = self.embedding(batch)
        valid, allowed = batch['device_mask'], allowed_edges(batch)
        for layer in self.layers:
            nodes = layer(nodes, allowed, batch['edge_roles'], valid)
        nodes = F.relu(self.layers[0].norm(nodes)).masked_fill(~valid.unsqueeze(-1), 0.)
        return graph_memory(nodes, valid)


class CrossAttentionPool(nn.Module):
    """Device-masked XAttn from the upstream Node2GraphEmb readout."""
    def __init__(self, dim, input_dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(input_dim, dim)
        self.value = nn.Linear(input_dim, dim)
        self.dim = dim

    def forward(self, nodes, state, padding):
        logits = torch.matmul(self.query(state), self.key(nodes).transpose(-1, -2))
        logits = logits / self.dim ** 0.5
        weights = logits.masked_fill(padding[:, None], -torch.inf).softmax(-1)
        return torch.matmul(weights, self.value(nodes))


class DeepGENReadout(nn.Module):
    """Native 16x16 query pooling (3 passes), then graph-regression MLP."""
    def __init__(self, cfg, num_metrics):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 16, 16))
        self.pool = nn.ModuleList(CrossAttentionPool(16, cfg.dim) for _ in range(3))
        width = cfg.dim if cfg.decoder_layers > 1 else num_metrics
        layers = [nn.Linear(256, width)]
        for index in range(cfg.decoder_layers - 1):
            target = cfg.dim if index < cfg.decoder_layers - 2 else num_metrics
            layers.extend((MaskedBatchNorm(cfg.dim), nn.ReLU(), nn.Dropout(cfg.dropout),
                           nn.Linear(cfg.dim, target)))
        self.output = nn.Sequential(*layers)

    def forward(self, memory, padding):
        state = self.query.expand(memory.shape[0], -1, -1)
        for layer in self.pool:
            state = layer(memory[:, 1:], state, padding[:, 1:])
        return self.output(state.flatten(1))
