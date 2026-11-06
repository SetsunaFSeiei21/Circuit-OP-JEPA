"""Device adapter of SetsunaFSeiei21/ACCFormer_CR.

Architectural source commit 63aa936255a5b3025bda777f0d94a74769894229:
src/models/accformer.py, src/layers/{encoder,decoder}.py. The pre-BatchNorm
residual blocks and metric-query decoder follow that source. Batched topology,
terminal-pair bias and padding-aware BatchNorm are explicit adaptations.
"""
import torch
from torch import nn

from .common import DeviceEmbedding, MaskedBatchNorm, graph_memory, model_config
from .graph import allowed_edges


def _initialize_block(attention, ffn):
    nn.init.kaiming_uniform_(ffn[0].weight, nonlinearity='relu')
    nn.init.zeros_(ffn[0].bias)
    nn.init.kaiming_uniform_(ffn[3].weight, mode='fan_in')
    nn.init.zeros_(ffn[3].bias)
    nn.init.xavier_uniform_(attention.in_proj_weight)
    nn.init.zeros_(attention.in_proj_bias)
    nn.init.xavier_uniform_(attention.out_proj.weight)
    nn.init.zeros_(attention.out_proj.bias)


def _ffn(cfg):
    # ACCFormer_CR's encoder/decoder use hidden_dim == model width, rather than
    # the common Transformer 4D feed-forward expansion.
    return nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.ReLU(),
                         nn.Dropout(cfg.dropout), nn.Linear(cfg.dim, cfg.dim),
                         nn.Dropout(cfg.dropout))


class ACCEncoderBlock(nn.Module):
    def __init__(self, cfg, edge_dim):
        super().__init__()
        self.heads = cfg.heads
        self.attention = nn.MultiheadAttention(cfg.dim, cfg.heads, cfg.dropout, batch_first=True)
        self.norm1, self.norm2 = MaskedBatchNorm(cfg.dim), MaskedBatchNorm(cfg.dim)
        self.ffn = _ffn(cfg)
        self.terminal_bias = nn.Linear(edge_dim, cfg.heads, bias=False)
        _initialize_block(self.attention, self.ffn)

    def forward(self, nodes, allowed, edge_roles, valid):
        normalized = self.norm1(nodes, valid)
        mask = self.terminal_bias(edge_roles).permute(0, 3, 1, 2)
        mask = mask.masked_fill(~allowed[:, None], -torch.inf).flatten(0, 1)
        attended = self.attention(normalized, normalized, normalized,
                                  attn_mask=mask, need_weights=False)[0] + nodes
        output = self.ffn(self.norm2(attended, valid)) + attended
        return output.masked_fill(~valid.unsqueeze(-1), 0.)


class ACCFormerEncoder(nn.Module):
    def __init__(self, config, prep):
        super().__init__()
        self.cfg = model_config(config)
        self.embedding = DeviceEmbedding(self.cfg, prep)
        edge_dim = len(prep['feature_spec']['terminal_names']) ** 2
        self.layers = nn.ModuleList(ACCEncoderBlock(self.cfg, edge_dim)
                                    for _ in range(self.cfg.encoder_layers))

    def forward(self, batch):
        nodes = self.embedding(batch)
        valid, allowed = batch['device_mask'], allowed_edges(batch)
        for layer in self.layers:
            nodes = layer(nodes, allowed, batch['edge_roles'], valid)
        return graph_memory(nodes, valid)


class ACCDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = nn.MultiheadAttention(cfg.dim, cfg.heads, cfg.dropout, batch_first=True)
        self.norm1, self.norm2 = MaskedBatchNorm(cfg.dim), MaskedBatchNorm(cfg.dim)
        self.ffn = _ffn(cfg)
        _initialize_block(self.attention, self.ffn)

    def forward(self, nodes, query, padding):
        normalized = self.norm1(query)
        attended = self.attention(normalized, nodes, nodes, key_padding_mask=padding,
                                  need_weights=False)[0] + query
        return attended + self.ffn(self.norm2(attended))


class ACCFormerDecoder(nn.Module):
    """Native per-metric learned queries, cross-attention, and scalar output."""
    def __init__(self, config, num_metrics):
        super().__init__()
        cfg = model_config(config)
        self.metric_queries = nn.Parameter(torch.empty(1, num_metrics, cfg.dim))
        nn.init.normal_(self.metric_queries, mean=0., std=0.02)
        self.layers = nn.ModuleList(ACCDecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        # Upstream get_mlp_layer(layer_num=1) is one linear transformation.
        self.output = nn.Linear(cfg.dim, 1)

    def forward(self, memory, padding):
        query = self.metric_queries.expand(memory.shape[0], -1, -1)
        # The graph token is an adapter interface artifact, not an ACCFormer
        # token. The native decoder attends only to whole-device encodings.
        for layer in self.layers:
            query = layer(memory[:, 1:], query, padding[:, 1:])
        return self.output(query).squeeze(-1)
