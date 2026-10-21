"""Original implementation of a ZeroSim-inspired device-token encoder.

This is not a byte-for-byte reproduction and imports no ZeroSim code. See
THIRD_PARTY_NOTICES.md and docs/architecture.md for correspondence/differences.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    dim: int = 128
    heads: int = 8
    encoder_layers: int = 6
    cross_layers: int = 4
    ffn_dim: int = 512
    decoder_layers: int = 2
    decoder_ffn_dim: int = 2048
    dropout: float = 0.1
    max_devices: int = 200
    terminal_bias: bool = True

    def __post_init__(self):
        if self.dim < 2 or self.heads < 1 or self.dim % self.heads:
            raise ValueError("dim must be >=2 and divisible by heads")
        if not 1 <= self.cross_layers <= self.encoder_layers:
            raise ValueError("At least one parameter cross-attention layer is required")
        if self.decoder_layers < 1 or self.max_devices < 1:
            raise ValueError("decoder_layers and max_devices must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)


class EncoderBlock(nn.Module):
    def __init__(self, cfg, cross):
        super().__init__()
        self.heads = cfg.heads
        self.self_attn = nn.MultiheadAttention(cfg.dim, cfg.heads, cfg.dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(cfg.dim)
        self.cross_attn = nn.MultiheadAttention(cfg.dim, cfg.heads, cfg.dropout, batch_first=True) if cross else None
        self.cross_norm = nn.LayerNorm(cfg.dim) if cross else None
        self.ffn = nn.Sequential(nn.Linear(cfg.dim, cfg.ffn_dim), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_dim, cfg.dim))
        self.ffn_norm = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, self_mask, parameter_tokens, cross_mask, valid):
        update = self.self_attn(x, x, x, attn_mask=self_mask, need_weights=False)[0]
        x = self.self_norm(x + self.drop(update))
        if self.cross_attn is not None:
            update = self.cross_attn(x, parameter_tokens, parameter_tokens,
                                     attn_mask=cross_mask, need_weights=False)[0]
            x = self.cross_norm(x + self.drop(update))
        x = self.ffn_norm(x + self.drop(self.ffn(x)))
        return x * valid.unsqueeze(-1)


class DeviceEncoder(nn.Module):
    """Return [graph token, every device token], never a pooled-only memory."""
    def __init__(self, cfg, num_types, num_parameters, num_terminals):
        super().__init__()
        self.cfg = cfg
        self.num_parameters = num_parameters
        self.type_embedding = nn.Embedding(num_types, cfg.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(cfg.max_devices, cfg.dim)
        self.graph_token = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        # Named feature embeddings disambiguate a zero W from a zero L.
        self.parameter_names = nn.Embedding(num_parameters, cfg.dim)
        self.parameter_value = nn.Linear(1, cfg.dim)
        self.null_parameter = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.port_projection = nn.Linear(num_terminals, cfg.dim, bias=False)
        self.role_bias = nn.Linear(num_terminals ** 2, cfg.heads, bias=False) if cfg.terminal_bias else None
        self.blocks = nn.ModuleList([
            EncoderBlock(cfg, cross=(i >= cfg.encoder_layers - cfg.cross_layers))
            for i in range(cfg.encoder_layers)
        ])
        self.output_norm = nn.LayerNorm(cfg.dim)
        self.output_projection = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, batch):
        types, params, observed = batch["device_types"], batch["parameters"], batch["parameter_mask"]
        valid_devices = batch["device_mask"]
        b, n = types.shape
        if n > self.cfg.max_devices:
            raise ValueError(f"{n} devices exceed max_devices={self.cfg.max_devices}; no truncation is allowed")
        if params.shape != (b, n, self.num_parameters):
            raise ValueError("Parameter schema does not match encoder")
        pos = self.position_embedding(torch.arange(n, device=types.device))[None]
        nodes = self.type_embedding(types) * self.cfg.dim ** 0.5 + pos
        nodes = nodes + self.port_projection(batch["port_roles"])
        nodes = nodes * valid_devices.unsqueeze(-1)
        x = torch.cat([self.graph_token.expand(b, -1, -1), nodes], dim=1)
        valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=types.device), valid_devices], 1)
        # Invalid numerical entries are already zeroed by the collator; mask again
        # here so a caller cannot accidentally feed missing-value garbage.
        safe_params = torch.where(observed, params, torch.zeros_like(params))
        p = self.parameter_value(safe_params.unsqueeze(-1))
        p = p + self.parameter_names.weight[None, None]
        p = p.reshape(b, n * self.num_parameters, self.cfg.dim)
        p = torch.cat([p, self.null_parameter.expand(b, -1, -1)], dim=1)
        cross_allowed = torch.zeros(b, n + 1, n * self.num_parameters + 1,
                                    dtype=torch.bool, device=types.device)
        live_params = observed & valid_devices.unsqueeze(-1)
        cross_allowed[:, 0, :-1] = live_params.flatten(1)
        for i in range(n):
            cross_allowed[:, i + 1, i * self.num_parameters:(i + 1) * self.num_parameters] = live_params[:, i]
        # A null token is visible only for rows without a real parameter. This
        # prevents all-masked attention NaNs for passive/missing/padded devices.
        cross_allowed[:, :, -1] = ~cross_allowed[:, :, :-1].any(-1)
        cross_mask = self._mask(cross_allowed)
        local = torch.ones(b, n + 1, n + 1, dtype=torch.bool, device=types.device)
        local[:, 1:, 1:] = batch["adjacency"] | torch.eye(n, dtype=torch.bool, device=types.device)[None]
        local &= valid[:, None, :]
        global_allowed = valid[:, None, :].expand(-1, n + 1, -1)
        bias = None
        if self.role_bias is not None:
            role = self.role_bias(batch["edge_roles"]).permute(0, 3, 1, 2)
            bias = x.new_zeros(b, self.cfg.heads, n + 1, n + 1)
            bias[:, :, 1:, 1:] = role
        for i, block in enumerate(self.blocks):
            mask = self._mask(local if i % 2 == 0 else global_allowed, bias)
            x = block(x, mask, p, cross_mask, valid)
        x = self.output_projection(self.output_norm(x)) * valid.unsqueeze(-1)
        return x, ~valid

    def _mask(self, allowed, bias=None):
        b, q, k = allowed.shape
        out = self.graph_token.new_zeros(b, self.cfg.heads, q, k)
        if bias is not None:
            out = out + bias
        return out.masked_fill(~allowed[:, None], float("-inf")).reshape(b * self.cfg.heads, q, k)


class SharedOPEncoder(nn.Module):
    """One trainable MLP reused for every device and every circuit.

    Missing OP values are represented by explicit observed-mask inputs. Entirely
    unobserved devices are excluded from alignment and all regularizers.
    """
    def __init__(self, num_op_features, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * num_op_features, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim))

    def forward(self, op, mask):
        values = torch.where(mask, op, torch.zeros_like(op))
        return self.net(torch.cat([values, mask.to(op.dtype)], -1))


class JEPAModel(nn.Module):
    def __init__(self, encoder, num_op_features):
        super().__init__()
        self.encoder = encoder
        d = encoder.cfg.dim
        self.predictor = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.op_encoder = SharedOPEncoder(num_op_features, d)

    def forward(self, batch):
        memory, _ = self.encoder(batch)
        predicted = self.predictor(memory[:, 1:])
        target = self.op_encoder(batch["op"], batch["op_mask"])
        valid = batch["device_mask"] & batch["op_mask"].any(-1)
        # Both branches are trainable by design. Not an EMA/frozen teacher.
        return predicted, target, valid


class MetricQueryDecoder(nn.Module):
    def __init__(self, cfg, num_metrics):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_metrics, cfg.dim) * 0.02)
        layer = nn.TransformerDecoderLayer(cfg.dim, cfg.heads, cfg.decoder_ffn_dim,
                                           cfg.dropout, activation="relu", batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers)
        self.regressor = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(),
                                       nn.Dropout(cfg.dropout), nn.Linear(cfg.dim, 1))

    def forward(self, memory, padding):
        query = self.query.expand(memory.shape[0], -1, -1)
        decoded = self.decoder(query, memory, memory_key_padding_mask=padding)
        return self.regressor(decoded).squeeze(-1)


class FrozenPerformanceModel(nn.Module):
    """Keep encoder parameters AND state behavior fixed even after train()."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, batch):
        with torch.no_grad():
            memory, padding = self.encoder(batch)
        return self.decoder(memory, padding)


def make_decoder(cfg, num_metrics):
    return MetricQueryDecoder(cfg, num_metrics)


def alignment_loss(predicted, target, valid, invariance=25.0, variance=25.0,
                   covariance=1.0, variance_floor=1.0):
    """Device-wise MSE + VICReg-style anti-collapse on BOTH learned branches.

    Regularizers treat observed device tokens as samples; tokens in the same
    circuit are correlated. This is a demo choice, not a claim of canonical JEPA.
    """
    a, b = predicted[valid], target[valid]
    if a.shape[0] < 2:
        raise ValueError("Alignment requires >=2 devices with observed OP per batch")
    mse = (a - b).square().mean()
    var, cov = a.new_zeros(()), a.new_zeros(())
    for z in (a, b):
        centered = z - z.mean(0)
        std = (centered.square().mean(0) + 1e-4).sqrt()
        var = var + torch.relu(variance_floor - std).mean() / 2
        matrix = centered.T @ centered / (z.shape[0] - 1)
        off_diagonal = matrix - torch.diag_embed(torch.diagonal(matrix))
        cov = cov + off_diagonal.square().sum() / z.shape[1] / 2
    total = invariance * mse + variance * var + covariance * cov
    return total, {"alignment_mse": mse.detach(), "variance": var.detach(),
                   "covariance": cov.detach(), "observed_devices": a.shape[0]}


def masked_mse(prediction, target, mask):
    if not mask.any():
        raise ValueError("No observed performance targets in batch")
    return (prediction[mask] - target[mask]).square().mean()
