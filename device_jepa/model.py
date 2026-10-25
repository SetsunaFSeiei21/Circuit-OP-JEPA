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
        if not 1 <= self.cross_layers < self.encoder_layers:
            raise ValueError("Require >=1 shared structure layer followed by >=1 parameter-fusion layer")
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


class StructureEncoder(nn.Module):
    """Entire pure-structure prefix before the first parameter-fusion layer.

    Consumes only type, position, connectivity/terminal and port metadata.
    Design parameters, OP values, states and labels are deliberately inaccessible.
    """
    def __init__(self, cfg, num_types, num_terminals):
        super().__init__()
        self.cfg = cfg
        self.type_embedding = nn.Embedding(num_types, cfg.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(cfg.max_devices, cfg.dim)
        self.graph_token = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        self.port_projection = nn.Linear(num_terminals, cfg.dim, bias=False)
        self.role_bias = nn.Linear(num_terminals ** 2, cfg.heads, bias=False) if cfg.terminal_bias else None
        self.blocks = nn.ModuleList([EncoderBlock(cfg, cross=False)
                                     for _ in range(cfg.encoder_layers - cfg.cross_layers)])

    def forward(self, batch):
        types, devices = batch["device_types"], batch["device_mask"]
        b, n = types.shape
        if n > self.cfg.max_devices:
            raise ValueError(f"{n} devices exceed max_devices={self.cfg.max_devices}; no truncation is allowed")
        pos = self.position_embedding(torch.arange(n, device=types.device))[None]
        nodes = self.type_embedding(types) * self.cfg.dim ** 0.5 + pos
        nodes = (nodes + self.port_projection(batch["port_roles"])) * devices.unsqueeze(-1)
        x = torch.cat([self.graph_token.expand(b, -1, -1), nodes], dim=1)
        valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=types.device), devices], 1)
        local = torch.ones(b, n + 1, n + 1, dtype=torch.bool, device=types.device)
        local[:, 1:, 1:] = batch["adjacency"] | torch.eye(n, dtype=torch.bool, device=types.device)[None]
        local &= valid[:, None, :]
        global_allowed = valid[:, None, :].expand(-1, n + 1, -1)
        bias = None
        if self.role_bias is not None:
            bias = x.new_zeros(b, self.cfg.heads, n + 1, n + 1)
            bias[:, :, 1:, 1:] = self.role_bias(batch["edge_roles"]).permute(0, 3, 1, 2)
        masks = (self.attention_mask(local, bias), self.attention_mask(global_allowed, bias))
        for i, block in enumerate(self.blocks):
            x = block(x, masks[i % 2], None, None, valid)
        return {"tokens": x, "valid": valid, "self_masks": masks}

    def attention_mask(self, allowed, bias=None):
        b, q, k = allowed.shape
        out = self.graph_token.new_zeros(b, self.cfg.heads, q, k)
        if bias is not None:
            out = out + bias
        return out.masked_fill(~allowed[:, None], float("-inf")).reshape(b * self.cfg.heads, q, k)


class DeviceEncoder(nn.Module):
    """Input branch: registered shared structure followed by design-only fusion.

    Default 6/4 means two shared structure blocks and four design-fusion blocks,
    preserving the original first-cross-attention boundary rather than adding
    six new layers. The later parameter-conditioned self-attention is not shared.
    """
    def __init__(self, cfg, num_types, num_parameters, num_terminals):
        super().__init__()
        self.cfg, self.num_parameters = cfg, num_parameters
        self.structure_encoder = StructureEncoder(cfg, num_types, num_terminals)
        self.parameter_names = nn.Embedding(num_parameters, cfg.dim)
        self.parameter_value = nn.Linear(1, cfg.dim)
        self.null_parameter = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.fusion_blocks = nn.ModuleList([EncoderBlock(cfg, cross=True) for _ in range(cfg.cross_layers)])
        self.output_norm = nn.LayerNorm(cfg.dim)
        self.output_projection = nn.Linear(cfg.dim, cfg.dim, bias=False)

    @property
    def graph_token(self):
        return self.structure_encoder.graph_token

    def forward(self, batch):
        return self.forward_from_structure(self.structure_encoder(batch), batch)

    def forward_from_structure(self, structure, batch):
        params, observed = batch["parameters"], batch["parameter_mask"]
        b, n = batch["device_types"].shape
        if params.shape != (b, n, self.num_parameters):
            raise ValueError("Parameter schema does not match encoder")
        safe_params = torch.where(observed, params, torch.zeros_like(params))
        p = self.parameter_value(safe_params.unsqueeze(-1)) + self.parameter_names.weight[None, None]
        p = torch.cat([p.reshape(b, n * self.num_parameters, self.cfg.dim),
                       self.null_parameter.expand(b, -1, -1)], 1)
        allowed = torch.zeros(b, n + 1, n * self.num_parameters + 1,
                              dtype=torch.bool, device=params.device)
        live = observed & batch["device_mask"].unsqueeze(-1)
        allowed[:, 0, :-1] = live.flatten(1)
        for i in range(n):
            allowed[:, i + 1, i * self.num_parameters:(i + 1) * self.num_parameters] = live[:, i]
        allowed[:, :, -1] = ~allowed[:, :, :-1].any(-1)
        cross_mask = self.structure_encoder.attention_mask(allowed)
        # Do not mutate or detach structure tokens: the other branch consumes
        # precisely this same dropout realization with its own gradient path.
        x, valid = structure["tokens"], structure["valid"]
        offset = len(self.structure_encoder.blocks)
        for i, block in enumerate(self.fusion_blocks):
            x = block(x, structure["self_masks"][(offset + i) % 2], p, cross_mask, valid)
        x = self.output_projection(self.output_norm(x)) * valid.unsqueeze(-1)
        return x, ~valid


class SharedOPEncoder(nn.Module):
    """One device-shared MLP combining shared structure and OP observations.

    Its structure_encoder property is the SAME module object as the input side.
    The input encoder alone owns registration, avoiding duplicate state entries.
    Optional categorical state values/masks bypass numeric normalization entirely.
    """
    def __init__(self, structure_encoder, num_op_features, num_state_features=0):
        super().__init__()
        # Deliberate non-registering reference. The module is registered exactly
        # once under DeviceEncoder; no copy, EMA, detach or state synchronization.
        object.__setattr__(self, "_shared_structure", structure_encoder)
        dim = structure_encoder.cfg.dim
        self.num_state_features = num_state_features
        self.net = nn.Sequential(nn.Linear(dim + 2 * num_op_features + 2 * num_state_features, 2 * dim),
                                 nn.GELU(), nn.Linear(2 * dim, dim))

    @property
    def structure_encoder(self):
        return self._shared_structure

    def forward(self, structure_tokens, op, mask, state=None, state_mask=None):
        values = torch.where(mask, op, torch.zeros_like(op))
        inputs = [structure_tokens, values, mask.to(op.dtype)]
        if self.num_state_features:
            if state is None or state_mask is None or state.shape[-1] != self.num_state_features:
                raise ValueError("Checkpoint requires explicitly encoded OP state features and masks")
            inputs.extend([torch.where(state_mask, state, torch.zeros_like(state)), state_mask.to(op.dtype)])
        return self.net(torch.cat(inputs, -1))


class JEPAModel(nn.Module):
    def __init__(self, encoder, num_op_features, num_state_features=0):
        super().__init__()
        self.encoder = encoder
        self.op_encoder = SharedOPEncoder(encoder.structure_encoder, num_op_features, num_state_features)

    @property
    def structure_encoder(self):
        return self.encoder.structure_encoder

    def forward(self, batch):
        # Exactly one shared forward, including one shared dropout realization.
        structure = self.structure_encoder(batch)
        memory, _ = self.encoder.forward_from_structure(structure, batch)
        target = self.op_encoder(structure["tokens"][:, 1:], batch["op"], batch["op_mask"],
                                 batch.get("op_state"), batch.get("op_state_mask"))
        eligible = batch.get("op_target_mask", batch["device_mask"])
        valid = batch["device_mask"] & eligible & batch["op_mask"].any(-1)
        # Direct hidden alignment. Both paths backpropagate into the same Sθ.
        # No predictor, EMA, teacher copy or stop-gradient exists.
        return memory[:, 1:], target, valid


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
