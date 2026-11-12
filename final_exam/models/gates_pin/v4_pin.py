"""Signed V3-style gate shared across fusion self-attention PE2 and later.

All topology and train-support protections remain inherited. Positive and
negative biases modify permitted logits without opening hard-masked edges.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .v3_pin import MoLinearPin, OPAttentionGateV3Pin


class OPAttentionGateV4Pin(OPAttentionGateV3Pin):
    version = 'v4'

    @property
    def architecture_contract(self):
        return dict(super().architecture_contract,
            initialization='xavier_hidden_small_random_output_zero_bias',
            shared_across_layers=True, gated_pe_layers=list(range(2, self.cfg.cross_layers + 1)),
            independent_gate_count=1, bias_transform='strength*tanh(logits)',
            bias_strength=self.settings['strength'])

    def _new_score_head(self):
        if self.cfg.cross_layers < 2:
            raise ValueError('V4/V5 gates require at least two fusion layers (PE2 or later)')
        dim, experts = self.cfg.dim, self.settings['n_experts']
        dropout = self.settings['mlp_dropout']
        network = nn.Sequential(
            MoLinearPin(self.input_dim, dim, experts), nn.Dropout(dropout), nn.GELU(),
            MoLinearPin(dim, dim, experts), nn.Dropout(dropout), nn.GELU(),
            MoLinearPin(dim, self.cfg.heads, experts),
        )
        # Near-neutral, not saturated: retain gradients through the OP inputs,
        # hidden experts and routers from the first optimization step.
        nn.init.normal_(network[-1].weight, std=1e-3)
        nn.init.zeros_(network[-1].bias)
        return network

    def _init_network(self):
        self.score_head = self._new_score_head()

    def iter_layer_score_heads(self):
        """Yield 1-based PE index and its scoring network (shared for V4)."""
        return ((index, self.score_head) for index in range(2, self.cfg.cross_layers + 1))

    def pair_inputs(self, batch, values, supported):
        safe = torch.where(supported, values, 0.)
        relative = self.relative_op(batch, values, supported)
        features = torch.cat((safe, relative), dim=-1)
        batch_size, devices, fields = features.shape
        op = features[:, :, None, None, :].expand(batch_size, devices, 4, 4, fields)
        one_hot = torch.eye(4, dtype=values.dtype, device=values.device)
        query = one_hot[None, None, :, None, :].expand(batch_size, devices, 4, 4, 4)
        key = one_hot[None, None, None, :, :].expand(batch_size, devices, 4, 4, 4)
        return torch.cat((op, query, key), dim=-1)

    def _bias_from_logits(self, logits, owners, roles, eligible):
        signed = self.settings['strength'] * torch.tanh(logits)
        batch_size, devices = logits.shape[:2]
        pins = owners.shape[1]
        table = signed.permute(0, 4, 1, 2, 3).reshape(batch_size, self.cfg.heads, devices * 16)
        indices = owners[:, :, None] * 16 + roles[:, :, None] * 4 + roles[:, None, :]
        result = table.gather(2, indices.reshape(batch_size, 1, pins * pins).expand(-1, self.cfg.heads, -1))
        result = result.reshape(batch_size, self.cfg.heads, pins, pins)
        result = torch.where(eligible[:, None], result, 0.)
        return F.pad(result, (1, 0, 1, 0))

    def forward(self, batch, values, supported):
        owners, roles, eligible = self._layout(batch, values, supported)
        logits = self.score_head(self.pair_inputs(batch, values, supported))
        bias = self._bias_from_logits(logits, owners, roles, eligible)
        # Compute once: both parameters and the dropout realization are shared.
        return (None,) + (bias,) * (self.cfg.cross_layers - 1)
