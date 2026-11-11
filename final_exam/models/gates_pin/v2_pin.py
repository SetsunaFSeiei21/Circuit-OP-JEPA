"""v2 gate: preserve absolute OP and add per-sample, per-field MOS contrasts.

This is a deterministic input transform, not BatchNorm: it has no running
statistics, never mixes batch samples, and uses only supplied predicted OP plus
the saved train-support/topology masks. It is not an extra OP prediction loss.
"""
import math

import torch
from torch import nn

from ..op_gate_pin import OPAttentionGatePin


class OPAttentionGateV2Pin(OPAttentionGatePin):
    """2F+4+4 -> d_model -> H, with one MLP shared by the selected layers."""
    version = 'v2'

    @property
    def input_dim(self):
        return 2 * len(self.field_names) + 8

    @property
    def normalization_contract(self):
        return dict(kind='sample_mos_field', device_scope='all_valid_mos',
                    field_scope='all_supported_op', variance='population',
                    norm_eps=self.settings['norm_eps'], global_channel=True,
                    min_count=2, constant_policy='zero')

    def _init_network(self):
        self.score_head = nn.Sequential(nn.Linear(self.input_dim, self.cfg.dim),
                                       nn.GELU(), nn.Linear(self.cfg.dim, self.cfg.heads))
        nn.init.normal_(self.score_head[-1].weight, std=1e-3)
        initial = self.settings['initial_gate']
        nn.init.constant_(self.score_head[-1].bias, math.log(initial / (1 - initial)))

    def relative_op(self, batch, values, supported):
        """Normalize each field across all live MOS owners in its own sample.

        Float64 centered moments with a per-field magnitude scale avoid both
        cancellation in E[x²]-E[x]² and overflow for extreme finite inputs.
        All operations stay on the input device. Statistics count physical
        devices once, independently of how many pin pairs use their OP vector.
        """
        valid = (supported & batch['owner_mask'].unsqueeze(-1)
                 & self.mos_types[batch['owner_device_types']].unsqueeze(-1))
        x = torch.where(valid, values.to(torch.float64), 0.)
        count = valid.sum(dim=1, keepdim=True)
        magnitude = x.abs().amax(dim=1, keepdim=True).clamp_min(1.)
        scaled = x / magnitude
        mean = scaled.sum(dim=1, keepdim=True) / count.clamp_min(1)
        centered = torch.where(valid, scaled - mean, 0.)
        variance = centered.square().sum(dim=1, keepdim=True) / count.clamp_min(1)
        eps = self.settings['norm_eps'] / magnitude / magnitude
        denominator = (variance + eps).sqrt().clamp_min(torch.finfo(torch.float64).tiny)
        lower = torch.where(valid, x, torch.inf).amin(dim=1, keepdim=True)
        upper = torch.where(valid, x, -torch.inf).amax(dim=1, keepdim=True)
        usable = valid & (count >= 2) & (upper > lower)
        relative = torch.where(usable, centered / denominator, 0.)
        return relative.to(dtype=values.dtype)

    def forward(self, batch, values, supported):
        owners, roles, eligible = self._layout(batch, values, supported)
        safe = torch.where(supported, values, 0.)
        relative = self.relative_op(batch, values, supported)
        features = torch.cat((safe, relative), dim=-1)
        batch_size, devices, fields = features.shape
        op = features[:, :, None, None, :].expand(batch_size, devices, 4, 4, fields)
        one_hot = torch.eye(4, dtype=values.dtype, device=values.device)
        query = one_hot[None, None, :, None, :].expand(batch_size, devices, 4, 4, 4)
        key = one_hot[None, None, None, :, :].expand(batch_size, devices, 4, 4, 4)
        logits = self.score_head(torch.cat((op, query, key), dim=-1))
        return self._bias_from_logits(logits, owners, roles, eligible)
