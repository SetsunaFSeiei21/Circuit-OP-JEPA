"""v1: normalized OP plus ordered terminal one-hots into one shared MLP."""
import math

import torch
from torch import nn

from ..op_gate_pin import OPAttentionGatePin


class OPAttentionGateV1Pin(OPAttentionGatePin):
    """F+4+4 -> d_model -> H; one network, one output per attention head.

    Supported-field masks sanitize unavailable values but are NOT MLP inputs.
    The inherited physical eligibility applies the output only within one MOS
    across different actual nets; R/C, ports, graph token and shorts are ungated.
    One forward computes the bias reused across all configured encoder layers.
    """
    version = 'v1'

    def _init_network(self):
        self.score_head = nn.Sequential(
            nn.Linear(len(self.field_names) + 8, self.cfg.dim),
            nn.GELU(), nn.Linear(self.cfg.dim, self.cfg.heads),
        )
        nn.init.normal_(self.score_head[-1].weight, std=1e-3)
        initial = self.settings['initial_gate']
        nn.init.constant_(self.score_head[-1].bias, math.log(initial / (1 - initial)))

    def forward(self, batch, values, supported):
        owners, roles, eligible = self._layout(batch, values, supported)
        safe = torch.where(supported, values, 0.)
        batch_size, devices, fields = safe.shape
        op = safe[:, :, None, None, :].expand(batch_size, devices, 4, 4, fields)
        one_hot = torch.eye(4, dtype=values.dtype, device=values.device)
        query = one_hot[None, None, :, None, :].expand(batch_size, devices, 4, 4, 4)
        key = one_hot[None, None, None, :, :].expand(batch_size, devices, 4, 4, 4)
        logits = self.score_head(torch.cat((op, query, key), dim=-1))
        return self._bias_from_logits(logits, owners, roles, eligible)
