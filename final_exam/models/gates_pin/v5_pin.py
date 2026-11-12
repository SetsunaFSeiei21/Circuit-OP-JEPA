"""Independent signed M-MLP gate for every fusion self-attention after PE1."""
from copy import deepcopy

from torch import nn

from .v4_pin import OPAttentionGateV4Pin


class OPAttentionGateV5Pin(OPAttentionGateV4Pin):
    version = 'v5'

    @property
    def architecture_contract(self):
        return dict(super().architecture_contract,
            shared_across_layers=False,
            independent_gate_count=self.cfg.cross_layers - 1,
            layer_initialization='identical_copies_independent_storage')

    def _init_network(self):
        first = self._new_score_head()
        self.score_heads = nn.ModuleList(
            [first] + [deepcopy(first) for _ in range(self.cfg.cross_layers - 2)])

    def iter_layer_score_heads(self):
        return enumerate(self.score_heads, start=2)

    def forward(self, batch, values, supported):
        owners, roles, eligible = self._layout(batch, values, supported)
        inputs = self.pair_inputs(batch, values, supported)
        return (None,) + tuple(self._bias_from_logits(network(inputs), owners, roles, eligible)
                              for network in self.score_heads)
