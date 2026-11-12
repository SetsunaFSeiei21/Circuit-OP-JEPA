"""Sample-relative OP gates with dense, input-routed mixtures of linears.

Independent PyTorch implementation of the mathematical operator described by
FlowNet (Feng et al., NeurIPS 2025). Architecture reference: Yutong-Feng/FlowNet,
commit 57ad6128c2a1980d0aa73325a844659262271797, src/layers/mo_linear.py and
src/blocks/hyper_flow.py::FlowBlock.build_mixer. No external source is vendored.

Three mixture layers retain its Dropout -> GELU order. Input/output widths and
the final expert initialization are adapted to the pin attention gate. V2's
normalization, physical-owner alignment, and protected edges are inherited.
"""
import math

import torch
from torch import nn

from .v2_pin import OPAttentionGateV2Pin


class MoLinearPin(nn.Module):
    """A dense convex mixture of E affine maps, routed separately per input.

    Each leading input location owns its routing distribution. All experts
    participate; no token mixing, top-k selection, or balancing loss is added.
    """

    def __init__(self, in_features, out_features, n_experts):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (in_features, out_features, n_experts)):
            raise ValueError('Mixture dimensions and n_experts must be positive integers')
        self.in_features = in_features
        self.out_features = out_features
        self.n_experts = n_experts
        self.router = nn.Linear(in_features, n_experts)
        self.weight = nn.Parameter(torch.empty(n_experts, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n_experts, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        for weight in self.weight.unbind(0):
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.bias)

    def forward(self, inputs):
        routing = self.router(inputs).softmax(dim=-1)
        expert_outputs = torch.einsum('...i,eio->...eo', inputs, self.weight) + self.bias
        return (expert_outputs * routing.unsqueeze(-1)).sum(dim=-2)


class OPAttentionGateV3Pin(OPAttentionGateV2Pin):
    """One three-layer mixture gate shared across the selected attention layers."""
    version = 'v3'

    @property
    def architecture_contract(self):
        return dict(
            kind='mixed_mlp',
            layer_dims=[self.input_dim, self.cfg.dim, self.cfg.dim, self.cfg.heads],
            n_experts=self.settings['n_experts'], dropout=self.settings['mlp_dropout'],
            activation='gelu', block_order='mo_linear_dropout_gelu',
            routing='input_softmax_dense', output='per_head_logits',
            shared_across_layers=True,
            initialization='xavier_hidden_small_random_output_logit_bias',
            source_repository='https://github.com/Yutong-Feng/FlowNet',
            source_commit='57ad6128c2a1980d0aa73325a844659262271797',
        )

    def _init_network(self):
        dim, experts = self.cfg.dim, self.settings['n_experts']
        dropout = self.settings['mlp_dropout']
        self.score_head = nn.Sequential(
            MoLinearPin(self.input_dim, dim, experts), nn.Dropout(dropout), nn.GELU(),
            MoLinearPin(dim, dim, experts), nn.Dropout(dropout), nn.GELU(),
            MoLinearPin(dim, self.cfg.heads, experts),
        )
        # A shared logit offset survives the convex expert mixture. Small,
        # independently sampled expert weights preserve router/OP gradients
        # from the start instead of imposing identical zero-output experts.
        nn.init.normal_(self.score_head[-1].weight, std=1e-3)
        initial = self.settings['initial_gate']
        nn.init.constant_(self.score_head[-1].bias, math.log(initial / (1 - initial)))
