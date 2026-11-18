"""Dynamic residual-stream mixing for pin PE blocks, after Feng et al.'s FlowNet.

Reference: Yutong-Feng/FlowNet@57ad6128, src/layers/hyper_connect.py and
src/blocks/hyper_flow.py. This adaptation keeps the pin encoder's post-norm
sublayers and adds a separate mixer for the parameter/OP cross-attention.
"""
import torch
from torch import nn


class HyperConnectionPin(nn.Module):
    """Mix full-width streams within one token. Alpha and beta are unconstrained."""

    def __init__(self, dim, streams, layer_index):
        super().__init__()
        # Column zero reads a rotating stream. The remaining columns carry the identity.
        alpha = torch.zeros(streams, streams + 1)
        alpha[layer_index % streams, 0] = 1.
        alpha[:, 1:] = torch.eye(streams)
        self.static_alpha = nn.Parameter(alpha)
        self.static_beta = nn.Parameter(torch.ones(streams))
        self.norm = nn.LayerNorm(dim)
        self.alpha_projection = nn.Parameter(torch.zeros(dim, streams + 1))
        self.beta_projection = nn.Parameter(torch.zeros(dim))
        self.alpha_scale = nn.Parameter(torch.full((1,), 0.01))
        self.beta_scale = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, streams):
        """Read one [B,N,D] sublayer input plus its bypass and write coefficients."""
        normalized = self.norm(streams)
        alpha = self.static_alpha + self.alpha_scale * (normalized @ self.alpha_projection).tanh()
        beta = self.static_beta + self.beta_scale * (normalized @ self.beta_projection).tanh()
        mixed = alpha.transpose(-2, -1) @ streams
        return mixed[..., 0, :], mixed[..., 1:, :], beta

    @staticmethod
    def write(bypass, beta, update):
        return bypass + beta.unsqueeze(-1) * update.unsqueeze(-2)


class HyperConnectionPEBlockPin(nn.Module):
    """One native PE branch computation per sublayer, with R persistent residual streams."""

    def __init__(self, block, streams, layer_index):
        super().__init__()
        self.block = block
        self.streams = streams
        dim = block.self_attn.embed_dim
        self.self_connection = HyperConnectionPin(dim, streams, layer_index)
        self.cross_connection = HyperConnectionPin(dim, streams, layer_index)
        self.ffn_connection = HyperConnectionPin(dim, streams, layer_index)

    @property
    def self_attn(self):
        return self.block.self_attn

    @staticmethod
    def readout_tokens(state):
        return state.sum(dim=-2)

    def forward(self, state, self_mask, parameter_tokens, cross_mask, valid, *, attention_observer=None):
        if state.ndim == 3:  # The SE prefix supplies one vector per pin at the first PE.
            state = state.unsqueeze(-2).expand(-1, -1, self.streams, -1)
        block = self.block
        x, bypass, beta = self.self_connection(state)
        update = block.self_attention_update(x, self_mask, valid, attention_observer)
        state = block.self_norm(self.self_connection.write(bypass, beta, block.drop(update)))

        x, bypass, beta = self.cross_connection(state)
        update = block.cross_attn(x, parameter_tokens, parameter_tokens, attn_mask=cross_mask,
                                  need_weights=False)[0]
        state = block.cross_norm(self.cross_connection.write(bypass, beta, block.drop(update)))

        x, bypass, beta = self.ffn_connection(state)
        state = block.ffn_norm(self.ffn_connection.write(bypass, beta, block.drop(block.ffn(x))))
        return state.masked_fill(~valid[:, :, None, None], 0.)
