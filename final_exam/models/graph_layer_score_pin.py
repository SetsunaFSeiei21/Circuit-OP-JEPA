"""Topology context, then per-node depth context, for headwise PE selection.

The input at each node contains its entire PE score trajectory plus its graph's
statistics. The first stack attends over graph nodes and retains every block's
output. The second encoder attends over those retained outputs for ONE node at
a time. It does not attend across samples or across different nodes.
"""
import torch
from torch import nn

from device_jepa.model import EncoderBlock


class GraphLayerScoreRouterPin(nn.Module):
    def __init__(self, input_dim, cfg, *, hidden_dim=128, dropout=0.):
        super().__init__()
        self.input_dim, self.layers, self.heads = input_dim, cfg.cross_layers, cfg.heads
        self.dim = cfg.dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, cfg.dim))
        self.graph_blocks = nn.ModuleList(EncoderBlock(cfg, cross=False) for _ in range(self.layers))
        self.layer_encoder = EncoderBlock(cfg, cross=False)
        self.head_projection = nn.Sequential(
            nn.Linear(cfg.dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, cfg.heads))

    def encode_layers(self, graph_states, valid):
        """[B,N,L,D] -> [B,N,L,D], independently encode each node's L tokens."""
        if (graph_states.ndim != 4 or graph_states.shape[2:] != (self.layers, self.dim)
                or valid.shape != graph_states.shape[:2] or valid.dtype != torch.bool
                or valid.device != graph_states.device):
            raise ValueError('Layer context requires aligned [B,N,L,D] states and validity')
        safe = graph_states.masked_fill(~valid[:, :, None, None], 0.)
        tokens = safe.flatten(0, 1)  # B*N,L,D; N is a batch axis, never a sequence axis here.
        token_valid = valid.flatten()[:, None].expand(-1, self.layers)
        # All L slots exist for each valid node. No causal mask and no graph NxN mask.
        # Invalid nodes use finite dummy tokens; their outputs are zeroed by the block.
        output = self.layer_encoder(tokens, None, None, None, token_valid)
        return output.reshape_as(graph_states)

    def project_features(self, features, valid):
        if (features.ndim != 3 or features.shape[-1] != self.input_dim
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool
                or valid.device != features.device):
            raise ValueError('Graph context requires aligned node features and boolean validity')
        safe = features.masked_fill(~valid.unsqueeze(-1), 0.)
        return self.input_projection(safe).masked_fill(~valid.unsqueeze(-1), 0.)

    def forward(self, features, valid, graph_mask, *, return_context=False):
        state = self.project_features(features, valid)
        b, n = valid.shape
        if (graph_mask is None or graph_mask.shape != (b*self.heads, n, n)
                or graph_mask.device != features.device or not graph_mask.is_floating_point()):
            raise ValueError('Graph context requires the original per-head additive circuit mask')
        memories = []
        for block in self.graph_blocks:
            # Reuse the native circuit mask, including self/GRAPH/padding and role bias.
            state = block(state, graph_mask, None, None, valid)
            memories.append(state)
        graph_states = torch.stack(memories, dim=2)  # B,N,L,D, retained in depth order.
        context = self.encode_layers(graph_states, valid)
        logits = self.head_projection(context).transpose(2, 3)  # B,N,H,L
        logits = logits.masked_fill(~valid[:, :, None, None], 0.)
        diagnostics = dict(layer_logits=logits)
        if return_context:
            diagnostics.update(graph_context_states=graph_states, layer_context_states=context)
        return logits, diagnostics


class MultiOrderGraphLayerScoreRouterPin(GraphLayerScoreRouterPin):
    """Independent order MLPs -> sum -> ONE existing graph/depth router."""
    def __init__(self, input_dim, cfg, *, distance, hidden_dim=128, dropout=0.):
        if type(distance) is not int or distance < 1:
            raise ValueError('Aggregation distance must be a positive integer')
        # Initialize all common parameters first, in the same order as v3.
        super().__init__(input_dim, cfg, hidden_dim=hidden_dim, dropout=dropout)
        self.distance = distance
        self.additional_input_projections = nn.ModuleList(nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, cfg.dim)) for _ in range(distance - 1))

    @property
    def order_projections(self):
        return (self.input_projection, *self.additional_input_projections)

    def project_features(self, features, valid):
        if (features.ndim != 4 or features.shape[2:] != (self.distance, self.input_dim)
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool
                or valid.device != features.device):
            raise ValueError('Multi-order context requires aligned [B,N,K,F] features')
        safe = features.masked_fill(~valid[:, :, None, None], 0.)
        state = self.input_projection(safe[:, :, 0])
        for order, projection in enumerate(self.additional_input_projections, start=1):
            state = state + projection(safe[:, :, order])
        # Sum, not average/concat or a new normalization. Mask trainable MLP biases.
        return state.masked_fill(~valid.unsqueeze(-1), 0.)
