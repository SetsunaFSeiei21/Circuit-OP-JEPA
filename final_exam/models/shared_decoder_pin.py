"""Train W_i = B_i A with one owned A per corresponding decoder projection.

The readout owns the shared modules exactly once in ``shared_down``. Projection
consumers hold ordinary Python references to those modules, not additional
registered children. Thus state_dict and optimizers contain one copy, while
deepcopy preserves the references and module-wide device/dtype moves reach A.
"""
from copy import deepcopy
import math

import torch
from torch import nn

from .low_rank_decoder_pin import LowRankLinearPin, LowRankMultiheadAttentionPin


class SharedDownLinearPin(LowRankLinearPin):
    """An independent B/bias consuming a down projection owned by the readout."""
    def __init__(self, down, out_features, bias=True, *, qkv=False):
        nn.Module.__init__(self)
        if not isinstance(down, nn.Linear) or down.bias is not None:
            raise ValueError('Shared down factor must be a bias-free Linear')
        self.in_features, self.out_features, self.rank = down.in_features, out_features, down.out_features
        self.qkv = qkv
        # Bypass Module registration intentionally; the readout owns A.
        self.__dict__['_down_reference'] = down
        self.up = nn.Linear(self.rank, out_features, bias=bias,
                            device=down.weight.device, dtype=down.weight.dtype)
        with torch.no_grad():
            self.up.weight.mul_(math.sqrt(4.5 if qkv else 3.0))
            if self.up.bias is not None:
                if qkv:
                    nn.init.zeros_(self.up.bias)
                else:
                    nn.init.uniform_(self.up.bias, -1 / math.sqrt(self.in_features),
                                     1 / math.sqrt(self.in_features))

    @property
    def down(self):
        return self.__dict__['_down_reference']

    def reset_parameters(self):
        # A must only be initialized by its single owner, never per consumer.
        self.up.reset_parameters()
        with torch.no_grad():
            self.up.weight.mul_(math.sqrt(4.5 if self.qkv else 3.0))
            if self.up.bias is not None:
                if self.qkv:
                    nn.init.zeros_(self.up.bias)
                else:
                    nn.init.uniform_(self.up.bias, -1 / math.sqrt(self.in_features),
                                     1 / math.sqrt(self.in_features))


class SharedDownAttentionPin(LowRankMultiheadAttentionPin):
    """SDPA with selected shared-factor projections and remaining dense ones."""
    def __init__(self, source, shared):
        nn.Module.__init__(self)
        if not isinstance(source, nn.MultiheadAttention) or not source.batch_first:
            raise ValueError('Shared decoder requires batch-first dense source attention')
        if not source._qkv_same_embed_dim or source.bias_k is not None or source.add_zero_attn:
            raise ValueError('Unsupported decoder attention source')
        self.embed_dim, self.num_heads = source.embed_dim, source.num_heads
        self.head_dim, self.dropout = source.head_dim, source.dropout
        self.batch_first, self.kdim, self.vdim = True, source.kdim, source.vdim
        self._qkv_same_embed_dim = True
        self.rank = next(iter(shared.values())).out_features
        factory = dict(device=source.out_proj.weight.device, dtype=source.out_proj.weight.dtype)
        weights = source.in_proj_weight.chunk(3)
        biases = source.in_proj_bias.chunk(3) if source.in_proj_bias is not None else (None,) * 3
        for part, weight, bias in zip(('q', 'k', 'v'), weights, biases):
            if part in shared:
                module = SharedDownLinearPin(shared[part], self.embed_dim, bias is not None, qkv=True)
            else:
                module = nn.Linear(self.embed_dim, self.embed_dim, bias=bias is not None, **factory)
                with torch.no_grad():
                    module.weight.copy_(weight)
                    if bias is not None:
                        module.bias.copy_(bias)
            setattr(self, part + '_proj', module)
        if 'out' in shared:
            self.out_proj = SharedDownLinearPin(shared['out'], self.embed_dim,
                                                source.out_proj.bias is not None)
            if self.out_proj.up.bias is not None:
                nn.init.zeros_(self.out_proj.up.bias)
        else:
            self.out_proj = source.out_proj
        self.train(source.training)


def share_decoder_down_pin(readout, spec):
    """Replace selected projections before training; no SVD or dense leftovers."""
    if hasattr(readout, 'shared_down'):
        raise ValueError('Shared decoder factors have already been installed')
    branches = readout.branches
    if len(branches) != spec['branch_count'] or any(
            len(branch.decoder.layers) != spec['decoder_layers'] for branch in branches):
        raise ValueError('Shared decoder specification disagrees with branch/layer counts')
    # Validate every selected source before mutating the readout.
    for branch in branches:
        for layer in branch.decoder.layers:
            for role in spec['shared_roles']:
                name = role.split('.')[0]
                module = getattr(layer, name)
                if not isinstance(module, nn.Linear if name.startswith('linear') else nn.MultiheadAttention):
                    raise ValueError(f'Shared decoder expects an unmodified source for {role}')
    shared = nn.ModuleDict()
    for layer_index in range(spec['decoder_layers']):
        source = branches[0].decoder.layers[layer_index]
        for role in spec['shared_roles']:
            ins = spec['decoder_ffn_dim'] if role == 'linear2' else spec['dim']
            rank = spec['ffn_rank'] if role.startswith('linear') else spec['attention_rank']
            parameter = next(getattr(source, role.split('.')[0]).parameters())
            key = f'layer_{layer_index + 1}__{role.replace(".", "__")}'
            shared[key] = nn.Linear(ins, rank, bias=False, device=parameter.device, dtype=parameter.dtype)
    readout.shared_down = shared
    readout.shared_down_spec = deepcopy(spec)
    for branch in branches:
        for layer_index, layer in enumerate(branch.decoder.layers):
            def down(role):
                return shared[f'layer_{layer_index + 1}__{role.replace(".", "__")}']
            for name in ('self_attn', 'multihead_attn'):
                chosen = {role.split('.')[1]: down(role) for role in spec['shared_roles']
                          if role.startswith(name + '.')}
                if chosen:
                    setattr(layer, name, SharedDownAttentionPin(getattr(layer, name), chosen))
            for name in ('linear1', 'linear2'):
                if name in spec['shared_roles']:
                    old = getattr(layer, name)
                    replacement = SharedDownLinearPin(down(name), old.out_features, old.bias is not None)
                    # Retain unchanged bias initialization; only weights change.
                    if old.bias is not None:
                        with torch.no_grad():
                            replacement.up.bias.copy_(old.bias)
                    replacement.train(old.training)
                    setattr(layer, name, replacement)
    return readout
