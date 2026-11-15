"""Independent per-PE decoders followed by a dense mixture of MLP experts.

Performance queries are metric-specific; OP queries retain physical device
identity. Neither architecture reads measured OP at deployment or uses gates.
"""
from copy import deepcopy

import torch
from torch import nn

from device_jepa.model import ModelConfig
from final_exam.width_registry_pin import WIDTH_MODELS_PIN, width_settings_pin
from .op_gate_pin import DeviceOPPredictorPin, DeviceOPGateModelPin
from .op_transfer_pin import design_input_pin, broadcast_owner_op_pin


class DenseMoEMLPPin(nn.Module):
    """All experts evaluate each metric/device; a per-token softmax mixes outputs."""
    def __init__(self, input_dim, output_dim, settings):
        super().__init__()
        count, hidden = settings['n_experts'], settings['hidden_dim']
        self.router = nn.Linear(input_dim, count)
        self.weight1 = nn.Parameter(torch.empty(count, input_dim, hidden))
        self.bias1 = nn.Parameter(torch.zeros(count, hidden))
        self.weight2 = nn.Parameter(torch.empty(count, hidden, output_dim))
        self.bias2 = nn.Parameter(torch.zeros(count, output_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(settings['dropout'])
        for weights in (self.weight1, self.weight2):
            for weight in weights.unbind(0):
                nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, values):
        routing = self.router(values).softmax(-1)
        hidden = torch.einsum('...i,eih->...eh', values, self.weight1) + self.bias1
        hidden = self.dropout(self.activation(hidden))
        outputs = torch.einsum('...eh,eho->...eo', hidden, self.weight2) + self.bias2
        return (outputs * routing.unsqueeze(-1)).sum(-2)


class DecoderFeaturesPin(nn.Module):
    """A feature decoder, with no intermediate scalar prediction head."""
    def __init__(self, cfg, metric_count=None):
        super().__init__()
        self.query = (nn.Parameter(torch.randn(1, metric_count, cfg.dim) * .02)
                      if metric_count is not None else None)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            cfg.dim, cfg.heads, cfg.decoder_ffn_dim, cfg.dropout,
            activation='relu', batch_first=True), cfg.decoder_layers)

    def forward(self, memory, padding):
        if self.query is not None:
            return self.decoder(self.query.expand(memory.shape[0], -1, -1), memory,
                                memory_key_padding_mask=padding)
        # Query j remains physical device j. Padding is also masked in query
        # self-attention; the GRAPH token is memory only, not an OP target.
        values = self.decoder(memory[:, 1:], memory,
                              tgt_key_padding_mask=padding[:, 1:],
                              memory_key_padding_mask=padding)
        return values.masked_fill(padding[:, 1:, None], 0.)


class MultiPEReadoutPin(nn.Module):
    def __init__(self, model_config, settings, *, output_dim, metric_count=None):
        super().__init__()
        self.cfg = ModelConfig(**model_config)
        self.settings = width_settings_pin(settings, model_config)
        self.metric_count = metric_count
        self.output_dim = output_dim
        self.branches = nn.ModuleList(DecoderFeaturesPin(self.cfg, metric_count)
                                      for _ in range(self.cfg.cross_layers))
        self.moe = DenseMoEMLPPin(self.cfg.cross_layers * self.cfg.dim,
                                  output_dim, self.settings)

    @property
    def architecture_contract(self):
        return dict(readout='per_pe_decoder_then_dense_moe_mlp',
            branch_count=len(self.branches), decoder_layers=self.cfg.decoder_layers,
            feature_dim=self.cfg.dim, concatenated_dim=len(self.branches) * self.cfg.dim,
            independent_decoders=True, pe_order=list(range(1, len(self.branches) + 1)),
            query_kind='metric' if self.metric_count is not None else 'physical_device_memory',
            metric_count=self.metric_count, output_dim=self.output_dim,
            output_projection='shared_encoder_norm_and_projection_per_pe',
            mixture=dict(**self.settings, expert='linear_gelu_dropout_linear',
                         routing='per_token_softmax_dense', auxiliary_loss=False))

    def features(self, memories, padding):
        if len(memories) != len(self.branches):
            raise ValueError('Wide readout requires one memory per PE block')
        if any(memory.shape != memories[0].shape for memory in memories):
            raise ValueError('PE memories must preserve graph/device/pin ordering')
        return torch.cat([branch(memory, padding)
                          for branch, memory in zip(self.branches, memories)], dim=-1)

    def forward(self, memories, padding):
        values = self.moe(self.features(memories, padding))
        if self.metric_count is not None:
            return values.squeeze(-1)
        return values.masked_fill(padding[:, 1:, None], 0.)


class WidthDeviceOPPredictorPin(DeviceOPPredictorPin):
    """Canonical device encoder; per-PE device decoders -> OP-field vector."""
    def __init__(self, config, prep, support, width_config=None):
        super().__init__(config, prep, support)
        self.width_config = width_settings_pin(width_config, self.model_config)
        # Replace the old final-device MLP; it is not kept as an unused parameter.
        self.head = MultiPEReadoutPin(self.model_config, self.width_config,
                                     output_dim=len(self.op_names))

    def forward(self, batch):
        from final_exam.op_gate_data_pin import device_input_pin
        inputs = device_input_pin(batch)
        memories, padding = self.encoder.forward_fusion_memories(inputs)
        prediction = self.head(memories, padding)
        supported = self.topology_mask(inputs) & ~padding[:, 1:, None]
        return torch.where(supported, prediction, 0.), supported


class WidthDecoderModelPin(DeviceOPGateModelPin):
    """Frozen device OP -> pin encoder -> independent metric decoders per PE."""
    def __init__(self, mode, config, prep, predictor, width_config=None):
        if mode not in WIDTH_MODELS_PIN:
            raise ValueError('Unknown width decoder mode')
        wide_op = isinstance(predictor, WidthDeviceOPPredictorPin)
        if wide_op != (mode == 'width_decoder_op_pin'):
            raise ValueError('Width model ID must match the OP predictor architecture')
        super().__init__('device_op_cascade_pin', config, prep, predictor)
        self.mode = mode
        self.width_config = width_settings_pin(width_config, self.model_config)
        self.decoder = MultiPEReadoutPin(self.model_config, self.width_config,
            output_dim=1, metric_count=len(prep['feature_spec']['metric_names']))

    def memory(self, batch):
        inputs = design_input_pin(batch)
        predicted, supported = self.predicted_devices(batch)
        values, support = broadcast_owner_op_pin(predicted, supported, batch)
        inputs.update(op=values, op_mask=support,
                      op_target_mask=support.any(-1) & inputs['device_mask'])
        return self.encoder.forward_fusion_memories(inputs)

    @property
    def architecture_contract(self):
        wide = isinstance(self.predictor, WidthDeviceOPPredictorPin)
        return dict(mode=self.mode, gate=False, requires_measured_op=False,
            predictor_granularity='device', performance_granularity='pin', predictor_frozen=True,
            performance=self.decoder.architecture_contract,
            op_predictor=(self.predictor.head.architecture_contract if wide else
                          dict(readout='last_pe_device_mlp', query_kind='none')))


def describe_width_model_pin(model):
    from .op_transfer_pin import describe_model_pin
    common = dict(model_config=deepcopy(model.model_config), width_config=deepcopy(model.width_config))
    if isinstance(model, WidthDeviceOPPredictorPin):
        return dict(kind='width_device_op_predictor_pin', **common,
                    support=deepcopy(model.support), architecture=model.head.architecture_contract)
    if isinstance(model, WidthDecoderModelPin):
        return dict(kind='width_decoder_pin', mode=model.mode, **common,
                    predictor=describe_model_pin(model.predictor), architecture=model.architecture_contract)
    raise TypeError('Unsupported wide decoder checkpoint')


def rebuild_width_model_pin(recipe, config, prep):
    from .op_transfer_pin import rebuild_model_pin
    cfg, width = recipe['model_config'], recipe['width_config']
    if recipe['kind'] == 'width_device_op_predictor_pin':
        model = WidthDeviceOPPredictorPin(cfg, prep, recipe['support'], width)
        contract = model.head.architecture_contract
    elif recipe['kind'] == 'width_decoder_pin':
        predictor = rebuild_model_pin(recipe['predictor'], cfg, prep)
        model = WidthDecoderModelPin(recipe['mode'], cfg, prep, predictor, width)
        contract = model.architecture_contract
    else:
        raise ValueError('Unknown wide decoder checkpoint kind')
    if recipe.get('architecture') != contract:
        raise ValueError('Wide decoder checkpoint architecture disagrees with saved configuration')
    return model
