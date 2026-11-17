"""Independent per-PE decoders followed by a dense mixture of MLP experts.

Performance queries are metric-specific; OP queries retain physical device
identity. Neither architecture reads measured OP at deployment or uses gates.
"""
from copy import deepcopy

import torch
from torch import nn

from device_jepa.model import ModelConfig
from final_exam.width_registry_pin import (
    ALL_WIDTH_MODELS_PIN as WIDTH_MODELS_PIN, WIDTH_DETACH_MODELS_PIN,
    width_settings_pin, detach_memory_spec_pin,
)
from final_exam.low_rank_registry_pin import LOW_RANK_MODELS_PIN, low_rank_settings_pin, low_rank_spec_pin
from final_exam.shared_decoder_registry_pin import (
    SHARED_DECODER_MODELS_PIN, shared_decoder_settings_pin, shared_decoder_spec_pin,
)
from final_exam.lasd_registry_pin import (
    ALL_LASD_MODELS_PIN as LASD_MODELS_PIN, LASD_DETACH_MODELS_PIN,
    lasd_settings_pin, lasd_spec_pin,
)
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
    def __init__(self, model_config, settings, *, output_dim, metric_count=None,
                 detach_shallow_memory=False):
        super().__init__()
        self.cfg = ModelConfig(**model_config)
        self.settings = width_settings_pin(settings, model_config)
        self.metric_count = metric_count
        self.output_dim = output_dim
        self.detach_shallow_memory = detach_shallow_memory
        self.branches = nn.ModuleList(DecoderFeaturesPin(self.cfg, metric_count)
                                      for _ in range(self.cfg.cross_layers))
        self.moe = DenseMoEMLPPin(self.cfg.cross_layers * self.cfg.dim,
                                  output_dim, self.settings)

    @property
    def architecture_contract(self):
        contract = dict(readout='per_pe_decoder_then_dense_moe_mlp',
            branch_count=len(self.branches), decoder_layers=self.cfg.decoder_layers,
            feature_dim=self.cfg.dim, concatenated_dim=len(self.branches) * self.cfg.dim,
            independent_decoders=True, pe_order=list(range(1, len(self.branches) + 1)),
            query_kind='metric' if self.metric_count is not None else 'physical_device_memory',
            metric_count=self.metric_count, output_dim=self.output_dim,
            output_projection='shared_encoder_norm_and_projection_per_pe',
            mixture=dict(**self.settings, expert='linear_gelu_dropout_linear',
                         routing='per_token_softmax_dense', auxiliary_loss=False))
        if hasattr(self, 'shared_down_spec'):
            contract['independent_decoders'] = False
            contract['shared_down_basis'] = deepcopy(self.shared_down_spec)
        if hasattr(self, 'lasd_spec'):
            contract['lasd'] = deepcopy(self.lasd_spec)
        if self.detach_shallow_memory:
            contract['memory_gradients'] = detach_memory_spec_pin(
                len(self.branches), getattr(self, 'lasd_spec', None))
        return contract

    def features(self, memories, padding, score_features=None):
        if len(memories) != len(self.branches):
            raise ValueError('Wide readout requires one memory per PE block')
        if any(memory.shape != memories[0].shape for memory in memories):
            raise ValueError('PE memories must preserve graph/device/pin ordering')
        if self.detach_shallow_memory:
            # Cut readout edges only, BEFORE adapters so every adapter still
            # trains. The original tensors fed the next PE normally. Detach
            # scores as well; otherwise their Q/K path bypasses memory detach.
            memories = tuple(memory.detach() if i < len(memories) - 1 else memory
                             for i, memory in enumerate(memories))
            if score_features is not None:
                score_features = tuple(score.detach() if i < len(memories) - 1 else score
                                       for i, score in enumerate(score_features))
        if hasattr(self, 'score_adapters'):
            if score_features is None or len(score_features) != len(memories):
                raise ValueError('LASD requires one score feature tensor per PE')
            memories = tuple(adapter(memory, score, padding) for adapter, memory, score in
                             zip(self.score_adapters, memories, score_features))
        elif score_features is not None:
            raise ValueError('Score features require an enabled LASD readout')
        return torch.cat([branch(memory, padding)
                          for branch, memory in zip(self.branches, memories)], dim=-1)

    def forward(self, memories, padding, score_features=None):
        values = self.moe(self.features(memories, padding, score_features))
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
    def __init__(self, mode, config, prep, predictor, width_config=None, low_rank_config=None,
                 shared_decoder_config=None, lasd_config=None):
        if mode not in WIDTH_MODELS_PIN:
            raise ValueError('Unknown width decoder mode')
        wide_op = isinstance(predictor, WidthDeviceOPPredictorPin)
        if wide_op != (mode == 'width_decoder_op_pin'):
            raise ValueError('Width model ID must match the OP predictor architecture')
        super().__init__('device_op_cascade_pin', config, prep, predictor)
        self.mode = mode
        self.width_config = width_settings_pin(width_config, self.model_config)
        self.decoder = MultiPEReadoutPin(self.model_config, self.width_config,
            output_dim=1, metric_count=len(prep['feature_spec']['metric_names']),
            detach_shallow_memory=mode in WIDTH_DETACH_MODELS_PIN + LASD_DETACH_MODELS_PIN)
        self.low_rank_config = (low_rank_settings_pin(low_rank_config)
                                if mode in LOW_RANK_MODELS_PIN else None)
        self.low_rank_spec = low_rank_spec_pin(mode, self.low_rank_config, self.model_config)
        if self.low_rank_spec is not None:
            from .low_rank_decoder_pin import factorize_decoder_pin
            for branch in self.decoder.branches:
                factorize_decoder_pin(branch.decoder,
                    attention_rank=self.low_rank_spec['attention_rank'],
                    ffn_rank=self.low_rank_spec['ffn_rank'])
        self.shared_decoder_config = (shared_decoder_settings_pin(shared_decoder_config)
                                      if mode in SHARED_DECODER_MODELS_PIN else None)
        self.shared_decoder_spec = shared_decoder_spec_pin(mode, self.shared_decoder_config,
                                                           self.model_config)
        if self.shared_decoder_spec is not None:
            from .shared_decoder_pin import share_decoder_down_pin
            share_decoder_down_pin(self.decoder, self.shared_decoder_spec)
        self.lasd_config = lasd_settings_pin(lasd_config) if mode in LASD_MODELS_PIN else None
        self.lasd_spec = lasd_spec_pin(mode, self.lasd_config, self.model_config)
        if self.lasd_spec is not None:
            self.decoder.lasd_spec = deepcopy(self.lasd_spec)
            if self.lasd_spec['enabled']:
                from .attention_score_pin import AttentionScoreExtractorPin, ScoreMemoryAdapterPin
                self.score_extractor = AttentionScoreExtractorPin(
                    self.lasd_config['score_chunk_size'], self.lasd_config['checkpoint_scores'])
                # Added after all original modules, preserving their initialization
                # under the same seed (including independent metric queries/MoE).
                self.decoder.score_adapters = nn.ModuleList(ScoreMemoryAdapterPin(
                    self.model_config['dim'], self.lasd_spec['score_dim'],
                    self.lasd_config['hidden_dim'], self.lasd_config['dropout'])
                    for _ in self.decoder.branches)

    def memory(self, batch, *, attention_observer=None):
        inputs = design_input_pin(batch)
        predicted, supported = self.predicted_devices(batch)
        values, support = broadcast_owner_op_pin(predicted, supported, batch)
        inputs.update(op=values, op_mask=support,
                      op_target_mask=support.any(-1) & inputs['device_mask'])
        return self.encoder.forward_fusion_memories(inputs, attention_observer=attention_observer)

    def forward(self, batch, *, return_scores=False):
        if self.lasd_spec is None or not self.lasd_spec['enabled']:
            if return_scores:
                raise ValueError('Score diagnostics require an enabled LASD model')
            return super().forward(batch)
        from .attention_score_pin import layer_score_features_pin
        scores = []
        def observe(index, attention, values, mask, valid):
            if index != len(scores):
                raise ValueError('LASD PE order must match the original encoder sequence')
            scores.append(self.score_extractor(attention, values, mask, valid))
        memories, padding = self.memory(batch, attention_observer=observe)
        features, differences = layer_score_features_pin(scores, dynamics=self.lasd_spec['dynamics'],
            detach_previous=self.mode in LASD_DETACH_MODELS_PIN)
        prediction = self.decoder(memories, padding, features)
        if return_scores:
            # Raw scores remain live observations; differences follow the
            # model's gradient policy. Serialization detaches diagnostic data.
            return prediction, dict(scores=torch.stack(scores), differences=torch.stack(differences),
                                    valid=~padding)
        return prediction

    @property
    def architecture_contract(self):
        wide = isinstance(self.predictor, WidthDeviceOPPredictorPin)
        performance = self.decoder.architecture_contract
        if self.low_rank_spec is not None:
            performance['low_rank'] = deepcopy(self.low_rank_spec)
        if self.shared_decoder_spec is not None:
            performance['independent_decoders'] = False
            performance['shared_down_basis'] = deepcopy(self.shared_decoder_spec)
        return dict(mode=self.mode, gate=False, requires_measured_op=False,
            predictor_granularity='device', performance_granularity='pin', predictor_frozen=True,
            performance=performance,
            op_predictor=(self.predictor.head.architecture_contract if wide else
                          dict(readout='last_pe_device_mlp', query_kind='none')))


def describe_width_model_pin(model):
    from .op_transfer_pin import describe_model_pin
    common = dict(model_config=deepcopy(model.model_config), width_config=deepcopy(model.width_config))
    if isinstance(model, WidthDeviceOPPredictorPin):
        return dict(kind='width_device_op_predictor_pin', **common,
                    support=deepcopy(model.support), architecture=model.head.architecture_contract)
    if isinstance(model, WidthDecoderModelPin):
        if model.low_rank_config is not None:
            common['low_rank_config'] = deepcopy(model.low_rank_config)
        if model.shared_decoder_config is not None:
            common['shared_decoder_config'] = deepcopy(model.shared_decoder_config)
        if model.lasd_config is not None:
            common['lasd_config'] = deepcopy(model.lasd_config)
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
        if recipe['mode'] in LOW_RANK_MODELS_PIN and 'low_rank_config' not in recipe:
            raise ValueError('Low-rank checkpoint requires its saved rank configuration')
        if recipe['mode'] in SHARED_DECODER_MODELS_PIN and 'shared_decoder_config' not in recipe:
            raise ValueError('Shared decoder checkpoint requires its saved rank configuration')
        if recipe['mode'] in LASD_MODELS_PIN and 'lasd_config' not in recipe:
            raise ValueError('LASD checkpoint requires its saved score configuration')
        model = WidthDecoderModelPin(recipe['mode'], cfg, prep, predictor, width,
                                     recipe.get('low_rank_config'), recipe.get('shared_decoder_config'),
                                     recipe.get('lasd_config'))
        contract = model.architecture_contract
    else:
        raise ValueError('Unknown wide decoder checkpoint kind')
    if recipe.get('architecture') != contract:
        raise ValueError('Wide decoder checkpoint architecture disagrees with saved configuration')
    return model
