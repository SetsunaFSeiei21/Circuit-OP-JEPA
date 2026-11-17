"""Per-node Top-2 expert routing of PE memories, with score-only stop-grad.

The original graph/attention masks and hidden-state chain are unchanged. Score
statistics are pre-dropout observations of the same Q/K and masks. One final
native graph block and one existing metric decoder consume the fused memory.
"""
from copy import deepcopy

import torch
from torch import nn

from device_jepa.model import EncoderBlock, ModelConfig
from final_exam.lasd_moe_registry_pin import (
    LASD_MOE_MODELS_PIN, lasd_moe_settings_pin, lasd_moe_spec_pin,
)
from .attention_score_pin import AttentionScoreExtractorPin, layer_score_features_pin
from .op_gate_pin import DeviceOPGateModelPin, DeviceOPPredictorPin
from .op_transfer_pin import design_input_pin, broadcast_owner_op_pin


def score_dynamics_pin(scores, *, node_ids=None):
    """[B,N,L,H,2] -> [B,N,4LH], [B,N,L,H,2]; no detach or head averaging."""
    if scores.ndim != 5 or scores.shape[-1] != 2 or scores.shape[2] < 1:
        raise ValueError('LASD-MoE scores must have shape [B,N,L,H,2] with L >= 1')
    _, differences = layer_score_features_pin(tuple(scores.unbind(2)), dynamics=True,
                                               node_ids=node_ids)
    differences = torch.stack(differences, dim=2)
    return torch.cat((scores, differences), dim=-1).flatten(2), differences


class Top2LayerMoEPin(nn.Module):
    """Four independent experts, computing only the two selected routes per node.

    Selection is discrete as in standard Top-k routing; selected router logits
    and expert outputs remain differentiable. There is no capacity truncation,
    token dropping, expert balancing penalty or hidden feature input to routing.
    """
    def __init__(self, input_dim, layers, settings):
        super().__init__()
        self.settings = lasd_moe_settings_pin(settings)
        self.input_dim, self.layers = input_dim, layers
        count, hidden = self.settings['n_experts'], self.settings['hidden_dim']
        self.router = nn.Linear(input_dim, count)
        self.experts = nn.ModuleList(nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(self.settings['dropout']),
            nn.Linear(hidden, layers)) for _ in range(count))

    def forward(self, features, valid):
        if (features.ndim != 3 or features.shape[-1] != self.input_dim
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool
                or valid.device != features.device):
            raise ValueError('MoE requires aligned node features and boolean validity')
        safe = features.masked_fill(~valid.unsqueeze(-1), 0.)
        logits = self.router(safe)
        chosen_logits, chosen = logits.topk(self.settings['top_k'], dim=-1)
        weights = chosen_logits.softmax(-1).masked_fill(~valid.unsqueeze(-1), 0.)
        flat = safe.flatten(0, 1)
        routes, route_weights = chosen.flatten(0, 1), weights.flatten(0, 1)
        flat_valid = valid.flatten()
        output = safe.new_zeros(flat.shape[0], self.layers)
        for index, expert in enumerate(self.experts):
            nodes, slots = torch.where((routes == index) & flat_valid[:, None])
            # Top-k returns distinct expert IDs, so node indices here are unique.
            # Unique-index writes avoid nondeterministic repeated-index scatter.
            values = expert(flat[nodes]) * route_weights[nodes, slots, None]
            contribution = torch.zeros_like(output)
            contribution[nodes] = values.to(contribution.dtype)
            output = output + contribution
        output = output.reshape(*features.shape[:2], self.layers)
        return output, dict(
            router_probs=logits.softmax(-1).masked_fill(~valid.unsqueeze(-1), 0.),
            selected_experts=chosen.masked_fill(~valid.unsqueeze(-1), -1),
            selected_weights=weights)


class LASDMoEModelPin(DeviceOPGateModelPin):
    """Frozen ordinary device OP -> pin encoder -> PE-only score-guided layer fusion."""
    def __init__(self, mode, config, prep, predictor, lasd_moe_config=None):
        if mode not in LASD_MOE_MODELS_PIN:
            raise ValueError('Unknown LASD-MoE model')
        if type(predictor) is not DeviceOPPredictorPin:
            raise ValueError('LASD-MoE requires the ordinary device-level OP predictor')
        # All original modules initialize first; the pair's only difference is
        # a forward stop-gradient operation and not a parameter/buffer/RNG change.
        super().__init__('device_op_cascade_pin', config, prep, predictor)
        self.mode = mode
        self.lasd_moe_config = lasd_moe_settings_pin(lasd_moe_config)
        self.lasd_moe_spec = lasd_moe_spec_pin(mode, self.lasd_moe_config, self.model_config)
        cfg = ModelConfig(**self.model_config)
        self.score_extractor = AttentionScoreExtractorPin(
            self.lasd_moe_config['score_chunk_size'], self.lasd_moe_config['checkpoint_scores'])
        self.layer_fusion = Top2LayerMoEPin(4*cfg.cross_layers*cfg.heads,
                                           cfg.cross_layers, self.lasd_moe_config)
        self.final_graph_encoder = EncoderBlock(cfg, cross=False)

    @property
    def architecture_contract(self):
        return deepcopy(self.lasd_moe_spec)

    def memory(self, batch, *, return_scores=False):
        inputs = design_input_pin(batch)
        predicted, supported = self.predicted_devices(batch)
        values, support = broadcast_owner_op_pin(predicted, supported, batch)
        inputs.update(op=values, op_mask=support,
                      op_target_mask=support.any(-1) & inputs['device_mask'])
        observed = []
        def observe(index, attention, hidden, mask, valid):
            if index != len(observed):
                raise ValueError('LASD-MoE PE attention layers are not aligned')
            observed.append(self.score_extractor(attention, hidden, mask, valid))
        memories, padding, masks = self.encoder.forward_fusion_memories(
            inputs, attention_observer=observe, return_self_masks=True)
        if len(memories) != len(observed) or len(memories) != self.model_config['cross_layers']:
            raise ValueError('LASD-MoE requires one score and memory per PE layer')
        if any(memory.shape != memories[0].shape for memory in memories):
            raise ValueError('Encoder memories must preserve the same node order and shape')
        scores = torch.stack(observed, dim=2)
        features, differences = score_dynamics_pin(scores)
        # The only gradient difference between the two model IDs. Never detach
        # memories, router/expert parameters, selected weights or layer weights.
        moe_input = features.detach() if self.lasd_moe_spec['score_detach'] else features
        valid = ~padding
        logits, routing = self.layer_fusion(moe_input, valid)
        weights = logits.softmax(-1).masked_fill(padding.unsqueeze(-1), 0.)
        fused = (torch.stack(memories, dim=2) * weights.unsqueeze(-1)).sum(2)
        memory = self.final_graph_encoder(fused, masks[0], None, None, valid)
        if return_scores:
            return memory, padding, dict(scores=scores, differences=differences,
                score_features=features, layer_weights=weights, valid=valid, **routing)
        return memory, padding

    def forward(self, batch, *, return_scores=False):
        if return_scores:
            memory, padding, diagnostics = self.memory(batch, return_scores=True)
            return self.decoder(memory, padding), diagnostics
        memory, padding = self.memory(batch)
        return self.decoder(memory, padding)


def describe_lasd_moe_model_pin(model):
    from .op_transfer_pin import describe_model_pin
    if not isinstance(model, LASDMoEModelPin):
        raise TypeError('Unsupported LASD-MoE checkpoint')
    return dict(kind='lasd_moe_pin', mode=model.mode,
                model_config=deepcopy(model.model_config),
                lasd_moe_config=deepcopy(model.lasd_moe_config),
                predictor=describe_model_pin(model.predictor),
                architecture=model.architecture_contract)


def rebuild_lasd_moe_model_pin(recipe, config, prep):
    from .op_transfer_pin import rebuild_model_pin
    if recipe.get('kind') != 'lasd_moe_pin' or 'lasd_moe_config' not in recipe:
        raise ValueError('LASD-MoE checkpoint requires its saved score fusion configuration')
    cfg = recipe['model_config']
    predictor = rebuild_model_pin(recipe['predictor'], cfg, prep)
    model = LASDMoEModelPin(recipe['mode'], cfg, prep, predictor, recipe['lasd_moe_config'])
    if recipe.get('architecture') != model.architecture_contract:
        raise ValueError('LASD-MoE architecture contract disagrees with saved configuration')
    return model
