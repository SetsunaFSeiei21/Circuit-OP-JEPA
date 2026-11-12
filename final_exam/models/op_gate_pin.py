"""Device-OP-guided pin attention, with matched feature and gate-only controls.

The OP predictor encodes canonical *devices*, not pooled pin representations.
Every deployed cascade uses its frozen predictions. Oracle gates are explicit
diagnostics requiring measured OP; they never replace the prediction features.
The ZeroSim-family structure/fusion/metric-query operators remain unchanged.
Only encoder self-attention logits receive a finite, directional soft penalty.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from final_exam.gate_version_registry_pin import VERSIONED_JOINT_GATE_MODELS_PIN, gate_version_pin
from device_jepa.model import ModelConfig
from device_jepa.training import state_digest
from experiments.op_transfer.models import OPPredictor, design_encoder, freeze
from .common_pin import require_pin_batch
from .op_transfer_pin import (
    _config, _owner_layout, _pin_prep, broadcast_owner_op_pin, decoder_pin,
    design_encoder_pin, design_input_pin, op_encoder_pin,
)


GATE_MODELS_PIN = (
    'device_op_cascade_pin', 'device_op_gate_pin',
    'device_op_gate_oracle_pin', 'device_op_gate_shuffle_pin',
    'device_op_gate_only_base_pin', 'device_op_gate_only_pin',
    'device_op_gate_only_oracle_pin', 'device_op_gate_only_shuffle_pin',
) + VERSIONED_JOINT_GATE_MODELS_PIN
GATE_ORACLES_PIN = frozenset((
    'device_op_gate_oracle_pin', 'device_op_gate_only_oracle_pin',
))
_FEATURE_MODES = frozenset(GATE_MODELS_PIN[:4] + VERSIONED_JOINT_GATE_MODELS_PIN)
_BASE_MODES = frozenset(('device_op_cascade_pin', 'device_op_gate_only_base_pin'))


def validate_gate_config_pin(value=None, *, version=None):
    """Validate the small gate namespace independently of model configuration."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError('op_gate_pin must be an object')
    defaults = dict(epsilon=1e-4, strength=1.0, initial_gate=0.99)
    allowed = set(defaults) | {'norm_eps', 'n_experts', 'mlp_dropout'}
    if set(value) - allowed:
        raise ValueError('Unknown op_gate_pin keys: ' + ', '.join(sorted(set(value)-allowed)))
    # Preserve legacy configuration/recipe dictionaries exactly unless the new
    # version (or an explicit caller) needs sample-normalization settings.
    if version in ('v2', 'v3'):
        defaults['norm_eps'] = 1e-5
    if version == 'v3':
        defaults.update(n_experts=16, mlp_dropout=0.1)
    result = {**defaults, **value}
    for name, number in result.items():
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError(f'op_gate_pin.{name} must be finite numeric data')
    if not 0 < result['epsilon'] < 1:
        raise ValueError('op_gate_pin.epsilon must be in (0, 1)')
    if not 0 < result['initial_gate'] < 1:
        raise ValueError('op_gate_pin.initial_gate must be in (0, 1)')
    if result['strength'] < 0:
        raise ValueError('op_gate_pin.strength must be nonnegative')
    if 'norm_eps' in result and result['norm_eps'] <= 0:
        raise ValueError('op_gate_pin.norm_eps must be positive')
    if 'n_experts' in result and (type(result['n_experts']) is not int or result['n_experts'] < 1):
        raise ValueError('op_gate_pin.n_experts must be a positive integer')
    if 'mlp_dropout' in result and not 0 <= result['mlp_dropout'] < 1:
        raise ValueError('op_gate_pin.mlp_dropout must be in [0, 1)')
    return {name: int(number) if name == 'n_experts' else float(number)
            for name, number in result.items()}


class DeviceOPPredictorPin(OPPredictor):
    """True D-token ZeroSim encoder with one OP prediction per physical device.

The suffix identifies the downstream pin experiment and its artifact family.
Its encoder itself always receives the original device vocabulary and graph.
"""
    granularity = 'device_to_pin'

    def __init__(self, config, prep, support):
        original = _pin_prep(prep)
        self.model_config = _config(config)
        super().__init__(design_encoder(self.model_config, original), original, support)

    def forward(self, batch):
        from final_exam.op_gate_data_pin import device_input_pin
        return super().forward(device_input_pin(batch))

    def predict_devices(self, batch):
        # Deliberately ignore any cached predictions; this is the fresh-model
        # path used while pretraining and while constructing frozen caches.
        return self.forward(batch)


class OPAttentionGatePin(nn.Module):
    """Shared d_model OP embedding, ordered terminal roles, per-head scores.

An F-field OP vector and its TRAIN-supported mask are first embedded to the
same d_model width as the encoder. A shared score trunk combines that embedding
with ordered query/key terminal embeddings. Its H output coordinates supply
different scores to the H attention heads; it is not H separate OP predictors.
"""
    version = 'v0'

    def __init__(self, config, prep, gate_config=None):
        super().__init__()
        self.cfg = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        original = _pin_prep(prep)
        self.settings = validate_gate_config_pin(gate_config, version=self.version)
        self.field_names = list(original['feature_spec']['op_names'])
        if not self.field_names:
            raise ValueError('OP attention requires at least one trained OP field')
        names = original['feature_spec']['terminal_names']
        role_table = [(['d', 'g', 's', 'b'].index(name) if name in ('d','g','s','b') else -1)
                      for name in names]
        self.register_buffer('terminal_roles', torch.tensor(role_table, dtype=torch.long))
        kinds = original['device_vocab']
        mos = torch.zeros(len(kinds), dtype=torch.bool)
        for name, index in kinds.items():
            mos[index] = name.lower() in ('nmos', 'pmos')
        self.register_buffer('mos_types', mos)
        self._init_network()

    def _init_network(self):
        dim, fields = self.cfg.dim, len(self.field_names)
        self.op_embedding = nn.Sequential(nn.Linear(2*fields, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query_roles = nn.Embedding(4, dim)
        self.key_roles = nn.Embedding(4, dim)
        self.score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                        nn.Linear(dim, self.cfg.heads))
        # Start near the ungated model, but retain nonzero OP-dependent
        # derivatives at initialization. Every ordered pair/head can diverge.
        nn.init.normal_(self.score_head[-1].weight, std=1e-3)
        initial = self.settings['initial_gate']
        nn.init.constant_(self.score_head[-1].bias, math.log(initial/(1-initial)))

    @classmethod
    def from_gate_layout(cls, source, gate_config=None):
        """Create a fresh version using identical physical layout metadata.

        Only used when a shared *ungated* base is copied into a new experiment.
        Encoder, decoder and predictor weights are never modified here.
        """
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.cfg = deepcopy(source.cfg)
        result.settings = validate_gate_config_pin(
            {**source.settings, **(gate_config or {})}, version=cls.version)
        result.field_names = list(source.field_names)
        for name in ('terminal_roles', 'mos_types'):
            result.register_buffer(name, getattr(source, name).detach().clone())
        result._init_network()
        reference = next(source.parameters())
        return result.to(device=reference.device, dtype=reference.dtype).train(source.training)

    def _layout(self, batch, values, supported):
        require_pin_batch(batch)
        selected, owned, owners = _owner_layout(batch)
        if (values.ndim != 3 or values.shape[:2] != owners.shape
                or values.shape[-1] != len(self.field_names)
                or supported.shape != values.shape or supported.dtype != torch.bool
                or not values.is_floating_point()):
            raise ValueError('OP gate requires physical-device OP vectors and boolean support')
        if values.device != owners.device or supported.device != owners.device:
            raise ValueError('OP gate data and graph must share one device')
        if not bool(torch.isfinite(torch.where(supported, values, 0.)).all()):
            raise ValueError('Supported gate OP values must be finite')
        terminal, nets = batch['pin_terminal_index'], batch['pin_net_index']
        if terminal.dtype != torch.long or nets.dtype != torch.long:
            raise ValueError('Pin terminal and net identities must use integer tensors')
        if bool(((terminal < -1) | (terminal >= self.terminal_roles.numel())).any()):
            raise ValueError('Pin terminal identity is outside the saved vocabulary')
        if nets.shape != owned.shape or bool((batch['device_mask'] & (nets < 0)).any()):
            raise ValueError('Every live pin/port requires a nonnegative net identity')
        types = batch['owner_device_types']
        if types.shape != owners.shape or types.dtype != torch.long:
            raise ValueError('OP gate owner types do not match physical devices')
        if bool(((types < 0) | (types >= self.mos_types.numel())).any()):
            raise ValueError('OP gate owner type is outside the saved vocabulary')
        roles = self.terminal_roles[terminal.clamp_min(0)]
        roles = torch.where(owned, roles, -1)
        active = (owned & (roles >= 0) & self.mos_types[types].gather(1, selected)
                  & supported.any(-1).gather(1, selected))
        # An actual net short takes priority over the artificial same-device
        # clique. This preserves diode connections (G=D), B=S, and self loops.
        eligible = (active[:, :, None] & active[:, None, :]
                    & (selected[:, :, None] == selected[:, None, :])
                    & (nets[:, :, None] != nets[:, None, :]))
        return selected, roles.clamp_min(0), eligible

    def forward(self, batch, values, supported):
        owners, roles, eligible = self._layout(batch, values, supported)
        safe = torch.where(supported, values, 0.)
        embedding = self.op_embedding(torch.cat((safe, supported.to(values.dtype)), -1))
        pairs = (embedding[:, :, None, None, :]
                 + self.query_roles.weight[None, None, :, None, :]
                 + self.key_roles.weight[None, None, None, :, :])
        logits = self.score_head(pairs)  # B,D,4,4,H
        return self._bias_from_logits(logits, owners, roles, eligible)

    def _bias_from_logits(self, logits, owners, roles, eligible):
        probability = torch.sigmoid(logits)
        epsilon = self.settings['epsilon']
        penalties = self.settings['strength'] * torch.log(epsilon+(1-epsilon)*probability)
        batch_size, devices = logits.shape[:2]
        pins = owners.shape[1]
        table = penalties.permute(0,4,1,2,3).reshape(batch_size,self.cfg.heads,devices*16)
        indices = owners[:, :, None]*16 + roles[:, :, None]*4 + roles[:, None, :]
        result = table.gather(2, indices.reshape(batch_size,1,pins*pins).expand(-1,self.cfg.heads,-1))
        result = result.reshape(batch_size,self.cfg.heads,pins,pins)
        result = torch.where(eligible[:,None],result,0.)
        return F.pad(result,(1,0,1,0))  # The graph token is never gated.


def gate_class_pin(version):
    """Lazy version factory preserves the historical module/class import path."""
    if version == 'v0':
        return OPAttentionGatePin
    if version == 'v1':
        from .gates_pin.v1_pin import OPAttentionGateV1Pin
        return OPAttentionGateV1Pin
    if version == 'v2':
        from .gates_pin.v2_pin import OPAttentionGateV2Pin
        return OPAttentionGateV2Pin
    if version == 'v3':
        from .gates_pin.v3_pin import OPAttentionGateV3Pin
        return OPAttentionGateV3Pin
    raise ValueError(f'Unknown OP gate version: {version}')


class DeviceOPGateModelPin(nn.Module):
    """Legacy paired controls and versioned joint device-to-pin OP gates.

All A variants inject correct predicted OP features; all B variants omit OP
feature tokens. Only the gate's OP source varies between predicted/oracle/shuffle.
The B ungated baseline has no predictor dependency or OP pretraining cost.
"""
    granularity = 'pin'

    def __init__(self, mode, config, prep, predictor=None, gate_config=None, shuffle_bank=None):
        super().__init__()
        if mode not in GATE_MODELS_PIN:
            raise ValueError(f'Unknown device-to-pin OP mode: {mode}')
        _pin_prep(prep)
        self.mode, self.model_config = mode, _config(config)
        self.gate_config = validate_gate_config_pin(gate_config, version=gate_version_pin(mode))
        self.use_op_features = mode in _FEATURE_MODES
        self.requires_measured_op = mode in GATE_ORACLES_PIN
        self.gate_source = ('none' if mode in _BASE_MODES else
                            'oracle' if self.requires_measured_op else
                            'shuffle' if '_shuffle_' in mode else 'predicted')
        self.attention_scope = ('none' if self.gate_source == 'none' else
            'fusion_self_attention' if mode in VERSIONED_JOINT_GATE_MODELS_PIN and '_fusion_' in mode else
            'all_encoder_self_attention')
        needs_predictor = mode != 'device_op_gate_only_base_pin'
        if needs_predictor != (predictor is not None):
            raise ValueError('Only the gate-only ungated baseline omits the device OP predictor')
        if predictor is not None and not isinstance(predictor, DeviceOPPredictorPin):
            raise ValueError('These controls require the true device-level OP predictor')
        if (self.gate_source == 'shuffle') != (shuffle_bank is not None):
            raise ValueError('Only shuffle gate modes require a saved training OP bank')
        self.predictor = freeze(predictor) if predictor is not None else None
        self._predictor_digest_pin = state_digest(self.predictor) if predictor is not None else None
        # Construct identical downstream modules before creating any gate, so
        # resetting the RNG gives matched initialization within the A/B families.
        if self.use_op_features:
            self.encoder = op_encoder_pin(self.model_config, prep, use_design=True).core
        else:
            self.encoder = design_encoder_pin(self.model_config, prep)
        self.decoder = decoder_pin(self.model_config, prep)
        gate_type = gate_class_pin(gate_version_pin(mode))
        self.gate = (None if self.gate_source == 'none' else
                     gate_type(self.model_config, prep, self.gate_config))
        self.shuffle_bank = shuffle_bank
        self.last_oracle_coverage = None

    @property
    def gate_version(self):
        return 'none' if self.gate is None else self.gate.version

    def train(self, mode=True):
        super().train(mode)
        if self.predictor is not None:
            self.predictor.eval()
        if self.shuffle_bank is not None:
            self.shuffle_bank.eval()
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.predictor is not None:
            freeze(self.predictor)
            self._predictor_digest_pin = state_digest(self.predictor)
        return result

    def predicted_devices(self, batch):
        from final_exam.op_gate_data_pin import device_input_pin
        if self.predictor is None:
            raise ValueError('This baseline has no OP predictor')
        cache_keys = ('_predicted_device_op','_predicted_device_op_mask','_predicted_device_op_digest')
        present = [name in batch for name in cache_keys]
        if any(present):
            if not all(present) or batch[cache_keys[2]] != self._predictor_digest_pin:
                raise ValueError('Frozen device OP cache does not match this predictor checkpoint')
            values, supported = batch[cache_keys[0]], batch[cache_keys[1]]
            expected = self.predictor.topology_mask(device_input_pin(batch))
            if (values.shape != expected.shape or supported.shape != expected.shape
                    or supported.dtype != torch.bool or not values.is_floating_point()
                    or values.device != expected.device or supported.device != expected.device
                    or not torch.equal(supported,expected)
                    or not bool(torch.isfinite(values).all())):
                raise ValueError('Invalid cached device OP shape, support or values')
            return values.detach(), supported
        with torch.no_grad():
            return self.predictor.predict_devices(batch)

    def gate_values(self, batch, predicted, supported):
        """Return OP for the gate only; the feature branch remains predicted."""
        if self.gate_source == 'shuffle':
            values, support = self.shuffle_bank(batch)
            if values.shape != predicted.shape or not torch.equal(support,supported):
                raise ValueError('Shuffle bank must preserve device identities and trained field support')
            return values.detach(), supported
        if self.gate_source != 'oracle':
            return predicted, supported
        required = ('owner_op','owner_op_mask','owner_op_target_mask')
        if not all(name in batch for name in required):
            raise ValueError('Oracle gate requires measured physical-device OP at evaluation')
        values, observed, eligible = (batch[name] for name in required)
        if (values.shape != predicted.shape or observed.shape != predicted.shape
                or observed.dtype != torch.bool or eligible.shape != predicted.shape[:2]
                or eligible.dtype != torch.bool):
            raise ValueError('Oracle OP values and masks do not match the device prediction schema')
        use_true = observed & supported & eligible.unsqueeze(-1) & batch['owner_mask'].unsqueeze(-1)
        if not bool(torch.isfinite(torch.where(use_true,values,0.)).all()):
            raise ValueError('Observed oracle gate OP must be finite')
        self.last_oracle_coverage = dict(observed=int(use_true.sum()), supported=int(supported.sum()))
        # The gate only sees the common train-supported mask, not measured
        # missingness as an additional privileged input.
        return torch.where(use_true,values,predicted).detach(), supported

    def memory(self, batch):
        require_pin_batch(batch)
        inputs = design_input_pin(batch)
        bias = None
        if self.predictor is not None:
            predicted, supported = self.predicted_devices(batch)
            if self.use_op_features:
                pin_values, pin_support = broadcast_owner_op_pin(predicted,supported,batch)
                inputs.update(op=pin_values,op_mask=pin_support,
                              op_target_mask=pin_support.any(-1) & inputs['device_mask'])
            if self.gate is not None:
                values, support = self.gate_values(batch,predicted,supported)
                bias = self.gate(batch,values,support)
        if self.attention_scope == 'fusion_self_attention':
            # OP continues to enter through feature fusion, while the entire
            # structural prefix sees topology only. The same gate values are
            # applied to each later fusion block's self-attention logits.
            structure = self.encoder.structure_encoder(inputs)
            return self.encoder.forward_from_structure(structure, inputs, fusion_attention_bias=bias)
        if self.attention_scope not in ('none', 'all_encoder_self_attention'):
            raise ValueError(f'Unknown OP attention scope: {self.attention_scope}')
        structure = self.encoder.structure_encoder(inputs,attention_bias=bias)
        return self.encoder.forward_from_structure(structure,inputs)

    def forward(self, batch):
        memory, padding = self.memory(batch)
        return self.decoder(memory,padding)


def describe_gate_model_pin(model):
    if isinstance(model, DeviceOPPredictorPin):
        return dict(kind='device_op_predictor_pin',model_config=deepcopy(model.model_config),
                    support=deepcopy(model.support),encoder_granularity='device')
    if not isinstance(model, DeviceOPGateModelPin):
        raise TypeError(f'Unsupported device-OP gate architecture: {type(model).__name__}')
    from final_exam.op_gate_data_pin import describe_shuffle_bank_pin
    recipe = dict(kind='device_op_gate_pin',mode=model.mode,model_config=deepcopy(model.model_config),
                gate_config=deepcopy(model.gate_config),
                gate_version=model.gate_version, attention_scope=model.attention_scope,
                predictor=None if model.predictor is None else describe_gate_model_pin(model.predictor),
                shuffle_bank=None if model.shuffle_bank is None else describe_shuffle_bank_pin(model.shuffle_bank),
                use_op_features=model.use_op_features,gate_source=model.gate_source,
                requires_measured_op=model.requires_measured_op)
    if model.gate_version in ('v2', 'v3'):
        recipe['gate_normalization'] = deepcopy(model.gate.normalization_contract)
    if model.gate_version == 'v3':
        recipe['gate_architecture'] = deepcopy(model.gate.architecture_contract)
    return recipe


def rebuild_gate_model_pin(recipe, config, prep):
    kind = recipe['kind']
    saved_config = recipe.get('model_config',_config(config))
    if kind == 'device_op_predictor_pin':
        if recipe.get('encoder_granularity') != 'device':
            raise ValueError('Expected a genuine device-token OP predictor recipe')
        return DeviceOPPredictorPin(saved_config,prep,recipe['support'])
    if kind != 'device_op_gate_pin':
        raise ValueError(f'Unknown device-OP gate checkpoint recipe: {kind}')
    from final_exam.op_gate_data_pin import rebuild_shuffle_bank_pin
    predictor = (None if recipe['predictor'] is None else
                 rebuild_gate_model_pin(recipe['predictor'],saved_config,prep))
    bank = None if recipe['shuffle_bank'] is None else rebuild_shuffle_bank_pin(recipe['shuffle_bank'])
    model = DeviceOPGateModelPin(recipe['mode'],saved_config,prep,predictor,
                                recipe['gate_config'],bank)
    saved_version = recipe.get('gate_version', 'none' if model.gate is None else 'v0')
    if saved_version != model.gate_version:
        raise ValueError('Gate checkpoint version disagrees with its model mode')
    saved_scope = recipe.get('attention_scope', model.attention_scope if saved_version == 'v0' or model.gate is None else None)
    if saved_scope != model.attention_scope:
        raise ValueError('Gate checkpoint scope disagrees with its model mode')
    if model.gate_version in ('v2', 'v3') and recipe.get('gate_normalization') != model.gate.normalization_contract:
        raise ValueError('Gate checkpoint normalization disagrees with its sample-relative contract')
    if model.gate_version == 'v3' and recipe.get('gate_architecture') != model.gate.architecture_contract:
        raise ValueError('Gate checkpoint architecture disagrees with its v3 contract')
    for name in ('use_op_features','gate_source','requires_measured_op'):
        if recipe.get(name) != getattr(model,name):
            raise ValueError(f'Gate checkpoint metadata disagrees with model mode: {name}')
    return model
