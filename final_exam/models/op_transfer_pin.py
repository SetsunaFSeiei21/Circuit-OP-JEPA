"""Pin encoders with a device-scoped OP prediction and supervision boundary.

A pin is a real device terminal or a named external circuit port. The encoder
retains all pin tokens. Only the OP head pools terminals belonging to a common
physical device: it predicts one OP vector per device, and broadcasts that
vector back to the same device's terminals for the downstream pin encoder.
External ports never acquire a measured or predicted device OP vector.
"""
from copy import deepcopy

import torch
from torch import nn

from device_jepa.baselines import _make_model
from device_jepa.model import ModelConfig
from experiments.op_transfer.distillation import (
    PredictedOPDistillation, PredictedOPEncoderDistillation,
)
from experiments.op_transfer.losses import LatentScaler, masked_field_mse
from experiments.op_transfer.models import (
    OPPredictor, PredictedOPCascade, ReadoutModel, _scaled, fit_op_support, freeze,
)
from .common_pin import require_pin_batch


_OWNER_TOPOLOGY = frozenset({
    'owner_index', 'pin_to_device', 'owner_mask', 'owner_device_types',
    'owner_device_models', 'owner_device_categories', 'owner_edge_roles',
    'owner_adjacency', 'pin_terminal_index', 'pin_net_index', 'pin_is_external_port',
})
_TOPOLOGY = frozenset({
    'device_types', 'device_categories', 'device_models', 'device_mask',
    'adjacency', 'edge_roles', 'port_roles', 'named_port_roles',
    'gtn_loop_membership_by_pin',
}) | _OWNER_TOPOLOGY
_DESIGN = frozenset({'parameters', 'parameter_mask', 'parameter_applicability'})
_OP = frozenset({'op', 'op_mask', 'op_target_mask'})


def design_input_pin(batch):
    """Strict design-only allowlist, including measured-data-free owner identity."""
    return {key: value for key, value in batch.items() if key in _TOPOLOGY | _DESIGN}


def op_input_pin(batch, *, use_design=False):
    """Measured/predicted pin OP contract; owner OP targets are never inputs."""
    allowed = _TOPOLOGY | _OP | (_DESIGN if use_design else frozenset())
    return {key: value for key, value in batch.items() if key in allowed}


def _pin_prep(prep):
    if prep.get('granularity') != 'pin' or 'device_preprocessing' not in prep:
        raise ValueError('Pin transfer models require saved pin preprocessing and device owner schema')
    if prep.get('pin_vocab') != prep['device_vocab']:
        raise ValueError('Pin token vocabulary disagrees with the saved model vocabulary')
    original = prep['device_preprocessing']
    for name in ('op_names', 'parameter_names', 'metric_names'):
        if prep['feature_spec'][name] != original['feature_spec'][name]:
            raise ValueError('Pin and device owner feature schemas disagree')
    return original


def _config(config):
    return config.to_dict() if isinstance(config, ModelConfig) else dict(config)


def design_encoder_pin(config, prep):
    _pin_prep(prep)
    from .models_pin import build_encoder_pin
    return build_encoder_pin('zerosim_supervised_pin', _config(config), prep)


def decoder_pin(config, prep):
    _pin_prep(prep)
    from .models_pin import build_decoder_pin
    return build_decoder_pin('zerosim_supervised_pin', _config(config), prep)


class OperatingPointEncoderPin(nn.Module):
    """Explicit pin-node instance of the repository's OP-fusion architecture.

    The upstream pin reference and device encoder share the same attention
    operators. Here their vocabulary and sequence are pin-level, so attention
    never contracts pins into a device token. OP-head pooling is separate.
    """
    granularity = 'pin'

    def __init__(self, config, prep, *, use_design=False):
        super().__init__()
        _pin_prep(prep)
        mode = 'zerosim_pin_op' if use_design else 'zerosim_pin_op_only'
        self.core = _make_model(mode, _config(config), prep,
                                pin_vocab=prep['device_vocab']).encoder
        self.cfg = self.core.cfg
        self.use_design_parameters = bool(use_design)

    def forward(self, batch):
        return self.core(require_pin_batch(batch))


def op_encoder_pin(config, prep, *, use_design=False):
    return OperatingPointEncoderPin(config, prep, use_design=use_design)


class ReadoutModelPin(ReadoutModel):
    """Graph-plus-pin memory and the native ZeroSim performance decoder."""
    granularity = 'pin'

    def memory(self, batch):
        inputs = (design_input_pin(batch) if self.source == 'design' else
                  op_input_pin(batch, use_design=self.source == 'design_op'))
        if self.frozen:
            with torch.no_grad():
                memory, padding = self.encoder(inputs)
        else:
            memory, padding = self.encoder(inputs)
        return _scaled(memory, padding, self.scaler, batch), padding


def fit_op_support_pin(train_rows, prep):
    """Fit physical type/model support on training devices, never repeated pins."""
    return fit_op_support(train_rows, _pin_prep(prep))


def _owner_layout(batch):
    index = batch.get('owner_index')
    owners = batch.get('owner_mask')
    pins = batch['device_mask']
    if index is None or owners is None:
        raise ValueError('Pin OP prediction requires owner_index and owner_mask')
    if (index.shape != pins.shape or index.dtype != torch.long or owners.ndim != 2
            or owners.shape[0] != pins.shape[0] or owners.dtype != torch.bool
            or owners.shape[1] < 1):
        raise ValueError('Invalid pin-to-device owner mapping shapes or dtypes')
    if bool(((index < -1) | (index >= owners.shape[1])).any()):
        raise ValueError('Pin owner index is outside the saved device layout')
    selected = index.clamp_min(0)
    owned = pins & (index >= 0)
    terminal = batch.get('pin_terminal_index')
    if terminal is None or terminal.shape != pins.shape:
        raise ValueError('Pin OP prediction requires terminal identity for every token')
    if not torch.equal(owned, pins & (terminal >= 0)):
        raise ValueError('External ports cannot own device OP; every terminal must have an owner')
    if 'pin_to_device' in batch and not torch.equal(index, batch['pin_to_device']):
        raise ValueError('Pin owner mapping aliases disagree')
    if bool((owned & ~owners.gather(1, selected)).any()):
        raise ValueError('A real pin cannot reference a padded device owner')
    if bool((~pins & (index >= 0)).any()):
        raise ValueError('Padding cannot reference a physical device owner')
    return selected, owned, owners


def pool_owner_memory_pin(memory, padding, batch):
    """Mean the real terminals of each owner; external and padding tokens excluded."""
    selected, owned, owners = _owner_layout(batch)
    if memory.ndim != 3 or memory.shape[:2] != (owned.shape[0], owned.shape[1] + 1):
        raise ValueError('Pin memory must contain graph token followed by all pin tokens')
    if padding.shape != memory.shape[:2] or padding.dtype != torch.bool:
        raise ValueError('Pin encoder padding does not match its memory')
    if not torch.equal(~padding[:, 1:], batch['device_mask']):
        raise ValueError('Pin encoder changed token identities or padding')
    real = owned & ~padding[:, 1:]
    values = torch.where(real.unsqueeze(-1), memory[:, 1:], 0.)
    # Dense owner incidence is tiny relative to attention and avoids atomic
    # scatter reductions under the project's deterministic CUDA policy.
    owner_ids = torch.arange(owners.shape[1], device=memory.device)
    incidence = ((selected.unsqueeze(-1) == owner_ids) & real.unsqueeze(-1)).to(memory.dtype)
    sums = torch.bmm(incidence.transpose(1, 2), values)
    counts = incidence.sum(1)
    if bool((owners & (counts == 0)).any()):
        raise ValueError('Each real device requires at least one observed terminal')
    return sums / counts.clamp_min(1).unsqueeze(-1), counts


def broadcast_owner_op_pin(values, mask, batch):
    """Broadcast one owner vector to its terminals, with no OP on external ports."""
    selected, owned, owners = _owner_layout(batch)
    if values.ndim != 3 or values.shape[:2] != owners.shape or mask.shape != values.shape:
        raise ValueError('Device OP vectors must match owner layout and field mask')
    if mask.dtype != torch.bool:
        raise ValueError('Device OP mask must be boolean')
    gather = selected.unsqueeze(-1).expand(*selected.shape, values.shape[-1])
    supported = mask.gather(1, gather) & owned.unsqueeze(-1)
    result = values.gather(1, gather)
    return torch.where(supported, result, 0.), supported


class OPPredictorPin(OPPredictor):
    """Pin encoder → owner terminal mean → one normalized OP vector per device."""
    granularity = 'pin'

    def __init__(self, encoder, prep, support, *, encoder_name='zerosim_supervised_pin'):
        original = _pin_prep(prep)
        if encoder_name not in {'zerosim_supervised_pin', 'deepgen_supervised_pin'}:
            raise ValueError('Unsupported pin OP predictor encoder architecture')
        super().__init__(encoder, original, support)
        self.encoder_name = encoder_name

    def topology_mask(self, batch):
        _, _, owners = _owner_layout(batch)
        kinds = batch['owner_device_types']
        models = batch['owner_device_models']
        if kinds.shape != owners.shape or models.shape != owners.shape:
            raise ValueError('Owner type/model tensors disagree with device layout')
        owner_batch = dict(device_types=kinds, device_models=models, device_mask=owners)
        return super().topology_mask(owner_batch)

    def predict_devices(self, batch):
        inputs = design_input_pin(batch)
        memory, padding = self.encoder(inputs)
        pooled, counts = pool_owner_memory_pin(memory, padding, inputs)
        predicted = self.head(pooled)
        supported = self.topology_mask(inputs) & (counts > 0).unsqueeze(-1)
        return torch.where(supported, predicted, 0.), supported

    def forward(self, batch):
        predicted, supported = self.predict_devices(batch)
        return broadcast_owner_op_pin(predicted, supported, batch)


class PredictedOPCascadePin(PredictedOPCascade):
    """Every split uses predicted device OP broadcast onto a genuine pin graph."""
    granularity = 'pin'

    def predicted_batch(self, batch):
        if self.joint:
            predicted, supported = self.predictor(batch)
        else:
            with torch.no_grad():
                predicted, supported = self.predictor(batch)
        result = design_input_pin(batch)
        result.update(op=predicted, op_mask=supported,
                      op_target_mask=supported.any(-1) & batch['device_mask'])
        if '_group_keys' in batch:
            result['_group_keys'] = batch['_group_keys']
        return result


class PredictedOPDistillationPin(PredictedOPDistillation):
    """Same joint objective as device study, aligned at eligible pin tokens."""
    granularity = 'pin'

    def _teacher_inputs(self, batch):
        return op_input_pin(batch, use_design=True)

    def _student_inputs(self, batch):
        return op_input_pin(batch, use_design=True)

    def objective(self, batch, **weights):
        return _pin_alignment_names(super().objective(batch, **weights))


class PredictedOPEncoderDistillationPin(PredictedOPEncoderDistillation):
    """Encoder-only pin latent alignment; inherited decoder and predictor freeze."""
    granularity = 'pin'

    def _teacher_inputs(self, batch):
        return op_input_pin(batch, use_design=True)

    def _student_inputs(self, batch):
        return op_input_pin(batch, use_design=True)

    def objective(self, batch):
        return _pin_alignment_names(super().objective(batch))


def _pin_alignment_names(result):
    if result is None:
        return None
    loss, count, detail = result
    names = {'device_alignment_mse': 'pin_alignment_mse',
             'raw_device_alignment_mse': 'raw_pin_alignment_mse'}
    return loss, count, {names.get(name, name): value for name, value in detail.items()}


def op_objective_pin(model, batch):
    """Equal OP-field MSE over observed physical devices, counted once per owner."""
    prediction, supported = model.predict_devices(batch)
    target, observed = batch['owner_op'], batch['owner_op_mask']
    eligible = batch['owner_op_target_mask'] & batch['owner_mask']
    if target.shape != prediction.shape or observed.shape != prediction.shape:
        raise ValueError('Device OP supervision schema does not match predictor')
    if observed.dtype != torch.bool or eligible.shape != prediction.shape[:2] or eligible.dtype != torch.bool:
        raise ValueError('Device OP supervision requires boolean observation and eligibility masks')
    mask = observed & supported & eligible.unsqueeze(-1)
    if not bool(mask.any()):
        return None
    # Select before arithmetic: arbitrary placeholders in unobserved fields do
    # not affect the loss, its gradients, or the validation field accumulators.
    safe_prediction = torch.where(mask, prediction.double(), 0.)
    safe_target = torch.where(mask, target.double(), 0.)
    sse = (safe_prediction - safe_target).square().sum((0, 1))
    counts = mask.sum((0, 1))
    return masked_field_mse(prediction, target, mask), int(mask.sum()), (sse, counts)


def describe_model_pin(model):
    """Self-contained pin architecture recipe, independent of teacher files."""
    from .op_gate_pin import DeviceOPPredictorPin, DeviceOPGateModelPin, describe_gate_model_pin
    if isinstance(model, (DeviceOPPredictorPin, DeviceOPGateModelPin)):
        return describe_gate_model_pin(model)
    scaling = lambda value: None if value is None else value.state_dict()
    if isinstance(model, ReadoutModelPin):
        return dict(kind='readout_pin', source=model.source, frozen=model.frozen,
                    scaler=scaling(model.scaler))
    if isinstance(model, OPPredictorPin):
        return dict(kind='op_predictor_pin', support=deepcopy(model.support),
                    encoder_name=model.encoder_name, owner_pool='terminal_mean')
    if isinstance(model, PredictedOPCascadePin):
        return dict(kind='cascade_pin', joint=model.joint,
                    predictor=describe_model_pin(model.predictor),
                    downstream=describe_model_pin(model.downstream))
    raise TypeError(f'Unsupported pin transfer checkpoint: {type(model).__name__}')


def rebuild_model_pin(recipe, config, prep):
    """Restore pin encoders, pooling, broadcast and frozen branches from metadata."""
    _pin_prep(prep)
    kind = recipe['kind']
    if kind in ('device_op_predictor_pin', 'device_op_gate_pin'):
        from .op_gate_pin import rebuild_gate_model_pin
        return rebuild_gate_model_pin(recipe, config, prep)
    if kind == 'readout_pin':
        source = recipe['source']
        encoder = (design_encoder_pin(config, prep) if source == 'design' else
                   op_encoder_pin(config, prep, use_design=source == 'design_op'))
        state = recipe.get('scaler')
        scaler = None if state is None else LatentScaler.from_state_dict(state)
        return ReadoutModelPin(encoder, decoder_pin(config, prep), source=source,
                               scaler=scaler, frozen=recipe['frozen'])
    if kind == 'op_predictor_pin':
        if recipe.get('owner_pool') != 'terminal_mean':
            raise ValueError('Unsupported pin-to-device OP pooling recipe')
        from .models_pin import build_encoder_pin
        name = recipe['encoder_name']
        return OPPredictorPin(build_encoder_pin(name, _config(config), prep), prep,
                              recipe['support'], encoder_name=name)
    if kind == 'cascade_pin':
        return PredictedOPCascadePin(rebuild_model_pin(recipe['predictor'], config, prep),
                                     rebuild_model_pin(recipe['downstream'], config, prep),
                                     joint=recipe['joint'])
    raise ValueError(f'Unknown pin checkpoint architecture: {kind}')


__all__ = [
    'design_input_pin', 'op_input_pin', 'design_encoder_pin', 'op_encoder_pin',
    'decoder_pin', 'fit_op_support_pin', 'pool_owner_memory_pin',
    'broadcast_owner_op_pin', 'OPPredictorPin', 'ReadoutModelPin',
    'PredictedOPCascadePin', 'PredictedOPDistillationPin',
    'PredictedOPEncoderDistillationPin', 'op_objective_pin',
    'describe_model_pin', 'rebuild_model_pin', 'freeze',
]
