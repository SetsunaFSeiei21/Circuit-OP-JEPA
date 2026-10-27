"""Separate EMA topology/fusion trunks with independently learned field encoders.

This is a hybrid alignment model, not a fully stop-gradient teacher. OP field
embeddings receive gradients through the frozen target fusion blocks. Only the
corresponding structure/fusion trunk is updated by the moving average.
"""
import copy
import math

import torch
from torch import nn

from .model import DeviceEncoder


EMA_MODE = "jepa_zerosim_ema"
EMA_ARCHITECTURE = "separate-structure-ema_device-op-cross-attention_v1"
EMA_TRUNK_MODULES = ("structure_encoder", "fusion_blocks", "output_norm", "output_projection")


def validate_ema_decay(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value < 1):
        raise ValueError("ema_decay must be finite and satisfy 0 <= ema_decay < 1")
    return float(value)


def ema_protocol(decay):
    return {"decay": validate_ema_decay(decay),
            "update_scope": [*EMA_TRUNK_MODULES, "null_parameter"],
            "update_timing": "after_optimizer_step",
            "gradient_policy": "independent_input_embeddings_frozen_op_trunk"}


def _trunk_parameters(encoder):
    result = {"null_parameter": encoder.null_parameter}
    for prefix in EMA_TRUNK_MODULES:
        result.update((f"{prefix}.{name}", value)
                      for name, value in getattr(encoder, prefix).named_parameters())
    return result


def _trunk_buffers(encoder):
    return {f"{prefix}.{name}": value for prefix in EMA_TRUNK_MODULES
            for name, value in getattr(encoder, prefix).named_buffers()}


class EMAOperatingPointEncoder(DeviceEncoder):
    """OP-only named-token encoder with an independent, frozen trunk copy.

The inherited own-node attention path never reads design parameters because
input_tokens is replaced here. Eligible missing observations retain named
tokens and an observed=0 bit. Ineligible devices and padding have no live OP
or state tokens. The graph token and real non-MOS nodes remain in the graph.
"""
    def __init__(self, encoder, num_op_features, num_state_features=0):
        # Do not run DeviceEncoder.__init__: that would allocate unused design
        # embeddings and redraw a trunk which must initially be an exact copy.
        nn.Module.__init__(self)
        if type(num_op_features) is not int or num_op_features < 1:
            raise ValueError("EMA OP encoder requires a positive OP feature count")
        if type(num_state_features) is not int or num_state_features < 0:
            raise ValueError("OP state feature count must be a nonnegative integer")
        self.cfg = copy.deepcopy(encoder.cfg)
        self.input_contract_version = encoder.input_contract_version
        self.num_op_features = num_op_features
        self.num_state_features = num_state_features
        for name in EMA_TRUNK_MODULES:
            component = copy.deepcopy(getattr(encoder, name))
            component.requires_grad_(False)
            setattr(self, name, component)
        self.null_parameter = nn.Parameter(encoder.null_parameter.detach().clone(), requires_grad=False)
        # Both modality-specific field identity and value/observation projections
        # are learned normally. They have no cross-modal EMA correspondence.
        self.op_names = nn.Embedding(num_op_features, self.cfg.dim)
        self.op_value = nn.Linear(2, self.cfg.dim)
        if num_state_features:
            self.state_names = nn.Embedding(num_state_features, self.cfg.dim)
            self.state_value = nn.Linear(2, self.cfg.dim)
        # Construction also supports an online encoder already moved to its
        # training device/dtype; new modality inputs must follow it immediately.
        self.to(device=encoder.null_parameter.device, dtype=encoder.null_parameter.dtype)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        # EMA targets use deterministic dropout behavior even when embeddings
        # are trained. eval() does not disable differentiation through fusion.
        for name in EMA_TRUNK_MODULES:
            getattr(self, name).eval()
        return self

    def input_tokens(self, batch):
        if not {"op", "op_mask", "op_target_mask"} <= batch.keys():
            raise ValueError("EMA OP encoder requires OP values, observation masks and eligibility")
        op, observed = batch["op"], batch["op_mask"]
        shape = batch["device_types"].shape
        expected = (*shape, self.num_op_features)
        if op.shape != expected or observed.shape != expected or observed.dtype != torch.bool:
            raise ValueError("OP values/mask do not match the EMA encoder schema")
        eligibility = batch["op_target_mask"]
        if eligibility.shape != shape or eligibility.dtype != torch.bool:
            raise ValueError("OP eligibility must be a boolean per-device mask")
        eligible = eligibility & batch["device_mask"]
        observed = observed & eligible.unsqueeze(-1)
        values = torch.stack((torch.where(observed, op, torch.zeros_like(op)), observed.to(op.dtype)), -1)
        tokens = self.op_value(values) + self.op_names.weight[None, None]
        live = eligible.unsqueeze(-1).expand_as(observed)
        if self.num_state_features:
            state, state_mask = batch.get("op_state"), batch.get("op_state_mask")
            expected_state = (*shape, self.num_state_features)
            if (state is None or state_mask is None or state.shape != expected_state
                    or state_mask.shape != expected_state or state_mask.dtype != torch.bool):
                raise ValueError("EMA OP encoder requires matching state features and observation masks")
            state_mask = state_mask & eligible.unsqueeze(-1)
            # State values arrive unnormalized, separately from electrical OP.
            state_values = torch.stack((torch.where(state_mask, state, torch.zeros_like(state)),
                                        state_mask.to(state.dtype)), -1)
            state_tokens = self.state_value(state_values) + self.state_names.weight[None, None]
            tokens = torch.cat((tokens, state_tokens), 2)
            live = torch.cat((live, eligible.unsqueeze(-1).expand_as(state_mask)), 2)
        return tokens, live

    def forward(self, batch):
        # Pure topology inputs are data, and every target SE weight is frozen.
        # Keep the fusion call OUTSIDE no_grad so OP/state embeddings can learn.
        with torch.no_grad():
            structure = self.structure_encoder(batch)
        return self.forward_from_structure(structure, batch)


class EMAJEPAModel(nn.Module):
    """Independent SE/fusion target EMA; independently gradient-trained OP inputs.

Call update_ema exactly once after a successful online optimizer step, never
after a forward/validation pass. The complete target output is not detached.
Variance/covariance penalties therefore also train the OP/state embeddings.
"""
    def __init__(self, encoder, num_op_features, num_state_features=0, ema_decay=0.99):
        super().__init__()
        decay = validate_ema_decay(ema_decay)
        self.encoder = encoder
        self.op_encoder = EMAOperatingPointEncoder(encoder, num_op_features, num_state_features)
        self.ema_decay = decay
        self.ema_updates = 0
        self.verify_ema_state()

    @property
    def structure_encoder(self):
        """The online structure encoder; the OP branch owns a separate copy."""
        return self.encoder.structure_encoder

    def ema_parameter_pairs(self):
        """Deterministic (name, online, target) map; excludes field embeddings."""
        online, target = _trunk_parameters(self.encoder), _trunk_parameters(self.op_encoder)
        if online.keys() != target.keys():
            raise ValueError("EMA trunk parameter names do not correspond")
        return tuple((name, online[name], target[name]) for name in sorted(online))

    def ema_buffer_pairs(self):
        online, target = _trunk_buffers(self.encoder), _trunk_buffers(self.op_encoder)
        if online.keys() != target.keys():
            raise ValueError("EMA trunk buffer names do not correspond")
        return tuple((name, online[name], target[name]) for name in sorted(online))

    def verify_ema_state(self):
        """Verify ownership and update boundaries without comparing weight values."""
        target_ids = set()
        for name, online, target in (*self.ema_parameter_pairs(), *self.ema_buffer_pairs()):
            if online.shape != target.shape or online.dtype != target.dtype or online.device != target.device:
                raise ValueError(f"EMA trunk tensor mismatch: {name}")
            if online is target or (online.numel() and online.data_ptr() == target.data_ptr()):
                raise ValueError(f"EMA target must own independent storage: {name}")
            if target.requires_grad or target.grad is not None:
                raise ValueError(f"EMA target trunk cannot receive parameter gradients: {name}")
            target_ids.add(id(target))
        for name, parameter in self.op_encoder.named_parameters():
            if id(parameter) not in target_ids and not parameter.requires_grad:
                raise ValueError(f"OP/state field embedding must remain trainable: {name}")
        if type(self.ema_updates) is not int or self.ema_updates < 0:
            raise ValueError("EMA update count must be a nonnegative integer")

    @torch.no_grad()
    def update_ema(self):
        if not self.training:
            raise RuntimeError("EMA updates are permitted only during training, after an optimizer step")
        self.verify_ema_state()
        for _, online, target in self.ema_parameter_pairs():
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)
        for _, online, target in self.ema_buffer_pairs():
            if target.is_floating_point() or target.is_complex():
                target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)
            else:
                target.copy_(online)
        self.ema_updates += 1

    def forward(self, batch):
        memory, _ = self.encoder(batch)
        target, _ = self.op_encoder(batch)
        valid = batch["device_mask"] & batch["op_target_mask"] & batch["op_mask"].any(-1)
        return memory[:, 1:], target[:, 1:], valid
