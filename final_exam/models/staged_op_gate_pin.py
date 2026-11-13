"""Post-training OP gates on supervised or distilled pin-level ZeroSim models.

The frozen physical-device OP predictor supplies both pin OP features and gate
inputs. Earlier stages bypass the gate completely. Later gate stages freeze
encoder weights, but deliberately retain the encoder's autograd graph: changing
an attention bias must still be differentiable through the frozen network.
"""
from copy import deepcopy

from torch import nn

from experiments.op_transfer.distillation import (
    PredictedOPEncoderDistillation, _signature, _storage,
)
from experiments.op_transfer.losses import LatentScaler
from experiments.op_transfer.models import freeze
from final_exam.staged_gate_registry_pin import (
    ALL_STAGED_GATE_MODELS_PIN, FUSION_STAGED_GATE_MODELS_PIN,
    STAGED_GATE_MODELS_PIN, staged_gate_spec_pin,
)
from final_exam.gate_version_registry_pin import gate_version_pin
from .common_pin import require_pin_batch
from .op_gate_pin import (
    DeviceOPGateModelPin, describe_gate_model_pin, rebuild_gate_model_pin, gate_class_pin,
    validate_gate_config_pin,
)
from .op_transfer_pin import (
    ReadoutModelPin, _pin_alignment_names, broadcast_owner_op_pin,
    design_input_pin, op_input_pin,
)


# Values identify exactly the modules which may be updated in each stage.
_PHASE_MODULES = {
    'supervised': frozenset({'encoder', 'decoder'}),
    'distill_encoder': frozenset({'encoder'}),
    'readout': frozenset({'decoder'}),
    'gate_only': frozenset({'gate'}),
    'gate_decoder': frozenset({'gate', 'decoder'}),
}


class StagedOPGateModelPin(DeviceOPGateModelPin):
    """One deployable architecture with an explicit, serializable stage policy."""

    def __init__(self, config, prep, predictor, *, mode, gate_config=None):
        if mode not in ALL_STAGED_GATE_MODELS_PIN:
            raise ValueError(f'Unknown staged OP gate mode: {mode}')
        version = gate_version_pin(mode)
        initial_mode = ('device_op_gate_fusion_pin' if version in ('v4', 'v5', 'v6', 'v7') else 'device_op_gate_pin') + (f'_{version}' if version != 'v0' else '')
        super().__init__(initial_mode, config, prep, predictor,
                         gate_config=gate_config)
        self.set_mode_pin(mode)
        self.set_phase('supervised')

    def set_mode_pin(self, mode, gate_config=None):
        """Select the gate location when copying a shared, ungated base."""
        spec = staged_gate_spec_pin(mode)
        version = gate_version_pin(mode)
        inherited = dict(self.gate.settings)
        if version in ('v4', 'v5', 'v6', 'v7'):
            inherited.pop('epsilon', None)
            inherited.pop('initial_gate', None)
        settings = validate_gate_config_pin({**inherited, **(gate_config or {})}, version=version)
        if self.gate.version != version or settings != self.gate.settings:
            if getattr(self, 'gate_enabled', False):
                raise ValueError('Gate versions may only change on an ungated base')
            self.gate = gate_class_pin(version).from_gate_layout(self.gate, settings)
            if hasattr(self, 'phase'):
                self.set_phase(self.phase)
        self.mode = mode
        self.gate_config = deepcopy(self.gate.settings)
        self.attention_scope = spec['attention_scope']
        return self

    @property
    def gate_enabled(self):
        return self.phase in ('gate_only', 'gate_decoder')

    def set_phase(self, phase):
        """Set trainability and module modes without detaching the forward path."""
        if phase not in _PHASE_MODULES:
            raise ValueError(f'Unknown staged OP gate phase: {phase}')
        self.phase = phase
        self.requires_grad_(False)
        for name in _PHASE_MODULES[phase]:
            getattr(self, name).requires_grad_(True)
        # A previous stage's gradient must not survive a freezing transition.
        self.zero_grad(set_to_none=True)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Also respect external freeze(model) used for shared base checkpoints.
        # Calling train() never re-enables gradients or a disabled gate.
        active = _PHASE_MODULES.get(getattr(self, 'phase', None), frozenset())
        for name in ('encoder', 'decoder', 'gate'):
            module = getattr(self, name)
            module.train(bool(mode and name in active
                              and any(p.requires_grad for p in module.parameters())))
        self.predictor.eval()
        return self

    def initialize_from_teacher(self, teacher):
        """Copy the complete teacher downstream weights, without shared storage."""
        if not isinstance(teacher, ReadoutModelPin) or teacher.source != 'design_op':
            raise ValueError('Staged distillation requires a design-plus-true-OP pin teacher')
        if teacher.scaler is not None or not hasattr(teacher.encoder, 'core'):
            raise ValueError('Teacher must expose the unscaled OP-fusion pin encoder')
        core = teacher.encoder.core
        if (core.cfg.to_dict() != self.encoder.cfg.to_dict()
                or _signature(core) != _signature(self.encoder)
                or _signature(teacher.decoder) != _signature(self.decoder)):
            raise ValueError('Teacher and student downstream architectures must match')
        if _storage(self) & _storage(teacher):
            raise ValueError('Teacher and student must not share parameter storage')
        self.encoder.load_state_dict(core.state_dict(), strict=True)
        self.decoder.load_state_dict(teacher.decoder.state_dict(), strict=True)
        return self

    def prepared_batch(self, batch):
        """Prediction-only pin features, independent of measured OP and labels."""
        require_pin_batch(batch)
        predicted, supported = self.predicted_devices(batch)
        pin_values, pin_support = broadcast_owner_op_pin(predicted, supported, batch)
        inputs = design_input_pin(batch)
        inputs.update(op=pin_values, op_mask=pin_support,
                      op_target_mask=pin_support.any(-1) & inputs['device_mask'])
        return inputs

    def memory(self, batch):
        if self.gate_enabled:
            # The parent keeps only the OP predictor in no_grad. Neither the
            # encoder nor decoder may be detached during either gate phase.
            return super().memory(batch)
        # Bypass rather than approximate with a sigmoid near one: this is the
        # exact same encoder computation as the ungated feature cascade.
        return self.encoder(self.prepared_batch(batch))


class StagedOPEncoderDistillationPin(PredictedOPEncoderDistillation):
    """Encoder-only alignment before any gate is enabled.

Inherits the audited latent objective and teacher-feature cache interface. The
initializer differs because the student's downstream modules are direct fields,
not nested in a PredictedOPCascade wrapper. The engine recognizes this class as
encoder-only distillation and caches teacher features on train and validation.
"""

    granularity = 'pin'

    def __init__(self, student, teacher, teacher_scaler):
        nn.Module.__init__(self)
        if not isinstance(student, StagedOPGateModelPin):
            raise ValueError('Expected a staged device-OP-to-pin student')
        if not isinstance(teacher, ReadoutModelPin) or teacher.source != 'design_op':
            raise ValueError('Expected a design-plus-true-OP pin teacher')
        if teacher.scaler is not None or not hasattr(teacher.encoder, 'core'):
            raise ValueError('Teacher alignment requires unscaled OP-fusion features')
        if (_signature(teacher.encoder.core) != _signature(student.encoder)
                or teacher.encoder.cfg.to_dict() != student.encoder.cfg.to_dict()
                or _signature(teacher.decoder) != _signature(student.decoder)):
            raise ValueError('Teacher/student pin architectures must match')
        if _storage(student) & _storage(teacher):
            raise ValueError('Teacher and student must not share parameter storage')
        if not isinstance(teacher_scaler, LatentScaler) or not teacher_scaler.finalized:
            raise ValueError('Teacher scaler must be finalized using training data')
        if teacher_scaler.hidden_dim != student.encoder.cfg.dim:
            raise ValueError('Teacher scaler width does not match the pin encoder')
        self.student = student.set_phase('distill_encoder')
        self.teacher = freeze(teacher)
        self.teacher_scaler = teacher_scaler

    def train(self, mode=True):
        # Avoid the inherited cascade-specific .student.downstream access.
        nn.Module.train(self, mode)
        self.teacher.eval()
        self.student.predictor.eval()
        return self

    def _teacher_inputs(self, batch):
        return op_input_pin(batch, use_design=True)

    def objective(self, batch):
        if self.student.phase != 'distill_encoder' or self.student.gate_enabled:
            raise ValueError('Latent distillation requires the encoder-only, ungated phase')
        prepared = self.student.prepared_batch(batch)
        memory, padding = self.student.encoder(prepared)
        return _pin_alignment_names(self._alignment(batch, prepared, memory, padding))


def describe_staged_gate_model_pin(model):
    if not isinstance(model, StagedOPGateModelPin):
        raise TypeError('Expected a staged pin OP gate model')
    recipe = dict(kind='staged_device_op_gate_pin', mode=model.mode,
                model_config=deepcopy(model.model_config),
                gate_config=deepcopy(model.gate_config), phase=model.phase,
                gate_enabled=model.gate_enabled,
                attention_scope=model.attention_scope, gate_version=model.gate_version,
                predictor=describe_gate_model_pin(model.predictor),
                use_op_features=True, gate_source='predicted_op',
                requires_measured_op=False)
    if model.gate_version in ('v2', 'v3', 'v4', 'v5', 'v6', 'v7'):
        recipe['gate_normalization'] = deepcopy(model.gate.normalization_contract)
    if model.gate_version in ('v3', 'v4', 'v5', 'v6', 'v7'):
        recipe['gate_architecture'] = deepcopy(model.gate.architecture_contract)
    return recipe


def rebuild_staged_gate_model_pin(recipe, config, prep):
    if recipe.get('kind') != 'staged_device_op_gate_pin':
        raise ValueError('Expected a staged pin gate checkpoint recipe')
    saved_config = recipe.get('model_config', config)
    predictor = rebuild_gate_model_pin(recipe['predictor'], saved_config, prep)
    model = StagedOPGateModelPin(saved_config, prep, predictor,
        mode=recipe['mode'], gate_config=recipe['gate_config']).set_phase(recipe['phase'])
    if recipe.get('gate_version', 'v0') != model.gate_version:
        raise ValueError('Staged gate recipe version disagrees with its model mode')
    # Active versioned checkpoints must preserve the trained module policy:
    # renaming a gate-only checkpoint cannot turn it into a decoder-tuned one.
    # Ungated preparatory stages remain valid for shared-base reconstruction.
    if (model.gate_version != 'v0' and model.gate_enabled
            and model.phase != staged_gate_spec_pin(model.mode)['final_phase']):
        raise ValueError('Staged gate recipe phase disagrees with its model mode')
    if model.gate_version in ('v2', 'v3', 'v4', 'v5', 'v6', 'v7') and recipe.get('gate_normalization') != model.gate.normalization_contract:
        raise ValueError('Staged gate recipe normalization disagrees with its sample-relative contract')
    if model.gate_version in ('v3', 'v4', 'v5', 'v6', 'v7') and recipe.get('gate_architecture') != model.gate.architecture_contract:
        raise ValueError('Staged gate recipe architecture disagrees with its versioned contract')
    expected = dict(gate_enabled=model.gate_enabled, use_op_features=True,
                    gate_source='predicted_op', requires_measured_op=False)
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError('Staged gate recipe input or phase metadata is inconsistent')
    # Historical all-layer recipes predate the explicit scope field. New
    # fusion-only recipes must declare it so ambiguity cannot change inference.
    saved_scope = recipe.get('attention_scope',
        'all_encoder_self_attention' if model.mode in STAGED_GATE_MODELS_PIN else None)
    if saved_scope != model.attention_scope:
        raise ValueError('Staged gate recipe attention scope disagrees with its model mode')
    return model


__all__ = ['STAGED_GATE_MODELS_PIN', 'FUSION_STAGED_GATE_MODELS_PIN',
           'ALL_STAGED_GATE_MODELS_PIN', 'staged_gate_spec_pin', 'StagedOPGateModelPin',
           'StagedOPEncoderDistillationPin', 'describe_staged_gate_model_pin',
           'rebuild_staged_gate_model_pin']
