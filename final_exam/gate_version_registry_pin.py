"""Torch-free identities for versioned pin OP attention gates.

Version belongs to the model name, not a mutable configuration switch. Saved
recipes can therefore reject an accidental architecture/version substitution.
"""
V1_JOINT_GATE_MODELS_PIN = (
    'device_op_gate_pin_v1',
    'device_op_gate_fusion_pin_v1',
)
V1_STAGED_GATE_MODELS_PIN = (
    'device_op_distill_gate_pin_v1',
    'device_op_distill_gate_fusion_pin_v1',
    'device_op_supervised_gate_pin_v1',
    'device_op_supervised_gate_fusion_pin_v1',
)
V1_GATE_MODELS_PIN = V1_JOINT_GATE_MODELS_PIN + V1_STAGED_GATE_MODELS_PIN

V2_JOINT_GATE_MODELS_PIN = (
    'device_op_gate_pin_v2',
    'device_op_gate_fusion_pin_v2',
)
V2_STAGED_GATE_MODELS_PIN = (
    'device_op_distill_gate_pin_v2',
    'device_op_distill_gate_fusion_pin_v2',
    'device_op_supervised_gate_pin_v2',
    'device_op_supervised_gate_fusion_pin_v2',
)
V2_GATE_MODELS_PIN = V2_JOINT_GATE_MODELS_PIN + V2_STAGED_GATE_MODELS_PIN
VERSIONED_JOINT_GATE_MODELS_PIN = V1_JOINT_GATE_MODELS_PIN + V2_JOINT_GATE_MODELS_PIN
VERSIONED_STAGED_GATE_MODELS_PIN = V1_STAGED_GATE_MODELS_PIN + V2_STAGED_GATE_MODELS_PIN
VERSIONED_GATE_MODELS_PIN = V1_GATE_MODELS_PIN + V2_GATE_MODELS_PIN


def gate_version_pin(mode):
    """Legacy names use v0; explicit registered names own their architecture."""
    if mode in V1_GATE_MODELS_PIN:
        return 'v1'
    if mode in V2_GATE_MODELS_PIN:
        return 'v2'
    if mode.endswith(('_v1', '_v2')):
        raise ValueError(f'Unknown versioned OP gate mode: {mode}')
    return 'v0'
