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
    'device_op_distill_gate_decoder_pin_v1',
    'device_op_distill_gate_decoder_fusion_pin_v1',
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
    'device_op_distill_gate_decoder_pin_v2',
    'device_op_distill_gate_decoder_fusion_pin_v2',
)
V2_GATE_MODELS_PIN = V2_JOINT_GATE_MODELS_PIN + V2_STAGED_GATE_MODELS_PIN
V3_JOINT_GATE_MODELS_PIN = (
    'device_op_gate_pin_v3',
    'device_op_gate_fusion_pin_v3',
)
V3_STAGED_GATE_MODELS_PIN = (
    'device_op_distill_gate_pin_v3',
    'device_op_distill_gate_fusion_pin_v3',
    'device_op_supervised_gate_pin_v3',
    'device_op_supervised_gate_fusion_pin_v3',
    'device_op_distill_gate_decoder_pin_v3',
    'device_op_distill_gate_decoder_fusion_pin_v3',
)
V3_GATE_MODELS_PIN = V3_JOINT_GATE_MODELS_PIN + V3_STAGED_GATE_MODELS_PIN
V4_JOINT_GATE_MODELS_PIN = ('device_op_gate_fusion_pin_v4',)
V4_STAGED_GATE_MODELS_PIN = (
    'device_op_distill_gate_fusion_pin_v4',
    'device_op_supervised_gate_fusion_pin_v4',
    'device_op_distill_gate_decoder_fusion_pin_v4',
)
V4_GATE_MODELS_PIN = V4_JOINT_GATE_MODELS_PIN + V4_STAGED_GATE_MODELS_PIN
V5_JOINT_GATE_MODELS_PIN = ('device_op_gate_fusion_pin_v5',)
V5_STAGED_GATE_MODELS_PIN = (
    'device_op_distill_gate_fusion_pin_v5',
    'device_op_supervised_gate_fusion_pin_v5',
    'device_op_distill_gate_decoder_fusion_pin_v5',
)
V5_GATE_MODELS_PIN = V5_JOINT_GATE_MODELS_PIN + V5_STAGED_GATE_MODELS_PIN
SIGNED_GATE_MODELS_PIN = V4_GATE_MODELS_PIN + V5_GATE_MODELS_PIN
VERSIONED_JOINT_GATE_MODELS_PIN = (V1_JOINT_GATE_MODELS_PIN + V2_JOINT_GATE_MODELS_PIN
                                 + V3_JOINT_GATE_MODELS_PIN + V4_JOINT_GATE_MODELS_PIN + V5_JOINT_GATE_MODELS_PIN)
VERSIONED_STAGED_GATE_MODELS_PIN = (V1_STAGED_GATE_MODELS_PIN + V2_STAGED_GATE_MODELS_PIN
                                  + V3_STAGED_GATE_MODELS_PIN + V4_STAGED_GATE_MODELS_PIN + V5_STAGED_GATE_MODELS_PIN)
VERSIONED_GATE_MODELS_PIN = V1_GATE_MODELS_PIN + V2_GATE_MODELS_PIN + V3_GATE_MODELS_PIN + SIGNED_GATE_MODELS_PIN


def gate_version_pin(mode):
    """Legacy names use v0; explicit registered names own their architecture."""
    if mode in V1_GATE_MODELS_PIN:
        return 'v1'
    if mode in V2_GATE_MODELS_PIN:
        return 'v2'
    if mode in V3_GATE_MODELS_PIN:
        return 'v3'
    if mode in V4_GATE_MODELS_PIN:
        return 'v4'
    if mode in V5_GATE_MODELS_PIN:
        return 'v5'
    if mode.endswith(('_v1', '_v2', '_v3', '_v4', '_v5')):
        raise ValueError(f'Unknown versioned OP gate mode: {mode}')
    return 'v0'
