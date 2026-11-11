"""Lightweight names and training contracts for staged pin OP gates.

Keep this module free of torch so login-node launch and recovery inspection can
validate model names without importing the training runtime.
"""

STAGED_GATE_MODELS_PIN = (
    'device_op_distill_gate_pin',
    'device_op_distill_gate_decoder_pin',
    'device_op_supervised_gate_pin',
)

FUSION_STAGED_GATE_MODELS_PIN = (
    'device_op_distill_gate_fusion_pin',
    'device_op_distill_gate_decoder_fusion_pin',
    'device_op_supervised_gate_fusion_pin',
)

ALL_STAGED_GATE_MODELS_PIN = STAGED_GATE_MODELS_PIN + FUSION_STAGED_GATE_MODELS_PIN


def staged_gate_spec_pin(mode):
    """Return a fresh contract for one model; scope follows its saved name."""
    if mode not in ALL_STAGED_GATE_MODELS_PIN:
        raise ValueError(f'Unknown staged OP gate mode: {mode}')
    fusion = mode in FUSION_STAGED_GATE_MODELS_PIN
    family = FUSION_STAGED_GATE_MODELS_PIN if fusion else STAGED_GATE_MODELS_PIN
    index = family.index(mode)
    return dict(
        attention_scope='fusion_self_attention' if fusion else 'all_encoder_self_attention',
        base_training='supervised' if index == 2 else 'distilled',
        final_phase='gate_decoder' if index == 1 else 'gate_only',
    )
