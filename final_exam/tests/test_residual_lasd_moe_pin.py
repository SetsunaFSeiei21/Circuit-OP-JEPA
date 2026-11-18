"""Final-PE residual addition and branch-local stop-gradient in LASD-MoE v1."""
import copy

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.lasd_moe_registry_pin import (
    RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN, validate_lasd_moe_contract_pin,
)
from final_exam.models.lasd_moe_pin import LASDMoEModelPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.tests.test_width_decoder_models_pin import inputs


def test_residual_addition_preserves_previous_headwise_parameter_count_and_initialization(inputs):
    cfg, prep, support, _ = inputs
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    torch.manual_seed(17)
    reference = LASDMoEModelPin('lasd_moe_headwise_pin', cfg, prep, predictor)
    for mode in RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN:
        torch.manual_seed(17)
        model = LASDMoEModelPin(mode, cfg, prep, predictor)
        assert state_digest(model) == state_digest(reference)
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())


@pytest.mark.parametrize('mode', RESIDUAL_HEADWISE_LASD_MOE_MODELS_PIN)
def test_final_encoder_receives_last_pe_plus_fusion_and_branch_gradients_are_exact(inputs, mode):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'encoder_layers': 6, 'cross_layers': 3}
    torch.manual_seed(43)
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support),
                           {'hidden_dim': 12, 'score_chunk_size': 3}).train()
    memories, block_outputs, final_inputs = [], [], []
    original = model.encoder.forward_fusion_memories

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        memories.extend(result[0])
        return result

    model.encoder.forward_fusion_memories = capture
    handles = [block.register_forward_hook(lambda module, args, result: block_outputs.append(result))
               for block in model.encoder.fusion_blocks]
    handles.append(model.final_graph_encoder.register_forward_pre_hook(
        lambda module, args: final_inputs.append(args[0])))
    try:
        output, info = model(batch, return_scores=True)
    finally:
        model.encoder.forward_fusion_memories = original
        for handle in handles:
            handle.remove()
    assert len(memories) == len(block_outputs) == 3 and len(final_inputs) == 1
    weights = info['layer_weights']
    heads, channels = cfg['heads'], cfg['dim'] // cfg['heads']
    expected_fusion = sum(
        memory.reshape(*memory.shape[:2], heads, channels) * weights[..., index, None]
        for index, memory in enumerate(memories)).flatten(-2)
    torch.testing.assert_close(final_inputs[0], memories[-1] + expected_fusion, atol=0, rtol=0)
    assert torch.count_nonzero(final_inputs[0][~info['valid']]) == 0

    # Isolate the input to the final encoder to check the residual's Jacobian.
    # The detached version gives exactly one identity path into the final PE tap.
    cotangent = torch.randn_like(final_inputs[0]).masked_fill(~info['valid'].unsqueeze(-1), 0.)
    objective = (final_inputs[0] * cotangent).sum()
    gradients = torch.autograd.grad(objective, (*memories, info['score_features']),
                                    allow_unused=True, retain_graph=True)
    if mode == 'lasd_moe_headwise_detach_pin_v1':
        assert gradients[0] is None and gradients[1] is None and gradients[3] is None
        torch.testing.assert_close(gradients[2], cotangent, atol=0, rtol=0)
    else:
        assert all(grad is not None and grad.abs().sum() > 0 for grad in gradients)

    # The actual performance loss still trains all raw PE outputs through the
    # sequential encoder, including when shallow readout taps are detached.
    for hidden in block_outputs:
        hidden.retain_grad()
    output.square().sum().backward()
    assert all(hidden.grad is not None and hidden.grad.abs().sum() > 0 for hidden in block_outputs)
    assert model.layer_fusion.router.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.layer_fusion.experts.parameters())
    assert all(p.grad is None for p in model.predictor.parameters())


@pytest.mark.parametrize('field,value', [
    ('residual_connection', 'none'), ('hidden_state_detach', False),
    ('hidden_state_detach_scope', 'encoder_chain'), ('final_pe_residual_detach', True),
])
def test_residual_detach_checkpoint_contract_rejects_changed_gradient_boundaries(inputs, field, value):
    cfg, prep, support, _ = inputs
    mode = 'lasd_moe_headwise_detach_pin_v1'
    model = LASDMoEModelPin(mode, cfg, prep, DeviceOPPredictorPin(cfg, prep, support))
    contract = model.architecture_contract
    validate_lasd_moe_contract_pin(mode, contract)
    corrupted = copy.deepcopy(contract)
    corrupted[field] = value
    with pytest.raises(ValueError, match='contract'):
        validate_lasd_moe_contract_pin(mode, corrupted)
