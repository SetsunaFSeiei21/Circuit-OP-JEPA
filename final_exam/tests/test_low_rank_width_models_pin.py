"""Low-rank replacements retain the cascade boundary and independent PE readers."""
import copy

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.low_rank_registry_pin import LOW_RANK_MODELS_PIN
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.width_decoder_pin import WidthDecoderModelPin
from final_exam.tests.test_width_decoder_models_pin import inputs


@pytest.mark.parametrize('mode', LOW_RANK_MODELS_PIN)
def test_prediction_boundary_gradients_and_exact_replay(inputs, mode):
    cfg, prep, support, batch = inputs
    torch.manual_seed(17)
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    digest = state_digest(predictor)
    model = WidthDecoderModelPin(mode, cfg, prep, predictor, {'n_experts': 2},
                                {'attention_rank': 2, 'ffn_rank': 4}).train()
    loss = model(batch).square().mean()
    loss.backward()
    for name, p in model.decoder.named_parameters():
        if 'down.weight' in name or 'up.weight' in name:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert p.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in predictor.parameters())
    assert state_digest(predictor) == digest
    model.eval()
    poisoned = copy.deepcopy(batch)
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        if key in poisoned: poisoned[key].fill_(float('nan'))
    for key in ('op_mask', 'owner_op_mask', 'op_target_mask', 'owner_op_target_mask'):
        if key in poisoned: poisoned[key] = ~poisoned[key]
    with torch.no_grad():
        expected = model(batch)
        torch.testing.assert_close(model(poisoned), expected, atol=0, rtol=0)
        recipe = describe_model_pin(model)
        restored = rebuild_model_pin(recipe, cfg, prep).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(batch), expected, atol=0, rtol=0)
    bad = copy.deepcopy(recipe)
    del bad['low_rank_config']
    with pytest.raises(ValueError, match='saved rank'):
        rebuild_model_pin(bad, cfg, prep)
    branches = list(model.decoder.branches)
    assert len(branches) >= 2
    assert branches[0].query.data_ptr() != branches[1].query.data_ptr()
    assert not ({p.data_ptr() for p in branches[0].parameters()} &
                {p.data_ptr() for p in branches[1].parameters()})


def test_nonreplaced_initial_weights_identical(inputs):
    cfg, prep, support, batch = inputs
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    def make(mode):
        torch.manual_seed(43)
        return WidthDecoderModelPin(mode, cfg, prep, predictor, {'n_experts': 2},
                                   {'attention_rank': 2, 'ffn_rank': 4})
    dense = make('width_decoder_pin')
    dense_state = dense.state_dict()
    for mode in LOW_RANK_MODELS_PIN:
        low = make(mode)
        assert state_digest(low.encoder) == state_digest(dense.encoder)
        assert state_digest(low.decoder.moe) == state_digest(dense.decoder.moe)
        for key, value in low.state_dict().items():
            if key in dense_state:
                torch.testing.assert_close(value, dense_state[key], atol=0, rtol=0)
