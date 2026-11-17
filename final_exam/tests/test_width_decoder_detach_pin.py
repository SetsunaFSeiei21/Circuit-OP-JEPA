"""Shallow readout stop-gradients must never detach the encoder depth chain."""
import copy

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.width_decoder_pin import WidthDecoderModelPin
from final_exam.tests.test_width_decoder_models_pin import inputs
from final_exam.width_registry_pin import validate_width_contract_pin

MODE = 'width_decoder_detach_pin'


def make_models(inputs, layers=3, dropout=0.):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'cross_layers': layers, 'encoder_layers': layers + 2, 'dropout': dropout}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    models = []
    for mode in ('width_decoder_pin', MODE):
        torch.manual_seed(17)
        models.append(WidthDecoderModelPin(mode, cfg, prep, copy.deepcopy(predictor),
                                          {'n_experts': 2}).train())
    return cfg, prep, batch, models


@pytest.mark.parametrize('layers', [1, 2, 3, 4])
def test_same_initialization_parameters_forward_and_dropout_rng(inputs, layers):
    _, _, batch, (base, detached) = make_models(inputs, layers, dropout=.1)
    assert state_digest(base) == state_digest(detached)
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in detached.parameters())
    # Native attention/dropout computations are identical; no extra RNG draws.
    torch.manual_seed(43)
    expected = base(batch)
    rng = torch.get_rng_state().clone()
    seen = []
    handles = [branch.register_forward_pre_hook(lambda m, a: seen.append(a[0].requires_grad))
               for branch in detached.decoder.branches]
    torch.manual_seed(43)
    actual = detached(batch)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert seen == [False] * (layers - 1) + [True]
    spec = detached.architecture_contract['performance']['memory_gradients']
    assert spec['detached_decoder_inputs'] == list(range(1, layers))
    assert spec['attached_decoder_inputs'] == [layers]
    validate_width_contract_pin(MODE, detached.architecture_contract)
    # A one-PE model has no shallow decoder edge: backward must also be identical.
    if layers == 1:
        target = torch.randn_like(expected)
        (expected - target).square().mean().backward()
        (actual - target).square().mean().backward()
        for a, b in zip(base.parameters(), detached.parameters()):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.parametrize('branch_index', [0, 1, 2])
def test_only_shallow_decoder_to_encoder_edges_are_cut(inputs, branch_index):
    cfg, _, batch, models = make_models(inputs)
    for model in models:
        memories, padding = model.memory(batch)
        features = model.decoder.features(memories, padding)
        chosen = features[..., branch_index * cfg['dim']:(branch_index + 1) * cfg['dim']]
        # A linear probe avoids the near-constant sum/squared norm of LayerNorm.
        (chosen * torch.randn_like(chosen)).sum().backward()
        encoder_grad = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters()
                           if p.grad is not None)
        if model.mode == MODE and branch_index < 2:
            assert encoder_grad == 0
        else:
            assert encoder_grad > 0
        decoder = model.decoder.branches[branch_index]
        for parameter in (decoder.query, decoder.decoder.layers[0].multihead_attn.in_proj_weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


def test_full_performance_loss_trains_all_decoders_and_all_encoder_depths(inputs):
    _, _, batch, (_, model) = make_models(inputs)
    raw_pe_outputs = []
    def observe(_, __, out):
        out.retain_grad()
        raw_pe_outputs.append(out)
    handles = [block.register_forward_hook(observe) for block in model.encoder.fusion_blocks]
    memories, padding = model.memory(batch)
    for handle in handles:
        handle.remove()
    for memory in memories:
        memory.retain_grad()
    before = state_digest(model.predictor)
    prediction = model.decoder(memories, padding)
    (prediction - batch['performance']).square().mean().backward()
    assert all(memory.grad is None for memory in memories[:-1])
    assert memories[-1].grad.abs().sum() > 0
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
               for x in raw_pe_outputs)
    for block in [*model.encoder.structure_encoder.blocks, *model.encoder.fusion_blocks]:
        grad = block.self_attn.in_proj_weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for branch in model.decoder.branches:
        assert branch.query.grad.abs().sum() > 0
        assert branch.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.decoder.moe.router.weight.grad.abs().sum() > 0
    assert not model.predictor.training
    assert all(not p.requires_grad and p.grad is None for p in model.predictor.parameters())
    assert state_digest(model.predictor) == before


def test_checkpoint_rebuild_retains_policy_and_rejects_wrong_contract(inputs):
    cfg, prep, batch, (_, model) = make_models(inputs)
    recipe = describe_model_pin(model)
    restored = rebuild_model_pin(recipe, cfg, prep).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.decoder.detach_shallow_memory is True
    torch.testing.assert_close(model.eval()(batch), restored(batch), rtol=0, atol=0)
    poison = copy.deepcopy(batch)
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        if key in poison:
            poison[key].fill_(float('nan'))
    torch.testing.assert_close(restored(poison), restored(batch), rtol=0, atol=0)
    bad = copy.deepcopy(recipe)
    del bad['architecture']['performance']['memory_gradients']
    with pytest.raises(ValueError, match='architecture'):
        rebuild_model_pin(bad, cfg, prep)
    with pytest.raises(ValueError, match='gradient policy'):
        validate_width_contract_pin(MODE, bad['architecture'])
    wrong = copy.deepcopy(recipe['architecture'])
    wrong['performance']['memory_gradients']['attached_decoder_inputs'] = []
    with pytest.raises(ValueError, match='gradient policy'):
        validate_width_contract_pin(MODE, wrong)
    wrong = copy.deepcopy(recipe['architecture'])
    wrong['mode'] = 'width_decoder_pin'
    with pytest.raises(ValueError, match='gradient policy'):
        validate_width_contract_pin('width_decoder_pin', wrong)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_backward_reaches_encoder_and_all_decoders(inputs):
    _, _, batch, (_, model) = make_models(inputs)
    model = model.cuda().train()
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model(batch).square().mean().backward()
    for block in model.encoder.fusion_blocks:
        assert torch.isfinite(block.self_attn.in_proj_weight.grad).all()
        assert block.self_attn.in_proj_weight.grad.abs().sum() > 0
    for branch in model.decoder.branches:
        assert branch.query.grad.abs().sum() > 0
