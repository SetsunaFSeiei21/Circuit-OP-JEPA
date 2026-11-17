"""Score adapters train after detach; no shallow Q/K or delta shortcut remains."""
import copy

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.lasd_registry_pin import LASD_DETACH_MODELS_PIN
from final_exam.models.attention_score_pin import layer_score_features_pin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.width_decoder_pin import WidthDecoderModelPin
from final_exam.tests.test_width_decoder_models_pin import inputs
from final_exam.width_registry_pin import validate_width_contract_pin


def make_pair(inputs, mode, layers=3, dropout=0., enabled=True):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, 'cross_layers': layers, 'encoder_layers': layers+2, 'dropout': dropout}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    models = []
    for name in (mode.replace('_detach_pin', '_pin'), mode):
        torch.manual_seed(17)
        models.append(WidthDecoderModelPin(name, cfg, prep, copy.deepcopy(predictor),
            {'n_experts': 2}, lasd_config=dict(enabled=enabled, hidden_dim=12,
                                             dropout=dropout, score_chunk_size=3)).train())
    return cfg, prep, batch, models


def assert_live(parameter):
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('mode', LASD_DETACH_MODELS_PIN)
@pytest.mark.parametrize('layers', [1, 2, 3])
def test_forward_initialization_parameter_count_rng_and_single_pe_gradients(inputs, mode, layers):
    _, _, batch, (base, model) = make_pair(inputs, mode, layers, dropout=.1)
    assert state_digest(base) == state_digest(model)
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in model.parameters())
    torch.manual_seed(43)
    expected, old_scores = base(batch, return_scores=True)
    rng = torch.get_rng_state().clone()
    seen = []
    handles = [adapter.register_forward_pre_hook(
        lambda m, args: seen.append((args[0].requires_grad, args[1].requires_grad)))
        for adapter in model.decoder.score_adapters]
    torch.manual_seed(43)
    actual, scores = model(batch, return_scores=True)
    for handle in handles:
        handle.remove()
    assert seen == [(False, False)] * (layers-1) + [(True, True)]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in ('scores', 'differences', 'valid'):
        torch.testing.assert_close(scores[key], old_scores[key], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    validate_width_contract_pin(mode, model.architecture_contract)
    if layers == 1:
        target = torch.randn_like(actual)
        (expected-target).square().sum().backward()
        (actual-target).square().sum().backward()
        for a, b in zip(base.parameters(), model.parameters()):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.parametrize('mode', LASD_DETACH_MODELS_PIN)
@pytest.mark.parametrize('branch_index', [0, 1, 2])
def test_isolated_readout_cuts_memory_scores_and_previous_delta_not_adapter(inputs, mode, branch_index):
    cfg, _, _, (_, model) = make_pair(inputs, mode)
    # Independent leaves make a previous-score shortcut visible independently
    # of legitimate backpropagation through the encoder's sequential PE chain.
    memories = [torch.randn(2, 4, cfg['dim'], requires_grad=True) for _ in range(3)]
    scores = [torch.randn(2, 4, cfg['heads'], 2, requires_grad=True) for _ in range(3)]
    padding = torch.tensor([[False]*4, [False]*3+[True]])
    features, _ = layer_score_features_pin(scores, dynamics=model.lasd_spec['dynamics'],
                                          detach_previous=True)
    output = model.decoder.features(memories, padding, features)
    selected = output[..., branch_index*cfg['dim']:(branch_index+1)*cfg['dim']]
    (selected*torch.randn_like(selected)).sum().backward()
    for i, (memory, score) in enumerate(zip(memories, scores)):
        if branch_index == i == 2:
            assert_live(memory)
            assert_live(score)
        else:
            assert memory.grad is None or memory.grad.count_nonzero() == 0
            assert score.grad is None or score.grad.count_nonzero() == 0
    for p in model.decoder.score_adapters[branch_index].parameters():
        assert_live(p)
    assert_live(model.decoder.branches[branch_index].query)
    assert_live(model.decoder.branches[branch_index].decoder.layers[0].multihead_attn.in_proj_weight)
    assert memories[2].grad is None or memories[2].grad[1, 3].count_nonzero() == 0


@pytest.mark.parametrize('mode', LASD_DETACH_MODELS_PIN)
def test_full_loss_keeps_encoder_chain_all_adapters_and_decoders_trainable(inputs, mode):
    _, _, batch, (_, model) = make_pair(inputs, mode)
    raw_pe, scores, memories = [], [], []
    def retain(destination):
        def hook(module, args, value):
            value.retain_grad()
            destination.append(value)
        return hook
    handles = [b.register_forward_hook(retain(raw_pe)) for b in model.encoder.fusion_blocks]
    handles += [model.score_extractor.register_forward_hook(retain(scores)),
                model.encoder.output_projection.register_forward_hook(retain(memories))]
    before = state_digest(model.predictor)
    output = model(batch)
    for h in handles:
        h.remove()
    (output*torch.randn_like(output)).sum().backward()
    assert len(scores) == len(memories) == len(raw_pe) == 3
    assert all(s.grad is None for s in scores[:-1])
    assert all(m.grad is None for m in memories[:-1])
    assert_live(scores[-1])
    assert_live(memories[-1])
    for value in raw_pe:
        assert_live(value)
    for block in [*model.encoder.structure_encoder.blocks, *model.encoder.fusion_blocks]:
        assert_live(block.self_attn.in_proj_weight)
    for adapter in model.decoder.score_adapters:
        for parameter in adapter.parameters():
            assert_live(parameter)
    for branch in model.decoder.branches:
        assert_live(branch.query)
        assert_live(branch.decoder.layers[0].multihead_attn.in_proj_weight)
    assert_live(model.decoder.moe.router.weight)
    assert all(not p.requires_grad and p.grad is None for p in model.predictor.parameters())
    assert state_digest(model.predictor) == before


@pytest.mark.parametrize('mode', LASD_DETACH_MODELS_PIN)
def test_rebuild_and_gradient_contract_tamper_detection(inputs, mode):
    cfg, prep, batch, (_, model) = make_pair(inputs, mode)
    model.eval()
    recipe = describe_model_pin(model)
    restored = rebuild_model_pin(recipe, cfg, prep).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(batch), model(batch), rtol=0, atol=0)
    poison = copy.deepcopy(batch)
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        if key in poison:
            poison[key].fill_(float('nan'))
    torch.testing.assert_close(restored(poison), restored(batch), rtol=0, atol=0)
    for field in ('memory_gradients', 'lasd'):
        wrong = copy.deepcopy(recipe)
        del wrong['architecture']['performance'][field]
        with pytest.raises(ValueError):
            validate_width_contract_pin(mode, wrong['architecture'])
        with pytest.raises(ValueError, match='architecture'):
            rebuild_model_pin(wrong, cfg, prep)
    wrong = copy.deepcopy(recipe)
    wrong['architecture']['performance']['lasd']['score_gradients']['previous_score_in_delta'] = 'attached'
    with pytest.raises(ValueError, match='LASD architecture'):
        validate_width_contract_pin(mode, wrong['architecture'])
    wrong = copy.deepcopy(recipe)
    del wrong['lasd_config']
    with pytest.raises(ValueError, match='saved score configuration'):
        rebuild_model_pin(wrong, cfg, prep)


@pytest.mark.parametrize('mode', LASD_DETACH_MODELS_PIN)
def test_disabled_scores_retain_ordinary_detach_behavior(inputs, mode):
    cfg, prep, batch, (_, model) = make_pair(inputs, mode, enabled=False)
    torch.manual_seed(17)
    baseline = WidthDecoderModelPin('width_decoder_detach_pin', cfg, prep,
                                    copy.deepcopy(model.predictor), {'n_experts': 2}).train()
    assert not hasattr(model.decoder, 'score_adapters')
    assert state_digest(model) == state_digest(baseline)
    a, b = model(batch), baseline(batch)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    coefficients = torch.randn_like(a)
    (a*coefficients).sum().backward()
    (b*coefficients).sum().backward()
    for p, q in zip(model.parameters(), baseline.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    validate_width_contract_pin(mode, model.architecture_contract)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('mode', LASD_DETACH_MODELS_PIN)
def test_cuda_detach_backward(inputs, mode):
    _, _, batch, (_, model) = make_pair(inputs, mode)
    model.cuda()
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model(batch).square().sum().backward()
    for b in model.encoder.fusion_blocks:
        assert_live(b.self_attn.in_proj_weight)
    for adapter in model.decoder.score_adapters:
        assert_live(adapter.network[0].weight)
