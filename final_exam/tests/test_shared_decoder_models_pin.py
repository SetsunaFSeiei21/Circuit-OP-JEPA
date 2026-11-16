"""Shared factors are real tied parameters, numerically correct and replayable."""
import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from device_jepa.training import state_digest
from final_exam.models.low_rank_decoder_pin import LowRankLinearPin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.models.op_transfer_pin import describe_model_pin, rebuild_model_pin
from final_exam.models.shared_decoder_pin import SharedDownLinearPin, SharedDownAttentionPin
from final_exam.models.width_decoder_pin import WidthDecoderModelPin
from final_exam.shared_decoder_registry_pin import (
    SHARED_DECODER_MODELS_PIN, shared_decoder_spec_pin, shared_decoder_settings_pin,
)
from final_exam.tests.test_width_decoder_models_pin import inputs
from final_exam.width_registry_pin import validate_width_contract_pin


def build(inputs, mode, **extra):
    cfg, prep, support, batch = inputs
    cfg = {**cfg, **extra}
    predictor = DeviceOPPredictorPin(cfg, prep, support)
    return WidthDecoderModelPin(mode, cfg, prep, predictor, {'n_experts': 2},
                               shared_decoder_config={'attention_rank': 2, 'ffn_rank': 4})


def consumers(model):
    return [m for m in model.modules() if isinstance(m, SharedDownLinearPin)]


@pytest.mark.parametrize('mode', SHARED_DECODER_MODELS_PIN)
def test_exact_scope_counts_and_single_parameter_ownership(inputs, mode):
    model = build(inputs, mode, decoder_layers=2)
    spec = model.shared_decoder_spec
    readout = model.decoder
    assert len(readout.shared_down) == spec['shared_group_count']
    assert len(consumers(model)) == spec['shared_group_count'] * spec['branch_count']
    assert sum(m.weight.numel() for m in readout.shared_down.values()) == spec['shared_down_parameters']
    assert sum(m.up.weight.numel() for m in consumers(model)) == spec['independent_up_parameters']
    all_parameters = list(model.named_parameters(remove_duplicate=False))
    assert len({id(p) for _, p in all_parameters}) == len(all_parameters)
    # Every A occurs exactly once in both state_dict and optimizer parameter list.
    saved = model.state_dict()
    for down in readout.shared_down.values():
        matches = [(n, t) for n, t in saved.items() if t.data_ptr() == down.weight.data_ptr()]
        assert len(matches) == 1 and '.shared_down.' in matches[0][0]
        owners = [m for m in consumers(model) if m.down is down]
        assert len(owners) == spec['branch_count']
        assert len({id(m.up.weight) for m in owners}) == len(owners)
    for layer_id in range(spec['decoder_layers']):
        for branch in readout.branches:
            layer = branch.decoder.layers[layer_id]
            for role in ('self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.out',
                         'multihead_attn.q', 'multihead_attn.k', 'multihead_attn.v',
                         'multihead_attn.out', 'linear1', 'linear2'):
                name, *part = role.split('.')
                operator = getattr(layer, name)
                if part:
                    operator = getattr(operator, part[0] + '_proj', None)
                assert isinstance(operator, SharedDownLinearPin) == (role in spec['shared_roles'])
    validate_width_contract_pin(mode, model.architecture_contract)
    assert readout.architecture_contract == model.architecture_contract['performance']
    a, b = readout.branches[:2]
    assert a.query is not b.query
    assert a.decoder.layers[0].norm1.weight is not b.decoder.layers[0].norm1.weight
    torch.manual_seed(31)
    dense = build(inputs, 'width_decoder_pin', decoder_layers=2)
    assert (sum(p.numel() for p in dense.parameters()) - sum(p.numel() for p in model.parameters())
            == spec['dense_weight_parameters'] - spec['shared_weight_parameters'])


@pytest.mark.parametrize('mode', SHARED_DECODER_MODELS_PIN)
def test_gradients_frozen_op_no_leak_replay_and_copy(inputs, mode):
    cfg, prep, support, batch = inputs
    model = build(inputs, mode).train()
    original_predictor = state_digest(model.predictor)
    model(batch).square().mean().backward()
    for name, p in model.decoder.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if 'shared_down' in name or '.up.weight' in name:
            assert p.grad.abs().sum() > 0, name
    assert all(p.grad is None and not p.requires_grad for p in model.predictor.parameters())
    assert state_digest(model.predictor) == original_predictor
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    assert all(int(optimizer.state[m.weight]['step']) == 1 for m in model.decoder.shared_down.values())
    model.eval()
    poisoned = copy.deepcopy(batch)
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        if key in poisoned:
            poisoned[key].fill_(float('nan'))
    for key in ('op_mask', 'owner_op_mask', 'op_target_mask', 'owner_op_target_mask'):
        if key in poisoned:
            poisoned[key] = ~poisoned[key]
    with torch.no_grad():
        expected = model(batch)
        torch.testing.assert_close(model(poisoned), expected, rtol=0, atol=0)
        recipe = describe_model_pin(model)
        restored = rebuild_model_pin(recipe, cfg, prep).eval()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        torch.testing.assert_close(restored(batch), expected, rtol=0, atol=0)
        duplicate = copy.deepcopy(model)
        torch.testing.assert_close(duplicate(batch), expected, rtol=0, atol=0)
    for copy_model in (restored, duplicate):
        bank = list(copy_model.decoder.shared_down.values())
        assert all(any(m.down is a for a in bank) for m in consumers(copy_model))
        assert not any(a is b for a in bank for b in model.decoder.shared_down.values())
    duplicate.to(dtype=torch.float64)
    assert all(m.down.weight.dtype == torch.float64 for m in consumers(duplicate))
    assert all(m.down.weight.dtype == torch.float32 for m in consumers(model))
    bad = copy.deepcopy(recipe)
    del bad['shared_decoder_config']
    with pytest.raises(ValueError, match='saved rank'):
        rebuild_model_pin(bad, cfg, prep)
    bad = copy.deepcopy(recipe)
    bad['architecture']['performance']['shared_down_basis']['shared_group_count'] += 1
    with pytest.raises(ValueError, match='architecture'):
        rebuild_model_pin(bad, cfg, prep)


def test_shared_gradient_equals_sum_of_branch_contributions():
    torch.manual_seed(4)
    bank = nn.Linear(8, 3, bias=False, dtype=torch.float64)
    projections = [SharedDownLinearPin(bank, 5) for _ in range(3)]
    xs = [torch.randn(2, 7, 8, dtype=torch.float64) for _ in projections]
    losses = [m(x).square().sum() for m, x in zip(projections, xs)]
    separate = [torch.autograd.grad(loss, bank.weight, retain_graph=True)[0] for loss in losses]
    sum(losses).backward()
    torch.testing.assert_close(bank.weight.grad, sum(separate), rtol=1e-12, atol=1e-12)
    for module, x in zip(projections, xs):
        torch.testing.assert_close(module(x), F.linear(x, module.up.weight @ bank.weight, module.up.bias))


def materialized(attention):
    dense = nn.MultiheadAttention(attention.embed_dim, attention.num_heads,
                                  dropout=attention.dropout, batch_first=True, dtype=torch.float64)
    def wb(p):
        return (p.up.weight @ p.down.weight, p.up.bias) if isinstance(p, LowRankLinearPin) else (p.weight, p.bias)
    with torch.no_grad():
        projections = [wb(getattr(attention, n + '_proj')) for n in ('q', 'k', 'v')]
        dense.in_proj_weight.copy_(torch.cat([w for w, _ in projections]))
        dense.in_proj_bias.copy_(torch.cat([b for _, b in projections]))
        w, b = wb(attention.out_proj)
        dense.out_proj.weight.copy_(w)
        dense.out_proj.bias.copy_(b)
    return dense


@pytest.mark.parametrize('parts', [('k', 'v'), ('q', 'k', 'v', 'out')])
@pytest.mark.parametrize('float_mask', [False, True])
@pytest.mark.parametrize('need_weights', [False, True])
def test_sdpa_partial_or_full_shared_matches_dense(parts, float_mask, need_weights):
    torch.manual_seed(19)
    source = nn.MultiheadAttention(16, 4, dropout=0., batch_first=True, dtype=torch.float64)
    shared = {part: nn.Linear(16, 8, bias=False, dtype=torch.float64) for part in parts}
    model = SharedDownAttentionPin(source, shared)
    q = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    mem = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(5, 7, dtype=torch.bool); mask[:, 1] = True
    padding = torch.zeros(2, 7, dtype=torch.bool); padding[1, -1] = True
    if float_mask:
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, float('-inf'))
        padding = torch.zeros_like(padding, dtype=q.dtype).masked_fill(padding, float('-inf'))
        mask[:, 2] = -.4
    kw = dict(attn_mask=mask, key_padding_mask=padding, need_weights=need_weights,
              average_attn_weights=False)
    actual, weights = model(q, mem, mem, **kw)
    expected, ew = materialized(model)(q, mem, mem, **kw)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    if need_weights:
        torch.testing.assert_close(weights, ew, rtol=1e-10, atol=1e-10)
    actual.square().mean().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(mem.grad).all()
    assert all(p.weight.grad is not None and torch.isfinite(p.weight.grad).all() for p in shared.values())


def test_all_masked_and_eval_dropout_are_finite():
    source = nn.MultiheadAttention(16, 4, dropout=.5, batch_first=True)
    bank = {p: nn.Linear(16, 4, bias=False) for p in ('k', 'v')}
    model = SharedDownAttentionPin(source, bank).eval()
    x = torch.randn(2, 5, 16)
    mask = torch.zeros(5, 5, dtype=torch.bool); mask[0] = True
    a, w = model(x, x, x, attn_mask=mask, need_weights=True)
    b, _ = model(x, x, x, attn_mask=mask, need_weights=False)
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all() and torch.count_nonzero(w[:, 0]) == 0
    a.square().mean().backward()
    assert all(torch.isfinite(p.weight.grad).all() for p in bank.values())


def test_rank_constraints_are_shared_group_constraints_and_not_independent():
    spec = shared_decoder_spec_pin('width_decoder_shared_both_pin',
        {'attention_rank': 64, 'ffn_rank': 64}, {'cross_layers': 3, 'decoder_layers': 1})
    assert spec['shared_weight_parameters'] == 819200
    assert spec['dense_weight_parameters'] == 1966080
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            shared_decoder_settings_pin({'attention_rank': value})
    for value in (96, 129):
        with pytest.raises(ValueError):
            shared_decoder_spec_pin('width_decoder_shared_attention_pin', {'attention_rank': value})
    with pytest.raises(ValueError, match='at least two'):
        shared_decoder_spec_pin('width_decoder_shared_attention_pin', model_config={'cross_layers': 1})


def test_nonselected_weights_start_identically_and_qo_remain_dense(inputs):
    def make(mode):
        torch.manual_seed(43)
        return build(inputs, mode)
    dense = make('width_decoder_pin')
    shared = make('width_decoder_shared_kv_pin')
    assert state_digest(dense.encoder) == state_digest(shared.encoder)
    assert state_digest(dense.decoder.moe) == state_digest(shared.decoder.moe)
    for db, sb in zip(dense.decoder.branches, shared.decoder.branches):
        torch.testing.assert_close(db.query, sb.query, rtol=0, atol=0)
        dl, sl = db.decoder.layers[0], sb.decoder.layers[0]
        assert state_digest(dl.self_attn) == state_digest(sl.self_attn)
        assert state_digest(dl.linear1) == state_digest(sl.linear1)
        assert state_digest(dl.linear2) == state_digest(sl.linear2)
        assert isinstance(sl.multihead_attn.q_proj, nn.Linear)
        assert isinstance(sl.multihead_attn.out_proj, nn.Linear)
        torch.testing.assert_close(sl.multihead_attn.q_proj.weight,
                                   dl.multihead_attn.in_proj_weight.chunk(3)[0], rtol=0, atol=0)
        assert state_digest(sl.multihead_attn.out_proj) == state_digest(dl.multihead_attn.out_proj)


def test_partial_attention_diagnostics_restore_original_weights_and_ties(inputs):
    from final_exam.shared_decoder_analysis_pin import decoder_groups_pin, apply_approximations_pin
    model = build(inputs, 'width_decoder_shared_kv_pin').eval()
    groups = decoder_groups_pin(model)
    original = state_digest(model)
    with torch.no_grad():
        expected = model(inputs[3])
    # Deliberately perturb every projection, including dense Q/O in the partial MHA.
    approximations = {group.name: {1: {'shared': [spec.weight * .5 for spec in group.matrices]}}
                      for group in groups}
    with pytest.raises(RuntimeError, match='synthetic exit'):
        with apply_approximations_pin(groups, approximations, 1):
            assert torch.isfinite(model(inputs[3])).all()
            raise RuntimeError('synthetic exit')
    assert state_digest(model) == original
    with torch.no_grad():
        torch.testing.assert_close(model(inputs[3]), expected, rtol=0, atol=0)
    assert all(any(m.down is a for a in model.decoder.shared_down.values()) for m in consumers(model))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_shared_bank_cuda_device_move_and_backward(inputs):
    model = build(inputs, 'width_decoder_shared_both_pin').cuda().train()
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs[3].items()}
    model(batch).square().mean().backward()
    assert all(m.down.weight.is_cuda and m.down.weight.grad is not None for m in consumers(model))
