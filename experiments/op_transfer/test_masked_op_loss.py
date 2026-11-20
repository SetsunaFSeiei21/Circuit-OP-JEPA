"""Equal-field OP loss values/gradients, including missing and strided data.

These tests check equivalence with the original selected-field formula. They
do not reproduce the cluster's internal PyTorch view-metadata error on CPU.
"""
import pytest
import torch

from experiments.op_transfer.losses import masked_field_mse


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA runtime unavailable'))]


def reference_loss(prediction, target, mask):
    terms = []
    for field in range(prediction.shape[-1]):
        selected = mask[..., field]
        if selected.any():
            terms.append((prediction[..., field][selected] - target[..., field][selected]).square().mean())
    return torch.stack(terms).mean()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(7, 5), (2, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize('strided', [False, True])
def test_equal_field_loss_and_both_operand_gradients_match_reference(device, dtype, shape, strided):
    generator = torch.Generator().manual_seed(71)
    # Slice the last axis so all three operands can have a noncontiguous layout.
    storage_shape = (*shape[:-1], shape[-1] * (2 if strided else 1))
    prediction = torch.randn(storage_shape, dtype=dtype, generator=generator).to(device)
    target = torch.randn(storage_shape, dtype=dtype, generator=generator).to(device)
    mask = (torch.rand(storage_shape, generator=generator) > .5).to(device)
    if strided:
        prediction, target, mask = prediction[..., ::2], target[..., ::2], mask[..., ::2]
        assert not prediction.is_contiguous() and not mask.is_contiguous()
    mask[..., 0] = True
    mask[..., -1] = False  # A field with no observations must not dilute the mean.
    prediction, target = prediction.detach().requires_grad_(), target.detach().requires_grad_()
    expected = reference_loss(prediction, target, mask)
    expected_grads = torch.autograd.grad(expected, (prediction, target))
    actual = masked_field_mse(prediction, target, mask)
    actual_grads = torch.autograd.grad(actual, (prediction, target))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
        assert (actual_grad[~mask] == 0).all()
    with torch.no_grad():
        evaluated = masked_field_mse(prediction, target, mask)
    torch.testing.assert_close(evaluated, actual.detach(), rtol=0, atol=0)
    assert not evaluated.requires_grad


@pytest.mark.parametrize('device', DEVICES)
def test_missing_nan_inf_and_unsupported_fields_have_zero_finite_gradients(device):
    prediction = torch.tensor([[1., 3., float('nan')], [1., float('inf'), float('-inf')],
                               [1., float('nan'), float('nan')]], device=device, requires_grad=True)
    target = torch.tensor([[0., 0., float('inf')], [0., float('nan'), float('nan')],
                          [0., float('-inf'), float('inf')]], device=device, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False], [True, False, False]], device=device)
    loss = masked_field_mse(prediction, target, mask)
    # MSE(field0)=1, MSE(field1)=9, field2 absent: (1+9)/2, not (1+9+1+1)/4.
    torch.testing.assert_close(loss, torch.tensor(5., device=device))
    loss.backward()
    for tensor in (prediction, target):
        assert torch.isfinite(tensor.grad).all()
        assert (tensor.grad[~mask] == 0).all()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
@pytest.mark.parametrize('operand', ['prediction', 'target'])
def test_observed_nonfinite_values_are_rejected(operand, bad):
    values = dict(prediction=torch.ones(2, 3), target=torch.zeros(2, 3))
    values[operand][0, 1] = bad
    with pytest.raises(ValueError, match='Nonfinite observed'):
        masked_field_mse(**values, mask=torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize('shape', [(2, 3), (0, 3), (2, 0), (2, 0, 3)])
def test_empty_supervision_is_rejected(shape):
    prediction = torch.zeros(shape)
    with pytest.raises(ValueError, match='No observed'):
        masked_field_mse(prediction, prediction, torch.zeros(shape, dtype=torch.bool))


def test_shape_and_boolean_mask_contracts_are_preserved():
    prediction, target = torch.ones(2, 3), torch.zeros(2, 3)
    with pytest.raises(ValueError, match='shapes must match'):
        masked_field_mse(prediction, target, torch.ones(2, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match='shapes must match'):
        masked_field_mse(prediction[0], target[0], torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match='boolean'):
        masked_field_mse(prediction, target, torch.ones_like(prediction))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_large_low_precision_field_reduces_without_sum_overflow(dtype):
    prediction = torch.full((70000, 2), 10., dtype=dtype, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    loss = masked_field_mse(prediction, target, mask)
    assert loss.dtype == dtype
    torch.testing.assert_close(loss, torch.tensor(100., dtype=dtype), rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and (prediction.grad > 0).all()


def test_expanded_boolean_mask_and_numerical_gradients():
    prediction = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(prediction)
    mask = torch.tensor([True, False, True, False]).expand_as(prediction)
    assert torch.autograd.gradcheck(lambda p: masked_field_mse(p, target, mask), (prediction,))
