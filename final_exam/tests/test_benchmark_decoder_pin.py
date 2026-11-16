"""Deployment benchmark boundaries, actual OP forwards and state preservation."""
import copy
from types import SimpleNamespace

import pytest
import torch

from device_jepa.training import state_digest
from final_exam.benchmark_decoder_pin import benchmark_low_rank_pin
from final_exam.models.op_gate_pin import DeviceOPPredictorPin
from final_exam.models.width_decoder_pin import WidthDecoderModelPin


def _context(batch):
    def batches(split, batch_size, seed=None, op=False):
        assert split == 'test' and seed is None and not op
        yield batch
    return SimpleNamespace(device=batch['device_mask'].device, batches=batches)


def test_benchmark_runs_uncached_predictor_and_preserves_weights_flags_rng_and_batch():
    from final_exam.tests.test_op_gate_models_pin import OPAttentionModelsPinTests as Fixtures
    Fixtures.setUpClass()
    batch = copy.deepcopy(Fixtures.batch)
    # These deliberately invalid caches would make a cached path fail. Targets
    # are poisoned too: none may enter this measurement's model input mapping.
    batch['_predicted_device_op'] = torch.full((1,), float('nan'))
    batch['_predicted_device_op_mask'] = torch.zeros(1, dtype=torch.bool)
    batch['_predicted_device_op_digest'] = 'invalid-cache'
    batch['_frozen_memory'] = torch.full((1,), float('nan'))
    for key in ('op', 'owner_op', 'performance', 'physical_performance'):
        if key in batch:
            batch[key].fill_(float('nan'))
    predictor = DeviceOPPredictorPin(Fixtures.cfg, Fixtures.prep, Fixtures.support)
    model = WidthDecoderModelPin('width_decoder_pin', Fixtures.cfg, Fixtures.prep,
                                 predictor, {'n_experts': 2}).train()
    before = state_digest(model)
    flags = [module.training for module in model.modules()]
    rng = torch.random.get_rng_state().clone()
    calls = []
    handle = predictor.encoder.register_forward_pre_hook(lambda module, args: calls.append(1))
    result = benchmark_low_rank_pin(_context(batch), model, {'batch_size': 32})
    handle.remove()
    assert len(calls) == 13
    assert result['batch_size'] == batch['device_mask'].shape[0]
    assert result['device'] == 'cpu'
    assert result['includes_op_predictor'] and not result['uses_cached_op']
    assert result['model_parameters'] == sum(p.numel() for p in model.parameters())
    assert result['frozen_parameters'] == sum(p.numel() for p in predictor.parameters())
    assert result['mean_latency_ms'] > 0 and result['samples_per_second'] > 0
    assert result['peak_allocated_bytes'] is None
    assert result['incremental_peak_allocated_bytes'] is None
    assert state_digest(model) == before
    assert flags == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert batch['_predicted_device_op_digest'] == 'invalid-cache'
    assert torch.isnan(batch['owner_op']).all()


def test_benchmark_restores_flags_and_rng_on_forward_failure():
    class Failure(torch.nn.Module):
        def forward(self, batch):
            assert not self.training
            torch.rand(2)
            raise RuntimeError('model failure')
    model = Failure().train()
    before = torch.random.get_rng_state().clone()
    ctx = _context({'device_mask': torch.ones(2, 3, dtype=torch.bool)})
    with pytest.raises(RuntimeError, match='model failure'):
        benchmark_low_rank_pin(ctx, model, {'batch_size': 2})
    assert model.training
    assert torch.equal(before, torch.random.get_rng_state())


def test_benchmark_rejects_empty_test_and_invalid_batch_size():
    empty = SimpleNamespace(device='cpu', batches=lambda *args, **kwargs: iter(()))
    with pytest.raises(ValueError, match='nonempty test'):
        benchmark_low_rank_pin(empty, torch.nn.Linear(2, 2), {'batch_size': 2})
    with pytest.raises(ValueError, match='positive integer'):
        benchmark_low_rank_pin(empty, torch.nn.Linear(2, 2), {'batch_size': True})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_memory_and_synchronized_timing():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(3, 3, device='cuda'))
        def forward(self, batch):
            return batch['parameters'] @ self.weight
    batch = dict(device_mask=torch.ones(2, 3, dtype=torch.bool, device='cuda'),
                 parameters=torch.randn(2, 3, 3, device='cuda'))
    result = benchmark_low_rank_pin(_context(batch), Model(), {'batch_size': 2})
    assert result['peak_allocated_bytes'] >= result['baseline_allocated_bytes'] > 0
    assert result['incremental_peak_allocated_bytes'] >= 0
    assert result['mean_latency_ms'] > 0
