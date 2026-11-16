"""Fixed resident-batch latency diagnostic for performance decoder ablations.

This measures deployment's complete OP-predictor + performance-model forward,
not the cached-OP training shortcut. It never selects a checkpoint. CUDA memory
figures belong to this process with its existing resident data/models; they are
not standalone model memory. Data preparation and transfer are outside timing.
"""
from __future__ import annotations

from statistics import mean, stdev
import time

import torch

from final_exam.models.op_transfer_pin import design_input_pin
from final_exam.op_gate_data_pin import _OWNER_INPUTS


WARMUP_ITERATIONS_PIN = 3
MEASURED_ITERATIONS_PIN = 10


def benchmark_low_rank_pin(ctx, model, options):
    """Return JSON-safe diagnostics without changing model/RNG/training state.

    The first deterministic test batch is reused for every measured forward.
    Only topology/design inputs are passed, so labels, measured OP and cached
    predictor/memory tensors cannot enter this path. A microbenchmark of one
    repeated resident batch is not whole-dataset throughput or I/O latency.
    """
    requested_batch = options['batch_size']
    if type(requested_batch) is not int or requested_batch < 1:
        raise ValueError('Benchmark batch_size must be a positive integer')
    device = torch.device(ctx.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Decoder benchmark supports CPU and CUDA devices')
    cuda_index = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == 'cuda' else None
    rng_devices = [] if cuda_index is None else [cuda_index]
    training_flags = [(module, module.training) for module in model.modules()]
    durations = []
    baseline_bytes = peak_bytes = incremental_bytes = None
    try:
        # Even a later implementation that samples internally must not advance
        # the next model's initialization/shuffling/dropout RNG state.
        with torch.random.fork_rng(devices=rng_devices), torch.inference_mode():
            batch = next(iter(ctx.batches('test', requested_batch, seed=None, op=False)), None)
            if batch is None:
                raise ValueError('Decoder benchmark requires a nonempty test split')
            inputs = design_input_pin(batch)
            inputs.update({key: batch[key] for key in _OWNER_INPUTS if key in batch})
            mask = inputs.get('device_mask')
            if not torch.is_tensor(mask) or mask.ndim != 2 or mask.shape[0] < 1:
                raise ValueError('Decoder benchmark requires a nonempty pin batch')
            actual_batch = int(mask.shape[0])
            if mask.device.type != device.type or (cuda_index is not None and mask.device.index != cuda_index):
                raise ValueError('Benchmark inputs must already reside on the context device')
            # Drop the original mapping before timing; ctx may retain caches,
            # but the model cannot access them through its input dictionary.
            del batch
            model.eval()
            for _ in range(WARMUP_ITERATIONS_PIN):
                prediction = model(inputs)
                if not torch.is_tensor(prediction) or not bool(torch.isfinite(prediction).all()):
                    raise ValueError('Benchmark model produced nonfinite/non-tensor predictions')
                del prediction
            if cuda_index is not None:
                torch.cuda.synchronize(device)
                baseline_bytes = int(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(MEASURED_ITERATIONS_PIN):
                start = time.perf_counter()
                prediction = model(inputs)
                if cuda_index is not None:
                    torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - start)
                del prediction
            if cuda_index is not None:
                peak_bytes = int(torch.cuda.max_memory_allocated(device))
                incremental_bytes = max(0, peak_bytes - baseline_bytes)
    finally:
        # Restore every flag directly: a blanket train(True) would unfreeze the
        # behavioral state of predictor submodules that intentionally stay eval.
        for module, training in training_flags:
            module.training = training
    average_seconds = mean(durations)
    if average_seconds <= 0:
        raise RuntimeError('Benchmark timer did not advance')
    parameters = list(model.parameters())
    total = sum(p.numel() for p in parameters)
    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    return dict(
        protocol='resident_first_test_batch_full_cascade_v1',
        device=str(mask.device), batch_size=actual_batch,
        warmup_iterations=WARMUP_ITERATIONS_PIN,
        measured_iterations=MEASURED_ITERATIONS_PIN,
        includes_op_predictor=True, uses_cached_op=False,
        input_transfer_included=False, checkpoint_selection=False,
        model_parameters=total, trainable_parameters=trainable,
        frozen_parameters=total - trainable,
        mean_latency_ms=1000 * average_seconds,
        latency_std_ms=1000 * stdev(durations),
        samples_per_second=actual_batch / average_seconds,
        baseline_allocated_bytes=baseline_bytes, peak_allocated_bytes=peak_bytes,
        incremental_peak_allocated_bytes=incremental_bytes,
        memory_scope='CUDA process with resident data/models; incremental peak is above post-warmup baseline',
        timing_scope='complete uncached cascade forward on one repeated resident test batch; excludes preprocessing, transfers and metrics',
    )
