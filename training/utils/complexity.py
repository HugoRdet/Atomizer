"""
Complexity measurement utilities.
Usage:
    from training.utils.complexity import measure_flops, measure_inference_time, batch_to_device
"""

import time
import torch


def batch_to_device(batch: dict, device) -> dict:
    """Recursively move a nested batch dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = batch_to_device(v, device)
        else:
            out[k] = v
    return out


def measure_flops(model, batch, device):
    """
    Measure FLOPs using torch.profiler.
    Handles einsum, custom attention, and other ops that fvcore misses.

    Args:
        model: nn.Module (already on device, in eval mode)
        batch: dict batch (will be moved to device)
        device: torch.device

    Returns:
        float: GFLOPs
    """
    batch = batch_to_device(batch, device)

    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True,
        ) as prof:
            _ = model(batch)

    total_flops = 0
    for event in prof.key_averages():
        if event.flops is not None and event.flops > 0:
            total_flops += event.flops

    return total_flops / 1e9


def measure_inference_time(model, samples, collate_fn, device, num_warmup=5, num_runs=30):
    """
    Measure average inference time per sample.

    Args:
        model: nn.Module (already on device, in eval mode)
        samples: list of dataset samples
        collate_fn: function to collate a single sample into a batch
        device: torch.device
        num_warmup: warmup iterations
        num_runs: timed iterations

    Returns:
        float: average inference time in milliseconds
    """
    num_samples = len(samples)

    with torch.no_grad():
        for i in range(num_warmup):
            b = collate_fn([samples[i % num_samples]])
            b = batch_to_device(b, device)
            _ = model(b)

        torch.cuda.synchronize()
        start = time.time()
        for i in range(num_runs):
            idx = (i + num_warmup) % num_samples
            b = collate_fn([samples[idx]])
            b = batch_to_device(b, device)
            _ = model(b)
            torch.cuda.synchronize()
        end = time.time()

    return (end - start) / num_runs * 1000