"""CUDA-synchronized detector latency and memory benchmark."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any

import torch

from lidar_perception.datasets.schemas import PointCloudFrame
from lidar_perception.detection.openpcdet_backend import OpenPCDetBackend, clone_batch


def _cuda_elapsed_ms(backend: OpenPCDetBackend, fn) -> float:
    if backend.device.type != "cuda":
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    fn()
    end_event.record()
    end_event.synchronize()
    return float(start_event.elapsed_time(end_event))


def benchmark_pointpillar(
    backend: OpenPCDetBackend,
    frame: PointCloudFrame,
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, Any]:
    """Measure model-only latency with CUDA events and end-to-end wall time."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")
    prepared = backend.prepare_frame(frame)
    # Convert once before model-only timing. This keeps CPU cloning and host to
    # device transfer out of the CUDA-event interval.
    if hasattr(backend, "_prepare_batch_for_device") and hasattr(backend, "_forward_prepared"):
        model_batch = backend._prepare_batch_for_device(prepared)
        timed_forward = lambda: backend._forward_prepared(model_batch)
        model_timing_method = "cuda_events" if backend.device.type == "cuda" else "perf_counter"
    else:
        # Small fakes used by unit tests may implement only the older boundary.
        timed_forward = lambda: backend._predict_prepared(clone_batch(prepared))
        model_timing_method = "cuda_events_including_prepare" if backend.device.type == "cuda" else "perf_counter_including_prepare"
    for _ in range(warmup):
        timed_forward()
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    model_times = [_cuda_elapsed_ms(backend, timed_forward) for _ in range(iterations)]
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)
        model_peak_allocated = int(torch.cuda.max_memory_allocated(backend.device))
        model_peak_reserved = int(torch.cuda.max_memory_reserved(backend.device))
    else:
        model_peak_allocated = model_peak_reserved = 0
    end_to_end_times: list[float] = []
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    for _ in range(iterations):
        start = time.perf_counter()
        backend.predict(frame)
        if backend.device.type == "cuda":
            torch.cuda.synchronize(backend.device)
        end_to_end_times.append((time.perf_counter() - start) * 1000.0)
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)
        end_to_end_peak_allocated = int(torch.cuda.max_memory_allocated(backend.device))
        end_to_end_peak_reserved = int(torch.cuda.max_memory_reserved(backend.device))
    else:
        end_to_end_peak_allocated = end_to_end_peak_reserved = 0
    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        return {
            "mean_ms": float(statistics.fmean(values)),
            "median_ms": float(statistics.median(values)),
            "p95_ms": float(ordered[p95_index]),
            "fps_batch1": float(1000.0 / statistics.fmean(values)),
        }
    return {
        "model": backend.name(),
        "gpu": torch.cuda.get_device_name(backend.device) if backend.device.type == "cuda" else "cpu",
        "device": str(backend.device),
        "batch_size": 1,
        "precision": "FP32",
        "warmup": warmup,
        "iterations": iterations,
        "input_frame": frame.frame_id,
        "model_only": summary(model_times),
        "end_to_end": summary(end_to_end_times),
        "timing_method": {
            "model_only": model_timing_method,
            "end_to_end": "perf_counter_plus_cuda_synchronize" if backend.device.type == "cuda" else "perf_counter",
        },
        "preprocessing_in_model_only": False,
        "preprocessing_in_end_to_end": True,
        "postprocessing_included": True,
        "peak_memory_allocated_bytes": end_to_end_peak_allocated,
        "peak_memory_reserved_bytes": end_to_end_peak_reserved,
        "model_only_peak_memory_allocated_bytes": model_peak_allocated,
        "model_only_peak_memory_reserved_bytes": model_peak_reserved,
        "end_to_end_peak_memory_allocated_bytes": end_to_end_peak_allocated,
        "end_to_end_peak_memory_reserved_bytes": end_to_end_peak_reserved,
        "peak_memory_scope": "end_to_end after warmup",
    }
