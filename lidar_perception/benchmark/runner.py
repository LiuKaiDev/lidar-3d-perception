"""Sequential, model-isolated Phase 5 benchmark orchestration."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import yaml

from lidar_perception.benchmark.latency import benchmark_pointpillar
from lidar_perception.datasets.schemas import PointCloudFrame

from .report import sha256_file


def isolate_cuda() -> None:
    """Release the previous model before loading the next one."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_cached_accuracy(
    path: str | Path,
    *,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Read a previously validated official summary with explicit provenance."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"cached accuracy summary not found: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    required = ("mAP", "NDS")
    if any(key not in value for key in required):
        raise ValueError(f"cached accuracy summary missing {required}: {source}")
    if value.get("dataset_version") != dataset.get("version") or value.get("eval_set") != dataset.get("split"):
        raise ValueError("cached accuracy dataset version/split does not match the Phase 5 protocol")
    return {
        "dataset": dataset["name"],
        "version": dataset["version"],
        "split": dataset["split"],
        "protocol": value.get("protocol"),
        "source": str(source),
        **{key: value.get(key) for key in ("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "per_class_ap")},
    }


def model_provenance(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    backend = config.get("backend", {})
    checkpoint = Path(str(backend.get("checkpoint", ""))).expanduser()
    config_file = Path(config_path).expanduser().resolve()
    opcdet_config = Path(str(backend.get("openpcdet_config", ""))).expanduser()
    if not opcdet_config.is_absolute():
        opcdet_config = (Path.cwd() / opcdet_config).resolve()
    class_names = list(config.get("classes", []))
    if not class_names and opcdet_config.is_file():
        try:
            raw = yaml.safe_load(opcdet_config.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("CLASS_NAMES"), list):
                class_names = [str(name) for name in raw["CLASS_NAMES"]]
        except (OSError, ValueError, yaml.YAMLError):
            class_names = []
    return {
        "model": backend.get("model"),
        "config": str(config_file),
        "config_sha256": sha256_file(config_file),
        "openpcdet_config": str(opcdet_config),
        "openpcdet_config_sha256": sha256_file(opcdet_config),
        "checkpoint": str(checkpoint),
        "checkpoint_source": backend.get("checkpoint_source"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "expected_checkpoint_sha256": backend.get("checkpoint_sha256"),
        "checkpoint_hash_matches_expected": (
            sha256_file(checkpoint) == backend.get("checkpoint_sha256")
            if backend.get("checkpoint_sha256") and checkpoint.is_file()
            else None
        ),
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "checkpoint_available": checkpoint.is_file(),
        "sweeps": config.get("dataset", {}).get("max_sweeps"),
        "class_names": class_names,
    }


def benchmark_model(
    spec: dict[str, Any],
    frame: PointCloudFrame,
    backend_factory: Callable[[dict[str, Any]], Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Load and benchmark one model, always cleaning it up before returning."""

    isolate_cuda()
    backend = None
    try:
        backend = backend_factory(spec)
        backend.load()
        benchmark = benchmark_pointpillar(backend, frame, warmup=warmup, iterations=iterations)
        return {
            **spec,
            "status": "completed",
            "benchmark": benchmark,
            "backend": backend.name(),
            "load_report": dict(getattr(backend, "load_report", {})),
        }
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        return {**spec, "status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        del backend
        isolate_cuda()


def run_sequential_benchmark(
    specs: Iterable[dict[str, Any]],
    frame: PointCloudFrame,
    backend_factory: Callable[[dict[str, Any]], Any],
    *,
    warmup: int = 20,
    iterations: int = 100,
) -> list[dict[str, Any]]:
    """Benchmark each configured detector one at a time under one protocol."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")
    return [
        benchmark_model(spec, frame, backend_factory, warmup=warmup, iterations=iterations)
        for spec in specs
    ]


__all__ = [
    "benchmark_model",
    "isolate_cuda",
    "load_cached_accuracy",
    "model_provenance",
    "run_sequential_benchmark",
]
