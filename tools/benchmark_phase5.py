#!/usr/bin/env python3
"""Run the reproducible Phase 5 nuScenes multi-model benchmark."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from lidar_perception.benchmark.report import build_report, collect_environment, write_reports
from lidar_perception.benchmark.runner import (
    benchmark_model,
    load_cached_accuracy,
    model_provenance,
)
from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter, NuScenesError
from lidar_perception.evaluation.nuscenes import evaluate_nuscenes, evaluation_sample_tokens
from lidar_perception.utils.config import load_yaml_config

from common import load_detector_config, make_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark/phase5_nuscenes.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--models", nargs="+", help="Logical model names to include")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--evaluate-missing", action="store_true", help="Run official accuracy for available non-cached models")
    parser.add_argument("--skip-runtime", action="store_true", help="Only assemble accuracy/provenance reports")
    return parser


def _model_name(config: dict[str, Any], fallback: str) -> str:
    model = str(config.get("backend", {}).get("model", fallback)).lower()
    return {"centerpoint": "centerpoint_pointpillar", "voxelnext": "voxelnext", "pointpillar": "pointpillar_nuscenes"}.get(model, fallback)


def _historical_reference() -> dict[str, Any] | None:
    evaluation_path = Path("outputs/phase2_pointpillar/evaluation/summary.json")
    benchmark_path = Path("outputs/phase2_pointpillar/benchmark.json")
    if not evaluation_path.is_file() and not benchmark_path.is_file():
        return None
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.is_file() else None
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.is_file() else None
    ap_dict = evaluation.get("ap_dict", {}) if evaluation else {}
    end_to_end = benchmark.get("end_to_end", {}) if benchmark else {}
    return {
        "label": "Historical/reference only: KITTI PointPillars",
        "dataset": "KITTI",
        "protocol": evaluation.get("protocol") if evaluation else "OpenPCDet official KITTI AP_R40",
        "accuracy": {
            key: ap_dict.get(key)
            for key in ("Car_3d/easy_R40", "Car_3d/moderate_R40", "Car_3d/hard_R40")
        },
        "runtime": {
            "mean_ms": end_to_end.get("mean_ms"),
            "median_ms": end_to_end.get("median_ms"),
            "p95_ms": end_to_end.get("p95_ms"),
            "fps_batch1": end_to_end.get("fps_batch1"),
            "peak_memory_allocated_bytes": benchmark.get("peak_memory_allocated_bytes") if benchmark else None,
            "peak_memory_reserved_bytes": benchmark.get("peak_memory_reserved_bytes") if benchmark else None,
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    benchmark_config = load_yaml_config(args.config)
    dataset = dict(benchmark_config.get("dataset", {}))
    protocol = dict(benchmark_config.get("protocol", {}))
    output_dir = Path(args.output_dir or benchmark_config.get("outputs", {}).get("directory", "outputs/phase5_benchmark")).expanduser()
    warmup = args.warmup if args.warmup is not None else int(protocol.get("warmup", 20))
    iterations = args.iterations if args.iterations is not None else int(protocol.get("iterations", 100))
    dataset_root = args.dataset_root or dataset.get("root")
    if not dataset_root:
        raise SystemExit("Phase 5 requires dataset.root")

    model_specs: list[dict[str, Any]] = []
    for entry in benchmark_config.get("models", []):
        config_path = Path(entry["config"]).expanduser().resolve()
        config, opcdet_config_path = load_detector_config(config_path)
        logical_name = str(entry.get("name") or _model_name(config, config_path.stem))
        model_specs.append({
            "name": logical_name,
            **dict(config.get("model_metadata", {})),
            "config": config,
            "config_path": str(config_path),
            "opcdet_config_path": str(opcdet_config_path),
            "provenance": model_provenance(config, config_path),
            "sweeps": int(config.get("dataset", {}).get("max_sweeps", dataset.get("max_sweeps", 10))),
            "status": str(entry.get("status", "checkpoint_required")),
            "checkpoint_acquisition": entry.get("checkpoint_acquisition"),
        })
    if args.models:
        requested = set(args.models)
        model_specs = [spec for spec in model_specs if spec["name"] in requested]
    if not model_specs:
        raise SystemExit("no Phase 5 models selected")

    try:
        adapter = NuScenesAdapter(dataset_root, version=dataset.get("version", "v1.0-mini"), max_sweeps=int(dataset.get("max_sweeps", 10)))
        tokens = evaluation_sample_tokens(adapter, eval_set=dataset.get("split", "mini_val"))
        frame = adapter.load_sample(tokens[0], max_sweeps=int(dataset.get("max_sweeps", 10)))
    except (NuScenesError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Phase 5 dataset setup failed: {exc}") from exc

    records: list[dict[str, Any]] = []
    for spec in model_specs:
        config = spec["config"]
        model_record = dict(spec)
        model_record["accuracy"] = None
        if spec["name"] == "centerpoint_pointpillar":
            cache_path = Path("outputs/phase3_centerpoint/evaluation/summary.json")
            checksum_ok = spec["provenance"].get("checkpoint_hash_matches_expected") is True
            if cache_path.is_file() and checksum_ok:
                try:
                    model_record["accuracy"] = load_cached_accuracy(cache_path, dataset=dataset)
                    model_record["status"] = "cached_accuracy"
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        checkpoint_available = bool(spec["provenance"]["checkpoint_available"])
        if args.evaluate_missing and checkpoint_available and model_record["accuracy"] is None:
            backend = make_backend(config, spec["opcdet_config_path"])
            try:
                backend.load()
                predictions = [backend.predict(adapter.load_sample(token, max_sweeps=spec["sweeps"])) for token in tokens]
                evaluation = evaluate_nuscenes(predictions, adapter, output_dir / spec["name"] / "evaluation", eval_set=dataset.get("split", "mini_val"))
                model_record["accuracy"] = {"dataset": dataset["name"], "version": dataset["version"], "split": dataset["split"], **evaluation}
                model_record["status"] = "accuracy_completed"
            except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
                model_record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                del backend
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if not checkpoint_available:
            model_record["status"] = "blocked"
            model_record["error"] = "matching official checkpoint is not available locally"
            model_record["load_report"] = {
                "load_result": "not_attempted_missing_checkpoint",
                "loaded_key_count": None,
                "missing_keys": None,
                "unexpected_keys": None,
                "shape_mismatch_keys": None,
            }
        if not args.skip_runtime and checkpoint_available:
            model_record = benchmark_model(
                model_record,
                frame,
                lambda current: make_backend(current["config"], current["opcdet_config_path"]),
                warmup=warmup,
                iterations=iterations,
            )
        model_record.pop("config", None)
        records.append(model_record)

    limitations = [
        "Current accuracy is nuScenes v1.0-mini pipeline validation, not full train/val.",
        "Runtime target is the local RTX 2060 6GB environment; no paper/model-zoo runtime numbers are used.",
    ]
    blocked = [record for record in records if record["status"] == "blocked"]
    limitations.extend(f"{record['name']}: {record.get('error')}" for record in blocked)
    report = build_report(
        dataset=dataset,
        protocol=protocol,
        models=records,
        environment=collect_environment(),
        historical_reference=_historical_reference(),
        limitations=limitations,
    )
    paths = write_reports(report, output_dir)
    print(json.dumps({"status": report["status"], "models": [record["name"] for record in records], "reports": {key: str(value) for key, value in paths.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
