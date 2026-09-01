#!/usr/bin/env python3
"""Run one nuScenes sample with the default detector or frozen E3 fusion."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from phase7_common import ROOT, load_project_config, load_system_config, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("voxelnext", "centerpoint", "e3"), default=None, help="Detector mode; defaults to VoxelNeXt")
    parser.add_argument("--sample-token", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--max-sweeps", type=int)
    parser.add_argument("--checkpoint", help="Optional checkpoint override for single-detector modes")
    parser.add_argument("--output")
    return parser


def _backend_config(portfolio: dict[str, Any], detector: str) -> Path:
    key = "voxelnext" if detector == "voxelnext" else "centerpoint"
    return resolve_path(portfolio["detectors"][key]["config"])


def select_detector(requested: str | None, portfolio: dict[str, Any]) -> str:
    detector = requested or str(portfolio["default_detector"])
    if detector not in {"voxelnext", "centerpoint", "e3"}:
        raise ValueError(f"unsupported detector: {detector}")
    return detector


def default_output_path(portfolio: dict[str, Any], detector: str, sample_token: str) -> Path:
    output_root = resolve_path(portfolio["dataset"].get("default_output_dir", "outputs/demo"))
    return output_root / detector / f"{sample_token}.json"


def _load_backend(detector: str, config_path: Path, checkpoint: str | None = None):
    # Heavy Torch/OpenPCDet imports happen only after argument parsing.
    from common import load_detector_config, make_backend

    config, opcdet_config = load_detector_config(config_path)
    backend = make_backend(config, opcdet_config, checkpoint)
    start = time.perf_counter()
    backend.load()
    return backend, config, (time.perf_counter() - start) * 1000.0


def _release_backend(backend: Any) -> None:
    if hasattr(backend, "model"):
        backend.model = None
    if hasattr(backend, "dataset"):
        backend.dataset = None
    del backend
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _prediction_payload(prediction: Any, *, detector: str, source: str, timing: dict[str, Any]) -> dict[str, Any]:
    boxes = prediction.to_dict()["boxes"]
    return {
        "schema_version": "lidar_perception.demo_nuscenes.v1",
        "sample_token": prediction.frame_id,
        "detector": detector,
        "source": source,
        "prediction": prediction.to_dict(),
        "summary": {
            "prediction_count": len(boxes),
            "classes": sorted({box["label"] for box in boxes}),
            "velocity_count": sum(box["velocity"] is not None for box in boxes),
            "scores": [box["score"] for box in boxes[:5]],
        },
        "timing": timing,
    }


def _run_single(args: argparse.Namespace, portfolio: dict[str, Any], detector: str, frame: Any) -> dict[str, Any]:
    config_path = _backend_config(portfolio, detector)
    load_start = time.perf_counter()
    backend, _config, load_ms = _load_backend(detector, config_path, args.checkpoint)
    load_wall_ms = (time.perf_counter() - load_start) * 1000.0
    try:
        import torch

        predict_start = time.perf_counter()
        with torch.inference_mode():
            prediction = backend.predict(frame)
        predict_wall_ms = (time.perf_counter() - predict_start) * 1000.0
        return _prediction_payload(
            prediction,
            detector=detector,
            source=detector,
            timing={
                "prediction_runtime_ms": prediction.runtime_ms,
                "predict_wall_ms": predict_wall_ms,
                "model_load_wall_ms": load_wall_ms,
                "model_load_ms": load_ms,
                "runtime_semantics": "PredictionBatch.runtime_ms is synchronized model forward/decode/NMS only; it is not end-to-end latency",
            },
        )
    finally:
        _release_backend(backend)


def _run_e3(args: argparse.Namespace, portfolio: dict[str, Any], frame: Any) -> dict[str, Any]:
    from lidar_perception.experiments.fusion import FusionConfig, fuse_predictions

    frozen_path = resolve_path(portfolio["e3"]["frozen_config"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    config = FusionConfig.from_dict(frozen["selected_config"])
    cp_config = _backend_config(portfolio, "centerpoint")
    vn_config = _backend_config(portfolio, "voxelnext")
    import torch

    stages: dict[str, Any] = {"frozen_config": str(frozen_path.relative_to(ROOT)), "fusion_config": config.to_dict()}
    cp_load_start = time.perf_counter()
    cp_backend, _cp_project, _cp_load_ms = _load_backend("centerpoint", cp_config)
    stages["centerpoint_model_load_wall_ms"] = (time.perf_counter() - cp_load_start) * 1000.0
    try:
        cp_start = time.perf_counter()
        with torch.inference_mode():
            cp_prediction = cp_backend.predict(frame)
        stages["centerpoint_predict_wall_ms"] = (time.perf_counter() - cp_start) * 1000.0
        stages["centerpoint_prediction_runtime_ms"] = cp_prediction.runtime_ms
    finally:
        _release_backend(cp_backend)
        cp_backend = None
    vn_load_start = time.perf_counter()
    vn_backend, _vn_project, _vn_load_ms = _load_backend("voxelnext", vn_config)
    stages["voxelnext_model_load_wall_ms"] = (time.perf_counter() - vn_load_start) * 1000.0
    try:
        vn_start = time.perf_counter()
        with torch.inference_mode():
            vn_prediction = vn_backend.predict(frame)
        stages["voxelnext_predict_wall_ms"] = (time.perf_counter() - vn_start) * 1000.0
        stages["voxelnext_prediction_runtime_ms"] = vn_prediction.runtime_ms
    finally:
        _release_backend(vn_backend)
        vn_backend = None
    fusion_start = time.perf_counter()
    fused, diagnostics = fuse_predictions(cp_prediction, vn_prediction, config, return_diagnostics=True)
    fused.runtime_ms = None
    stages["fusion_wall_ms"] = (time.perf_counter() - fusion_start) * 1000.0
    stages["fusion_diagnostics"] = {
        "association_count": len(diagnostics.associations),
        "input_centerpoint_count": diagnostics.input_centerpoint_count,
        "input_voxelnext_count": diagnostics.input_voxelnext_count,
        "candidate_count_before_limit": diagnostics.candidate_count_before_limit,
        "candidate_count_after_limit": diagnostics.candidate_count_after_limit,
        "truncated": diagnostics.truncated,
        "association_time_ms": diagnostics.association_time_ms,
        "fusion_time_ms": diagnostics.fusion_time_ms,
        "sorting_time_ms": diagnostics.sorting_time_ms,
        "total_fusion_time_ms": diagnostics.association_time_ms + diagnostics.fusion_time_ms + diagnostics.sorting_time_ms,
    }
    stages["cli_wall_scope"] = "dataset initialization, frame loading, model loads, synchronized detector prediction, and CPU fusion; process startup excluded"
    stages["detector_execution_model"] = "sequential; CenterPoint released before VoxelNeXt load"
    stages["detector_runtime_sum_ms"] = float(cp_prediction.runtime_ms or 0.0) + float(vn_prediction.runtime_ms or 0.0)
    stages["detector_predict_wall_sum_ms"] = stages["centerpoint_predict_wall_ms"] + stages["voxelnext_predict_wall_ms"]
    stages["e3_total_wall_ms"] = stages["centerpoint_model_load_wall_ms"] + stages["voxelnext_model_load_wall_ms"] + stages["detector_predict_wall_sum_ms"] + stages["fusion_wall_ms"]
    return _prediction_payload(fused, detector="e3", source="centerpoint+voxelnext", timing=stages)


def main() -> int:
    args = build_parser().parse_args()
    portfolio = load_system_config()
    detector = select_detector(args.detector, portfolio)
    dataset = portfolio["dataset"]
    root = resolve_path(args.dataset_root or __import__("os").environ.get("NUSCENES_ROOT", dataset.get("root", "~/datasets/nuscenes")))
    version = str(dataset.get("version", "v1.0-mini"))
    sweeps = int(args.max_sweeps or dataset.get("max_sweeps", 10))
    output = resolve_path(args.output) if args.output else default_output_path(portfolio, detector, args.sample_token)
    overall_start = time.perf_counter()
    NuScenesError = RuntimeError
    try:
        from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter, NuScenesError

        adapter = NuScenesAdapter(root, version=version, max_sweeps=sweeps)
        frame_start = time.perf_counter()
        frame = adapter.load_sample(args.sample_token, max_sweeps=sweeps)
        frame_load_ms = (time.perf_counter() - frame_start) * 1000.0
        if detector == "e3" and args.checkpoint:
            raise ValueError("--checkpoint is only supported for single-detector modes; E3 uses frozen detector configs")
        payload = _run_e3(args, portfolio, frame) if detector == "e3" else _run_single(args, portfolio, detector, frame)
        payload["timing"]["frame_load_wall_ms"] = frame_load_ms
        payload["timing"]["demo_wall_ms_before_serialization"] = (time.perf_counter() - overall_start) * 1000.0
        payload["timing"]["demo_wall_scope"] = "includes dataset initialization, frame loading, model loading, prediction, and fusion where applicable"
        payload["timing"]["dataset_root"] = str(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "detector": detector, "sample_token": args.sample_token, "prediction_count": payload["summary"]["prediction_count"], "output": str(output), "timing": payload["timing"]}, sort_keys=True))
        return 0
    except (NuScenesError, FileNotFoundError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAIL: nuScenes demo {detector}: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
