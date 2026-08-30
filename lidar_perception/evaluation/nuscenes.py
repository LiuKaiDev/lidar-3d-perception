"""Thin project wrapper around the official nuScenes detection evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter
from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.geometry.transforms import transform_points


def _attribute_name(label: str, velocity_global: np.ndarray) -> str:
    speed = float(np.linalg.norm(velocity_global[:2]))
    if speed > 0.2:
        if label in {"car", "construction_vehicle", "bus", "truck", "trailer"}:
            return "vehicle.moving"
        if label in {"bicycle", "motorcycle"}:
            return "cycle.with_rider"
    if label == "pedestrian":
        return "pedestrian.standing"
    if label == "bus":
        return "vehicle.stopped"
    defaults = {
        "barrier": "cycle.with_rider",
        "bicycle": "cycle.without_rider",
        "car": "vehicle.parked",
        "construction_vehicle": "vehicle.parked",
        "motorcycle": "cycle.without_rider",
        "pedestrian": "pedestrian.standing",
        "traffic_cone": "cycle.with_rider",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
    }
    return defaults[label]


def prediction_to_nuscenes_results(
    prediction: PredictionBatch,
    adapter: NuScenesAdapter,
) -> list[dict[str, Any]]:
    """Convert project LiDAR boxes into official nuScenes result records."""

    sample_token = str(prediction.metadata.get("sample_token", prediction.frame_id))
    sample_data_token = adapter.sample_data_token(sample_token)
    global_from_sensor = adapter.sensor_to_global(sample_data_token)
    rotation = global_from_sensor[:3, :3]
    results: list[dict[str, Any]] = []
    from pyquaternion import Quaternion

    for box in prediction.boxes:
        center_global = transform_points(box.center.reshape(1, 3), global_from_sensor)[0]
        heading_sensor = np.array([np.cos(box.yaw), np.sin(box.yaw), 0.0], dtype=np.float64)
        heading_global = rotation @ heading_sensor
        yaw_global = float(np.arctan2(heading_global[1], heading_global[0]))
        velocity_sensor = np.zeros(3, dtype=np.float64) if box.velocity is None else box.velocity
        velocity_global = rotation @ velocity_sensor
        results.append(
            {
                "sample_token": sample_token,
                "translation": center_global.tolist(),
                "size": [float(box.size[1]), float(box.size[0]), float(box.size[2])],
                "rotation": Quaternion(axis=[0, 0, 1], radians=yaw_global).elements.tolist(),
                "velocity": velocity_global[:2].tolist(),
                "detection_name": box.label,
                "detection_score": float(box.score if box.score is not None else 0.0),
                "attribute_name": _attribute_name(box.label, velocity_global),
            }
        )
    return results


def evaluation_sample_tokens(adapter: NuScenesAdapter, eval_set: str = "mini_val") -> list[str]:
    """Resolve official split scene names to sample tokens."""

    try:
        from nuscenes.utils import splits
    except ImportError as exc:
        raise RuntimeError("nuscenes-devkit is required for official evaluation") from exc
    scene_names = getattr(splits, eval_set, None)
    if scene_names is None:
        raise ValueError(f"unsupported nuScenes evaluation split: {eval_set}")
    available = {record["name"] for record in adapter.scene_records}
    tokens: list[str] = []
    for scene_name in scene_names:
        if scene_name in available:
            tokens.extend(adapter.sample_tokens(scene_name))
    if not tokens:
        raise RuntimeError(f"no samples from {eval_set} are available under {adapter.root}")
    return tokens


def build_nuscenes_result_json(
    predictions: Iterable[PredictionBatch],
    adapter: NuScenesAdapter,
    sample_tokens: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the official result JSON, including empty entries for eval frames."""

    tokens = list(sample_tokens) if sample_tokens is not None else []
    results = {token: [] for token in tokens}
    for prediction in predictions:
        token = str(prediction.metadata.get("sample_token", prediction.frame_id))
        results[token] = prediction_to_nuscenes_results(prediction, adapter)
    return {
        "results": results,
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
    }


def evaluate_nuscenes(
    predictions: Iterable[PredictionBatch],
    adapter: NuScenesAdapter,
    output_dir: str | Path,
    eval_set: str = "mini_val",
) -> dict[str, Any]:
    """Run official nuScenes detection evaluation and return structured metrics."""

    try:
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval
    except ImportError as exc:
        raise RuntimeError("nuscenes-devkit is required for official evaluation") from exc
    sample_tokens = evaluation_sample_tokens(adapter, eval_set=eval_set)
    result_json = build_nuscenes_result_json(predictions, adapter, sample_tokens=sample_tokens)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "results_nusc.json"
    result_path.write_text(json.dumps(result_json), encoding="utf-8")
    try:
        eval_config = config_factory("detection_cvpr_2019")
        eval_version = "detection_cvpr_2019"
    except Exception:
        eval_config = config_factory("cvpr_2019")
        eval_version = "cvpr_2019"
    evaluator = NuScenesEval(
        adapter.nusc,
        config=eval_config,
        result_path=str(result_path),
        eval_set=eval_set,
        output_dir=str(output_path),
        verbose=False,
    )
    evaluator.main(plot_examples=0, render_curves=False)
    metrics_path = output_path / "metrics_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    tp_errors = metrics.get("tp_errors", {})
    result = {
        "label": f"nuScenes {adapter.version} / pipeline validation",
        "dataset_version": adapter.version,
        "eval_set": eval_set,
        "sample_count": len(sample_tokens),
        "protocol": eval_version,
        "mAP": metrics.get("mean_ap"),
        "NDS": metrics.get("nd_score"),
        "mATE": tp_errors.get("trans_err"),
        "mASE": tp_errors.get("scale_err"),
        "mAOE": tp_errors.get("orient_err"),
        "mAVE": tp_errors.get("vel_err"),
        "mAAE": tp_errors.get("attr_err"),
        "per_class_ap": metrics.get("label_aps", {}),
        "per_class_tp_errors": metrics.get("label_tp_errors", {}),
        "result_path": str(result_path),
    }
    (output_path / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = [
    "build_nuscenes_result_json",
    "evaluate_nuscenes",
    "evaluation_sample_tokens",
    "prediction_to_nuscenes_results",
]
