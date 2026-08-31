"""Inference-only predicted-box features for future Phase 6 policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.geometry.boxes3d import Box3D, count_points_in_box


POINT_SOURCES = {"current_keyframe", "multi_sweep"}


@dataclass(frozen=True)
class PredictedBoxFeatures:
    prediction_index: int
    range_m: float
    point_count: int
    point_source: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        feature_name = (
            "predicted_box_keyframe_point_count"
            if self.point_source == "current_keyframe"
            else "predicted_box_multi_sweep_point_count"
        )
        value[feature_name] = self.point_count
        return value

    @property
    def predicted_box_keyframe_point_count(self) -> int:
        if self.point_source != "current_keyframe":
            raise ValueError("feature was not computed from current-keyframe points")
        return self.point_count


def predicted_box_range_m(box: Box3D) -> float:
    """Compute inference-time range from a predicted LiDAR-frame center."""

    if not isinstance(box, Box3D):
        raise TypeError("box must be a predicted Box3D")
    return float(np.hypot(box.center[0], box.center[1]))


def _select_points(points: np.ndarray, point_source: str) -> np.ndarray:
    if point_source not in POINT_SOURCES:
        raise ValueError(f"point_source must be one of {sorted(POINT_SOURCES)}")
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < 3 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("points must have numeric shape (N, 3+)")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    if point_source == "current_keyframe" and array.shape[1] >= 5:
        array = array[np.isclose(array[:, 4], 0.0, atol=1e-7)]
    return np.asarray(array[:, :3], dtype=np.float64)


def _count_points_in_predicted_box(points: np.ndarray, box: Box3D, tolerance: float = 1e-6) -> int:
    """Apply an exact AABB prefilter before the shared oriented-box test."""

    corners = box.corners
    lower = np.min(corners, axis=0) - tolerance
    upper = np.max(corners, axis=0) + tolerance
    candidates = points[np.all((points >= lower) & (points <= upper), axis=1)]
    return count_points_in_box(candidates, box, tolerance=tolerance)


def extract_prediction_features(
    prediction: PredictionBatch,
    sensor_points: np.ndarray,
    *,
    point_source: str = "current_keyframe",
) -> list[PredictedBoxFeatures]:
    """Extract range and predicted-box point count without accepting GT input."""

    if not isinstance(prediction, PredictionBatch):
        raise TypeError("prediction must be a PredictionBatch")
    xyz = _select_points(sensor_points, point_source)
    return [
        PredictedBoxFeatures(
            prediction_index=index,
            range_m=predicted_box_range_m(box),
            point_count=_count_points_in_predicted_box(xyz, box),
            point_source=point_source,
        )
        for index, box in enumerate(prediction.boxes)
    ]


def predicted_box_keyframe_point_counts(
    prediction: PredictionBatch,
    sensor_points: np.ndarray,
) -> np.ndarray:
    """Return inference-time current-keyframe counts aligned to predictions."""

    features = extract_prediction_features(prediction, sensor_points, point_source="current_keyframe")
    return np.asarray([feature.predicted_box_keyframe_point_count for feature in features], dtype=np.int64)


__all__ = [
    "POINT_SOURCES",
    "PredictedBoxFeatures",
    "extract_prediction_features",
    "predicted_box_keyframe_point_counts",
    "predicted_box_range_m",
]
