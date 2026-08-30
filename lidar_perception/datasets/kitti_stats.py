"""Ground-truth statistics for a KITTI Object Detection split."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from lidar_perception.geometry.boxes3d import count_points_in_box
from .kitti_adapter import KittiAdapter


def _summary(values: list[float], width: int) -> dict[str, object]:
    if not values:
        return {"count": 0, "mean": [None] * width, "min": [None] * width, "max": [None] * width}
    array = np.asarray(values, dtype=np.float64)
    if width == 1:
        return {"count": int(len(array)), "mean": float(array.mean()), "min": float(array.min()), "max": float(array.max())}
    return {"count": int(len(array)), "mean": array.mean(axis=0).tolist(), "min": array.min(axis=0).tolist(), "max": array.max(axis=0).tolist()}


def compute_kitti_statistics(
    adapter: KittiAdapter,
    frame_ids: Iterable[str] | None = None,
    classes: set[str] | None = None,
) -> dict[str, object]:
    """Compute deterministic ground-truth counts without detector evaluation."""

    selected_ids = list(adapter.frame_ids() if frame_ids is None else frame_ids)
    class_counts: Counter[str] = Counter()
    distances: list[float] = []
    sizes: list[list[float]] = []
    points_per_box: list[float] = []
    per_class_points: dict[str, list[float]] = {}
    per_class_sizes: dict[str, list[list[float]]] = {}
    distance_bins = ((0.0, 10.0, "0-10m"), (10.0, 20.0, "10-20m"), (20.0, 30.0, "20-30m"), (30.0, 40.0, "30-40m"), (40.0, 50.0, "40-50m"), (50.0, float("inf"), "50m+"))
    distance_counts = {name: 0 for _, _, name in distance_bins}
    for frame_id in selected_ids:
        points = adapter.load_points(frame_id)[:, :3]
        for box in adapter.load_boxes(frame_id, classes=classes):
            class_counts[box.label] += 1
            distance = float(np.linalg.norm(box.center[:2]))
            distances.append(distance)
            sizes.append(box.size.tolist())
            count = float(count_points_in_box(points, box))
            points_per_box.append(count)
            per_class_points.setdefault(box.label, []).append(count)
            per_class_sizes.setdefault(box.label, []).append(box.size.tolist())
            for lower, upper, name in distance_bins:
                if lower <= distance < upper:
                    distance_counts[name] += 1
                    break
    return {
        "dataset": "KITTI Object Detection",
        "split": adapter.split,
        "frame_count": len(selected_ids),
        "class_counts": dict(sorted(class_counts.items())),
        "distance_definition": "sqrt(x^2 + y^2) of the internal LiDAR box center",
        "distance_distribution": distance_counts,
        "distance_summary_m": _summary(distances, 1),
        "box_size_definition": "internal [length, width, height] in meters",
        "box_size_statistics": _summary(sizes, 3),
        "box_size_statistics_by_class": {key: _summary(value, 3) for key, value in sorted(per_class_sizes.items())},
        "points_per_gt_box_statistics": _summary(points_per_box, 1),
        "points_per_gt_box_by_class": {key: _summary(value, 1) for key, value in sorted(per_class_points.items())},
    }
