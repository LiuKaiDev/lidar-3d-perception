"""Project-owned data schemas shared by dataset and geometry code."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validate_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass
class PointCloudFrame:
    """A single point-cloud frame in the project's canonical representation.

    KITTI does not provide the ego/global poses used by nuScenes, so its adapter
    stores identity transforms and records that limitation in ``metadata``.
    Points contain at least ``x, y, z, intensity`` in that order.
    """

    frame_id: str
    points: np.ndarray
    timestamp: int | float | None = None
    lidar_to_ego: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    ego_to_global: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        points = np.asarray(self.points)
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError(
                f"points must have shape (N, 4+) with x/y/z/intensity, got {points.shape}"
            )
        if not np.issubdtype(points.dtype, np.number):
            raise TypeError("points must contain numeric values")
        points = points.astype(np.float32, copy=False)
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")
        self.points = points
        self.lidar_to_ego = _validate_matrix(self.lidar_to_ego, "lidar_to_ego")
        self.ego_to_global = _validate_matrix(self.ego_to_global, "ego_to_global")
        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
