"""Yaw-only oriented 3D boxes in the project's LiDAR convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box3D:
    """An oriented box with ``center``, ``size=[length,width,height]`` and yaw."""

    center: np.ndarray
    size: np.ndarray
    yaw: float
    label: str
    score: float | None = None
    velocity: np.ndarray | None = None
    track_id: str | None = None

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.size = np.asarray(self.size, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {self.center.shape}")
        if self.size.shape != (3,):
            raise ValueError(f"size must have shape (3,) as [length, width, height], got {self.size.shape}")
        if not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.size)):
            raise ValueError("center and size must contain only finite values")
        if np.any(self.size <= 0):
            raise ValueError("size values must be positive and ordered [length, width, height]")
        self.yaw = float(self.yaw)
        if not np.isfinite(self.yaw):
            raise ValueError("yaw must be finite")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.score is not None:
            self.score = float(self.score)
            if not np.isfinite(self.score):
                raise ValueError("score must be finite when provided")
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float64)
            if self.velocity.shape != (3,):
                raise ValueError(f"velocity must have shape (3,), got {self.velocity.shape}")
            if not np.all(np.isfinite(self.velocity)):
                raise ValueError("velocity must contain only finite values")
        if self.track_id is not None and not isinstance(self.track_id, str):
            raise TypeError("track_id must be a string or None")

    @property
    def corners(self) -> np.ndarray:
        return box3d_corners(self)

    @property
    def bev(self) -> np.ndarray:
        return bev_corners(self)


def box3d_corners(box: Box3D) -> np.ndarray:
    """Return eight corners, bottom four followed by top four, in LiDAR frame."""

    if not isinstance(box, Box3D):
        raise TypeError("box must be a Box3D")
    length, width, height = box.size
    local = np.array(
        [
            [length / 2, width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [-length / 2, -width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [length / 2, width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [-length / 2, -width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ],
        dtype=np.float64,
    )
    cos_yaw, sin_yaw = np.cos(box.yaw), np.sin(box.yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    corners = local.copy()
    corners[:, :2] = local[:, :2] @ rotation.T
    corners += box.center
    return corners


def bev_corners(box: Box3D) -> np.ndarray:
    """Return the four oriented XY corners in a consistent cyclic order."""

    return box3d_corners(box)[:4, :2]


def point_in_box(points: np.ndarray, box: Box3D, tolerance: float = 1e-6) -> np.ndarray:
    """Return an inclusive mask for points inside an oriented 3D box.

    Points on a face are inside when they are within ``tolerance`` of that
    face. The test rotates world/LiDAR points into the box-local frame first,
    so non-zero yaw is handled identically to axis-aligned boxes.
    """

    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points_array.shape}")
    if not np.all(np.isfinite(points_array)):
        raise ValueError("points must contain only finite values")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative value")
    relative = points_array - box.center
    cos_yaw, sin_yaw = np.cos(box.yaw), np.sin(box.yaw)
    local_x = cos_yaw * relative[:, 0] + sin_yaw * relative[:, 1]
    local_y = -sin_yaw * relative[:, 0] + cos_yaw * relative[:, 1]
    local = np.column_stack((local_x, local_y, relative[:, 2]))
    return np.all(np.abs(local) <= box.size / 2 + tolerance, axis=1)


def points_in_box(points: np.ndarray, box: Box3D, tolerance: float = 1e-6) -> np.ndarray:
    """Alias for :func:`point_in_box` using plural naming."""

    return point_in_box(points, box, tolerance=tolerance)


def count_points_in_box(points: np.ndarray, box: Box3D, tolerance: float = 1e-6) -> int:
    """Count points inside an oriented box."""

    return int(np.count_nonzero(point_in_box(points, box, tolerance=tolerance)))
