"""Small point-cloud utilities used by the KITTI tools."""

from __future__ import annotations

import numpy as np


def validate_points(points: np.ndarray, min_features: int = 3) -> np.ndarray:
    """Validate and return a floating point ``(N, min_features+)`` array."""

    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < min_features:
        raise ValueError(f"points must have shape (N, {min_features}+), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("points must contain numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    return array


def point_ranges(points: np.ndarray) -> np.ndarray:
    """Return Euclidean sensor range for each point using its XYZ columns."""

    array = validate_points(points, min_features=3)
    return np.linalg.norm(array[:, :3], axis=1)


def filter_points_by_range(points: np.ndarray, min_range: float = 0.0, max_range: float = np.inf) -> np.ndarray:
    """Keep points whose sensor range lies in the inclusive range."""

    if np.isnan(min_range) or np.isnan(max_range) or min_range < 0 or max_range < min_range:
        raise ValueError("range bounds must satisfy 0 <= min_range <= max_range")
    array = validate_points(points, min_features=3)
    ranges = point_ranges(array)
    return array[(ranges >= min_range) & (ranges <= max_range)]


def filter_points_by_bounds(points: np.ndarray, bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    """Keep points inside inclusive ``(xmin, xmax, ymin, ymax, zmin, zmax)`` bounds."""

    if len(bounds) != 6:
        raise ValueError("bounds must contain xmin, xmax, ymin, ymax, zmin, zmax")
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds)
    if xmin > xmax or ymin > ymax or zmin > zmax:
        raise ValueError("each lower bound must be <= its upper bound")
    array = validate_points(points, min_features=3)
    xyz = array[:, :3]
    mask = (
        (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)
        & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
        & (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    )
    return array[mask]
