"""Explicit homogeneous transform and quaternion utilities.

Transforms follow the convention ``p_destination = T_destination_source p_source``.
Quaternions use ``[w, x, y, z]`` ordering throughout this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .boxes3d import Box3D


def _array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_transform(transform: object, name: str = "transform") -> np.ndarray:
    matrix = _array(transform, name, (4, 4))
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must be a homogeneous transform with last row [0, 0, 0, 1]")
    return matrix


def make_transform(rotation: object | None = None, translation: object | None = None) -> np.ndarray:
    """Build a 4x4 transform from a 3x3 rotation and 3-vector translation."""

    rotation_array = np.eye(3, dtype=np.float64) if rotation is None else _array(rotation, "rotation", (3, 3))
    translation_array = (
        np.zeros(3, dtype=np.float64)
        if translation is None
        else _array(translation, "translation", (3,))
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_array
    result[:3, 3] = translation_array
    return result


def invert_transform(transform: object) -> np.ndarray:
    """Return the inverse of a homogeneous transform."""

    matrix = _validate_transform(transform)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("transform must be invertible") from exc
    return _validate_transform(inverse, "inverse_transform")


def compose_transform(*transforms: object) -> np.ndarray:
    """Compose transforms left-to-right using matrix multiplication.

    ``compose_transform(T_ab, T_bc)`` returns ``T_ab @ T_bc``.
    """

    if not transforms:
        raise ValueError("at least one transform is required")
    result = np.eye(4, dtype=np.float64)
    for index, transform in enumerate(transforms):
        result = result @ _validate_transform(transform, f"transform[{index}]")
    return _validate_transform(result, "composed_transform")


def transform_points(points: object, transform: object) -> np.ndarray:
    """Apply a homogeneous transform to points with shape ``(N, 3)``."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {point_array.shape}")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must contain only finite values")
    matrix = _validate_transform(transform)
    return point_array @ matrix[:3, :3].T + matrix[:3, 3]


def quaternion_to_rotation_matrix(quaternion: object) -> np.ndarray:
    """Convert a finite ``[w, x, y, z]`` quaternion to a 3x3 matrix."""

    q = _array(quaternion, "quaternion", (4,))
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation: object) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized ``[w, x, y, z]`` quaternion."""

    matrix = _array(rotation, "rotation", (3, 3))
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError("rotation must have determinant +1")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0))
            q = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2], 0.0))
            q = np.array([(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(max(1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2], 0.0))
            q = np.array([(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def transform_box(box: "Box3D", transform: object) -> "Box3D":
    """Transform a yaw-only :class:`Box3D` while preserving ``[l, w, h]`` size.

    A general 3D rotation can introduce roll/pitch, which this yaw-only schema
    cannot represent. The transformed heading is therefore projected onto the
    destination XY plane and rejected if it has no horizontal component.
    """

    from .boxes3d import Box3D

    if not isinstance(box, Box3D):
        raise TypeError("box must be a Box3D")
    matrix = _validate_transform(transform)
    center = transform_points(box.center.reshape(1, 3), matrix)[0]
    heading = np.array([np.cos(box.yaw), np.sin(box.yaw), 0.0], dtype=np.float64)
    heading_transformed = matrix[:3, :3] @ heading
    horizontal_norm = np.linalg.norm(heading_transformed[:2])
    if horizontal_norm <= 1e-12:
        raise ValueError("transform produces a heading without a horizontal component")
    yaw = float(np.arctan2(heading_transformed[1], heading_transformed[0]))
    velocity = None if box.velocity is None else matrix[:3, :3] @ box.velocity
    return Box3D(center=center, size=box.size.copy(), yaw=yaw, label=box.label, score=box.score, velocity=velocity, track_id=box.track_id)


def transform_boxes(boxes: list["Box3D"], transform: object) -> list["Box3D"]:
    """Transform each box in a list."""

    return [transform_box(box, transform) for box in boxes]
