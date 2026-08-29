"""KITTI LiDAR/camera/image projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .boxes3d import Box3D, box3d_corners
from .transforms import make_transform, transform_points

if TYPE_CHECKING:
    from lidar_perception.datasets.kitti_adapter import KittiCalibration


@dataclass(frozen=True)
class ProjectionResult:
    """Projection arrays that can also be unpacked as a 3-tuple."""

    pixels: np.ndarray
    depth: np.ndarray
    valid_mask: np.ndarray
    inside_image: np.ndarray | None = None
    bbox: np.ndarray | None = None

    def __iter__(self):
        yield self.pixels
        yield self.depth
        yield self.valid_mask


def _matrices(
    calibration: "KittiCalibration | np.ndarray",
    r0_rect: np.ndarray | None,
    tr_velo_to_cam: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(calibration, "P2"):
        p2 = np.asarray(calibration.P2, dtype=np.float64)
        rect = np.asarray(calibration.R0_rect, dtype=np.float64)
        velo = np.asarray(calibration.Tr_velo_to_cam, dtype=np.float64)
    else:
        p2 = np.asarray(calibration, dtype=np.float64)
        if r0_rect is None or tr_velo_to_cam is None:
            raise ValueError("r0_rect and tr_velo_to_cam are required when calibration is a matrix")
        rect = np.asarray(r0_rect, dtype=np.float64)
        velo = np.asarray(tr_velo_to_cam, dtype=np.float64)
    if p2.shape != (3, 4):
        raise ValueError(f"P2 must have shape (3, 4), got {p2.shape}")
    if rect.shape == (4, 4):
        rect = rect[:3, :3]
    if velo.shape == (4, 4):
        velo = velo[:3, :]
    if rect.shape != (3, 3):
        raise ValueError(f"R0_rect must have shape (3, 3) or (4, 4), got {rect.shape}")
    if velo.shape != (3, 4):
        raise ValueError(f"Tr_velo_to_cam must have shape (3, 4) or (4, 4), got {velo.shape}")
    if not all(np.all(np.isfinite(value)) for value in (p2, rect, velo)):
        raise ValueError("calibration matrices must contain only finite values")
    return p2, rect, velo


def project_points_to_image(
    points: np.ndarray,
    calibration: "KittiCalibration | np.ndarray",
    r0_rect: np.ndarray | None = None,
    tr_velo_to_cam: np.ndarray | None = None,
    image_shape: tuple[int, int] | None = None,
    eps: float = 1e-6,
) -> ProjectionResult:
    """Project Velodyne points to image pixels using ``P2 R0_rect Tr_velo_to_cam``."""

    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points_array.shape}")
    if not np.all(np.isfinite(points_array)):
        raise ValueError("points must contain only finite values")
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    p2, rect, velo = _matrices(calibration, r0_rect, tr_velo_to_cam)
    rect_transform = make_transform(rect) @ make_transform(velo[:, :3], velo[:, 3])
    points_rect = transform_points(points_array, rect_transform)
    homogeneous = np.column_stack((points_rect, np.ones(len(points_rect))))
    image_h = homogeneous @ p2.T
    depth = points_rect[:, 2]
    denominator = image_h[:, 2]
    valid = np.isfinite(depth) & (depth > eps) & np.isfinite(denominator) & (np.abs(denominator) > eps)
    pixels = np.full((len(points_array), 2), np.nan, dtype=np.float64)
    pixels[valid] = image_h[valid, :2] / denominator[valid, None]
    inside = None
    if image_shape is not None:
        if len(image_shape) != 2:
            raise ValueError("image_shape must be (height, width)")
        height, width = map(int, image_shape)
        if height <= 0 or width <= 0:
            raise ValueError("image_shape values must be positive")
        inside = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return ProjectionResult(pixels=pixels, depth=depth, valid_mask=valid, inside_image=inside)


def project_box_to_image(
    box: Box3D,
    calibration: "KittiCalibration | np.ndarray",
    r0_rect: np.ndarray | None = None,
    tr_velo_to_cam: np.ndarray | None = None,
    image_shape: tuple[int, int] | None = None,
    eps: float = 1e-6,
) -> ProjectionResult:
    """Project all valid corners of a LiDAR-frame box to an image.

    Corners behind the camera are marked invalid; valid corners still produce a
    finite partial projection and never contaminate the result with NaNs.
    """

    if not isinstance(box, Box3D):
        raise TypeError("box must be a Box3D")
    result = project_points_to_image(box3d_corners(box), calibration, r0_rect, tr_velo_to_cam, image_shape, eps)
    if np.any(result.valid_mask):
        valid_pixels = result.pixels[result.valid_mask]
        bbox = np.array([valid_pixels[:, 0].min(), valid_pixels[:, 1].min(), valid_pixels[:, 0].max(), valid_pixels[:, 1].max()])
    else:
        bbox = None
    return ProjectionResult(result.pixels, result.depth, result.valid_mask, result.inside_image, bbox)


project_box3d_to_image = project_box_to_image
