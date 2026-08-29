import numpy as np

from lidar_perception.datasets.kitti_adapter import KittiCalibration
from lidar_perception.geometry.boxes3d import Box3D
from lidar_perception.geometry.projection import project_box_to_image, project_points_to_image


def synthetic_calibration() -> KittiCalibration:
    return KittiCalibration(
        P2=np.array([[100.0, 0, 0, 0], [0, 100.0, 0, 0], [0, 0, 1, 0]]),
        R0_rect=np.eye(3),
        Tr_velo_to_cam=np.hstack((np.eye(3), np.zeros((3, 1)))),
    )


def test_positive_negative_and_batch_projection() -> None:
    result = project_points_to_image(np.array([[1, 2, 10], [0, 0, -1], [2, 1, 5], [1, 1, 1e-8]]), synthetic_calibration(), image_shape=(100, 100))
    assert np.allclose(result.pixels[0], [10, 20])
    assert np.allclose(result.depth, [10, -1, 5, 1e-8])
    assert result.valid_mask.tolist() == [True, False, True, False]
    assert result.inside_image.tolist() == [True, False, True, False]
    assert np.isnan(result.pixels[1]).all()
    assert np.isnan(result.pixels[3]).all()


def test_projection_accepts_raw_calibration_matrices() -> None:
    calibration = synthetic_calibration()
    pixels, depth, valid = project_points_to_image(np.array([[1, 2, 10]]), calibration.P2, calibration.R0_rect, calibration.Tr_velo_to_cam)
    assert np.allclose(pixels, [[10, 20]])
    assert depth.tolist() == [10.0]
    assert valid.tolist() == [True]


def test_box_projection_returns_finite_valid_corners_and_bbox() -> None:
    result = project_box_to_image(Box3D([0, 0, 10], [2, 2, 2], 0, "Car"), synthetic_calibration(), image_shape=(100, 100))
    assert result.pixels.shape == (8, 2)
    assert result.valid_mask.all()
    assert np.all(np.isfinite(result.pixels))
    assert np.allclose(result.bbox, [-(100 / 9), -(100 / 9), 100 / 9, 100 / 9])


def test_box_projection_handles_corners_behind_camera() -> None:
    result = project_box_to_image(Box3D([0, 0, 0.5], [2, 2, 2], 0, "Car"), synthetic_calibration())
    assert result.valid_mask.sum() == 4
    assert np.isfinite(result.pixels[result.valid_mask]).all()
    assert np.isnan(result.pixels[~result.valid_mask]).all()
    assert result.bbox is not None
