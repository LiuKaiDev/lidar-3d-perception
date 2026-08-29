import numpy as np
import pytest

from lidar_perception.datasets.schemas import PointCloudFrame
from lidar_perception.geometry.boxes3d import Box3D


def test_point_cloud_frame_valid_construction() -> None:
    frame = PointCloudFrame("000001", np.zeros((2, 4), dtype=np.float32))
    assert frame.points.shape == (2, 4)
    assert np.array_equal(frame.lidar_to_ego, np.eye(4))
    assert np.array_equal(frame.ego_to_global, np.eye(4))


@pytest.mark.parametrize("points", [np.zeros((2, 3)), np.zeros((2, 5, 1)), np.array([["bad"]])])
def test_point_cloud_frame_rejects_malformed_points(points: np.ndarray) -> None:
    with pytest.raises((ValueError, TypeError)):
        PointCloudFrame("000001", points)


def test_box3d_validates_shapes_and_optional_fields() -> None:
    box = Box3D([1, 2, 3], [4, 2, 1], 0.5, "Car", score=0.9, velocity=[1, 0, 0], track_id="a")
    assert box.size.tolist() == [4, 2, 1]
    assert box.score == 0.9
    assert box.velocity.tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
        Box3D([1, 2], [4, 2, 1], 0, "Car")
    with pytest.raises(ValueError):
        Box3D([1, 2, 3], [4, -1, 1], 0, "Car")
