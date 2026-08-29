import numpy as np

from lidar_perception.geometry.boxes3d import Box3D, bev_corners, box3d_corners, count_points_in_box, point_in_box


def test_axis_aligned_box_inclusive_boundary_and_count() -> None:
    box = Box3D([0, 0, 0], [2, 2, 2], 0, "box")
    points = np.array([[0, 0, 0], [1, 1, 1], [1.00001, 0, 0], [-1, 0, 0]])
    assert point_in_box(points, box).tolist() == [True, True, False, True]
    assert count_points_in_box(points, box) == 3


def test_rotated_box_supports_ninety_degrees_and_arbitrary_yaw() -> None:
    box = Box3D([0, 0, 0], [4, 2, 2], np.pi / 2, "box")
    assert point_in_box(np.array([[0, 1.9, 0], [1.9, 0, 0]]), box).tolist() == [True, False]
    arbitrary = Box3D([2, -1, 0.5], [4, 2, 2], 0.37, "box")
    inside = arbitrary.center.reshape(1, 3)
    outside = inside + np.array([[3, 3, 0]])
    assert point_in_box(inside, arbitrary).item()
    assert not point_in_box(outside, arbitrary).item()


def test_corners_are_unique_and_centered() -> None:
    box = Box3D([1, 2, 3], [4, 2, 6], 0.4, "box")
    corners = box3d_corners(box)
    assert corners.shape == (8, 3)
    assert len(np.unique(np.round(corners, 10), axis=0)) == 8
    assert np.allclose(corners.mean(axis=0), box.center)
    assert bev_corners(box).shape == (4, 2)
