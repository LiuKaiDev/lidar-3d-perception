import numpy as np

from lidar_perception.geometry.transforms import (
    compose_transform,
    invert_transform,
    make_transform,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    transform_points,
)
from lidar_perception.geometry.boxes3d import Box3D
from lidar_perception.geometry.transforms import transform_box


def test_identity_transform() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
    assert np.allclose(transform_points(points, np.eye(4)), points)


def test_translation_and_rotation() -> None:
    points = np.array([[1.0, 0.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = make_transform(rotation, [2.0, 3.0, 4.0])
    assert np.allclose(transform_points(points, transform), [[2.0, 4.0, 4.0]])


def test_composition_matches_sequential_application() -> None:
    first = make_transform(translation=[1, 2, 3])
    second = make_transform(translation=[4, 5, 6])
    points = np.array([[0.5, 1.0, 2.0]])
    sequential = transform_points(transform_points(points, second), first)
    composed = transform_points(points, compose_transform(first, second))
    assert np.allclose(sequential, composed)


def test_inverse_transform() -> None:
    transform = make_transform(np.eye(3), [1.0, -2.0, 4.0])
    assert np.allclose(compose_transform(transform, invert_transform(transform)), np.eye(4))


def test_quaternion_rotation_round_trip() -> None:
    angle = np.deg2rad(35.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    quaternion = rotation_matrix_to_quaternion(rotation)
    assert np.allclose(quaternion_to_rotation_matrix(quaternion), rotation)


def test_transform_box_updates_center_heading_and_velocity() -> None:
    box = Box3D([1, 2, 3], [4, 2, 1], 0.0, "Car", velocity=[1, 0, 0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = transform_box(box, make_transform(rotation, [10, 0, 0]))
    assert np.allclose(transformed.center, [8, 1, 3])
    assert np.isclose(transformed.yaw, np.pi / 2)
    assert np.allclose(transformed.velocity, [0, 1, 0])
    assert np.allclose(transformed.size, box.size)
