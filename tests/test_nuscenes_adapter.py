import numpy as np

from lidar_perception.datasets.nuscenes_adapter import (
    GENERAL_TO_DETECTION,
    NuScenesAdapter,
    ego_to_global_transform,
    sensor_to_ego_transform,
    sensor_to_global_transform,
    sweep_to_reference_transform,
)
from lidar_perception.geometry.transforms import invert_transform, transform_points


def _record(rotation=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
    return {"rotation": list(rotation), "translation": list(translation)}


def test_quaternion_transform_composition_round_trip() -> None:
    sensor = _record((0.9238795, 0.0, 0.0, 0.3826834), (1.0, 2.0, 0.5))
    pose = _record((0.9659258, 0.0, 0.258819, 0.0), (10.0, -3.0, 1.0))
    composed = sensor_to_global_transform(sensor, pose)
    point = np.array([[2.0, -1.0, 0.5]])
    restored = transform_points(transform_points(point, composed), invert_transform(composed))
    assert np.allclose(restored, point, atol=1e-6)
    assert np.allclose(composed, ego_to_global_transform(pose) @ sensor_to_ego_transform(sensor))


def test_sweep_to_reference_transform_applies_relative_translation() -> None:
    reference_sensor = _record(translation=(10.0, 0.0, 0.0))
    reference_pose = _record()
    sweep_sensor = _record(translation=(8.0, 0.0, 0.0))
    sweep_pose = _record()
    transform = sweep_to_reference_transform(reference_sensor, reference_pose, sweep_sensor, sweep_pose)
    assert np.allclose(transform_points(np.array([[0.0, 0.0, 0.0]]), transform), [[-2.0, 0.0, 0.0]])


def test_nuscenes_mapping_uses_official_detection_names() -> None:
    assert GENERAL_TO_DETECTION["vehicle.car"] == "car"
    assert GENERAL_TO_DETECTION["movable_object.trafficcone"] == "traffic_cone"
    assert GENERAL_TO_DETECTION["human.pedestrian.wheelchair"] == "ignore"


def test_real_mini_sample_and_multisweep_schema() -> None:
    adapter = NuScenesAdapter("~/datasets/nuscenes", version="v1.0-mini", max_sweeps=2)
    scene_name = adapter.scene_records[0]["name"]
    token = adapter.sample_tokens(scene_name)[1]
    frame = adapter.load_sample(token, max_sweeps=2)
    assert frame.points.ndim == 2 and frame.points.shape[1] == 5
    assert frame.metadata["sample_token"] == token
    assert len(frame.metadata["sweep_tokens"]) <= 1
    boxes = adapter.load_boxes(token)
    assert boxes
    assert all(box.size.shape == (3,) and box.velocity.shape == (3,) for box in boxes)
