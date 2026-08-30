"""Project-owned nuScenes table, geometry, and multi-sweep adapter.

The nuScenes devkit is used for table access and file resolution.  This module
owns the conversion into the project's canonical ``PointCloudFrame`` and
``Box3D`` schemas; it does not copy OpenPCDet's dataset implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lidar_perception.geometry.boxes3d import Box3D
from lidar_perception.geometry.transforms import invert_transform, make_transform, transform_points

from .schemas import PointCloudFrame


NUSCENES_DETECTION_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

GENERAL_TO_DETECTION = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


class NuScenesError(RuntimeError):
    """Base error for missing or malformed nuScenes data."""


def _rotation_matrix(quaternion: Iterable[float]) -> np.ndarray:
    """Convert a nuScenes ``[w, x, y, z]`` quaternion to a rotation matrix."""

    from lidar_perception.geometry.transforms import quaternion_to_rotation_matrix

    return quaternion_to_rotation_matrix(quaternion)


def sensor_to_ego_transform(calibrated_sensor: dict[str, Any]) -> np.ndarray:
    """Return ``T_ego_sensor`` from a calibrated-sensor record."""

    return make_transform(
        _rotation_matrix(calibrated_sensor["rotation"]),
        calibrated_sensor["translation"],
    )


def ego_to_global_transform(ego_pose: dict[str, Any]) -> np.ndarray:
    """Return ``T_global_ego`` from an ego-pose record."""

    return make_transform(_rotation_matrix(ego_pose["rotation"]), ego_pose["translation"])


def sensor_to_global_transform(calibrated_sensor: dict[str, Any], ego_pose: dict[str, Any]) -> np.ndarray:
    """Compose ``T_global_sensor = T_global_ego @ T_ego_sensor``."""

    return ego_to_global_transform(ego_pose) @ sensor_to_ego_transform(calibrated_sensor)


def sweep_to_reference_transform(
    reference_calibrated_sensor: dict[str, Any],
    reference_ego_pose: dict[str, Any],
    sweep_calibrated_sensor: dict[str, Any],
    sweep_ego_pose: dict[str, Any],
) -> np.ndarray:
    """Return the rigid transform from a sweep sensor into reference LiDAR."""

    reference_from_global = invert_transform(
        sensor_to_global_transform(reference_calibrated_sensor, reference_ego_pose)
    )
    global_from_sweep = sensor_to_global_transform(sweep_calibrated_sensor, sweep_ego_pose)
    return reference_from_global @ global_from_sweep


class NuScenesAdapter:
    """Load nuScenes samples into the project's canonical LiDAR convention."""

    def __init__(
        self,
        root: str | Path,
        version: str = "v1.0-mini",
        max_sweeps: int = 1,
        class_names: Iterable[str] = NUSCENES_DETECTION_CLASSES,
    ) -> None:
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be at least 1")
        self.root = Path(root).expanduser().resolve()
        self.version = version
        self.max_sweeps = int(max_sweeps)
        self.class_names = tuple(class_names)
        unknown = set(self.class_names) - set(NUSCENES_DETECTION_CLASSES)
        if unknown:
            raise ValueError(f"unsupported nuScenes detection classes: {sorted(unknown)}")
        if not (self.root / version).is_dir():
            raise NuScenesError(f"nuScenes version directory not found: {self.root / version}")
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise NuScenesError("nuscenes-devkit is required for NuScenesAdapter") from exc
        self.nusc = NuScenes(version=version, dataroot=str(self.root), verbose=False)

    @property
    def scene_count(self) -> int:
        return len(self.nusc.scene)

    @property
    def sample_count(self) -> int:
        return len(self.nusc.sample)

    @property
    def scene_records(self) -> list[dict[str, Any]]:
        return list(self.nusc.scene)

    def scene_token(self, scene: str) -> str:
        """Resolve a scene token or human-readable scene name."""

        for record in self.nusc.scene:
            if record["token"] == scene or record["name"] == scene:
                return record["token"]
        raise NuScenesError(f"scene not found: {scene}")

    def sample_record(self, sample_token: str) -> dict[str, Any]:
        return self.nusc.get("sample", sample_token)

    def sample_tokens(self, scene: str | None = None) -> list[str]:
        """Return sample tokens in chronological order, optionally per scene."""

        if scene is None:
            return [record["token"] for record in self.nusc.sample]
        scene_record = self.nusc.get("scene", self.scene_token(scene))
        tokens: list[str] = []
        token = scene_record["first_sample_token"]
        while token:
            tokens.append(token)
            token = self.nusc.get("sample", token)["next"]
        return tokens

    def sample_data_token(self, sample_token: str, channel: str = "LIDAR_TOP") -> str:
        sample = self.sample_record(sample_token)
        try:
            return sample["data"][channel]
        except KeyError as exc:
            raise NuScenesError(f"sample {sample_token} has no channel {channel}") from exc

    def _sample_data_records(self, sample_data_token: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        sample_data = self.nusc.get("sample_data", sample_data_token)
        calibrated = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
        return sample_data, calibrated, ego_pose

    def sensor_to_ego(self, sample_data_token: str) -> np.ndarray:
        _, calibrated, _ = self._sample_data_records(sample_data_token)
        return sensor_to_ego_transform(calibrated)

    def ego_to_global(self, sample_data_token: str) -> np.ndarray:
        _, _, ego_pose = self._sample_data_records(sample_data_token)
        return ego_to_global_transform(ego_pose)

    def sensor_to_global(self, sample_data_token: str) -> np.ndarray:
        _, calibrated, ego_pose = self._sample_data_records(sample_data_token)
        return sensor_to_global_transform(calibrated, ego_pose)

    def global_to_sensor(self, sample_data_token: str) -> np.ndarray:
        return invert_transform(self.sensor_to_global(sample_data_token))

    def _load_lidar_xyzi(self, sample_data_token: str) -> np.ndarray:
        _, _, _ = self._sample_data_records(sample_data_token)
        path = Path(self.nusc.get_sample_data_path(sample_data_token))
        if not path.is_file():
            raise NuScenesError(f"LiDAR file not found: {path}")
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 5 != 0:
            raise NuScenesError(f"LiDAR file has {raw.size} values; expected 5 per point: {path}")
        points = raw.reshape(-1, 5)[:, :4]
        if not np.all(np.isfinite(points)):
            raise NuScenesError(f"LiDAR file contains non-finite values: {path}")
        return points

    def _sweep_tokens(self, reference_token: str, max_sweeps: int) -> list[str]:
        sample_data, _, _ = self._sample_data_records(reference_token)
        tokens: list[str] = []
        previous = sample_data["prev"]
        while previous and len(tokens) < max_sweeps - 1:
            tokens.append(previous)
            previous = self.nusc.get("sample_data", previous)["prev"]
        return tokens

    def load_sample(self, sample_token: str, max_sweeps: int | None = None) -> PointCloudFrame:
        """Load one sample and transform past sweeps into its LiDAR frame."""

        sweep_count = self.max_sweeps if max_sweeps is None else int(max_sweeps)
        if sweep_count < 1:
            raise ValueError("max_sweeps must be at least 1")
        sample = self.sample_record(sample_token)
        reference_token = self.sample_data_token(sample_token)
        reference_sd, reference_cs, reference_pose = self._sample_data_records(reference_token)
        reference_timestamp = int(reference_sd["timestamp"])
        points = self._load_lidar_xyzi(reference_token)
        chunks = [np.column_stack((points, np.zeros((len(points), 1), dtype=np.float32)))]
        sweep_tokens = self._sweep_tokens(reference_token, sweep_count)
        for sweep_token in sweep_tokens:
            sweep_sd, sweep_cs, sweep_pose = self._sample_data_records(sweep_token)
            sweep_points = self._load_lidar_xyzi(sweep_token)
            transform = sweep_to_reference_transform(reference_cs, reference_pose, sweep_cs, sweep_pose)
            transformed_xyz = transform_points(sweep_points[:, :3], transform).astype(np.float32)
            time_lag = (reference_timestamp - int(sweep_sd["timestamp"])) * 1e-6
            chunks.append(
                np.column_stack(
                    (transformed_xyz, sweep_points[:, 3], np.full(len(sweep_points), time_lag, dtype=np.float32))
                )
            )
        scene = self.nusc.get("scene", sample["scene_token"])
        lidar_to_ego = sensor_to_ego_transform(reference_cs)
        ego_to_global = ego_to_global_transform(reference_pose)
        return PointCloudFrame(
            frame_id=sample_token,
            points=np.concatenate(chunks, axis=0),
            timestamp=reference_timestamp,
            lidar_to_ego=lidar_to_ego,
            ego_to_global=ego_to_global,
            metadata={
                "dataset": "nuScenes",
                "version": self.version,
                "sample_token": sample_token,
                "scene_token": sample["scene_token"],
                "scene_name": scene["name"],
                "sample_data_token": reference_token,
                "channel": "LIDAR_TOP",
                "timestamp_us": reference_timestamp,
                "max_sweeps": sweep_count,
                "sweep_tokens": sweep_tokens,
                "time_lag_convention": "reference_timestamp - sweep_timestamp, seconds",
            },
        )

    def load_scene(self, scene: str, max_sweeps: int | None = None) -> list[PointCloudFrame]:
        """Load consecutive samples from a scene in chronological order."""

        return [self.load_sample(token, max_sweeps=max_sweeps) for token in self.sample_tokens(scene)]

    def _category_label(self, annotation: dict[str, Any]) -> str | None:
        instance = self.nusc.get("instance", annotation["instance_token"])
        category = self.nusc.get("category", instance["category_token"])["name"]
        label = GENERAL_TO_DETECTION.get(category)
        return label if label in self.class_names else None

    def load_boxes(self, sample_token: str, classes: set[str] | None = None) -> list[Box3D]:
        """Load sample annotations as LiDAR-frame ``Box3D`` objects."""

        sample = self.sample_record(sample_token)
        reference_token = self.sample_data_token(sample_token)
        reference_from_global = self.global_to_sensor(reference_token)
        result: list[Box3D] = []
        for annotation_token in sample["anns"]:
            annotation = self.nusc.get("sample_annotation", annotation_token)
            label = self._category_label(annotation)
            if label is None or (classes is not None and label not in classes):
                continue
            center = transform_points(np.asarray(annotation["translation"], dtype=np.float64).reshape(1, 3), reference_from_global)[0]
            heading_global = _rotation_matrix(annotation["rotation"]) @ np.array([1.0, 0.0, 0.0])
            heading_sensor = reference_from_global[:3, :3] @ heading_global
            yaw = float(np.arctan2(heading_sensor[1], heading_sensor[0]))
            velocity_global = np.asarray(self.nusc.box_velocity(annotation_token), dtype=np.float64)
            velocity_global[~np.isfinite(velocity_global)] = 0.0
            velocity_sensor = reference_from_global[:3, :3] @ velocity_global
            width, length, height = np.asarray(annotation["size"], dtype=np.float64)
            result.append(
                Box3D(
                    center=center,
                    size=np.array([length, width, height]),
                    yaw=yaw,
                    label=label,
                    velocity=velocity_sensor,
                    track_id=annotation.get("instance_token"),
                )
            )
        return result


__all__ = [
    "GENERAL_TO_DETECTION",
    "NUSCENES_DETECTION_CLASSES",
    "NuScenesAdapter",
    "NuScenesError",
    "ego_to_global_transform",
    "sensor_to_ego_transform",
    "sensor_to_global_transform",
    "sweep_to_reference_transform",
]
