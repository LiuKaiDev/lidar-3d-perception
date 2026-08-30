"""Project-owned KITTI Object Detection format adapter.

This module intentionally parses KITTI files directly. OpenPCDet's KITTI
loader is a third-party reference only and is not used by this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from lidar_perception.geometry.boxes3d import Box3D
from lidar_perception.geometry.transforms import invert_transform, make_transform, transform_points
from .schemas import PointCloudFrame


class KittiError(RuntimeError):
    """Base class for clear KITTI data and format errors."""


class KittiDatasetError(KittiError):
    """Raised when a configured KITTI root or required file is unavailable."""


class KittiFormatError(KittiError):
    """Raised when a KITTI file has malformed content."""


@dataclass(frozen=True)
class KittiObject:
    """One KITTI label row in its native rectified-camera convention."""

    class_name: str
    truncation: float
    occlusion: int
    alpha: float
    bbox_2d: np.ndarray
    dimensions_hwl: np.ndarray
    location_rect: np.ndarray
    rotation_y: float
    score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_2d", np.asarray(self.bbox_2d, dtype=np.float64))
        object.__setattr__(self, "dimensions_hwl", np.asarray(self.dimensions_hwl, dtype=np.float64))
        object.__setattr__(self, "location_rect", np.asarray(self.location_rect, dtype=np.float64))
        if self.bbox_2d.shape != (4,):
            raise ValueError("bbox_2d must have shape (4,)")
        if not np.all(np.isfinite(self.bbox_2d)):
            raise ValueError("bbox_2d must contain finite values")
        if self.dimensions_hwl.shape != (3,):
            raise ValueError("dimensions_hwl must have shape (3,)")
        if not np.all(np.isfinite(self.dimensions_hwl)):
            raise ValueError("dimensions_hwl must contain finite values")
        if self.class_name != "DontCare" and np.any(self.dimensions_hwl <= 0):
            raise ValueError("dimensions_hwl must be positive for labeled objects")
        if self.location_rect.shape != (3,) or not np.all(np.isfinite(self.location_rect)):
            raise ValueError("location_rect must have shape (3,) and finite values")
        if not all(np.isfinite(value) for value in (self.truncation, self.alpha, self.rotation_y)):
            raise ValueError("truncation, alpha, and rotation_y must be finite")

    @property
    def height(self) -> float:
        return float(self.dimensions_hwl[0])

    @property
    def width(self) -> float:
        return float(self.dimensions_hwl[1])

    @property
    def length(self) -> float:
        return float(self.dimensions_hwl[2])


def _as_matrix(values: Iterable[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    matrix = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if matrix.size != int(np.prod(shape)):
        raise KittiFormatError(f"{name} must contain {int(np.prod(shape))} values, got {matrix.size}")
    matrix = matrix.reshape(shape)
    if not np.all(np.isfinite(matrix)):
        raise KittiFormatError(f"{name} contains non-finite values")
    return matrix


@dataclass(frozen=True)
class KittiCalibration:
    """KITTI camera/LiDAR calibration with explicit source/destination methods."""

    P2: np.ndarray
    R0_rect: np.ndarray
    Tr_velo_to_cam: np.ndarray

    def __post_init__(self) -> None:
        p2 = np.asarray(self.P2, dtype=np.float64)
        rect = np.asarray(self.R0_rect, dtype=np.float64)
        velo = np.asarray(self.Tr_velo_to_cam, dtype=np.float64)
        if p2.shape != (3, 4):
            raise ValueError(f"P2 must have shape (3, 4), got {p2.shape}")
        if rect.shape == (4, 4):
            rect = rect[:3, :3]
        if velo.shape == (4, 4):
            velo = velo[:3, :]
        if rect.shape != (3, 3):
            raise ValueError(f"R0_rect must have shape (3, 3), got {rect.shape}")
        if velo.shape != (3, 4):
            raise ValueError(f"Tr_velo_to_cam must have shape (3, 4), got {velo.shape}")
        if not all(np.all(np.isfinite(value)) for value in (p2, rect, velo)):
            raise ValueError("calibration matrices must contain only finite values")
        object.__setattr__(self, "P2", p2)
        object.__setattr__(self, "R0_rect", rect)
        object.__setattr__(self, "Tr_velo_to_cam", velo)

    @classmethod
    def from_file(cls, path: str | Path) -> "KittiCalibration":
        """Parse a KITTI calibration text file."""

        path = Path(path).expanduser()
        if not path.is_file():
            raise KittiDatasetError(f"calibration file not found: {path}")
        values: dict[str, list[float]] = {}
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, raw_values = line.split(":", 1)
            try:
                values[key.strip()] = [float(item) for item in raw_values.split()]
            except ValueError as exc:
                raise KittiFormatError(f"invalid numeric value in {path}:{line_number}") from exc
        p2_values = values.get("P2")
        rect_values = values.get("R0_rect", values.get("R_rect_00"))
        velo_values = values.get("Tr_velo_to_cam", values.get("Tr_velo2cam"))
        missing = [name for name, value in (("P2", p2_values), ("R0_rect", rect_values), ("Tr_velo_to_cam", velo_values)) if value is None]
        if missing:
            raise KittiFormatError(f"calibration file {path} is missing: {', '.join(missing)}")
        return cls(_as_matrix(p2_values, (3, 4), "P2"), _as_matrix(rect_values, (3, 3), "R0_rect"), _as_matrix(velo_values, (3, 4), "Tr_velo_to_cam"))

    @property
    def rect_from_velo(self) -> np.ndarray:
        """Homogeneous transform mapping Velodyne points to rectified camera."""

        return make_transform(self.R0_rect) @ make_transform(self.Tr_velo_to_cam[:, :3], self.Tr_velo_to_cam[:, 3])

    @property
    def velo_from_rect(self) -> np.ndarray:
        """Inverse homogeneous transform mapping rectified camera to Velodyne."""

        return invert_transform(self.rect_from_velo)

    @property
    def cam_from_velo(self) -> np.ndarray:
        """Homogeneous unrectified-camera transform from Velodyne."""

        return make_transform(self.Tr_velo_to_cam[:, :3], self.Tr_velo_to_cam[:, 3])

    @property
    def velo_from_cam(self) -> np.ndarray:
        return invert_transform(self.cam_from_velo)

    def velo_to_rect(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.rect_from_velo)

    def rect_to_velo(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.velo_from_rect)

    def velo_to_cam(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.cam_from_velo)

    def cam_to_velo(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.velo_from_cam)

    def rect_vector_to_velo(self, vectors: np.ndarray) -> np.ndarray:
        """Transform direction vectors without applying translation."""

        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError(f"vectors must have shape (N, 3), got {vectors.shape}")
        return vectors @ self.velo_from_rect[:3, :3].T


def parse_kitti_label_line(line: str, source: str = "label", line_number: int = 1) -> KittiObject:
    """Parse one 15- or 16-field KITTI Object Detection label row."""

    fields = line.strip().split()
    if len(fields) not in (15, 16):
        raise KittiFormatError(f"{source}:{line_number} must contain 15 or 16 fields, got {len(fields)}")
    try:
        return KittiObject(
            class_name=fields[0],
            truncation=float(fields[1]),
            occlusion=int(fields[2]),
            alpha=float(fields[3]),
            bbox_2d=np.array([float(value) for value in fields[4:8]]),
            dimensions_hwl=np.array([float(value) for value in fields[8:11]]),
            location_rect=np.array([float(value) for value in fields[11:14]]),
            rotation_y=float(fields[14]),
            score=float(fields[15]) if len(fields) == 16 else None,
        )
    except (TypeError, ValueError) as exc:
        raise KittiFormatError(f"invalid numeric value in {source}:{line_number}") from exc


def camera_object_to_lidar_box(obj: KittiObject, calibration: KittiCalibration) -> Box3D:
    """Convert KITTI camera bottom-center ``h,w,l`` labels to internal LiDAR boxes.

    KITTI labels locate the bottom center in rectified camera coordinates and
    rotate around camera +Y with ``rotation_y``. The internal box uses its
    geometric center, ``[length,width,height]``, and yaw around LiDAR +Z.
    """

    center_rect = obj.location_rect + np.array([0.0, -obj.height / 2.0, 0.0])
    center_lidar = calibration.rect_to_velo(center_rect.reshape(1, 3))[0]
    heading_rect = np.array([[np.cos(obj.rotation_y), 0.0, -np.sin(obj.rotation_y)]], dtype=np.float64)
    heading_lidar = calibration.rect_vector_to_velo(heading_rect)[0]
    horizontal_norm = np.linalg.norm(heading_lidar[:2])
    if horizontal_norm <= 1e-12:
        raise KittiFormatError(f"cannot convert {obj.class_name} box with degenerate horizontal heading")
    yaw = float(np.arctan2(heading_lidar[1], heading_lidar[0]))
    return Box3D(center=center_lidar, size=np.array([obj.length, obj.width, obj.height]), yaw=yaw, label=obj.class_name, score=obj.score)


class KittiAdapter:
    """Load one standard KITTI Object Detection split without fixed local paths."""

    def __init__(self, root: str | Path, split: str = "training") -> None:
        self.root = Path(root).expanduser()
        self.split_alias = split
        self.split = {"train": "training", "val": "training", "test": "testing"}.get(split, split)
        if self.split not in {"training", "testing"}:
            raise ValueError("split must be training, testing, train, val, or test")

    @property
    def split_root(self) -> Path:
        return self.root / self.split

    def _require_root(self) -> None:
        if not self.root.is_dir():
            raise KittiDatasetError(f"KITTI dataset root not found: {self.root}")
        if not self.split_root.is_dir():
            raise KittiDatasetError(f"KITTI split directory not found: {self.split_root}")

    def _path(self, directory: str, frame_id: str, suffix: str = ".txt") -> Path:
        if not frame_id or Path(frame_id).name != frame_id:
            raise ValueError("frame_id must be a non-empty file stem without path separators")
        return self.split_root / directory / f"{frame_id}{suffix}"

    def frame_ids(self) -> list[str]:
        self._require_root()
        candidates = [self.root / "ImageSets" / f"{self.split_alias}.txt"]
        if self.split_alias != self.split:
            candidates.append(self.root / "ImageSets" / f"{self.split}.txt")
        for list_path in candidates:
            if list_path.is_file():
                return [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        velo_dir = self.split_root / "velodyne"
        if not velo_dir.is_dir():
            raise KittiDatasetError(f"Velodyne directory not found: {velo_dir}")
        return sorted(path.stem for path in velo_dir.glob("*.bin"))

    def _require_file(self, path: Path, description: str) -> Path:
        if not path.is_file():
            raise KittiDatasetError(f"{description} not found: {path}")
        return path

    def load_points(self, frame_id: str) -> np.ndarray:
        """Load Velodyne float32 records as ``(N,4)`` XYZ-intensity."""

        self._require_root()
        path = self._require_file(self._path("velodyne", frame_id, ".bin"), "point cloud")
        byte_count = path.stat().st_size
        if byte_count % (4 * 4) != 0:
            raise KittiFormatError(f"point cloud {path} has {byte_count} bytes; expected a multiple of 16")
        points = np.fromfile(path, dtype=np.float32)
        if points.size % 4 != 0:
            raise KittiFormatError(f"point cloud {path} cannot be reshaped into XYZ-intensity records")
        points = points.reshape(-1, 4)
        if not np.all(np.isfinite(points)):
            raise KittiFormatError(f"point cloud {path} contains non-finite values")
        return points

    def load_calibration(self, frame_id: str) -> KittiCalibration:
        self._require_root()
        return KittiCalibration.from_file(self._path("calib", frame_id))

    def load_labels(self, frame_id: str, include_dontcare: bool = False) -> list[KittiObject]:
        self._require_root()
        path = self._require_file(self._path("label_2", frame_id), "label file")
        labels: list[KittiObject] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            obj = parse_kitti_label_line(line, str(path), line_number)
            if include_dontcare or obj.class_name != "DontCare":
                labels.append(obj)
        return labels

    def load_boxes(self, frame_id: str, classes: set[str] | None = None) -> list[Box3D]:
        calibration = self.load_calibration(frame_id)
        labels = self.load_labels(frame_id)
        return [camera_object_to_lidar_box(obj, calibration) for obj in labels if classes is None or obj.class_name in classes]

    def load_image(self, frame_id: str) -> np.ndarray:
        self._require_root()
        path = self._require_file(self._path("image_2", frame_id, ".png"), "image")
        try:
            import cv2
        except ImportError as exc:
            raise KittiDatasetError("OpenCV is required to load KITTI images") from exc
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise KittiDatasetError(f"image could not be decoded: {path}")
        return image

    def load_frame(self, frame_id: str) -> PointCloudFrame:
        points = self.load_points(frame_id)
        calibration_path = self._path("calib", frame_id)
        return PointCloudFrame(
            frame_id=frame_id,
            points=points,
            timestamp=None,
            lidar_to_ego=np.eye(4),
            ego_to_global=np.eye(4),
            metadata={
                "dataset": "KITTI Object Detection",
                "split": self.split,
                "calibration_path": str(calibration_path),
                "has_true_ego_global_pose": False,
                "pose_note": "KITTI Phase 1 does not provide true ego/global temporal poses.",
            },
        )
