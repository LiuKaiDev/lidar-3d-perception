"""Unified detector prediction schema and JSON serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lidar_perception.geometry.boxes3d import Box3D


@dataclass
class PredictionBatch:
    """Predictions in the project LiDAR convention.

    ``Box3D`` sizes are always ``[length, width, height]`` regardless of the
    tensor layout used by a third-party detector.
    """

    frame_id: str
    boxes: list[Box3D] = field(default_factory=list)
    runtime_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        if not isinstance(self.boxes, list) or not all(isinstance(box, Box3D) for box in self.boxes):
            raise TypeError("boxes must be a list of Box3D instances")
        if self.runtime_ms is not None:
            self.runtime_ms = float(self.runtime_ms)
            if not np.isfinite(self.runtime_ms) or self.runtime_ms < 0:
                raise ValueError("runtime_ms must be finite and non-negative")
        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data without tensors or NumPy scalars."""

        return {
            "frame_id": self.frame_id,
            "runtime_ms": self.runtime_ms,
            "boxes": [
                {
                    "label": box.label,
                    "score": box.score,
                    "center": box.center.tolist(),
                    "size": box.size.tolist(),
                    "yaw": box.yaw,
                    "velocity": None if box.velocity is None else box.velocity.tolist(),
                    "track_id": box.track_id,
                }
                for box in self.boxes
            ],
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PredictionBatch":
        """Build a prediction batch from the project JSON representation."""

        if not isinstance(value, dict):
            raise TypeError("prediction JSON must contain an object")
        boxes = []
        for index, raw_box in enumerate(value.get("boxes", [])):
            if not isinstance(raw_box, dict):
                raise ValueError(f"boxes[{index}] must be an object")
            boxes.append(
                Box3D(
                    center=raw_box["center"],
                    size=raw_box["size"],
                    yaw=raw_box["yaw"],
                    label=raw_box["label"],
                    score=raw_box.get("score"),
                    velocity=raw_box.get("velocity"),
                    track_id=raw_box.get("track_id"),
                )
            )
        return cls(value["frame_id"], boxes, value.get("runtime_ms"), value.get("metadata", {}))


def _json_safe(value: Any) -> Any:
    """Convert NumPy values in metadata to ordinary JSON values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
