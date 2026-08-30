"""Detector backend boundary for project-owned code."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lidar_perception.datasets.schemas import PointCloudFrame

from .schemas import PredictionBatch


class DetectorBackend(ABC):
    """Stable project interface hiding a concrete detector implementation."""

    @abstractmethod
    def load(self, config_path: str, checkpoint_path: str) -> None:
        """Load model configuration and checkpoint."""

    @abstractmethod
    def predict(self, frame: PointCloudFrame) -> PredictionBatch:
        """Run inference for one project-owned point-cloud frame."""

    @abstractmethod
    def name(self) -> str:
        """Return a stable backend/model name."""
