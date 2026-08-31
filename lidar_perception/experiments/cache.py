"""Exact-provenance prediction cache for lightweight Phase 6 experiments."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.utils.io import save_json


CACHE_SCHEMA_VERSION = "lidar_perception.phase6_prediction_cache.v1"
PREDICTION_SCHEMA_VERSION = "lidar_perception.prediction_batch.v1"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PredictionCacheProvenance:
    dataset: str
    dataset_version: str
    split: str
    sample_token: str
    detector: str
    detector_config: str
    detector_config_sha256: str
    checkpoint_sha256: str
    sweeps: int
    candidate_threshold: float
    score_filtering_policy: str
    prediction_schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("dataset", "dataset_version", "split", "sample_token", "detector"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or not _SAFE_COMPONENT.fullmatch(value):
                raise ValueError(f"{name} must be a non-empty cache-safe identifier")
        if not isinstance(self.detector_config, str) or not self.detector_config:
            raise ValueError("detector_config must be a non-empty path/name")
        for name in ("detector_config_sha256", "checkpoint_sha256"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if isinstance(self.sweeps, bool) or not isinstance(self.sweeps, int) or self.sweeps < 1:
            raise ValueError("sweeps must be a positive integer")
        threshold = float(self.candidate_threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("candidate_threshold must be in [0, 1]")
        object.__setattr__(self, "candidate_threshold", threshold)
        if not self.score_filtering_policy:
            raise ValueError("score_filtering_policy must be non-empty")
        if self.prediction_schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError("unsupported prediction schema version")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PredictionCacheProvenance":
        if not isinstance(value, dict):
            raise TypeError("cache provenance must be a mapping")
        return cls(**value)


class PredictionCache:
    """Read/write PredictionBatch payloads only when every identity field matches."""

    def __init__(self, root: str | Path = "outputs/phase6_prediction_cache") -> None:
        self.root = Path(root).expanduser()

    def path_for(self, provenance: PredictionCacheProvenance) -> Path:
        return (
            self.root
            / f"{provenance.dataset}-{provenance.dataset_version}"
            / provenance.split
            / provenance.detector
            / f"{provenance.sample_token}.json"
        )

    def save(self, prediction: PredictionBatch, provenance: PredictionCacheProvenance) -> Path:
        if prediction.frame_id != provenance.sample_token:
            raise ValueError("prediction frame_id must match cache sample_token")
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "provenance": provenance.to_dict(),
            "prediction": prediction.to_dict(),
        }
        return save_json(payload, self.path_for(provenance))

    def load(self, expected: PredictionCacheProvenance) -> PredictionBatch | None:
        path = self.path_for(expected)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                return None
            actual = PredictionCacheProvenance.from_dict(payload["provenance"])
            if actual != expected:
                return None
            prediction = PredictionBatch.from_dict(payload["prediction"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if prediction.frame_id != expected.sample_token:
            return None
        return prediction

    def is_compatible(self, expected: PredictionCacheProvenance) -> bool:
        return self.load(expected) is not None


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "PredictionCache",
    "PredictionCacheProvenance",
]
