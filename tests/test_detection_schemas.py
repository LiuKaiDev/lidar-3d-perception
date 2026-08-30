import numpy as np
import pytest

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.geometry.boxes3d import Box3D


def test_prediction_batch_json_round_trip() -> None:
    prediction = PredictionBatch(
        "000123",
        [Box3D([1, 2, 3], [4, 2, 1], 0.25, "Car", score=0.91)],
        runtime_ms=12.5,
        metadata={"labels": np.array([1, 2]), "threshold": np.float32(0.1)},
    )
    restored = PredictionBatch.from_dict(prediction.to_dict())
    assert restored.frame_id == prediction.frame_id
    assert restored.runtime_ms == prediction.runtime_ms
    assert restored.boxes[0].label == "Car"
    assert np.allclose(restored.boxes[0].center, prediction.boxes[0].center)
    assert restored.metadata["labels"] == [1, 2]


def test_prediction_batch_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        PredictionBatch("frame", [], runtime_ms=-1)
    with pytest.raises(TypeError):
        PredictionBatch("frame", [object()])


def test_prediction_batch_preserves_optional_box_fields() -> None:
    prediction = PredictionBatch(
        "frame",
        [Box3D([0, 0, 0], [1, 1, 1], 0, "Car", score=0.8, velocity=None, track_id=None)],
    )
    payload = prediction.to_dict()
    assert payload["boxes"][0]["velocity"] is None
    assert payload["boxes"][0]["track_id"] is None
