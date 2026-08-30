import numpy as np

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.nuscenes import build_nuscenes_result_json, prediction_to_nuscenes_results
from lidar_perception.geometry.boxes3d import Box3D


class FakeAdapter:
    version = "v1.0-mini"

    def sample_data_token(self, sample_token):
        assert sample_token == "sample"
        return "lidar"

    def sensor_to_global(self, sample_data_token):
        assert sample_data_token == "lidar"
        transform = np.eye(4)
        transform[:3, 3] = [10.0, 0.0, 0.0]
        return transform


def test_prediction_to_nuscenes_result_preserves_geometry_and_velocity() -> None:
    prediction = PredictionBatch(
        "sample",
        [Box3D([1, 2, 3], [4, 2, 1], 0.25, "car", score=0.9, velocity=[3, -1, 0])],
        metadata={"sample_token": "sample"},
    )
    result = prediction_to_nuscenes_results(prediction, FakeAdapter())[0]
    assert np.allclose(result["translation"], [11, 2, 3])
    assert result["size"] == [2.0, 4.0, 1.0]
    assert result["velocity"] == [3.0, -1.0]
    assert result["detection_name"] == "car"


def test_build_nuscenes_result_json_includes_empty_eval_samples() -> None:
    prediction = PredictionBatch("sample", [], metadata={"sample_token": "sample"})
    payload = build_nuscenes_result_json([prediction], FakeAdapter(), sample_tokens=["sample", "empty"])
    assert set(payload["results"]) == {"sample", "empty"}
    assert payload["results"]["sample"] == []
    assert payload["results"]["empty"] == []
    assert payload["meta"]["use_lidar"] is True
