import numpy as np
import pytest
import torch

from lidar_perception.detection.openpcdet_backend import OpenPCDetBackend, _move_batch_to_device


def test_native_openpcdet_prediction_maps_to_project_schema() -> None:
    backend = OpenPCDetBackend(device="cpu", score_threshold=0.5)
    backend.class_names = ["Car", "Pedestrian", "Cyclist"]
    native = {
        "pred_boxes": torch.tensor([[1, 2, 3, 4, 2, 1, 0.25], [0, 0, 0, 1, 1, 1, 0]], dtype=torch.float32),
        "pred_scores": torch.tensor([0.9, 0.1], dtype=torch.float32),
        "pred_labels": torch.tensor([1, 2], dtype=torch.long),
    }
    prediction = backend.native_prediction_to_batch("000001", native)
    assert prediction.frame_id == "000001"
    assert len(prediction.boxes) == 1
    assert prediction.boxes[0].label == "Car"
    assert prediction.boxes[0].size.tolist() == [4.0, 2.0, 1.0]
    assert np.isclose(prediction.boxes[0].yaw, 0.25)


def test_cpu_batch_conversion_does_not_force_cuda() -> None:
    batch = {
        "points": np.zeros((2, 5), dtype=np.float32),
        "voxel_coords": np.zeros((1, 4), dtype=np.int32),
        "image_shape": np.array([[375, 1242]], dtype=np.int32),
        "frame_id": ["frame"],
    }
    converted = _move_batch_to_device(batch, torch.device("cpu"))
    assert converted["points"].device.type == "cpu"
    assert converted["voxel_coords"].dtype == torch.float32
    assert converted["image_shape"].dtype == torch.int32
    assert converted["frame_id"] == ["frame"]


def test_score_threshold_must_be_finite_and_bounded() -> None:
    with pytest.raises(ValueError):
        OpenPCDetBackend(device="cpu", score_threshold=float("nan"))
    backend = OpenPCDetBackend(device="cpu", score_threshold=0.5)
    with pytest.raises(ValueError):
        backend.native_prediction_to_batch(
            "frame",
            {
                "pred_boxes": torch.zeros((0, 7)),
                "pred_scores": torch.zeros((0,)),
                "pred_labels": torch.zeros((0,), dtype=torch.long),
            },
            score_threshold=1.1,
        )
