import numpy as np

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.official import prediction_to_kitti_anno
from lidar_perception.geometry.boxes3d import Box3D


class FakeCalibration:
    def lidar_to_rect(self, points):
        return points

    def rect_to_img(self, points):
        return points[:, :2] / points[:, 2:3], points[:, 2]


class FakeDataset:
    def get_calib(self, frame_id):
        assert frame_id == "frame"
        return FakeCalibration()

    def get_image_shape(self, frame_id):
        return np.array([100, 100], dtype=np.int32)


def test_prediction_to_kitti_annotation_uses_project_box_layout() -> None:
    prediction = PredictionBatch("frame", [Box3D([1, 2, 10], [4, 2, 1], 0, "Car", score=0.9)])
    anno = prediction_to_kitti_anno(prediction, FakeDataset())
    assert anno["name"].tolist() == ["Car"]
    assert anno["boxes_lidar"].shape == (1, 7)
    assert anno["dimensions"].shape == (1, 3)
    assert np.isclose(anno["score"][0], 0.9)
    assert anno["truncated"].shape == (1,)
    assert anno["occluded"].shape == (1,)
