from pathlib import Path

import numpy as np
import pytest

from lidar_perception.datasets.kitti_adapter import (
    KittiAdapter,
    KittiCalibration,
    KittiFormatError,
    camera_object_to_lidar_box,
    parse_kitti_label_line,
)


def test_kitti_label_conversion_uses_bottom_center_and_hwl() -> None:
    calibration = KittiCalibration(np.eye(3, 4), np.eye(3), np.eye(3, 4))
    obj = parse_kitti_label_line("Car 0 0 0 0 0 10 10 2 3 4 1 2 10 0")
    box = camera_object_to_lidar_box(obj, calibration)
    assert np.allclose(box.center, [1, 1, 10])
    assert np.allclose(box.size, [4, 3, 2])
    assert np.isclose(box.yaw, 0)


def test_kitti_adapter_loads_bin_calib_labels_and_frame(tmp_path: Path) -> None:
    root = tmp_path / "kitti"
    for directory in ("velodyne", "calib", "label_2", "image_2"):
        (root / "training" / directory).mkdir(parents=True)
    (root / "training" / "velodyne" / "000001.bin").write_bytes(np.array([[1, 2, 3, 0.5]], dtype=np.float32).tobytes())
    calibration = "P2: " + " ".join(str(x) for x in np.eye(3, 4).ravel()) + "\n"
    calibration += "R0_rect: " + " ".join(str(x) for x in np.eye(3).ravel()) + "\n"
    calibration += "Tr_velo_to_cam: " + " ".join(str(x) for x in np.eye(3, 4).ravel()) + "\n"
    (root / "training" / "calib" / "000001.txt").write_text(calibration)
    (root / "training" / "label_2" / "000001.txt").write_text("Car 0 0 0 0 0 10 10 2 2 4 1 2 10 0\nDontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10\n")
    adapter = KittiAdapter(root)
    assert adapter.frame_ids() == ["000001"]
    assert adapter.load_points("000001").shape == (1, 4)
    assert len(adapter.load_labels("000001")) == 1
    assert len(adapter.load_labels("000001", include_dontcare=True)) == 2
    frame = adapter.load_frame("000001")
    assert frame.metadata["has_true_ego_global_pose"] is False
    assert len(adapter.load_boxes("000001")) == 1


def test_training_split_does_not_silently_use_train_subset(tmp_path: Path) -> None:
    root = tmp_path / "kitti"
    velo = root / "training" / "velodyne"
    velo.mkdir(parents=True)
    for frame_id in ("000001", "000002"):
        (velo / f"{frame_id}.bin").write_bytes(np.zeros((1, 4), dtype=np.float32).tobytes())
    (root / "ImageSets").mkdir()
    (root / "ImageSets" / "train.txt").write_text("000001\n")
    assert KittiAdapter(root, split="training").frame_ids() == ["000001", "000002"]
    assert KittiAdapter(root, split="train").frame_ids() == ["000001"]


def test_kitti_adapter_rejects_malformed_bin(tmp_path: Path) -> None:
    root = tmp_path / "kitti" / "training" / "velodyne"
    root.mkdir(parents=True)
    (root / "000001.bin").write_bytes(b"bad")
    with pytest.raises(KittiFormatError, match="multiple of 16"):
        KittiAdapter(tmp_path / "kitti").load_points("000001")
