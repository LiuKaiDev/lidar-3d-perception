"""Thin project wrapper around OpenPCDet's official KITTI evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from lidar_perception.detection.schemas import PredictionBatch


def prediction_to_kitti_anno(prediction: PredictionBatch, dataset: Any) -> dict[str, np.ndarray]:
    """Convert project boxes to the annotation dictionary expected by KITTI eval."""

    from pcdet.utils import box_utils

    frame_id = prediction.frame_id
    calib = dataset.get_calib(frame_id)
    image_shape = dataset.get_image_shape(frame_id)
    if prediction.boxes:
        boxes_lidar = np.asarray(
            [np.r_[box.center, box.size, box.yaw] for box in prediction.boxes], dtype=np.float32
        )
        scores = np.asarray([box.score if box.score is not None else 0.0 for box in prediction.boxes], dtype=np.float32)
        names = np.asarray([box.label for box in prediction.boxes])
        boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(boxes_lidar, calib)
        boxes_image = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes_camera, calib, image_shape=image_shape)
        alpha = -np.arctan2(-boxes_lidar[:, 1], boxes_lidar[:, 0]) + boxes_camera[:, 6]
        return {
            "name": names,
            "truncated": np.zeros((len(prediction.boxes),), dtype=np.float32),
            "occluded": np.zeros((len(prediction.boxes),), dtype=np.int32),
            "alpha": alpha,
            "bbox": boxes_image,
            "dimensions": boxes_camera[:, 3:6],
            "location": boxes_camera[:, 0:3],
            "rotation_y": boxes_camera[:, 6],
            "score": scores,
            "boxes_lidar": boxes_lidar,
            "frame_id": frame_id,
        }
    return {
        "name": np.asarray([], dtype="<U1"),
        "truncated": np.empty((0,), dtype=np.float32),
        "occluded": np.empty((0,), dtype=np.int32),
        "alpha": np.empty((0,), dtype=np.float32),
        "bbox": np.empty((0, 4), dtype=np.float32),
        "dimensions": np.empty((0, 3), dtype=np.float32),
        "location": np.empty((0, 3), dtype=np.float32),
        "rotation_y": np.empty((0,), dtype=np.float32),
        "score": np.empty((0,), dtype=np.float32),
        "boxes_lidar": np.empty((0, 7), dtype=np.float32),
        "frame_id": frame_id,
    }


def evaluate_kitti(
    backend: Any,
    dataset_root: str | Path,
    split: str = "val",
    output_dir: str | Path | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    """Run OpenPCDet's official KITTI evaluation on its configured val split."""

    if split not in {"val", "test"}:
        raise ValueError("official KITTI evaluation requires split='val' or split='test'")
    if backend.cfg is None:
        raise RuntimeError("backend must be loaded before evaluation")
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset

    configured_root = Path(dataset_root).expanduser().resolve()
    generated_root = backend.opcdet_root / "data" / "kitti"
    # Prefer a caller-provided OpenPCDet data root when it contains generated
    # infos; otherwise use the prepared project tree whose raw directories are
    # symlinks to the configured KITTI dataset.
    root = configured_root if (configured_root / f"kitti_infos_{'val' if split == 'val' else 'test'}.pkl").is_file() else generated_root.resolve()
    # ``EasyDict.copy()`` returns a plain dict in this OpenPCDet revision;
    # deepcopy preserves the attribute-access mapping expected by KittiDataset.
    data_cfg = copy.deepcopy(backend.cfg.DATA_CONFIG)
    data_cfg.DATA_SPLIT = {"train": "train", "test": split}
    data_cfg.INFO_PATH = {"train": ["kitti_infos_train.pkl"], "test": [f"kitti_infos_{'val' if split == 'val' else 'test'}.pkl"]}
    dataset = KittiDataset(data_cfg, backend.class_names, training=False, root_path=root, logger=None)
    if not len(dataset):
        raise RuntimeError(f"KITTI {split} split contains no frames under {root}")
    predictions: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        frame_id = dataset.kitti_infos[index]["point_cloud"]["lidar_idx"]
        points = dataset.get_lidar(frame_id)
        from lidar_perception.datasets.schemas import PointCloudFrame

        frame = PointCloudFrame(frame_id, points)
        prediction = backend.predict(frame)
        predictions.append(prediction_to_kitti_anno(prediction, dataset))
    ap_result_str, ap_dict = dataset.evaluation(predictions, backend.class_names)
    result = {
        "split": split,
        "frame_count": len(dataset),
        "class_names": backend.class_names,
        "checkpoint": backend.load_report.get("checkpoint"),
        "checkpoint_source": backend.load_report.get("checkpoint_source"),
        "protocol": "OpenPCDet official KITTI AP_R40",
        "ap_result": ap_result_str,
        "ap_dict": ap_dict,
    }
    if output_dir is not None:
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "kitti_official_result.txt").write_text(ap_result_str or "", encoding="utf-8")
    return result
