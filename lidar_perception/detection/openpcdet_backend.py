"""OpenPCDet PointPillars adapter.

OpenPCDet remains a third-party implementation. This module owns only the
boundary conversion from :class:`PointCloudFrame` to the project's
:class:`PredictionBatch`.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lidar_perception.datasets.schemas import PointCloudFrame
from lidar_perception.geometry.boxes3d import Box3D

from .base import DetectorBackend
from .schemas import PredictionBatch

LOGGER = logging.getLogger(__name__)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _SingleFrameDataset:
    """Small DatasetTemplate-compatible processor for project frames."""

    def __init__(self, dataset_cfg: Any, class_names: list[str], root_path: Path):
        from pcdet.datasets.dataset import DatasetTemplate

        class _Dataset(DatasetTemplate):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise IndexError(index)

        self._dataset = _Dataset(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=root_path,
            logger=None,
        )

    def __getattr__(self, name: str):
        return getattr(self._dataset, name)

    def prepare(self, frame: PointCloudFrame) -> dict[str, Any]:
        return self._dataset.prepare_data({"points": frame.points.copy(), "frame_id": frame.frame_id})

    def collate(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._dataset.collate_batch([data])


class OpenPCDetBackend(DetectorBackend):
    """Project adapter for the fixed OpenPCDet PointPillars implementation."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        score_threshold: float = 0.1,
        opcdet_root: str | Path = "third_party/OpenPCDet",
        checkpoint_source: str | None = None,
    ) -> None:
        self.opcdet_root = Path(opcdet_root).expanduser().resolve()
        self.config_path = None if config_path is None else Path(config_path).expanduser().resolve()
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path).expanduser().resolve()
        self.checkpoint_source = checkpoint_source
        self.device = torch.device(device)
        self.score_threshold = self._validate_score_threshold(score_threshold)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; PointPillars inference requires a CUDA device")
        self.cfg = None
        self.model = None
        self.dataset = None
        self.model_name: str | None = None
        self.class_names: list[str] = []
        self.load_report: dict[str, Any] = {}

    def name(self) -> str:
        return "openpcdet_pointpillar"

    @staticmethod
    def _validate_score_threshold(value: float) -> float:
        threshold = float(value)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("score_threshold must be a finite value in [0, 1]")
        return threshold

    def load(self, config_path: str | Path | None = None, checkpoint_path: str | Path | None = None) -> None:
        """Load a fixed-revision OpenPCDet config, model, and checkpoint."""

        if self.device.type != "cuda":
            raise RuntimeError("OpenPCDet PointPillars backend requires device='cuda'; CPU is supported only for schema tests")

        if config_path is not None:
            self.config_path = Path(config_path).expanduser().resolve()
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if self.config_path is None:
            raise ValueError("OpenPCDet detector config path is required")
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            model_hint = "PointPillars" if "pointpillar" in str(self.config_path).lower() else "detector"
            raise FileNotFoundError(
                f"{model_hint} checkpoint not found: {self.checkpoint_path}. "
                "Download the official Model Zoo checkpoint and pass --checkpoint."
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(f"OpenPCDet config not found: {self.config_path}")
        if not (self.opcdet_root / "pcdet").is_dir():
            raise FileNotFoundError(f"OpenPCDet source not found: {self.opcdet_root}")

        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network

        cfg.clear()
        cfg.ROOT_DIR = self.opcdet_root
        cfg.LOCAL_RANK = 0
        with _working_directory(self.opcdet_root / "tools"):
            cfg_from_yaml_file(str(self.config_path), cfg)
        self.cfg = cfg
        self.model_name = str(cfg.MODEL.NAME)
        self.class_names = list(cfg.CLASS_NAMES)
        self.dataset = _SingleFrameDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=self.class_names,
            root_path=self.opcdet_root / "data" / ("nuscenes" if "nuscenes" in str(cfg.DATA_CONFIG.get("DATASET", "")).lower() else "kitti"),
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(self.class_names), dataset=self.dataset)
        self.model.to(self.device)
        self.model.eval()

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError("checkpoint is not an OpenPCDet checkpoint containing model_state")
        model_state = checkpoint["model_state"]
        current_state = self.model.state_dict()
        checkpoint_keys = set(model_state)
        model_keys = set(current_state)
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        shape_mismatch = sorted(
            key for key in checkpoint_keys & model_keys if tuple(model_state[key].shape) != tuple(current_state[key].shape)
        )
        self.load_report = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_source": self.checkpoint_source,
            "config": str(self.config_path),
            "checkpoint_version": checkpoint.get("version"),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatch_keys": shape_mismatch,
            "checkpoint_key_count": len(checkpoint_keys),
            "model_key_count": len(model_keys),
        }
        if shape_mismatch or missing or unexpected:
            details = []
            if shape_mismatch:
                details.append(f"shape mismatches={shape_mismatch[:5]}")
            if missing:
                details.append(f"missing keys={missing[:5]}")
            if unexpected:
                details.append(f"unexpected keys={unexpected[:5]}")
            raise ValueError("checkpoint is incompatible; " + "; ".join(details))
        logger = logging.getLogger("openpcdet.checkpoint")
        self.model.load_params_from_file(filename=str(self.checkpoint_path), logger=logger, to_cpu=self.device.type == "cpu")
        self.model.eval()
        self.load_report["load_result"] = "loaded"

    def architecture_summary(self) -> dict[str, Any]:
        """Return concise names for the configured VFE/BEV/backbone/head."""

        if self.cfg is None or self.model is None:
            raise RuntimeError("backend must be loaded before architecture_summary")
        return {
            "backend": self.name(),
            "openpcdet_config": str(self.config_path),
            "class_names": self.class_names,
            "model_name": self.cfg.MODEL.NAME,
            "vfe": self.cfg.MODEL.VFE.NAME,
            "map_to_bev": self.cfg.MODEL.MAP_TO_BEV.NAME,
            "backbone_2d": self.cfg.MODEL.BACKBONE_2D.NAME,
            "dense_head": self.cfg.MODEL.DENSE_HEAD.NAME,
            "point_cloud_range": list(self.cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
            "voxel_size": list(self.cfg.DATA_CONFIG.DATA_PROCESSOR[-1].VOXEL_SIZE),
        }

    def prepare_frame(self, frame: PointCloudFrame) -> dict[str, Any]:
        if self.model is None or self.dataset is None:
            raise RuntimeError("backend must be loaded before inference")
        return self.dataset.collate(self.dataset.prepare(frame))

    def _predict_prepared(self, batch_dict: dict[str, Any]) -> tuple[dict[str, torch.Tensor], float]:
        batch_dict = self._prepare_batch_for_device(batch_dict)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        pred_dict = self._forward_prepared(batch_dict)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return pred_dict, (time.perf_counter() - start) * 1000.0

    def _prepare_batch_for_device(self, batch_dict: dict[str, Any]) -> dict[str, Any]:
        """Move a collated batch to the backend device using OpenPCDet rules."""

        if self.device.type == "cuda":
            # Reuse OpenPCDet's tested conversion for its native CUDA path.
            from pcdet.models import load_data_to_gpu

            load_data_to_gpu(batch_dict)
        else:
            batch_dict = _move_batch_to_device(batch_dict, self.device)
        return batch_dict

    def _forward_prepared(self, batch_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run the already-device-resident model, including decode and NMS."""

        with torch.no_grad():
            pred_dicts, _ = self.model(batch_dict)
        return pred_dicts[0]

    def predict_native(self, frame: PointCloudFrame) -> tuple[dict[str, torch.Tensor], float]:
        """Run inference and return native tensors for benchmark/evaluation wrappers."""

        return self._predict_prepared(self.prepare_frame(frame))

    def predict(self, frame: PointCloudFrame) -> PredictionBatch:
        """Run the configured OpenPCDet detector and convert native boxes."""

        pred_dict, runtime_ms = self.predict_native(frame)
        prediction = self.native_prediction_to_batch(frame.frame_id, pred_dict, runtime_ms=runtime_ms)
        if "sample_token" in frame.metadata:
            prediction.metadata["sample_token"] = frame.metadata["sample_token"]
        return prediction

    def native_prediction_to_batch(
        self,
        frame_id: str,
        pred_dict: dict[str, torch.Tensor],
        runtime_ms: float | None = None,
        score_threshold: float | None = None,
    ) -> PredictionBatch:
        threshold = self.score_threshold if score_threshold is None else self._validate_score_threshold(score_threshold)
        boxes = pred_dict.get("pred_boxes")
        scores = pred_dict.get("pred_scores")
        labels = pred_dict.get("pred_labels")
        if boxes is None or scores is None or labels is None:
            raise ValueError("OpenPCDet prediction is missing pred_boxes/pred_scores/pred_labels")
        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        if boxes_np.ndim != 2 or boxes_np.shape[1] < 7 or scores_np.ndim != 1 or labels_np.ndim != 1:
            raise ValueError("unexpected OpenPCDet prediction tensor shapes")
        if not (len(boxes_np) == len(scores_np) == len(labels_np)):
            raise ValueError("OpenPCDet prediction tensor lengths do not match")
        result_boxes: list[Box3D] = []
        for native_box, score, label_index in zip(boxes_np, scores_np, labels_np):
            if not np.isfinite(score) or score < threshold:
                continue
            label_index = int(label_index)
            if label_index < 1 or label_index > len(self.class_names):
                raise ValueError(f"OpenPCDet class index out of range: {label_index}")
            if not np.all(np.isfinite(native_box[:7])):
                raise ValueError("OpenPCDet prediction contains non-finite box values")
            velocity = None
            if len(native_box) >= 9:
                if not np.all(np.isfinite(native_box[7:9])):
                    raise ValueError("OpenPCDet prediction contains non-finite velocity values")
                velocity = np.array([native_box[7], native_box[8], 0.0], dtype=np.float64)
            result_boxes.append(
                Box3D(
                    center=native_box[:3],
                    size=native_box[3:6],
                    yaw=float(native_box[6]),
                    label=self.class_names[label_index - 1],
                    score=float(score),
                    velocity=velocity,
                )
            )
        return PredictionBatch(
            frame_id=frame_id,
            boxes=result_boxes,
            runtime_ms=runtime_ms,
            metadata={
                "backend": self.name(),
                "class_names": self.class_names,
                "score_threshold": threshold,
                "native_box_convention": "[x,y,z,dx,dy,dz,heading] == [center,l,w,h,yaw]",
            },
        )


def clone_batch(value: Any) -> Any:
    """Clone a prepared batch for repeated model-only benchmark iterations."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_batch(item) for item in value]
    return copy.deepcopy(value)


def _move_batch_to_device(batch_dict: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Convert a collated OpenPCDet NumPy batch without forcing CUDA."""

    skip_keys = {"frame_id", "metadata", "calib", "image_paths", "ori_shape", "img_process_infos"}
    result: dict[str, Any] = {}
    for key, value in batch_dict.items():
        if key in skip_keys:
            result[key] = value
        elif isinstance(value, np.ndarray):
            dtype = torch.int32 if key == "image_shape" else torch.float32
            result[key] = torch.from_numpy(value).to(device=device, dtype=dtype)
        elif isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


class CenterPointBackend(OpenPCDetBackend):
    """CenterPoint specialization using the shared OpenPCDet boundary."""

    def name(self) -> str:
        return "openpcdet_centerpoint"
