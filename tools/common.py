"""Shared Phase 2 CLI loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lidar_perception.detection.openpcdet_backend import CenterPointBackend, OpenPCDetBackend, VoxelNeXtBackend
from lidar_perception.utils.config import load_yaml_config
from lidar_perception.utils.io import save_json


def load_detector_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    config = load_yaml_config(config_path)
    backend = config.get("backend")
    dataset = config.get("dataset")
    if not isinstance(backend, dict) or not isinstance(dataset, dict):
        raise ValueError("detector config must contain backend and dataset mappings")
    opcdet_config = Path(backend.get("openpcdet_config", "")).expanduser()
    if not opcdet_config.is_absolute():
        opcdet_config = (Path.cwd() / opcdet_config).resolve()
    return config, opcdet_config


def load_pointpillar_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Backward-compatible name used by the Phase 2/3 tools."""

    return load_detector_config(path)


def make_backend(config: dict[str, Any], opcdet_config: Path, checkpoint: str | Path | None = None) -> OpenPCDetBackend:
    backend_cfg = config["backend"]
    checkpoint_path = checkpoint or backend_cfg.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("checkpoint is required in config or --checkpoint")
    model = str(backend_cfg.get("model", "pointpillar")).lower()
    backend_class = {
        "centerpoint": CenterPointBackend,
        "voxelnext": VoxelNeXtBackend,
    }.get(model, OpenPCDetBackend)
    return backend_class(
        config_path=opcdet_config,
        checkpoint_path=checkpoint_path,
        checkpoint_source=backend_cfg.get("checkpoint_source"),
        device=backend_cfg.get("device", "cuda"),
        score_threshold=float(backend_cfg.get("score_threshold", 0.1)),
        opcdet_root=Path("third_party/OpenPCDet"),
        model_type=model,
    )


def write_prediction(prediction, path: str | Path) -> Path:
    return save_json(prediction.to_dict(), path)
