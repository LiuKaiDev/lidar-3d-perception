"""YAML configuration loading without machine-specific paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and fail clearly for missing or malformed files."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a YAML mapping: {config_path}")
    return value


def dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return and validate the dataset section of a project config."""

    section = config.get("dataset")
    if not isinstance(section, dict):
        raise ValueError("config must contain a dataset mapping")
    if section.get("name", "kitti").lower() != "kitti":
        raise ValueError("this Phase 1 tool only supports dataset.name: kitti")
    if "root" not in section:
        raise ValueError("dataset.root is required")
    return section
