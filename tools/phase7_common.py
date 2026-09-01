"""Small, dependency-light helpers shared by Phase 7 entrypoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CONFIG = ROOT / "configs/system/portfolio.yaml"
DEFAULT_DATASET_ROOT = "~/datasets/nuscenes"


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_system_config(path: str | Path = SYSTEM_CONFIG) -> dict[str, Any]:
    target = resolve_path(path)
    with target.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("portfolio"), dict):
        raise ValueError(f"system config must contain portfolio mapping: {target}")
    return value["portfolio"]


def load_project_config(path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    with target.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"detector config must contain a mapping: {target}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_report(payload: dict[str, Any], output: str | Path | None) -> None:
    if output is None:
        return
    target = resolve_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default
