"""Minimal structured-output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return its expanded path."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_json(value: Any, path: str | Path) -> Path:
    """Write JSON with stable indentation and sorted keys."""

    target = ensure_parent(path)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
