#!/usr/bin/env python3
"""Read-only detector, dataset, checkpoint, and optional cache validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase7_common import DEFAULT_DATASET_ROOT, ROOT, env_or_default, json_report, load_project_config, load_system_config, resolve_path, sha256_file


CACHE_DETECTOR = {"centerpoint": "centerpoint_pointpillar", "voxelnext": "voxelnext"}
CACHE_CONFIG = {
    "centerpoint": ("configs/detectors/centerpoint/nuscenes_mini.yaml", "58b5b8b2e4303a4563b6635c7d9f75a41acc8714635914b4d2cfb95ba8e40fc0"),
    "voxelnext": ("cbgs_voxel0075_voxelnext.yaml", "84f5ddcab780ef108af77412f9bad587c21299419530c8708121442449115a0d"),
}


def _item(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **extra}


def _detector_config(portfolio: dict[str, Any], detector: str) -> Path:
    return resolve_path(portfolio["detectors"][detector]["config"])


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _asset_status(detector: str, config_path: Path, checkpoint_override: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    config = load_project_config(config_path)
    checks.append(_item("detector config", "PASS", str(config_path)))
    backend = config.get("backend", {})
    checkpoint_value = checkpoint_override or backend.get("checkpoint")
    if not checkpoint_value:
        checks.append(_item("checkpoint", "FAIL", "backend.checkpoint is missing"))
        checkpoint_path = resolve_path("missing-checkpoint")
    else:
        checkpoint_path = resolve_path(checkpoint_value)
        if not checkpoint_path.is_file():
            checks.append(_item("checkpoint", "FAIL", f"missing file: {checkpoint_path}"))
        else:
            actual = sha256_file(checkpoint_path)
            expected = str(backend.get("checkpoint_sha256", ""))
            if len(expected) != 64:
                checks.append(_item("checkpoint hash declaration", "FAIL", f"backend.checkpoint_sha256 missing or malformed in {config_path}"))
            checks.append(_item("checkpoint", "PASS" if actual == expected and len(expected) == 64 else "FAIL", str(checkpoint_path), expected_sha256=expected, actual_sha256=actual))
    return checks, {"config": _display_path(config_path), "checkpoint": str(checkpoint_path), "checkpoint_sha256": str(backend.get("checkpoint_sha256", ""))}


def _dataset_checks(portfolio: dict[str, Any], dataset_root_arg: str | None) -> list[dict[str, Any]]:
    configured = dataset_root_arg or env_or_default("NUSCENES_ROOT", str(portfolio["dataset"].get("root", DEFAULT_DATASET_ROOT)))
    root = resolve_path(configured)
    version = portfolio["dataset"]["version"]
    version_root = root / version
    checks = [_item("nuScenes root", "PASS" if root.is_dir() else "FAIL", str(root)), _item("nuScenes version", "PASS" if version_root.is_dir() else "FAIL", str(version_root))]
    for filename in ("sample.json", "scene.json", "sample_data.json", "calibrated_sensor.json", "ego_pose.json"):
        path = version_root / filename
        checks.append(_item(f"metadata {filename}", "PASS" if path.is_file() else "FAIL", str(path)))
    return checks


def _cache_checks(detector: str, split: str, require: bool, cache_base: Path, expected_checkpoint_sha256: str | None = None, expected_detector_config: tuple[str, str] | None = None) -> list[dict[str, Any]]:
    cache_root = cache_base / "nuscenes-v1.0-mini" / split / CACHE_DETECTOR[detector]
    paths = sorted(cache_root.glob("*.json")) if cache_root.is_dir() else []
    expected_count = 323 if split == "mini_train" else 81
    if not paths:
        return [_item(f"{split} cache", "FAIL" if require else "WARN", f"not present: {cache_root}", entries=0)]
    statuses = []
    tokens: set[str] = set()
    provenance_ok = True
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8")); provenance = payload["provenance"]
            token = str(provenance["sample_token"]); tokens.add(token)
            prediction = payload.get("prediction", {})
            provenance_ok &= path.stem == token and prediction.get("frame_id") == token and provenance["dataset"] == "nuscenes" and provenance["dataset_version"] == "v1.0-mini" and provenance["split"] == split and provenance["detector"] == CACHE_DETECTOR[detector] and provenance["sweeps"] == 10 and float(provenance["candidate_threshold"]) == 0.1 and provenance["prediction_schema_version"] == "lidar_perception.prediction_batch.v1"
            if expected_checkpoint_sha256 is not None:
                provenance_ok &= provenance.get("checkpoint_sha256") == expected_checkpoint_sha256
            if expected_detector_config is not None:
                provenance_ok &= (provenance.get("detector_config"), provenance.get("detector_config_sha256")) == expected_detector_config
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            provenance_ok = False
    count_ok = len(paths) == expected_count and len(tokens) == len(paths)
    statuses.append(_item(f"{split} cache", "PASS" if count_ok and provenance_ok else ("FAIL" if require else "WARN"), str(cache_root), entries=len(paths), expected_entries=expected_count, unique_tokens=len(tokens), provenance_valid=provenance_ok))
    return statuses


def validate(detector: str, dataset_root: str | None, splits: list[str], require_cache: bool, *, checkpoints: dict[str, str | None] | None = None, cache_root: str | None = None) -> dict[str, Any]:
    portfolio = load_system_config()
    selected = ("centerpoint", "voxelnext") if detector == "e3" else (detector,)
    checks: list[dict[str, Any]] = []
    assets: dict[str, Any] = {}
    checkpoints = checkpoints or {}
    for name in selected:
        detector_checks, descriptor = _asset_status(name, _detector_config(portfolio, name), checkpoints.get(name) or __import__("os").environ.get(f"{name.upper()}_CHECKPOINT"))
        checks.extend([dict(item, detector=name) for item in detector_checks]); assets[name] = descriptor
    checks.extend(_dataset_checks(portfolio, dataset_root))
    if detector == "e3":
        frozen = resolve_path(portfolio["e3"]["frozen_config"])
        checks.append(_item("E3 frozen config", "PASS" if frozen.is_file() else "FAIL", str(frozen)))
    cache_base = resolve_path(cache_root or env_or_default("PHASE6_CACHE_ROOT", str(portfolio["dataset"].get("phase6_cache_root", "outputs/phase6_prediction_cache"))))
    if not splits:
        checks.append(_item("experiment cache", "WARN", "not checked; optional for a demo, use --split and optionally --require-cache for experiment reproduction"))
    for split in splits:
        for name in selected:
            checks.extend(_cache_checks(name, split, require_cache, cache_base, assets[name]["checkpoint_sha256"], CACHE_CONFIG[name]))
    failures = [item for item in checks if item["status"] == "FAIL"]
    return {"schema_version": "lidar_perception.phase7_assets.v1", "detector": detector, "checks": checks, "assets": assets, "status": "FAIL" if failures else "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("voxelnext", "centerpoint", "e3"), default="voxelnext")
    parser.add_argument("--dataset-root")
    parser.add_argument("--checkpoint", help="Checkpoint override for a single-detector selection")
    parser.add_argument("--centerpoint-checkpoint")
    parser.add_argument("--voxelnext-checkpoint")
    parser.add_argument("--cache-root")
    parser.add_argument("--split", action="append", choices=("mini_train", "mini_val"), default=[])
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.require_cache and not args.split:
        parser.error("--require-cache requires at least one --split")
    if args.detector == "e3" and args.checkpoint:
        parser.error("--checkpoint is ambiguous for E3; use both detector-specific checkpoint options")
    checkpoints = {
        "centerpoint": args.centerpoint_checkpoint or (args.checkpoint if args.detector == "centerpoint" else None),
        "voxelnext": args.voxelnext_checkpoint or (args.checkpoint if args.detector == "voxelnext" else None),
    }
    try:
        payload = validate(args.detector, args.dataset_root, args.split, args.require_cache, checkpoints=checkpoints, cache_root=args.cache_root)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"FAIL: asset validation: {exc}")
        return 1
    json_report(payload, args.output)
    for check in payload["checks"]:
        suffix = f" [{check['detector']}]" if "detector" in check else ""
        print(f"{check['status']}: {check['name']}{suffix}: {check['detail']}")
    print(f"FINAL: {payload['status']}")
    return 1 if payload["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
