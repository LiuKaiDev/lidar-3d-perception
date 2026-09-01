#!/usr/bin/env python3
"""Generate the compact Phase 6 closure provenance summary."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

from phase7_common import ROOT, load_project_config, load_system_config, resolve_path, sha256_file


CLOSURE_COMMIT = "3d25ca08fbd66f366ecc215c6c0a79da40c91c38"
CLOSURE_TAG = "v0.6.0-phase6-closure"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": __import__("platform").python_version(),
        "torch": _version("torch"),
        "torch_cuda": None,
        "cuda_available": None,
        "spconv": _version("spconv-cu124") or _version("spconv"),
        "nuscenes_devkit": _version("nuscenes-devkit"),
        "verified_on_current_machine": False,
    }
    try:
        import torch

        result.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
    except ImportError:
        pass
    try:
        import spconv

        result["spconv"] = spconv.__version__
    except ImportError:
        pass
    result["verified_on_current_machine"] = all(result[key] is not None for key in ("torch", "spconv", "nuscenes_devkit"))
    return result


def _checkpoint(config_path: Path) -> dict[str, Any]:
    config = load_project_config(config_path)
    backend = config["backend"]
    checkpoint = resolve_path(backend["checkpoint"])
    expected = backend["checkpoint_sha256"]
    actual = sha256_file(checkpoint) if checkpoint.is_file() else None
    return {"config": str(config_path.relative_to(ROOT)), "expected_sha256": expected, "present": checkpoint.is_file(), "verified": actual == expected if actual else False}


def _cache_summary(detector: str, split: str) -> dict[str, Any]:
    directory_name = "centerpoint_pointpillar" if detector == "centerpoint" else "voxelnext"
    root = ROOT / "outputs/phase6_prediction_cache/nuscenes-v1.0-mini" / split / directory_name
    paths = sorted(root.glob("*.json")) if root.is_dir() else []
    valid = 0
    key_provenance: dict[str, Any] | None = None
    tokens: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8")); provenance = payload["provenance"]
            token = str(provenance["sample_token"]); tokens.add(token)
            if provenance["dataset"] == "nuscenes" and provenance["dataset_version"] == "v1.0-mini" and provenance["split"] == split and provenance["sweeps"] == 10 and float(provenance["candidate_threshold"]) == 0.1:
                valid += 1
            if key_provenance is None:
                key_provenance = {key: provenance[key] for key in ("dataset", "dataset_version", "split", "detector", "detector_config", "detector_config_sha256", "checkpoint_sha256", "sweeps", "candidate_threshold", "score_filtering_policy", "prediction_schema_version")}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return {"entries": len(paths), "compatible_entries": valid, "unique_tokens": len(tokens), "verified_on_current_machine": bool(paths) and valid == len(paths) == len(tokens), "key_provenance": key_provenance}


def build() -> dict[str, Any]:
    portfolio = load_system_config()
    cp_config = resolve_path(portfolio["detectors"]["centerpoint"]["config"])
    vn_config = resolve_path(portfolio["detectors"]["voxelnext"]["config"])
    e4 = json.loads((ROOT / "experiments/e4_repeat_validation/metrics.json").read_text(encoding="utf-8"))
    e4_ablation = json.loads((ROOT / "experiments/e4_repeat_validation/ablation.json").read_text(encoding="utf-8"))
    tag_target = _git("rev-list", "-n", "1", CLOSURE_TAG)
    return {
        "schema_version": "lidar_perception.phase6_provenance.v1",
        "dataset_label": "nuScenes v1.0-mini exploratory experiment",
        "phase6": {"status": "PASS", "classification": "DIRECTIONAL", "closure_commit": CLOSURE_COMMIT, "closure_tag": CLOSURE_TAG, "closure_tag_target": tag_target, "tag_target_verified": tag_target == CLOSURE_COMMIT},
        "third_party": {"openpcdet_revision": _git("-C", "third_party/OpenPCDet", "rev-parse", "HEAD"), "openpcdet_modified": bool(_git("-C", "third_party/OpenPCDet", "status", "--short"))},
        "environment": _environment(),
        "models": {"default": "voxelnext", "centerpoint": _checkpoint(cp_config), "voxelnext": _checkpoint(vn_config), "e3": {"role": "directional late-fusion ablation; not the default", "frozen_config": portfolio["e3"]["frozen_config"]}},
        "cache": {split: {detector: _cache_summary(detector, split) for detector in ("centerpoint", "voxelnext")} for split in ("mini_train", "mini_val")},
        "e4_repeat_validation": {"status": e4["status"], "classification": e4["classification"], "custom_exact_repeat": e4_ablation["e3_repeat_exact_match"], "official_exact_repeat": e4_ablation["official_repeat_exact_match"]},
        "limitations": {"mini_val_scene_count": 2, "mini_val_previously_exposed": True, "full_nuscenes_claim": False, "sota_claim": False},
    }


def main() -> int:
    output = ROOT / "reports/phase6_provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved: {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
