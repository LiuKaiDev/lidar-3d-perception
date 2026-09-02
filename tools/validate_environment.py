#!/usr/bin/env python3
"""Read-only Phase 7 environment validation for CPU and GPU profiles."""

from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from phase7_common import ROOT, json_report, load_project_config, load_system_config, resolve_path


def _check(name: str, fn: Callable[[], str]) -> dict[str, Any]:
    try:
        return {"name": name, "status": "PASS", "detail": fn()}
    except Exception as exc:  # validators intentionally report concise diagnostics
        return {"name": name, "status": "FAIL", "detail": str(exc)}


def _warn(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "WARN", "detail": detail}


def _python_check() -> str:
    if sys.version_info < (3, 12):
        raise RuntimeError(f"Python {platform.python_version()} found; >=3.12 is required")
    return platform.python_version()


def _project_check() -> str:
    # CPU validation is intentionally usable from a source checkout that has
    # not initialized the optional detector submodule.  GPU validation checks
    # the submodule explicitly in ``_openpcdet_check`` below.
    required = [ROOT / "lidar_perception", ROOT / "configs/system/portfolio.yaml"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing project paths: " + ", ".join(missing))
    return "project package and system config present"


def _config_check() -> str:
    portfolio = load_system_config()
    for detector in ("centerpoint", "voxelnext"):
        path = resolve_path(portfolio["detectors"][detector]["config"])
        config = load_project_config(path)
        if not isinstance(config.get("backend"), dict) or not isinstance(config.get("dataset"), dict):
            raise RuntimeError(f"invalid detector config: {path}")
    return "CenterPoint and VoxelNeXt project configs parse"


def _module_check(module_name: str, purpose: str) -> str:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    return f"{module_name} available" + (f" ({version})" if version else "") + f"; {purpose}"


def _openpcdet_check() -> str:
    source = ROOT / "third_party/OpenPCDet"
    if not (source / "pcdet/__init__.py").is_file():
        raise RuntimeError("OpenPCDet submodule is not initialized; run git submodule update --init third_party/OpenPCDet")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    expected = "233f849829b6ac19afb8af8837a0246890908755"
    if revision != expected:
        raise RuntimeError(f"OpenPCDet revision {revision}; expected {expected}")
    sys.path.insert(0, str(source))
    importlib.import_module("pcdet")
    return f"OpenPCDet importable at fixed revision {revision}"


def _compiled_ops_check() -> str:
    sys.path.insert(0, str(ROOT / "third_party/OpenPCDet"))
    for module_name in ("pcdet.ops.iou3d_nms.iou3d_nms_cuda", "pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda"):
        importlib.import_module(module_name)
    return "OpenPCDet compiled CUDA op modules import"


def validate(profile: str) -> dict[str, Any]:
    checks = [
        _check("python", _python_check),
        _check("project structure", _project_check),
        _check("project configuration", _config_check),
        _check("PyYAML", lambda: _module_check("yaml", "configuration parsing")),
        _check("NumPy", lambda: _module_check("numpy", "geometry and schema arrays")),
        _check("Torch", lambda: _module_check("torch", "PredictionBatch backend dependency")),
    ]
    if profile == "cpu":
        checks.append(_warn("CUDA requirement", "not required for CPU profile; no detector inference is claimed"))
    else:
        def cuda_check() -> str:
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; install a CUDA-enabled Torch build and driver")
            return f"CUDA available ({torch.version.cuda}); {torch.cuda.get_device_name(0)}"
        checks.extend([
            _check("CUDA", cuda_check),
            _check("spconv", lambda: _module_check("spconv", "sparse voxel inference")),
            _check("nuScenes devkit", lambda: _module_check("nuscenes", "official mini evaluation and dataset access")),
            _check("OpenPCDet", _openpcdet_check),
            _check("OpenPCDet compiled ops", _compiled_ops_check),
        ])
    failures = [item for item in checks if item["status"] == "FAIL"]
    payload = {"schema_version": "lidar_perception.phase7_environment.v1", "profile": profile, "checks": checks, "status": "FAIL" if failures else "PASS"}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        payload = validate(args.profile)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAIL: environment validation: {exc}")
        return 1
    json_report(payload, args.output)
    for check in payload["checks"]:
        print(f"{check['status']}: {check['name']}: {check['detail']}")
    print(f"FINAL: {payload['status']}")
    return 1 if payload["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
