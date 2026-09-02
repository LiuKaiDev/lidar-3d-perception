import argparse
import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import demo_nuscenes as demo  # noqa: E402
import validate_assets as assets  # noqa: E402
import validate_environment as environment  # noqa: E402
from phase7_common import load_system_config  # noqa: E402
from lidar_perception.detection.schemas import PredictionBatch  # noqa: E402
from lidar_perception.geometry.boxes3d import Box3D  # noqa: E402


def _prediction(token: str, center: float, runtime: float) -> PredictionBatch:
    return PredictionBatch(token, [Box3D([center, 0, 0], [4, 2, 1.5], 0, "car", score=0.8, velocity=[1, 0, 0])], runtime_ms=runtime)


def test_portfolio_defaults_to_voxelnext_and_output_path() -> None:
    portfolio = load_system_config()
    assert portfolio["default_detector"] == "voxelnext"
    assert demo.select_detector(None, portfolio) == "voxelnext"
    assert demo.select_detector("centerpoint", portfolio) == "centerpoint"
    assert demo.select_detector("e3", portfolio) == "e3"
    assert demo.default_output_path(portfolio, "voxelnext", "abc").as_posix().endswith("outputs/demo/voxelnext/abc.json")
    with pytest.raises(ValueError, match="unsupported detector"):
        demo.select_detector("bad", portfolio)


def test_demo_help_does_not_load_backend(monkeypatch) -> None:
    monkeypatch.setattr(demo, "_load_backend", lambda *_args, **_kwargs: pytest.fail("backend loaded"))
    monkeypatch.setattr(sys, "argv", ["demo_nuscenes.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        demo.main()
    assert exc.value.code == 0


def test_prediction_payload_preserves_required_schema_fields() -> None:
    payload = demo._prediction_payload(_prediction("token", 0, 12.5), detector="voxelnext", source="voxelnext", timing={})
    assert payload["sample_token"] == "token"
    assert payload["summary"]["prediction_count"] == 1
    box = payload["prediction"]["boxes"][0]
    assert {"label", "score", "center", "size", "yaw", "velocity"} <= set(box)
    assert payload["prediction"]["runtime_ms"] == 12.5


def test_e3_uses_frozen_config_and_sequential_backend_order(monkeypatch) -> None:
    calls = []

    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def predict(self, frame):
            calls.append(f"predict:{self.name}")
            return _prediction(frame, 0.0 if self.name == "centerpoint" else 0.1, 10.0)

    def fake_load(name, _config, _checkpoint=None):
        calls.append(f"load:{name}")
        return FakeBackend(name), {}, 1.0

    monkeypatch.setattr(demo, "_load_backend", fake_load)
    monkeypatch.setattr(demo, "_release_backend", lambda backend: calls.append(f"release:{backend.name}"))
    args = argparse.Namespace(checkpoint=None)
    payload = demo._run_e3(args, load_system_config(), "token")
    assert calls == ["load:centerpoint", "predict:centerpoint", "release:centerpoint", "load:voxelnext", "predict:voxelnext", "release:voxelnext"]
    assert payload["timing"]["fusion_config"]["association_threshold_m"] == 0.5
    assert payload["timing"]["fusion_config"]["centerpoint_weight"] == 1.2
    assert payload["timing"]["detector_execution_model"].startswith("sequential")
    assert "associations" not in payload["timing"]["fusion_diagnostics"]
    assert payload["timing"]["fusion_diagnostics"]["association_count"] == 1
    assert payload["prediction"]["runtime_ms"] is None


def test_environment_profiles_and_failure_exit(monkeypatch) -> None:
    cpu = environment.validate("cpu")
    assert cpu["status"] == "PASS"
    original = environment._module_check
    monkeypatch.setattr(environment, "_module_check", lambda name, purpose: (_ for _ in ()).throw(RuntimeError("missing NumPy for geometry")) if name == "numpy" else original(name, purpose))
    failed = environment.validate("cpu")
    assert failed["status"] == "FAIL"
    assert any(item["status"] == "FAIL" and item["name"] == "NumPy" for item in failed["checks"])
    monkeypatch.setattr(environment, "validate", lambda _profile: {"checks": [], "status": "FAIL"})
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", "--profile", "cpu"])
    assert environment.main() == 1


def test_gpu_environment_missing_dependency_is_reported(monkeypatch) -> None:
    original = environment._module_check
    monkeypatch.setattr(environment, "_module_check", lambda name, purpose: (_ for _ in ()).throw(RuntimeError("spconv is required for sparse voxel inference")) if name == "spconv" else original(name, purpose))
    failed = environment.validate("gpu")
    assert failed["status"] == "FAIL"
    assert any(item["name"] == "spconv" and item["status"] == "FAIL" for item in failed["checks"])


def test_openpcdet_check_reports_uninitialized_submodule(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(environment, "ROOT", tmp_path)
    (tmp_path / "third_party/OpenPCDet").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="submodule is not initialized"):
        environment._openpcdet_check()


def _write_detector_config(path: Path, checkpoint: Path, expected_hash: str) -> None:
    path.write_text(yaml.safe_dump({"backend": {"checkpoint": str(checkpoint), "checkpoint_sha256": expected_hash}, "dataset": {}}), encoding="utf-8")


def test_asset_checkpoint_missing_and_hash_mismatch(tmp_path: Path) -> None:
    config = tmp_path / "detector.yaml"
    checkpoint = tmp_path / "model.pth"
    _write_detector_config(config, checkpoint, "a" * 64)
    checks, _ = assets._asset_status("voxelnext", config, None)
    assert any(item["status"] == "FAIL" and "missing file" in item["detail"] for item in checks)
    checkpoint.write_bytes(b"checkpoint")
    checks, _ = assets._asset_status("voxelnext", config, None)
    assert any(item["status"] == "FAIL" and item["name"] == "checkpoint" for item in checks)
    expected = hashlib.sha256(b"checkpoint").hexdigest()
    _write_detector_config(config, checkpoint, expected)
    checks, _ = assets._asset_status("voxelnext", config, None)
    assert any(item["status"] == "PASS" and item["name"] == "checkpoint" for item in checks)


def test_asset_validator_failure_returns_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(assets, "validate", lambda *_args, **_kwargs: {"checks": [], "status": "FAIL"})
    monkeypatch.setattr(sys, "argv", ["validate_assets.py", "--detector", "voxelnext"])
    assert assets.main() == 1


def test_optional_and_required_cache_are_read_only(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    optional = assets._cache_checks("voxelnext", "mini_val", False, cache_root)
    required = assets._cache_checks("voxelnext", "mini_val", True, cache_root)
    assert optional[0]["status"] == "WARN"
    assert required[0]["status"] == "FAIL"
    assert not cache_root.exists()

    directory = cache_root / "nuscenes-v1.0-mini/mini_val/voxelnext"
    directory.mkdir(parents=True)
    for index in range(81):
        token = f"token{index:03d}"
        payload = {"prediction": {"frame_id": token}, "provenance": {"dataset": "nuscenes", "dataset_version": "v1.0-mini", "split": "mini_val", "detector": "voxelnext", "sample_token": token, "sweeps": 10, "candidate_threshold": 0.1, "prediction_schema_version": "lidar_perception.prediction_batch.v1"}}
        (directory / f"{token}.json").write_text(json.dumps(payload), encoding="utf-8")
    assert assets._cache_checks("voxelnext", "mini_val", True, cache_root)[0]["status"] == "PASS"


def test_cache_checkpoint_provenance_mismatch_is_rejected(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    directory = cache_root / "nuscenes-v1.0-mini/mini_val/voxelnext"
    directory.mkdir(parents=True)
    for index in range(81):
        token = f"token{index:03d}"
        payload = {"prediction": {"frame_id": token}, "provenance": {"dataset": "nuscenes", "dataset_version": "v1.0-mini", "split": "mini_val", "detector": "voxelnext", "sample_token": token, "sweeps": 10, "candidate_threshold": 0.1, "prediction_schema_version": "lidar_perception.prediction_batch.v1", "checkpoint_sha256": "wrong"}}
        (directory / f"{token}.json").write_text(json.dumps(payload), encoding="utf-8")
    result = assets._cache_checks("voxelnext", "mini_val", True, cache_root, "expected")
    assert result[0]["status"] == "FAIL"
