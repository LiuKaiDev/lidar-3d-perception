import json
from pathlib import Path

import numpy as np

from lidar_perception.benchmark.report import build_report, write_reports
from lidar_perception.benchmark.runner import benchmark_model, load_cached_accuracy, model_provenance, run_sequential_benchmark
from lidar_perception.datasets.schemas import PointCloudFrame
from lidar_perception.detection.openpcdet_backend import OpenPCDetBackend, VoxelNeXtBackend


def _dataset():
    return {"name": "nuScenes", "version": "v1.0-mini", "split": "mini_val"}


def _accuracy():
    return {
        "dataset": "nuScenes",
        "version": "v1.0-mini",
        "split": "mini_val",
        "mAP": 0.4,
        "NDS": 0.5,
        "mATE": 0.2,
        "mASE": 0.3,
        "mAOE": 0.4,
        "mAVE": 0.5,
        "mAAE": 0.6,
    }


def test_report_keeps_only_same_dataset_accuracy_comparable(tmp_path: Path) -> None:
    models = [
        {"name": "centerpoint", "status": "cached_accuracy", "sweeps": 10, "accuracy": _accuracy()},
        {"name": "voxelnext", "status": "completed", "sweeps": 10, "accuracy": {**_accuracy(), "version": "v1.0-trainval"}},
    ]
    report = build_report(dataset=_dataset(), protocol={"batch_size": 1}, models=models, environment={})
    assert report["status"] == "BLOCKED"
    assert [row["model"] for row in report["accuracy_comparison"]] == ["centerpoint"]


def test_voxelnext_selects_shared_openpcdet_backend_identity() -> None:
    assert OpenPCDetBackend(device="cpu", model_type="voxelnext").name() == "openpcdet_voxelnext"
    assert VoxelNeXtBackend(device="cpu").name() == "openpcdet_voxelnext"


def test_report_pass_requires_two_accuracy_and_runtime_complete_models() -> None:
    runtime = {"end_to_end": {"mean_ms": 10.0}, "model_only": {"mean_ms": 8.0}}
    models = [
        {"name": "one", "status": "completed", "sweeps": 10, "accuracy": _accuracy(), "benchmark": runtime},
        {"name": "two", "status": "completed", "sweeps": 10, "accuracy": _accuracy(), "benchmark": runtime},
    ]
    report = build_report(dataset=_dataset(), protocol={"batch_size": 1}, models=models, environment={})
    assert report["status"] == "PASS"


def test_report_writes_required_machine_and_human_views(tmp_path: Path) -> None:
    report = build_report(
        dataset=_dataset(),
        protocol={"batch_size": 1},
        models=[{"name": "centerpoint", "status": "cached_accuracy", "sweeps": 10, "accuracy": _accuracy(), "provenance": {}}],
        environment={"torch": "test"},
    )
    paths = write_reports(report, tmp_path)
    assert {path.name for path in paths.values()} == {"benchmark.json", "accuracy.json", "environment.json", "benchmark.csv", "README.md"}
    assert json.loads((tmp_path / "benchmark.json").read_text())["schema_version"].startswith("lidar_perception.phase5")
    assert "KITTI AP_R40" in (tmp_path / "README.md").read_text()


def test_model_provenance_records_checkpoint_availability_and_classes(tmp_path: Path) -> None:
    config_path = tmp_path / "wrapper.yaml"
    opcdet = tmp_path / "model.yaml"
    checkpoint = tmp_path / "weights.pth"
    opcdet.write_text("CLASS_NAMES: [car, pedestrian]\n", encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    config = {"backend": {"model": "voxelnext", "openpcdet_config": str(opcdet), "checkpoint": str(checkpoint)}, "dataset": {"max_sweeps": 10}}
    config_path.write_text("test", encoding="utf-8")
    provenance = model_provenance(config, config_path)
    assert provenance["checkpoint_available"] is True
    assert provenance["checkpoint_sha256"]
    assert provenance["class_names"] == ["car", "pedestrian"]


def test_cached_accuracy_rejects_different_split(tmp_path: Path) -> None:
    cache = tmp_path / "summary.json"
    cache.write_text(json.dumps({
        "dataset_version": "v1.0-mini",
        "eval_set": "mini_train",
        "protocol": "detection_cvpr_2019",
        "mAP": 0.4,
        "NDS": 0.5,
    }), encoding="utf-8")
    try:
        load_cached_accuracy(cache, dataset=_dataset())
    except ValueError:
        return
    raise AssertionError("expected a mismatched cached split to be rejected")


def test_sequential_runner_isolates_models_and_records_missing_checkpoint() -> None:
    frame = PointCloudFrame("sample", np.zeros((1, 4), dtype=np.float32))
    calls = []

    class Backend:
        device = type("Device", (), {"type": "cpu"})()

        def __init__(self, name):
            self._name = name
            self.load_report = {"loaded_key_count": 1}

        def name(self):
            return self._name

        def load(self):
            calls.append(self._name)

        def prepare_frame(self, frame):
            return {"points": frame.points}

        def _predict_prepared(self, batch):
            return {}, 0.1

        def predict(self, frame):
            return object()

    records = run_sequential_benchmark(
        [{"name": "first"}, {"name": "second"}],
        frame,
        lambda spec: Backend(spec["name"]),
        warmup=0,
        iterations=1,
    )
    assert calls == ["first", "second"]
    assert [record["status"] for record in records] == ["completed", "completed"]


def test_benchmark_model_records_missing_checkpoint_without_fabricated_metrics() -> None:
    frame = PointCloudFrame("sample", np.zeros((1, 4), dtype=np.float32))

    class MissingBackend:
        def load(self):
            raise FileNotFoundError("official checkpoint missing")

    record = benchmark_model(
        {"name": "voxelnext"},
        frame,
        lambda spec: MissingBackend(),
        warmup=0,
        iterations=1,
    )
    assert record["status"] == "blocked"
    assert "official checkpoint missing" in record["error"]
    assert "benchmark" not in record
