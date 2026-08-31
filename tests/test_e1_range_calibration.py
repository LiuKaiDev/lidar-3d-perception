import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.experiments.calibration import (
    CalibrationParameters,
    LogisticCalibrator,
    RangeAwareCalibrator,
    ScoreOnlyCalibrator,
    calibrate_prediction,
    fit_logistic_calibrator,
    search_calibrators,
)
from lidar_perception.experiments.features import predicted_box_range_m
from lidar_perception.experiments.manifest import ExperimentManifest
from lidar_perception.geometry.boxes3d import Box3D


def _fit_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([0.1, 0.2, 0.3, 0.65, 0.8, 0.9]),
        np.asarray([5.0, 45.0, 15.0, 55.0, 25.0, 65.0]),
        np.asarray([0, 0, 0, 1, 1, 1]),
    )


def test_e1_predicted_range_uses_only_predicted_xy_center() -> None:
    near = Box3D([3, 4, 999], [1, 1, 1], 0, "car", score=0.5)
    assert predicted_box_range_m(near) == 5.0


def test_e1_calibration_is_finite_bounded_and_deterministic() -> None:
    scores, ranges, labels = _fit_data()
    first = fit_logistic_calibrator(scores, ranges, labels, family="score_range", ridge=1.0)
    second = fit_logistic_calibrator(scores, ranges, labels, family="score_range", ridge=1.0)
    assert first.to_dict() == second.to_dict()
    calibrated = first.predict([0.0, 0.1, 0.9, 1.0], [0, 20, 40, 100])
    assert np.all(np.isfinite(calibrated))
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_e1_frozen_parameters_json_round_trip() -> None:
    scores, ranges, labels = _fit_data()
    fitted = fit_logistic_calibrator(scores, ranges, labels, family="score_range", ridge=0.1)
    encoded = json.loads(json.dumps(fitted.to_dict()))
    restored = LogisticCalibrator.from_dict(encoded)
    assert restored.to_dict() == fitted.to_dict()
    assert np.array_equal(restored.predict(scores, ranges), fitted.predict(scores, ranges))


def test_e1_inference_api_rejects_ground_truth_and_preserves_boxes() -> None:
    assert "ground_truth" not in inspect.signature(calibrate_prediction).parameters
    params = CalibrationParameters("score_range", intercept=0.2, score_weight=1.1, range_weight=0.3)
    calibrator = RangeAwareCalibrator(params)
    source = PredictionBatch(
        "sample",
        [Box3D([3, 4, 1], [4, 2, 1.5], 0.4, "car", score=0.6, velocity=[1, 2, 0], track_id="id")],
        runtime_ms=12.0,
        metadata={"sample_token": "sample"},
    )
    calibrated = calibrate_prediction(source, calibrator)
    before, after = source.boxes[0], calibrated.boxes[0]
    assert np.array_equal(after.center, before.center)
    assert np.array_equal(after.size, before.size)
    assert np.array_equal(after.velocity, before.velocity)
    assert (after.yaw, after.label, after.track_id) == (before.yaw, before.label, before.track_id)
    assert after.score != before.score
    with pytest.raises(TypeError):
        calibrate_prediction(source, calibrator, ground_truth=[])


def test_e1_search_records_every_config_and_score_only_control() -> None:
    scores, ranges, labels = _fit_data()
    selected, records, control = search_calibrators(scores, ranges, labels, ridge_values=(0.1, 1.0, 10.0))
    repeated, repeated_records, repeated_control = search_calibrators(scores, ranges, labels, ridge_values=(0.1, 1.0, 10.0))
    assert isinstance(selected, RangeAwareCalibrator)
    assert isinstance(control, ScoreOnlyCalibrator)
    assert len(records) == 6
    assert sum(record.selected for record in records) == 1
    assert [record.to_dict() for record in records] == [record.to_dict() for record in repeated_records]
    assert selected.to_dict() == repeated.to_dict()
    assert control.to_dict() == repeated_control.to_dict()


def test_e1_manifest_freezes_split_threshold_and_candidate_policy() -> None:
    manifest = ExperimentManifest.load("experiments/e1_range_calibration/config.yaml")
    data = manifest.to_dict()
    assert data["dataset"]["split"] == "mini_train"
    assert data["tuning"]["confirmation_split"] == "mini_val"
    assert data["tuning"]["mini_val_used_for_tuning"] is False
    assert data["prediction_candidate_threshold"] == 0.1
    assert data["selection"]["downstream_threshold"] is None
    assert data["models"][0]["sweeps"] == 10

    changed = dict(data)
    changed["prediction_candidate_threshold"] = 0.05
    with pytest.raises(ValueError, match="0.1"):
        ExperimentManifest(changed)


def test_e1_search_and_report_artifacts_are_serializable() -> None:
    experiment = Path("experiments/e1_range_calibration")
    with (experiment / "search_results.json").open(encoding="utf-8") as handle:
        search = json.load(handle)
    with (experiment / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert search["fit_split"] == "mini_train"
    assert search["selected_parameters"]["family"] == "score_range"
    assert metrics["fit_split"] == "mini_train"
    assert metrics["confirmation_split"] == "mini_val"
    assert metrics["candidate_threshold"] == 0.1
    assert (experiment / "analysis.md").read_text(encoding="utf-8").startswith("# Experiment\n")


def test_e1_config_is_one_detector_and_global_not_class_specific() -> None:
    config = yaml.safe_load(Path("experiments/e1_range_calibration/config.yaml").read_text(encoding="utf-8"))
    assert len(config["models"]) == 1
    assert config["calibration"]["parameter_count"] == 3
    assert "predicted_box_range" in config["inference_policy"]["allowed_features"]
    assert all("class" not in name for name in config["calibration"]["inference_features"])
