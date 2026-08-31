import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.experiments.calibration import ScoreOnlyCalibrator
from lidar_perception.experiments.features import (
    extract_prediction_features,
    predicted_box_keyframe_point_counts,
)
from lidar_perception.experiments.manifest import ExperimentManifest
from lidar_perception.experiments.sparsity import (
    SparsityAwareCalibrator,
    apply_sparsity_policy,
    fit_sparsity_calibrator,
    search_sparsity_calibrators,
)
from lidar_perception.geometry.boxes3d import Box3D


def _prediction() -> PredictionBatch:
    return PredictionBatch("sample", [
        Box3D([0, 0, 0], [4, 2, 2], np.pi / 4, "car", score=0.7),
        Box3D([10, 10, 0], [2, 2, 2], 0, "pedestrian", score=0.2),
    ])


def _search_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    scores = np.tile(np.asarray([0.1, 0.25, 0.65, 0.9]), 4)
    counts = np.tile(np.asarray([0, 2, 15, 60]), 4)
    labels = np.tile(np.asarray([0, 0, 1, 1]), 4)
    scenes = [scene for scene in ("a", "b", "c", "d") for _ in range(4)]
    return scores, counts, labels, scenes


def test_predicted_box_keyframe_count_rotated_boundary_and_sweep_separation() -> None:
    prediction = _prediction()
    boundary = np.sqrt(2.0)
    points = np.asarray([
        [0, 0, 0, 1, 0.0],
        [boundary, 0, 0, 1, 0.0],  # Rotated-box boundary.
        [3, 3, 0, 1, 0.0],
        [0, 0, 0, 1, 0.5],  # Inside, but historical sweep.
    ], dtype=np.float64)
    counts = predicted_box_keyframe_point_counts(prediction, points)
    assert counts.tolist() == [2, 0]
    multi = extract_prediction_features(prediction, points, point_source="multi_sweep")
    assert multi[0].point_count == 3
    assert multi[0].to_dict()["predicted_box_multi_sweep_point_count"] == 3


def test_predicted_box_keyframe_count_handles_empty_and_zero_supported_points() -> None:
    prediction = _prediction()
    assert predicted_box_keyframe_point_counts(prediction, np.empty((0, 5))).tolist() == [0, 0]
    outside = np.asarray([[100, 100, 100, 1, 0]], dtype=np.float64)
    features = extract_prediction_features(prediction, outside)
    assert all(feature.point_count == 0 for feature in features)
    assert all(feature.predicted_box_keyframe_point_count == 0 for feature in features)


def test_sparsity_feature_api_accepts_no_ground_truth() -> None:
    assert "ground_truth" not in inspect.signature(predicted_box_keyframe_point_counts).parameters
    with pytest.raises(TypeError):
        predicted_box_keyframe_point_counts(_prediction(), np.empty((0, 5)), ground_truth=[])


def test_sparsity_fit_is_deterministic_finite_and_bounded() -> None:
    scores, counts, labels, _ = _search_data()
    first = fit_sparsity_calibrator(scores, counts, labels, ridge=1.0)
    second = fit_sparsity_calibrator(scores, counts, labels, ridge=1.0)
    assert first.to_dict() == second.to_dict()
    result = first.predict([0.0, 0.5, 1.0], [0, 5, 100])
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))
    with pytest.raises(ValueError, match="non-negative integers"):
        first.predict([0.5], [-1])


def test_sparsity_serialization_round_trip() -> None:
    scores, counts, labels, _ = _search_data()
    model = fit_sparsity_calibrator(scores, counts, labels, ridge=0.1)
    restored = SparsityAwareCalibrator.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.to_dict() == model.to_dict()
    assert np.array_equal(restored.predict(scores, counts), model.predict(scores, counts))


def test_scene_level_search_is_deterministic_and_reuses_score_only_control() -> None:
    scores, counts, labels, scenes = _search_data()
    selected, records, control = search_sparsity_calibrators(scores, counts, labels, scenes)
    repeated, repeated_records, repeated_control = search_sparsity_calibrators(scores, counts, labels, scenes)
    assert isinstance(selected, SparsityAwareCalibrator)
    assert isinstance(control, ScoreOnlyCalibrator)
    assert len(records) == 6 and sum(record.selected for record in records) == 1
    assert all(record.fold_count == 4 and record.validation_strategy == "leave-one-scene-out" for record in records)
    assert [record.to_dict() for record in records] == [record.to_dict() for record in repeated_records]
    assert selected.to_dict() == repeated.to_dict()
    assert control.to_dict() == repeated_control.to_dict()


def test_path_a_policy_preserves_membership_geometry_and_classes() -> None:
    prediction = _prediction()
    points = np.asarray([[0, 0, 0, 1, 0], [10, 10, 0, 1, 0]], dtype=np.float64)
    scores, counts, labels, _ = _search_data()
    model = fit_sparsity_calibrator(scores, counts, labels)
    output = apply_sparsity_policy(prediction, points, model)
    assert len(output.boxes) == len(prediction.boxes)
    for before, after in zip(prediction.boxes, output.boxes):
        assert np.array_equal(before.center, after.center)
        assert np.array_equal(before.size, after.size)
        assert (before.yaw, before.label) == (after.yaw, after.label)
    assert output.metadata["point_count_source"] == "current_keyframe"
    with pytest.raises(TypeError):
        apply_sparsity_policy(prediction, points, model, ground_truth=[])


def test_e2_manifest_and_artifacts_freeze_train_val_boundary() -> None:
    experiment = Path("experiments/e2_density_policy")
    manifest = ExperimentManifest.load(experiment / "config.yaml")
    data = manifest.to_dict()
    assert data["experiment_id"] == "E2"
    assert data["dataset"]["split"] == "mini_train"
    assert data["prediction_candidate_threshold"] == 0.1
    assert data["operating_policy"]["path"] == "A"
    assert data["inference_policy"]["point_count_source"] == "current_keyframe"
    assert data["tuning"]["mini_val_used_for_tuning"] is False

    changed = dict(data)
    changed["prediction_candidate_threshold"] = 0.05
    with pytest.raises(ValueError, match="0.1"):
        ExperimentManifest(changed)

    with (experiment / "search_results.json").open(encoding="utf-8") as handle:
        search = json.load(handle)
    with (experiment / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert search["fit_split"] == "mini_train"
    assert metrics["confirmation_split"] == "mini_val"
    assert metrics["candidate_threshold"] == 0.1
    assert (experiment / "analysis.md").read_text(encoding="utf-8").startswith("# Experiment\n")


def test_e2_config_is_global_sparsity_only_without_range() -> None:
    config = yaml.safe_load(Path("experiments/e2_density_policy/config.yaml").read_text(encoding="utf-8"))
    assert len(config["models"]) == 1
    assert config["policy"]["parameter_count"] == 3
    assert "predicted_box_keyframe_point_count" in config["policy"]["inference_features"]
    assert all("range" not in name for name in config["policy"]["inference_features"])
