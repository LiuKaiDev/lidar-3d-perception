from pathlib import Path

import numpy as np
import pytest

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.experiments.bootstrap import (
    SceneMetricCounts,
    SceneMetricRecord,
    group_scene_counts,
    paired_scene_bootstrap,
    scene_level_bootstrap,
)
from lidar_perception.experiments.cache import PredictionCache, PredictionCacheProvenance
from lidar_perception.experiments.features import extract_prediction_features, predicted_box_range_m
from lidar_perception.experiments.manifest import ExperimentManifest
from lidar_perception.geometry.boxes3d import Box3D


def test_scene_grouping_is_additive_and_deterministic() -> None:
    grouped = group_scene_counts([
        SceneMetricRecord("scene-b", "recall_50m_plus", 1, 2),
        SceneMetricRecord("scene-a", "recall_50m_plus", 2, 3),
        SceneMetricRecord("scene-a", "recall_50m_plus", 1, 2),
        SceneMetricRecord("scene-a", "overall_custom_recall", 4, 5),
    ])
    assert [scene.scene_id for scene in grouped] == ["scene-a", "scene-b"]
    assert grouped[0].counts["recall_50m_plus"] == (3, 5)
    assert grouped[0].counts["overall_custom_recall"] == (4, 5)


def test_scene_bootstrap_is_deterministic_with_fixed_seed() -> None:
    scenes = [
        SceneMetricCounts("scene-a", {"recall_50m_plus": (1, 4)}),
        SceneMetricCounts("scene-b", {"recall_50m_plus": (3, 6)}),
        SceneMetricCounts("scene-c", {"recall_50m_plus": (7, 10)}),
    ]
    first = scene_level_bootstrap(scenes, metrics=["recall_50m_plus"], repetitions=200, seed=42)
    second = scene_level_bootstrap(reversed(scenes), metrics=["recall_50m_plus"], repetitions=200, seed=42)
    assert first == second
    interval = first["recall_50m_plus"]
    assert interval.point_estimate == pytest.approx(11 / 20)
    assert interval.valid_repetitions == 200
    assert interval.lower <= interval.point_estimate <= interval.upper


def test_bootstrap_empty_bin_no_scene_and_one_scene_edges() -> None:
    no_scenes = scene_level_bootstrap([], metrics=["recall_50m_plus"], repetitions=10)
    assert no_scenes["recall_50m_plus"].point_estimate is None
    assert no_scenes["recall_50m_plus"].lower is None
    assert no_scenes["recall_50m_plus"].valid_repetitions == 0

    empty_bin = scene_level_bootstrap(
        [SceneMetricCounts("scene", {"recall_50m_plus": (0, 0)})],
        metrics=["recall_50m_plus"],
        repetitions=10,
    )["recall_50m_plus"]
    assert empty_bin.point_estimate is None and empty_bin.valid_repetitions == 0

    one_scene = scene_level_bootstrap(
        [SceneMetricCounts("scene", {"recall_50m_plus": (5, 10)})],
        metrics=["recall_50m_plus"],
        repetitions=10,
    )["recall_50m_plus"]
    assert one_scene.point_estimate == one_scene.lower == one_scene.upper == 0.5


def test_paired_scene_bootstrap_uses_delta_and_frozen_language() -> None:
    baseline = [
        SceneMetricCounts("scene-a", {"recall_50m_plus": (2, 10)}),
        SceneMetricCounts("scene-b", {"recall_50m_plus": (4, 10)}),
    ]
    experiment = [
        SceneMetricCounts("scene-a", {"recall_50m_plus": (3, 10)}),
        SceneMetricCounts("scene-b", {"recall_50m_plus": (5, 10)}),
    ]
    interval = paired_scene_bootstrap(
        baseline,
        experiment,
        metrics=["recall_50m_plus"],
        repetitions=100,
        seed=42,
    )["recall_50m_plus"]
    assert interval.delta == pytest.approx(0.1)
    assert interval.lower == pytest.approx(0.1)
    assert interval.claim == "bootstrap-supported improvement on mini"

    uncertain = paired_scene_bootstrap(
        [
            SceneMetricCounts("scene-a", {"recall_50m_plus": (5, 10)}),
            SceneMetricCounts("scene-b", {"recall_50m_plus": (5, 10)}),
        ],
        [
            SceneMetricCounts("scene-a", {"recall_50m_plus": (10, 10)}),
            SceneMetricCounts("scene-b", {"recall_50m_plus": (1, 10)}),
        ],
        metrics=["recall_50m_plus"],
        repetitions=100,
        seed=42,
    )["recall_50m_plus"]
    assert uncertain.delta == pytest.approx(0.05)
    assert uncertain.lower < 0 < uncertain.upper
    assert uncertain.claim == "directional improvement; uncertainty overlaps zero"

    changed_denominator = [SceneMetricCounts("scene-a", {"recall_50m_plus": (3, 11)}), experiment[1]]
    with pytest.raises(ValueError, match="denominators differ"):
        paired_scene_bootstrap(baseline, changed_denominator, metrics=["recall_50m_plus"])


def test_e0_manifest_round_trip_and_protocol_guards(tmp_path: Path) -> None:
    manifest = ExperimentManifest.load("experiments/e0_baseline_protocol/config.yaml")
    assert manifest.experiment_id == "E0"
    manifest.verify_model_configs()
    saved = manifest.save(tmp_path / "config.yaml")
    assert ExperimentManifest.load(saved).to_dict() == manifest.to_dict()

    leaked = manifest.to_dict()
    leaked["inference_policy"]["ground_truth_at_inference"] = True
    with pytest.raises(ValueError, match="ground truth"):
        ExperimentManifest(leaked)

    changed_match = manifest.to_dict()
    changed_match["matching"]["threshold_m"] = 2.1
    with pytest.raises(ValueError, match="frozen"):
        ExperimentManifest(changed_match)


def _cache_provenance(**changes) -> PredictionCacheProvenance:
    values = {
        "dataset": "nuscenes",
        "dataset_version": "v1.0-mini",
        "split": "mini_train",
        "sample_token": "abc123",
        "detector": "openpcdet_centerpoint",
        "detector_config": "configs/detectors/centerpoint/nuscenes_mini.yaml",
        "detector_config_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "sweeps": 10,
        "candidate_threshold": 0.1,
        "score_filtering_policy": "fixed upstream and project threshold",
    }
    values.update(changes)
    return PredictionCacheProvenance(**values)


def test_prediction_cache_requires_exact_provenance_and_rejects_stale_entries(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path)
    provenance = _cache_provenance()
    prediction = PredictionBatch("abc123", [Box3D([1, 2, 3], [4, 2, 1], 0.2, "car", score=0.8)])
    cache.save(prediction, provenance)
    loaded = cache.load(provenance)
    assert loaded is not None and loaded.to_dict() == prediction.to_dict()

    assert cache.load(_cache_provenance(candidate_threshold=0.2)) is None
    assert cache.load(_cache_provenance(checkpoint_sha256="c" * 64)) is None
    assert cache.load(_cache_provenance(split="mini_val")) is None


@pytest.mark.parametrize(
    ("detector", "split"),
    [
        ("openpcdet_centerpoint", "mini_train"),
        ("openpcdet_centerpoint", "mini_val"),
        ("openpcdet_voxelnext", "mini_train"),
        ("openpcdet_voxelnext", "mini_val"),
    ],
)
def test_prediction_cache_paths_cover_both_models_and_official_mini_splits(tmp_path: Path, detector: str, split: str) -> None:
    provenance = _cache_provenance(detector=detector, split=split)
    path = PredictionCache(tmp_path).path_for(provenance)
    assert detector in path.parts and split in path.parts


def test_predicted_range_and_density_use_prediction_and_sensor_points_only() -> None:
    box = Box3D([3, 4, 0], [2, 2, 2], 0, "car", score=0.7)
    prediction = PredictionBatch("sample", [box])
    points = np.asarray([
        [3, 4, 0, 1, 0.0],
        [3, 4, 0, 1, 0.5],
        [10, 10, 0, 1, 0.0],
    ], dtype=np.float32)
    assert predicted_box_range_m(box) == 5.0
    current = extract_prediction_features(prediction, points, point_source="current_keyframe")[0]
    accumulated = extract_prediction_features(prediction, points, point_source="multi_sweep")[0]
    assert (current.range_m, current.point_count, current.point_source) == (5.0, 1, "current_keyframe")
    assert (accumulated.point_count, accumulated.point_source) == (2, "multi_sweep")

    with pytest.raises(TypeError):
        extract_prediction_features(prediction, points, ground_truth=[box])
