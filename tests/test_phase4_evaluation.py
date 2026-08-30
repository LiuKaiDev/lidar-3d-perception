import numpy as np

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.density_eval import DensityAwareEvaluator
from lidar_perception.evaluation.distance_eval import DistanceAwareEvaluator
from lidar_perception.evaluation.matching import SampleEvaluation, match_prediction_to_ground_truth
from lidar_perception.geometry.boxes3d import Box3D, count_points_in_box


def make_sample(centers, counts):
    gt = [Box3D(center, [2, 2, 2], 0, "car") for center in centers]
    prediction = PredictionBatch("sample", [Box3D(center, [2, 2, 2], 0, "car", score=0.8) for center in centers])
    match = match_prediction_to_ground_truth(prediction, gt, distance_threshold_m=0.1, gt_point_counts=counts)
    return SampleEvaluation(prediction, gt, list(counts), match)


def test_distance_boundaries_and_zero_bins_are_explicit() -> None:
    sample = make_sample([[0, 0, 0], [10, 0, 0], [50, 0, 0]], [1, 6, 51])
    report = DistanceAwareEvaluator().evaluate([sample])
    rows = {row["bin"]: row for row in report["overall"]}
    assert rows["0-10m"]["gt_count"] == 1
    assert rows["10-20m"]["gt_count"] == 1
    assert rows["40-50m"]["gt_count"] == 0
    assert rows["50m+"]["gt_count"] == 1
    assert rows["0-10m"]["recall"] == 1.0


def test_density_boundaries_and_empty_denominators() -> None:
    sample = make_sample([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], [5, 6, 10, 11, 51])
    report = DensityAwareEvaluator().evaluate([sample])
    rows = {row["bin"]: row for row in report["overall"]}
    assert [rows[name]["gt_count"] for name in ("0-5", "6-10", "11-20", "21-50", "51+")] == [1, 2, 1, 0, 1]
    assert rows["21-50"]["recall"] is None
    assert rows["0-5"]["average_matched_confidence"] == 0.8


def test_density_requires_aligned_counts() -> None:
    sample = make_sample([[0, 0, 0]], [2])
    sample.gt_point_counts = []
    try:
        DensityAwareEvaluator().evaluate([sample])
    except ValueError:
        return
    raise AssertionError("expected misaligned point counts to fail")


def test_evaluators_accept_prediction_and_ground_truth_inputs() -> None:
    gt = [[Box3D([0, 0, 0], [2, 2, 2], 0, "car")]]
    predictions = [PredictionBatch("sample", [Box3D([0, 0, 0], [2, 2, 2], 0, "car", score=0.9)])]
    distance_report = DistanceAwareEvaluator().evaluate(predictions, gt)
    density_report = DensityAwareEvaluator().evaluate(predictions, gt, [[5]])
    assert distance_report["overall"][0]["matched_count"] == 1
    assert density_report["overall"][0]["matched_count"] == 1


def test_density_rejects_missing_point_counts_for_raw_inputs() -> None:
    gt = [[Box3D([0, 0, 0], [2, 2, 2], 0, "car")]]
    predictions = [PredictionBatch("sample", [])]
    try:
        DensityAwareEvaluator().evaluate(predictions, gt)
    except ValueError:
        return
    raise AssertionError("expected density point counts to be required")


def test_phase4_point_count_uses_oriented_gt_box() -> None:
    box = Box3D([0, 0, 0], [4, 2, 2], np.pi / 2, "car")
    points = np.asarray([[0, 1, 0], [0, -1, 0], [2.1, 0, 0], [0, 0, 1.1]], dtype=np.float64)
    assert count_points_in_box(points, box) == 2
