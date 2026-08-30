import numpy as np
import pytest

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.matching import match_prediction_to_ground_truth
from lidar_perception.geometry.boxes3d import Box3D


def box(center, label="car", score=None):
    return Box3D(center, [2, 2, 2], 0, label, score=score)


def test_perfect_match_and_exact_threshold_are_true_positives() -> None:
    result = match_prediction_to_ground_truth(
        PredictionBatch("sample", [box([2, 0, 0], score=0.8)]),
        [box([0, 0, 0])],
        distance_threshold_m=2.0,
        gt_point_counts=[5],
    )
    assert len(result.matches) == 1
    assert result.matches[0].localization_error_m == pytest.approx(2.0)
    assert result.matches[0].gt_point_count == 5


def test_matching_is_class_aware_one_to_one_and_deterministic() -> None:
    prediction = PredictionBatch("sample", [box([0.5, 0, 0], score=0.2), box([0.1, 0, 0], score=0.9)])
    result = match_prediction_to_ground_truth(prediction, [box([0, 0, 0])], distance_threshold_m=2)
    assert [(item.gt_index, item.prediction_index) for item in result.matches] == [(0, 1)]
    assert [item.prediction_index for item in result.false_positives] == [0]

    wrong_class = match_prediction_to_ground_truth(PredictionBatch("s", [box([0, 0, 0], "pedestrian")]), [box([0, 0, 0], "car")])
    assert len(wrong_class.false_negatives) == 1 and len(wrong_class.false_positives) == 1


@pytest.mark.parametrize(
    ("ground_truth", "predictions", "expected_fp", "expected_fn"),
    [([], [], 0, 0), ([box([0, 0, 0])], [], 0, 1), ([], [box([0, 0, 0], score=0.5)], 1, 0)],
)
def test_empty_and_unmatched_cases(ground_truth, predictions, expected_fp, expected_fn) -> None:
    result = match_prediction_to_ground_truth(PredictionBatch("s", predictions), ground_truth)
    assert len(result.false_positives) == expected_fp
    assert len(result.false_negatives) == expected_fn


def test_invalid_point_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        match_prediction_to_ground_truth(PredictionBatch("s", []), [box([0, 0, 0])], gt_point_counts=[])
    with pytest.raises(ValueError):
        match_prediction_to_ground_truth(PredictionBatch("s", []), [box([0, 0, 0])], gt_point_counts=[-1])
