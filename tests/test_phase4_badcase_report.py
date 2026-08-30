import json

from lidar_perception.analysis.badcase_mining import mine_bad_cases
from lidar_perception.analysis.report import write_json
from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.matching import SampleEvaluation, match_prediction_to_ground_truth
from lidar_perception.geometry.boxes3d import Box3D


def test_badcase_categories_and_deterministic_selection(tmp_path) -> None:
    gt = [
        Box3D([0, 0, 0], [2, 2, 2], 0, "car"),
        Box3D([1, 0, 0], [2, 2, 2], 0, "car"),
        Box3D([60, 0, 0], [2, 2, 2], 0, "pedestrian"),
    ]
    prediction = PredictionBatch(
        "sample",
        [
            Box3D([0, 0, 0], [2, 2, 2], 0, "car", score=0.2),
            Box3D([2.5, 0, 0], [2, 2, 2], 0, "car", score=0.8),
            Box3D([5, 0, 0], [2, 2, 2], 0, "truck", score=0.9),
        ],
    )
    match = match_prediction_to_ground_truth(prediction, gt, distance_threshold_m=2, gt_point_counts=[5, 30, 2])
    sample = SampleEvaluation(prediction, gt, [5, 30, 2], match)
    cases, counts = mine_bad_cases([sample], max_per_category=2)
    categories = {case.category for case in cases}
    assert {"false_positive", "low_confidence_true_positive", "far_range_miss", "low_density_miss"} <= categories
    assert counts["false_negative"] == 1
    assert all(case.gt_box or case.prediction_box for case in cases)

    path = write_json({"upper": float("inf"), "cases": [case.to_dict() for case in cases]}, tmp_path / "report.json")
    assert json.loads(path.read_text())["upper"] is None
