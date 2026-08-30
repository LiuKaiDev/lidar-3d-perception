"""Deterministic project-owned matching for stratified detector analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.geometry.boxes3d import Box3D


def center_distance(first: Box3D, second: Box3D) -> float:
    """Return horizontal center distance in the shared LiDAR frame."""

    return float(np.linalg.norm(first.center[:2] - second.center[:2]))


def box_to_dict(box: Box3D | None) -> dict[str, Any] | None:
    """Serialize a project box for reports and bad-case records."""

    if box is None:
        return None
    return {
        "center": box.center.tolist(),
        "size": box.size.tolist(),
        "yaw": box.yaw,
        "label": box.label,
        "score": box.score,
        "velocity": None if box.velocity is None else box.velocity.tolist(),
        "track_id": box.track_id,
    }


@dataclass(frozen=True)
class MatchRecord:
    sample_id: str
    gt_index: int
    prediction_index: int
    label: str
    prediction_score: float | None
    gt_distance_m: float
    prediction_distance_m: float
    localization_error_m: float
    gt_point_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "gt_index": self.gt_index,
            "prediction_index": self.prediction_index,
            "class": self.label,
            "prediction_score": self.prediction_score,
            "gt_distance_m": self.gt_distance_m,
            "prediction_distance_m": self.prediction_distance_m,
            "localization_error_m": self.localization_error_m,
            "gt_point_count": self.gt_point_count,
            "state": "TP",
        }


@dataclass(frozen=True)
class FalsePositiveRecord:
    sample_id: str
    prediction_index: int
    label: str
    prediction_score: float | None
    prediction_distance_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prediction_index": self.prediction_index,
            "class": self.label,
            "prediction_score": self.prediction_score,
            "prediction_distance_m": self.prediction_distance_m,
            "state": "FP",
        }


@dataclass(frozen=True)
class FalseNegativeRecord:
    sample_id: str
    gt_index: int
    label: str
    gt_distance_m: float
    gt_point_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "gt_index": self.gt_index,
            "class": self.label,
            "gt_distance_m": self.gt_distance_m,
            "gt_point_count": self.gt_point_count,
            "state": "FN",
        }


@dataclass
class MatchResult:
    """One-to-one result for one sample and its project boxes."""

    sample_id: str
    protocol: dict[str, Any]
    matches: list[MatchRecord]
    false_positives: list[FalsePositiveRecord]
    false_negatives: list[FalseNegativeRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "protocol": self.protocol,
            "matches": [item.to_dict() for item in self.matches],
            "false_positives": [item.to_dict() for item in self.false_positives],
            "false_negatives": [item.to_dict() for item in self.false_negatives],
        }


@dataclass
class SampleEvaluation:
    """A prediction/GT sample with its reusable matching result."""

    prediction: PredictionBatch
    ground_truth: list[Box3D]
    gt_point_counts: list[int]
    match: MatchResult

    @property
    def sample_id(self) -> str:
        return self.match.sample_id


def match_prediction_to_ground_truth(
    prediction: PredictionBatch,
    ground_truth: Sequence[Box3D],
    *,
    distance_threshold_m: float = 2.0,
    gt_point_counts: Sequence[int] | None = None,
) -> MatchResult:
    """Match same-class boxes by deterministic greedy center distance.

    Candidate pairs are sorted by ``(distance, gt_index, prediction_index)``
    and accepted once per GT and prediction.  The threshold is inclusive and
    therefore a pair at exactly ``distance_threshold_m`` is a match.
    """

    threshold = float(distance_threshold_m)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("distance_threshold_m must be finite and non-negative")
    gt_boxes = list(ground_truth)
    if gt_point_counts is None:
        point_counts = [None] * len(gt_boxes)
    else:
        if len(gt_point_counts) != len(gt_boxes):
            raise ValueError("gt_point_counts must have one entry per ground-truth box")
        point_counts = [int(value) for value in gt_point_counts]
        if any(value < 0 for value in point_counts):
            raise ValueError("gt_point_counts must be non-negative")

    sample_id = str(prediction.metadata.get("sample_token", prediction.frame_id))
    candidates: list[tuple[float, int, int]] = []
    gt_distances = [float(np.linalg.norm(box.center[:2])) for box in gt_boxes]
    pred_distances = [float(np.linalg.norm(box.center[:2])) for box in prediction.boxes]
    for gt_index, gt_box in enumerate(gt_boxes):
        for prediction_index, prediction_box in enumerate(prediction.boxes):
            if gt_box.label != prediction_box.label:
                continue
            distance = center_distance(gt_box, prediction_box)
            if distance <= threshold:
                candidates.append((distance, gt_index, prediction_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[MatchRecord] = []
    for distance, gt_index, prediction_index in candidates:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append(
            MatchRecord(
                sample_id=sample_id,
                gt_index=gt_index,
                prediction_index=prediction_index,
                label=gt_boxes[gt_index].label,
                prediction_score=prediction.boxes[prediction_index].score,
                gt_distance_m=gt_distances[gt_index],
                prediction_distance_m=pred_distances[prediction_index],
                localization_error_m=distance,
                gt_point_count=point_counts[gt_index],
            )
        )

    false_positives = [
        FalsePositiveRecord(
            sample_id=sample_id,
            prediction_index=index,
            label=box.label,
            prediction_score=box.score,
            prediction_distance_m=pred_distances[index],
        )
        for index, box in enumerate(prediction.boxes)
        if index not in used_predictions
    ]
    false_negatives = [
        FalseNegativeRecord(
            sample_id=sample_id,
            gt_index=index,
            label=box.label,
            gt_distance_m=gt_distances[index],
            gt_point_count=point_counts[index],
        )
        for index, box in enumerate(gt_boxes)
        if index not in used_gt
    ]
    false_positives.sort(key=lambda item: item.prediction_index)
    false_negatives.sort(key=lambda item: item.gt_index)
    return MatchResult(
        sample_id=sample_id,
        protocol={
            "strategy": "center_distance",
            "threshold_m": threshold,
            "comparison": "distance <= threshold",
            "distance_definition": "sqrt((x_gt-x_pred)^2 + (y_gt-y_pred)^2) in LiDAR frame",
            "class_aware": True,
            "one_to_one": True,
            "assignment": "greedy sorted by distance, gt_index, prediction_index",
        },
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def match_samples(
    predictions: Iterable[PredictionBatch],
    ground_truths: Iterable[Sequence[Box3D]],
    *,
    distance_threshold_m: float = 2.0,
    gt_point_counts: Iterable[Sequence[int]] | None = None,
) -> list[SampleEvaluation]:
    """Build reusable matching records for aligned prediction/GT samples."""

    prediction_list = list(predictions)
    ground_truth_list = [list(boxes) for boxes in ground_truths]
    if len(prediction_list) != len(ground_truth_list):
        raise ValueError("predictions and ground_truths must have equal lengths")
    count_list = None if gt_point_counts is None else [list(counts) for counts in gt_point_counts]
    if count_list is not None and len(count_list) != len(prediction_list):
        raise ValueError("gt_point_counts must align with predictions")
    result: list[SampleEvaluation] = []
    for index, (prediction, boxes) in enumerate(zip(prediction_list, ground_truth_list)):
        counts = [0] * len(boxes) if count_list is None else count_list[index]
        match = match_prediction_to_ground_truth(
            prediction,
            boxes,
            distance_threshold_m=distance_threshold_m,
            gt_point_counts=counts,
        )
        result.append(SampleEvaluation(prediction, boxes, counts, match))
    return result


__all__ = [
    "FalseNegativeRecord",
    "FalsePositiveRecord",
    "MatchRecord",
    "MatchResult",
    "SampleEvaluation",
    "box_to_dict",
    "center_distance",
    "match_prediction_to_ground_truth",
    "match_samples",
]
