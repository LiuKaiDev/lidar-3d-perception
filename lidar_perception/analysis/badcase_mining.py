"""Deterministic representative bad-case mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from lidar_perception.evaluation.matching import SampleEvaluation, box_to_dict


@dataclass(frozen=True)
class BadCase:
    category: str
    sample_id: str
    label: str
    distance_m: float
    gt_point_count: int | None
    confidence: float | None
    localization_error_m: float | None
    gt_index: int | None
    prediction_index: int | None
    gt_box: dict[str, Any] | None
    prediction_box: dict[str, Any] | None
    reason: str
    rank_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "sample_id": self.sample_id,
            "class": self.label,
            "distance_m": self.distance_m,
            "gt_point_count": self.gt_point_count,
            "confidence": self.confidence,
            "localization_error_m": self.localization_error_m,
            "gt_index": self.gt_index,
            "prediction_index": self.prediction_index,
            "gt_box": self.gt_box,
            "prediction_box": self.prediction_box,
            "reason": self.reason,
            "rank_value": self.rank_value,
        }


def mine_bad_cases(
    samples: Iterable[SampleEvaluation],
    *,
    max_per_category: int = 5,
    far_distance_m: float = 50.0,
    low_density_max_points: int = 5,
    low_confidence_threshold: float = 0.3,
    high_localization_error_m: float = 1.0,
) -> tuple[list[BadCase], dict[str, int]]:
    """Mine and rank FN/FP/TP categories with explicit deterministic rules."""

    if max_per_category < 1:
        raise ValueError("max_per_category must be positive")
    categories: dict[str, list[tuple[tuple[Any, ...], BadCase]]] = {name: [] for name in (
        "false_negative",
        "false_positive",
        "low_confidence_true_positive",
        "high_localization_error_true_positive",
        "far_range_miss",
        "low_density_miss",
    )}
    for sample in samples:
        prediction_boxes = sample.prediction.boxes
        gt_boxes = sample.ground_truth
        for item in sample.match.false_negatives:
            gt_box = gt_boxes[item.gt_index]
            base = dict(
                sample_id=item.sample_id,
                label=item.label,
                distance_m=item.gt_distance_m,
                gt_point_count=item.gt_point_count,
                confidence=None,
                localization_error_m=None,
                gt_index=item.gt_index,
                prediction_index=None,
                gt_box=box_to_dict(gt_box),
                prediction_box=None,
            )
            categories["false_negative"].append(
                ((-item.gt_distance_m, item.sample_id, item.gt_index), BadCase("false_negative", reason="unmatched ground truth", rank_value=item.gt_distance_m, **base))
            )
            if item.gt_distance_m >= far_distance_m:
                categories["far_range_miss"].append(
                    ((-item.gt_distance_m, item.sample_id, item.gt_index), BadCase("far_range_miss", reason=f"unmatched GT at distance >= {far_distance_m:g} m", rank_value=item.gt_distance_m, **base))
                )
            if item.gt_point_count is not None and item.gt_point_count <= low_density_max_points:
                categories["low_density_miss"].append(
                    ((item.gt_point_count, -item.gt_distance_m, item.sample_id, item.gt_index), BadCase("low_density_miss", reason=f"unmatched GT with <= {low_density_max_points} points", rank_value=float(item.gt_point_count), **base))
                )
        for item in sample.match.false_positives:
            pred_box = prediction_boxes[item.prediction_index]
            base = dict(
                sample_id=item.sample_id,
                label=item.label,
                distance_m=item.prediction_distance_m,
                gt_point_count=None,
                confidence=item.prediction_score,
                localization_error_m=None,
                gt_index=None,
                prediction_index=item.prediction_index,
                gt_box=None,
                prediction_box=box_to_dict(pred_box),
            )
            score = -1.0 if item.prediction_score is None else item.prediction_score
            categories["false_positive"].append(
                ((-score, item.sample_id, item.prediction_index), BadCase("false_positive", reason="unmatched prediction", rank_value=score, **base))
            )
        for item in sample.match.matches:
            gt_box = gt_boxes[item.gt_index]
            pred_box = prediction_boxes[item.prediction_index]
            base = dict(
                sample_id=item.sample_id,
                label=item.label,
                distance_m=item.gt_distance_m,
                gt_point_count=item.gt_point_count,
                confidence=item.prediction_score,
                localization_error_m=item.localization_error_m,
                gt_index=item.gt_index,
                prediction_index=item.prediction_index,
                gt_box=box_to_dict(gt_box),
                prediction_box=box_to_dict(pred_box),
            )
            score = 1.0 if item.prediction_score is None else item.prediction_score
            if item.prediction_score is not None and item.prediction_score < low_confidence_threshold:
                categories["low_confidence_true_positive"].append(
                    ((item.prediction_score, item.sample_id, item.gt_index, item.prediction_index), BadCase("low_confidence_true_positive", reason=f"matched prediction confidence < {low_confidence_threshold:g}", rank_value=item.prediction_score, **base))
                )
            if item.localization_error_m >= high_localization_error_m:
                categories["high_localization_error_true_positive"].append(
                    ((-item.localization_error_m, item.sample_id, item.gt_index, item.prediction_index), BadCase("high_localization_error_true_positive", reason=f"center error >= {high_localization_error_m:g} m", rank_value=item.localization_error_m, **base))
                )

    counts = {category: len(items) for category, items in categories.items()}
    result: list[BadCase] = []
    for category, items in categories.items():
        items.sort(key=lambda item: item[0])
        result.extend(case for _, case in items[:max_per_category])
    return result, counts


__all__ = ["BadCase", "mine_bad_cases"]
