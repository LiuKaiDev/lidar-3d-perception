"""GT LiDAR point-density stratified evaluation for project analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .matching import SampleEvaluation, match_samples
from .metrics import DEFAULT_DENSITY_BINS, MetricBin, coerce_bins, find_bin, mean_or_none, safe_ratio


def _empty_row(metric_bin: MetricBin) -> dict[str, Any]:
    return {
        "bin": metric_bin.name,
        "lower_points": int(metric_bin.lower),
        "upper_points_exclusive": None if metric_bin.upper == float("inf") else int(metric_bin.upper),
        "gt_count": 0,
        "matched_count": 0,
        "false_negatives": 0,
        "recall": None,
        "average_matched_confidence": None,
        "matched_localization_error_m": None,
    }


class DensityAwareEvaluator:
    """Evaluate matches by current-keyframe points inside each GT 3D box."""

    def __init__(self, *, bins: Iterable[MetricBin | dict[str, Any]] | None = None, distance_threshold_m: float = 2.0) -> None:
        self.bins = coerce_bins(bins, DEFAULT_DENSITY_BINS)
        self.distance_threshold_m = float(distance_threshold_m)

    def evaluate(
        self,
        samples: Iterable[SampleEvaluation],
        ground_truths: Iterable | None = None,
        gt_point_counts: Iterable | None = None,
    ) -> dict[str, Any]:
        if ground_truths is not None and gt_point_counts is None:
            raise ValueError("density evaluation requires gt_point_counts")
        sample_list = (
            match_samples(
                samples,
                ground_truths,
                distance_threshold_m=self.distance_threshold_m,
                gt_point_counts=gt_point_counts,
            )
            if ground_truths is not None
            else list(samples)
        )
        classes = sorted({box.label for sample in sample_list for box in [*sample.ground_truth, *sample.prediction.boxes]})
        overall = {metric_bin.name: _empty_row(metric_bin) for metric_bin in self.bins}
        per_class = {label: {metric_bin.name: _empty_row(metric_bin) for metric_bin in self.bins} for label in classes}
        confidences: dict[str, list[float]] = defaultdict(list)
        errors: dict[str, list[float]] = defaultdict(list)
        class_confidences: dict[tuple[str, str], list[float]] = defaultdict(list)
        class_errors: dict[tuple[str, str], list[float]] = defaultdict(list)
        for sample in sample_list:
            if len(sample.gt_point_counts) != len(sample.ground_truth):
                raise ValueError("sample.gt_point_counts must align with ground_truth")
            for box, point_count in zip(sample.ground_truth, sample.gt_point_counts):
                metric_bin = find_bin(float(point_count), self.bins)
                if metric_bin is not None:
                    overall[metric_bin.name]["gt_count"] += 1
                    per_class[box.label][metric_bin.name]["gt_count"] += 1
            for match in sample.match.matches:
                if match.gt_point_count is None:
                    raise ValueError("density evaluation requires point counts in matching records")
                metric_bin = find_bin(float(match.gt_point_count), self.bins)
                if metric_bin is not None:
                    overall[metric_bin.name]["matched_count"] += 1
                    per_class[match.label][metric_bin.name]["matched_count"] += 1
                    errors[metric_bin.name].append(match.localization_error_m)
                    class_errors[(match.label, metric_bin.name)].append(match.localization_error_m)
                    if match.prediction_score is not None:
                        confidences[metric_bin.name].append(match.prediction_score)
                        class_confidences[(match.label, metric_bin.name)].append(match.prediction_score)
            for false_negative in sample.match.false_negatives:
                if false_negative.gt_point_count is None:
                    raise ValueError("density evaluation requires point counts in matching records")
                metric_bin = find_bin(float(false_negative.gt_point_count), self.bins)
                if metric_bin is not None:
                    overall[metric_bin.name]["false_negatives"] += 1
                    per_class[false_negative.label][metric_bin.name]["false_negatives"] += 1

        def finalize(row: dict[str, Any], name: str, class_name: str | None = None) -> dict[str, Any]:
            row["recall"] = safe_ratio(row["matched_count"], row["gt_count"])
            row["average_matched_confidence"] = mean_or_none(
                confidences[name] if class_name is None else class_confidences[(class_name, name)]
            )
            row["matched_localization_error_m"] = mean_or_none(
                errors[name] if class_name is None else class_errors[(class_name, name)]
            )
            return row

        overall_rows = [finalize(overall[metric_bin.name], metric_bin.name) for metric_bin in self.bins]
        per_class_rows = {
            label: [finalize(per_class[label][metric_bin.name], metric_bin.name, label) for metric_bin in self.bins]
            for label in classes
        }
        return {
            "protocol": {
                "metric": "point-density-aware custom analysis",
                "point_count_definition": "number of current keyframe LiDAR points inside each GT oriented 3D box",
                "multi_sweep_policy": "current keyframe only (time_lag == 0), for physical observability and reproducibility",
                "bins": [metric_bin.to_dict() for metric_bin in self.bins],
                "matching_threshold_m": self.distance_threshold_m,
            },
            "sample_count": len(sample_list),
            "overall": overall_rows,
            "per_class": per_class_rows,
        }


__all__ = ["DensityAwareEvaluator"]
