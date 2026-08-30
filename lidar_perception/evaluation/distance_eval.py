"""Distance-stratified evaluation for project analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .matching import SampleEvaluation, match_samples
from .metrics import DEFAULT_DISTANCE_BINS, MetricBin, coerce_bins, find_bin, mean_or_none, safe_ratio


def _empty_row(metric_bin: MetricBin) -> dict[str, Any]:
    return {
        "bin": metric_bin.name,
        "lower_m": metric_bin.lower,
        "upper_m": metric_bin.upper,
        "gt_count": 0,
        "prediction_count": 0,
        "matched_count": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "recall": None,
        "precision": None,
        "matched_localization_error_m": None,
    }


def _finalize(row: dict[str, Any], localization_errors: list[float]) -> dict[str, Any]:
    row["recall"] = safe_ratio(row["matched_count"], row["gt_count"])
    row["precision"] = safe_ratio(row["matched_count"], row["prediction_count"])
    row["matched_localization_error_m"] = mean_or_none(localization_errors)
    return row


class DistanceAwareEvaluator:
    """Compute GT/prediction counts and custom matching metrics by range."""

    def __init__(
        self,
        *,
        bins: Iterable[MetricBin | dict[str, Any]] | None = None,
        distance_threshold_m: float = 2.0,
    ) -> None:
        self.bins = coerce_bins(bins, DEFAULT_DISTANCE_BINS)
        self.distance_threshold_m = float(distance_threshold_m)

    def evaluate(
        self,
        samples: Iterable[SampleEvaluation],
        ground_truths: Iterable | None = None,
        gt_point_counts: Iterable | None = None,
    ) -> dict[str, Any]:
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
        classes = sorted(
            {
                box.label
                for sample in sample_list
                for box in [*sample.ground_truth, *sample.prediction.boxes]
            }
        )
        overall = {metric_bin.name: _empty_row(metric_bin) for metric_bin in self.bins}
        per_class = {
            label: {metric_bin.name: _empty_row(metric_bin) for metric_bin in self.bins}
            for label in classes
        }
        errors: dict[str, list[float]] = defaultdict(list)
        class_errors: dict[tuple[str, str], list[float]] = defaultdict(list)
        for sample in sample_list:
            for box in sample.ground_truth:
                metric_bin = find_bin(float((box.center[:2] ** 2).sum() ** 0.5), self.bins)
                if metric_bin is not None:
                    overall[metric_bin.name]["gt_count"] += 1
                    per_class[box.label][metric_bin.name]["gt_count"] += 1
            for match in sample.match.matches:
                metric_bin = find_bin(match.gt_distance_m, self.bins)
                if metric_bin is not None:
                    # A matched prediction is assigned to its target's range
                    # bin so precision uses the same target-conditioned scope
                    # as recall. Unmatched predictions use their own range.
                    overall[metric_bin.name]["prediction_count"] += 1
                    overall[metric_bin.name]["matched_count"] += 1
                    per_class[match.label][metric_bin.name]["prediction_count"] += 1
                    per_class[match.label][metric_bin.name]["matched_count"] += 1
                    errors[metric_bin.name].append(match.localization_error_m)
                    class_errors[(match.label, metric_bin.name)].append(match.localization_error_m)
            for false_negative in sample.match.false_negatives:
                metric_bin = find_bin(false_negative.gt_distance_m, self.bins)
                if metric_bin is not None:
                    overall[metric_bin.name]["false_negatives"] += 1
                    per_class[false_negative.label][metric_bin.name]["false_negatives"] += 1
            for false_positive in sample.match.false_positives:
                metric_bin = find_bin(false_positive.prediction_distance_m, self.bins)
                if metric_bin is not None:
                    overall[metric_bin.name]["prediction_count"] += 1
                    overall[metric_bin.name]["false_positives"] += 1
                    per_class[false_positive.label][metric_bin.name]["prediction_count"] += 1
                    per_class[false_positive.label][metric_bin.name]["false_positives"] += 1
        overall_rows = [_finalize(overall[metric_bin.name], errors[metric_bin.name]) for metric_bin in self.bins]
        per_class_rows = {
            label: [_finalize(per_class[label][metric_bin.name], class_errors[(label, metric_bin.name)]) for metric_bin in self.bins]
            for label in classes
        }
        return {
            "protocol": {
                "metric": "distance-aware custom analysis",
                "distance_definition": "sqrt(x^2 + y^2) of the GT/prediction center in reference LiDAR frame",
                "prediction_bin_definition": "matched predictions use the matched GT range; false positives use prediction range",
                "bins": [metric_bin.to_dict() for metric_bin in self.bins],
                "matching_threshold_m": self.distance_threshold_m,
            },
            "sample_count": len(sample_list),
            "overall": overall_rows,
            "per_class": per_class_rows,
        }


__all__ = ["DistanceAwareEvaluator"]
