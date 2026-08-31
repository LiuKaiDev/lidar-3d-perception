"""Scene-level bootstrap intervals for project-owned recall metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


PHASE6_PRIMARY_METRICS = ("recall_50m_plus", "recall_0_5_points")
PHASE6_BOOTSTRAP_METRICS = (*PHASE6_PRIMARY_METRICS, "recall_40_50m", "overall_custom_recall")


@dataclass(frozen=True)
class SceneMetricRecord:
    """One additive recall numerator/denominator contribution from a scene."""

    scene_id: str
    metric: str
    matched_count: int
    gt_count: int

    def __post_init__(self) -> None:
        if not self.scene_id or not self.metric:
            raise ValueError("scene_id and metric must be non-empty")
        if isinstance(self.matched_count, bool) or isinstance(self.gt_count, bool):
            raise TypeError("counts must be integers")
        if not isinstance(self.matched_count, int) or not isinstance(self.gt_count, int):
            raise TypeError("counts must be integers")
        if self.matched_count < 0 or self.gt_count < 0 or self.matched_count > self.gt_count:
            raise ValueError("counts require 0 <= matched_count <= gt_count")


@dataclass(frozen=True)
class SceneMetricCounts:
    """All additive recall counts for one independently resampled scene."""

    scene_id: str
    counts: dict[str, tuple[int, int]]

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        for metric, (matched, total) in self.counts.items():
            SceneMetricRecord(self.scene_id, metric, matched, total)


@dataclass(frozen=True)
class BootstrapInterval:
    metric: str
    point_estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    repetitions: int
    valid_repetitions: int
    seed: int
    resampling_unit: str = "scene"

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class PairedBootstrapInterval:
    metric: str
    baseline: float | None
    experiment: float | None
    delta: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    repetitions: int
    valid_repetitions: int
    seed: int
    claim: str
    resampling_unit: str = "paired scene"

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


def group_scene_counts(records: Iterable[SceneMetricRecord]) -> list[SceneMetricCounts]:
    """Aggregate sample contributions into deterministic scene-level counts."""

    grouped: dict[str, dict[str, list[int]]] = {}
    for record in records:
        if not isinstance(record, SceneMetricRecord):
            raise TypeError("records must contain SceneMetricRecord instances")
        metric_counts = grouped.setdefault(record.scene_id, {}).setdefault(record.metric, [0, 0])
        metric_counts[0] += record.matched_count
        metric_counts[1] += record.gt_count
    return [
        SceneMetricCounts(scene_id, {metric: tuple(values) for metric, values in sorted(metrics.items())})
        for scene_id, metrics in sorted(grouped.items())
    ]


def _validate_bootstrap(repetitions: int, seed: int, confidence_level: float) -> tuple[int, int, float]:
    if isinstance(repetitions, bool) or int(repetitions) != repetitions or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    return int(repetitions), int(seed), confidence


def _recall(scenes: list[SceneMetricCounts], indices: np.ndarray, metric: str) -> float | None:
    matched = 0
    total = 0
    for index in indices:
        scene_matched, scene_total = scenes[int(index)].counts.get(metric, (0, 0))
        matched += scene_matched
        total += scene_total
    return None if total == 0 else matched / total


def _interval(values: list[float], confidence_level: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    tail = (1.0 - confidence_level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def _sorted_unique_scenes(scenes: Iterable[SceneMetricCounts]) -> list[SceneMetricCounts]:
    scene_list = sorted(list(scenes), key=lambda item: item.scene_id)
    scene_ids = [scene.scene_id for scene in scene_list]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("bootstrap input must contain one aggregate per scene ID")
    return scene_list


def scene_level_bootstrap(
    scenes: Iterable[SceneMetricCounts],
    *,
    metrics: Iterable[str] = PHASE6_BOOTSTRAP_METRICS,
    repetitions: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, BootstrapInterval]:
    """Resample complete scenes with replacement and aggregate recall counts."""

    repetitions, seed, confidence_level = _validate_bootstrap(repetitions, seed, confidence_level)
    scene_list = _sorted_unique_scenes(scenes)
    metric_list = list(dict.fromkeys(metrics))
    if any(not metric for metric in metric_list):
        raise ValueError("metric names must be non-empty")
    rng = np.random.default_rng(seed)
    sampled_indices = [
        rng.integers(0, len(scene_list), size=len(scene_list))
        for _ in range(repetitions)
    ] if scene_list else []
    full_indices = np.arange(len(scene_list), dtype=np.int64)
    result: dict[str, BootstrapInterval] = {}
    for metric in metric_list:
        values = [value for indices in sampled_indices if (value := _recall(scene_list, indices, metric)) is not None]
        lower, upper = _interval(values, confidence_level)
        result[metric] = BootstrapInterval(
            metric=metric,
            point_estimate=_recall(scene_list, full_indices, metric),
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            repetitions=repetitions,
            valid_repetitions=len(values),
            seed=seed,
        )
    return result


def _claim(delta: float | None, lower: float | None) -> str:
    if delta is None:
        return "insufficient data"
    if delta < 0:
        return "regression on mini"
    if delta == 0:
        return "no change on mini"
    if lower is not None and lower > 0:
        return "bootstrap-supported improvement on mini"
    return "directional improvement; uncertainty overlaps zero"


def paired_scene_bootstrap(
    baseline: Iterable[SceneMetricCounts],
    experiment: Iterable[SceneMetricCounts],
    *,
    metrics: Iterable[str] = PHASE6_BOOTSTRAP_METRICS,
    repetitions: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, PairedBootstrapInterval]:
    """Use identical resampled scene indices for baseline/experiment deltas."""

    repetitions, seed, confidence_level = _validate_bootstrap(repetitions, seed, confidence_level)
    baseline_list = _sorted_unique_scenes(baseline)
    experiment_list = _sorted_unique_scenes(experiment)
    baseline_map = {item.scene_id: item for item in baseline_list}
    experiment_map = {item.scene_id: item for item in experiment_list}
    if set(baseline_map) != set(experiment_map):
        raise ValueError("paired bootstrap requires identical scene IDs")
    scene_ids = sorted(baseline_map)
    baseline_scenes = [baseline_map[scene_id] for scene_id in scene_ids]
    experiment_scenes = [experiment_map[scene_id] for scene_id in scene_ids]
    metric_list = list(dict.fromkeys(metrics))
    for scene_id in scene_ids:
        for metric in metric_list:
            baseline_total = baseline_map[scene_id].counts.get(metric, (0, 0))[1]
            experiment_total = experiment_map[scene_id].counts.get(metric, (0, 0))[1]
            if baseline_total != experiment_total:
                raise ValueError(f"paired metric denominators differ for {scene_id}/{metric}")
    rng = np.random.default_rng(seed)
    sampled_indices = [
        rng.integers(0, len(scene_ids), size=len(scene_ids))
        for _ in range(repetitions)
    ] if scene_ids else []
    full_indices = np.arange(len(scene_ids), dtype=np.int64)
    result: dict[str, PairedBootstrapInterval] = {}
    for metric in metric_list:
        baseline_point = _recall(baseline_scenes, full_indices, metric)
        experiment_point = _recall(experiment_scenes, full_indices, metric)
        point_delta = None if baseline_point is None or experiment_point is None else experiment_point - baseline_point
        deltas: list[float] = []
        for indices in sampled_indices:
            baseline_value = _recall(baseline_scenes, indices, metric)
            experiment_value = _recall(experiment_scenes, indices, metric)
            if baseline_value is not None and experiment_value is not None:
                deltas.append(experiment_value - baseline_value)
        lower, upper = _interval(deltas, confidence_level)
        result[metric] = PairedBootstrapInterval(
            metric=metric,
            baseline=baseline_point,
            experiment=experiment_point,
            delta=point_delta,
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            repetitions=repetitions,
            valid_repetitions=len(deltas),
            seed=seed,
            claim=_claim(point_delta, lower),
        )
    return result


__all__ = [
    "BootstrapInterval",
    "PHASE6_BOOTSTRAP_METRICS",
    "PHASE6_PRIMARY_METRICS",
    "PairedBootstrapInterval",
    "SceneMetricCounts",
    "SceneMetricRecord",
    "group_scene_counts",
    "paired_scene_bootstrap",
    "scene_level_bootstrap",
]
