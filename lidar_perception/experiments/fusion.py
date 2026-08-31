"""Prediction-only CenterPoint/VoxelNeXt late fusion for Phase 6 E3.

The implementation deliberately contains no ground-truth access.  GT matching
belongs to :mod:`lidar_perception.evaluation.matching` and is used only by the
analysis helpers in this module.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.matching import match_prediction_to_ground_truth
from lidar_perception.geometry.boxes3d import Box3D

FUSION_SCHEMA_VERSION = "lidar_perception.e3_late_fusion.v1"


@dataclass(frozen=True)
class FusionConfig:
    """Small, serializable E3 policy.

    Scores are source-adjusted by a global weight, clipped to ``[0, 1]``, and
    matched pairs use probabilistic OR.  Geometry is copied from the stronger
    adjusted source, avoiding yaw interpolation and preserving velocity.
    """

    association_threshold_m: float = 1.0
    centerpoint_weight: float = 1.0
    voxelnext_weight: float = 1.0
    score_rule: str = "probabilistic_or"
    geometry_policy: str = "winner_take_all"
    candidate_floor: float = 0.1
    max_boxes: int = 500
    schema_version: str = FUSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        threshold = float(self.association_threshold_m)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("association_threshold_m must be finite and non-negative")
        for name in ("centerpoint_weight", "voxelnext_weight"):
            weight = float(getattr(self, name))
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        floor = float(self.candidate_floor)
        if not 0 <= floor <= 1:
            raise ValueError("candidate_floor must be in [0, 1]")
        if isinstance(self.max_boxes, bool) or not isinstance(self.max_boxes, int) or self.max_boxes < 1:
            raise ValueError("max_boxes must be a positive integer")
        if self.score_rule != "probabilistic_or":
            raise ValueError("unsupported E3 score rule")
        if self.geometry_policy != "winner_take_all":
            raise ValueError("unsupported E3 geometry policy")
        if self.schema_version != FUSION_SCHEMA_VERSION:
            raise ValueError("unsupported E3 fusion schema version")
        object.__setattr__(self, "association_threshold_m", threshold)
        object.__setattr__(self, "centerpoint_weight", float(self.centerpoint_weight))
        object.__setattr__(self, "voxelnext_weight", float(self.voxelnext_weight))
        object.__setattr__(self, "candidate_floor", floor)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FusionConfig":
        if not isinstance(value, dict):
            raise TypeError("fusion config must be a mapping")
        return cls(**value)


@dataclass(frozen=True)
class Association:
    centerpoint_index: int
    voxelnext_index: int
    distance_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusionDiagnostics:
    associations: tuple[Association, ...]
    input_centerpoint_count: int
    input_voxelnext_count: int
    candidate_count_before_limit: int
    candidate_count_after_limit: int
    truncated: bool
    association_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    sorting_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["associations"] = [item.to_dict() for item in self.associations]
        value["total_fusion_time_ms"] = self.association_time_ms + self.fusion_time_ms + self.sorting_time_ms
        return value


def _score(box: Box3D, weight: float) -> float:
    if box.score is None:
        raise ValueError("all fusion candidates require a score")
    value = float(box.score)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("fusion candidate scores must be finite and in [0, 1]")
    return float(np.clip(value * weight, 0.0, 1.0))


def _sort_key(box: Box3D, score: float, source: str, index: int) -> tuple[Any, ...]:
    # Explicit keys make ties stable across Python versions and input order.
    return (-score, box.label, tuple(float(x) for x in box.center), source, index)


def associate_predictions(
    centerpoint: PredictionBatch,
    voxelnext: PredictionBatch,
    *,
    threshold_m: float = 1.0,
    centerpoint_weight: float = 1.0,
    voxelnext_weight: float = 1.0,
) -> list[Association]:
    """Class-aware one-to-one prediction association by center distance."""

    if not isinstance(centerpoint, PredictionBatch) or not isinstance(voxelnext, PredictionBatch):
        raise TypeError("centerpoint and voxelnext must be PredictionBatch instances")
    threshold = float(threshold_m)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold_m must be finite and non-negative")
    candidates: list[tuple[float, float, float, int, int]] = []
    for ci, cp in enumerate(centerpoint.boxes):
        for vi, vn in enumerate(voxelnext.boxes):
            if cp.label != vn.label:
                continue
            distance = float(np.linalg.norm(cp.center[:2] - vn.center[:2]))
            if distance <= threshold:
                # score keys only break equal-distance ties; indices finish the order.
                candidates.append((distance, -_score(cp, centerpoint_weight), -_score(vn, voxelnext_weight), ci, vi))
    candidates.sort()
    used_cp: set[int] = set()
    used_vn: set[int] = set()
    result: list[Association] = []
    for distance, _cp_score, _vn_score, ci, vi in candidates:
        if ci in used_cp or vi in used_vn:
            continue
        used_cp.add(ci)
        used_vn.add(vi)
        result.append(Association(ci, vi, distance))
    return result


def _copy_with_score(box: Box3D, score: float, *, source: str) -> Box3D:
    return Box3D(
        center=box.center.copy(), size=box.size.copy(), yaw=box.yaw,
        label=box.label, score=float(np.clip(score, 0.0, 1.0)),
        velocity=None if box.velocity is None else box.velocity.copy(), track_id=box.track_id,
    )


def fuse_predictions(
    centerpoint: PredictionBatch,
    voxelnext: PredictionBatch,
    config: FusionConfig | None = None,
    *,
    return_diagnostics: bool = False,
) -> PredictionBatch | tuple[PredictionBatch, FusionDiagnostics]:
    """Fuse two frozen detector outputs using prediction-only information."""

    if not isinstance(centerpoint, PredictionBatch) or not isinstance(voxelnext, PredictionBatch):
        raise TypeError("centerpoint and voxelnext must be PredictionBatch instances")
    if centerpoint.frame_id != voxelnext.frame_id:
        raise ValueError("centerpoint and voxelnext frame_id must match")
    cfg = FusionConfig() if config is None else config
    if not isinstance(cfg, FusionConfig):
        raise TypeError("config must be FusionConfig")
    t0 = perf_counter()
    associations = associate_predictions(
        centerpoint, voxelnext,
        threshold_m=cfg.association_threshold_m,
        centerpoint_weight=cfg.centerpoint_weight,
        voxelnext_weight=cfg.voxelnext_weight,
    )
    association_ms = (perf_counter() - t0) * 1000.0
    t1 = perf_counter()
    matched_cp = {pair.centerpoint_index for pair in associations}
    matched_vn = {pair.voxelnext_index for pair in associations}
    output: list[tuple[Box3D, float, str, int]] = []
    for pair in associations:
        cp = centerpoint.boxes[pair.centerpoint_index]
        vn = voxelnext.boxes[pair.voxelnext_index]
        cp_score = _score(cp, cfg.centerpoint_weight)
        vn_score = _score(vn, cfg.voxelnext_weight)
        fused_score = 1.0 - (1.0 - cp_score) * (1.0 - vn_score)
        # Winner-take-all geometry preserves a valid box and its velocity/yaw.
        if cp_score >= vn_score:
            geometry = cp
            source = "centerpoint"
        else:
            geometry = vn
            source = "voxelnext"
        output.append((_copy_with_score(geometry, fused_score, source=source), fused_score, source, pair.centerpoint_index))
    for index, box in enumerate(centerpoint.boxes):
        if index not in matched_cp:
            score = _score(box, cfg.centerpoint_weight)
            if score >= cfg.candidate_floor:
                output.append((_copy_with_score(box, score, source="centerpoint"), score, "centerpoint", index))
    for index, box in enumerate(voxelnext.boxes):
        if index not in matched_vn:
            score = _score(box, cfg.voxelnext_weight)
            if score >= cfg.candidate_floor:
                output.append((_copy_with_score(box, score, source="voxelnext"), score, "voxelnext", index))
    fusion_ms = (perf_counter() - t1) * 1000.0
    before = len(output)
    t2 = perf_counter()
    output.sort(key=lambda item: _sort_key(item[0], item[1], item[2], item[3]))
    output = output[: cfg.max_boxes]
    sorting_ms = (perf_counter() - t2) * 1000.0
    metadata = dict(centerpoint.metadata)
    metadata.update({
        "fusion_policy": "e3_centerpoint_voxelnext_late_fusion",
        "fusion_config": cfg.to_dict(),
        "fusion_sources": ["centerpoint", "voxelnext"],
        "candidate_count_before_limit": before,
        "candidate_count_after_limit": len(output),
    })
    prediction = PredictionBatch(centerpoint.frame_id, [item[0] for item in output], centerpoint.runtime_ms, metadata)
    diagnostics = FusionDiagnostics(
        associations=tuple(associations), input_centerpoint_count=len(centerpoint.boxes),
        input_voxelnext_count=len(voxelnext.boxes), candidate_count_before_limit=before,
        candidate_count_after_limit=len(output), truncated=before > cfg.max_boxes,
        association_time_ms=association_ms, fusion_time_ms=fusion_ms, sorting_time_ms=sorting_ms,
    )
    return (prediction, diagnostics) if return_diagnostics else prediction


def naive_union(
    centerpoint: PredictionBatch,
    voxelnext: PredictionBatch,
    *,
    duplicate_threshold_m: float = 0.01,
    candidate_floor: float = 0.1,
    max_boxes: int = 500,
) -> PredictionBatch:
    """Unoptimized union control with only near-identical duplicate removal."""
    if centerpoint.frame_id != voxelnext.frame_id:
        raise ValueError("prediction frame_id values must match")
    if duplicate_threshold_m < 0 or not np.isfinite(duplicate_threshold_m):
        raise ValueError("duplicate_threshold_m must be finite and non-negative")
    if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or max_boxes < 1:
        raise ValueError("max_boxes must be a positive integer")
    items: list[tuple[Box3D, str, int]] = []
    for source, batch in (("centerpoint", centerpoint), ("voxelnext", voxelnext)):
        for index, box in enumerate(batch.boxes):
            if box.score is None or not 0 <= float(box.score) <= 1:
                raise ValueError("union candidates require scores in [0, 1]")
            if float(box.score) >= candidate_floor:
                items.append((box, source, index))
    kept: list[tuple[Box3D, str, int]] = []
    for box, source, index in items:
        duplicate = any(source != other_source and box.label == other.label and np.linalg.norm(box.center[:2] - other.center[:2]) <= duplicate_threshold_m for other, other_source, _i in kept)
        if not duplicate:
            kept.append((box, source, index))
    kept.sort(key=lambda item: _sort_key(item[0], float(item[0].score), item[1], item[2]))
    metadata = dict(centerpoint.metadata)
    metadata.update({"fusion_policy": "naive_union", "candidate_count_before_limit": len(kept), "candidate_count_after_limit": min(len(kept), max_boxes)})
    return PredictionBatch(centerpoint.frame_id, [item[0] for item in kept[:max_boxes]], centerpoint.runtime_ms, metadata)


late_fuse_predictions = fuse_predictions
fuse_prediction_batches = fuse_predictions
late_fuse = fuse_predictions


def analyze_complementarity(
    centerpoint_predictions: Iterable[PredictionBatch],
    voxelnext_predictions: Iterable[PredictionBatch],
    ground_truths: Iterable[Sequence[Box3D]],
    *,
    gt_point_counts: Iterable[Sequence[int]] | None = None,
    distance_threshold_m: float = 2.0,
) -> dict[str, Any]:
    """Evaluation-only both/CP-only/VN-only/neither coverage summary."""
    cp = list(centerpoint_predictions)
    vn = list(voxelnext_predictions)
    gt = [list(x) for x in ground_truths]
    counts = None if gt_point_counts is None else [list(x) for x in gt_point_counts]
    if not (len(cp) == len(vn) == len(gt)):
        raise ValueError("prediction and ground-truth sequences must align")
    if counts is not None and len(counts) != len(gt):
        raise ValueError("gt_point_counts must align with ground_truths")
    records: list[dict[str, Any]] = []
    for i, (cp_batch, vn_batch, gt_boxes) in enumerate(zip(cp, vn, gt)):
        cp_match = match_prediction_to_ground_truth(cp_batch, gt_boxes, distance_threshold_m=distance_threshold_m, gt_point_counts=None if counts is None else counts[i])
        vn_match = match_prediction_to_ground_truth(vn_batch, gt_boxes, distance_threshold_m=distance_threshold_m, gt_point_counts=None if counts is None else counts[i])
        cp_set = {m.gt_index for m in cp_match.matches}
        vn_set = {m.gt_index for m in vn_match.matches}
        for j, box in enumerate(gt_boxes):
            records.append({"sample_index": i, "gt_index": j, "label": box.label, "distance_m": float(np.linalg.norm(box.center[:2])), "point_count": None if counts is None else int(counts[i][j]), "category": "detected_by_both" if j in cp_set and j in vn_set else "centerpoint_only" if j in cp_set else "voxelnext_only" if j in vn_set else "neither"})
    def subset(predicate):
        values = [r for r in records if predicate(r)]
        total = len(values)
        categories = {name: sum(r["category"] == name for r in values) for name in ("detected_by_both", "centerpoint_only", "voxelnext_only", "neither")}
        return {"total": total, "counts": categories, "coverage": None if total == 0 else (total - categories["neither"]) / total}
    sections = {
        "overall": subset(lambda r: True),
        "50m_plus": subset(lambda r: r["distance_m"] >= 50),
        "40_50m": subset(lambda r: 40 <= r["distance_m"] < 50),
        "0_5_points": subset(lambda r: r["point_count"] is not None and 0 <= r["point_count"] <= 5),
        "6_10_points": subset(lambda r: r["point_count"] is not None and 6 <= r["point_count"] <= 10),
    }
    for label in sorted({r["label"] for r in records}):
        label_rows = lambda r, label=label: r["label"] == label
        sections.setdefault("classes", {})[label] = subset(label_rows)
    return {"protocol": {"matching": "class-aware one-to-one center distance", "threshold_m": float(distance_threshold_m), "comparison": "distance <= threshold", "gt_used_for": "evaluation-only"}, "sections": sections}


def save_frozen_config(config: FusionConfig, path: str | Path) -> Path:
    if not isinstance(config, FusionConfig):
        raise TypeError("config must be FusionConfig")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_frozen_config(path: str | Path) -> FusionConfig:
    return FusionConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fusion_recall_metrics(
    predictions: Iterable[PredictionBatch],
    ground_truths: Iterable[Sequence[Box3D]],
    *,
    gt_point_counts: Iterable[Sequence[int]] | None = None,
    distance_threshold_m: float = 2.0,
) -> dict[str, float | int | None]:
    """Compute the preregistered custom recall/precision metrics."""
    pred = list(predictions)
    gt = [list(x) for x in ground_truths]
    counts = None if gt_point_counts is None else [list(x) for x in gt_point_counts]
    if len(pred) != len(gt) or (counts is not None and len(counts) != len(gt)):
        raise ValueError("predictions, ground_truths, and point counts must align")
    matched = total = fp = 0
    far_matched = far_total = sparse_matched = sparse_total = mid_matched = mid_total = 0
    for i, (batch, truth) in enumerate(zip(pred, gt)):
        result = match_prediction_to_ground_truth(batch, truth, distance_threshold_m=distance_threshold_m, gt_point_counts=None if counts is None else counts[i])
        matched += len(result.matches); total += len(truth); fp += len(result.false_positives)
        matched_indices = {item.gt_index for item in result.matches}
        for index, box in enumerate(truth):
            distance = float(np.linalg.norm(box.center[:2]))
            is_match = index in matched_indices
            if distance >= 50: far_total += 1; far_matched += int(is_match)
            if 40 <= distance < 50: mid_total += 1; mid_matched += int(is_match)
            if counts is not None and 0 <= counts[i][index] <= 5: sparse_total += 1; sparse_matched += int(is_match)
    return {"recall_50m_plus": None if far_total == 0 else far_matched / far_total, "recall_0_5_points": None if sparse_total == 0 else sparse_matched / sparse_total, "recall_40_50m": None if mid_total == 0 else mid_matched / mid_total, "overall_custom_recall": None if total == 0 else matched / total, "precision": None if matched + fp == 0 else matched / (matched + fp), "fp_count": fp, "matched_count": matched, "gt_count": total}


def search_fusion_configs(
    centerpoint_predictions: Sequence[PredictionBatch],
    voxelnext_predictions: Sequence[PredictionBatch],
    ground_truths: Sequence[Sequence[Box3D]],
    *,
    gt_point_counts: Sequence[Sequence[int]] | None = None,
    association_thresholds_m: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    centerpoint_weights: Sequence[float] = (0.8, 1.0, 1.2),
    voxelnext_weight: float = 1.0,
    candidate_floor: float = 0.1,
    max_boxes: int = 500,
) -> tuple[FusionConfig, list[dict[str, Any]]]:
    """Run the preregistered low-dimensional mini-train grid."""
    if not (len(centerpoint_predictions) == len(voxelnext_predictions) == len(ground_truths)):
        raise ValueError("prediction and ground-truth sequences must align")
    configs = [FusionConfig(float(threshold), float(weight), float(voxelnext_weight), candidate_floor=float(candidate_floor), max_boxes=max_boxes) for threshold in association_thresholds_m for weight in centerpoint_weights]
    if not configs:
        raise ValueError("search grid must not be empty")
    records: list[dict[str, Any]] = []
    for index, config in enumerate(configs, start=1):
        fused = [fuse_predictions(cp, vn, config) for cp, vn in zip(centerpoint_predictions, voxelnext_predictions)]
        metrics = fusion_recall_metrics(fused, ground_truths, gt_point_counts=gt_point_counts)
        records.append({"configuration_id": f"e3_{index:03d}", "parameters": config.to_dict(), "metrics": metrics, "selected": False, "selection_reason": "not selected"})
    def metric_value(record: dict[str, Any], key: str) -> float:
        value = record["metrics"].get(key)
        return -math.inf if value is None else float(value)
    winner_index = max(range(len(records)), key=lambda i: (metric_value(records[i], "recall_50m_plus"), metric_value(records[i], "recall_0_5_points"), -float(records[i]["metrics"].get("fp_count", 0)), -records[i]["parameters"]["association_threshold_m"], -abs(records[i]["parameters"]["centerpoint_weight"] - 1.0)))
    records[winner_index]["selected"] = True
    records[winner_index]["selection_reason"] = "lexicographic primary recall rule"
    return FusionConfig.from_dict(records[winner_index]["parameters"]), records


__all__ = [
    "Association", "FUSION_SCHEMA_VERSION", "FusionConfig", "FusionDiagnostics",
    "analyze_complementarity", "associate_predictions", "fuse_predictions",
    "late_fuse_predictions", "fuse_prediction_batches", "late_fuse", "load_frozen_config", "naive_union", "save_frozen_config",
    "fusion_recall_metrics", "search_fusion_configs",
]
