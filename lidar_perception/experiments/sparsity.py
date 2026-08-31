"""Prediction-only sparsity calibration for Phase 6 E2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.experiments.calibration import (
    CalibrationParameters,
    ScoreOnlyCalibrator,
    _finite_vector,
    _fit_newton,
    _logit,
    _sigmoid,
    calibration_metrics,
    fit_logistic_calibrator,
)
from lidar_perception.experiments.features import predicted_box_keyframe_point_counts
from lidar_perception.geometry.boxes3d import Box3D


SPARSITY_SCHEMA_VERSION = "lidar_perception.e2_sparsity_calibration.v1"


def _point_counts(values: Iterable[int | float], expected_length: int | None = None) -> np.ndarray:
    counts = _finite_vector(values, "predicted_box_keyframe_point_counts")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("predicted-box point counts must be non-negative integers")
    if expected_length is not None and len(counts) != expected_length:
        raise ValueError("predicted-box point counts must align with raw scores")
    return counts


@dataclass(frozen=True)
class SparsityCalibrationParameters:
    """Frozen coefficients for score plus predicted-box point count."""

    intercept: float
    score_weight: float
    sparsity_weight: float
    ridge: float = 1.0
    point_count_source: str = "current_keyframe"
    point_count_transform: str = "log1p"
    family: str = "score_sparsity"
    schema_version: str = SPARSITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("intercept", "score_weight", "sparsity_weight", "ridge"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.point_count_source != "current_keyframe":
            raise ValueError("E2 primary policy requires current_keyframe point counts")
        if self.point_count_transform != "log1p":
            raise ValueError("unsupported point-count transform")
        if self.family != "score_sparsity" or self.schema_version != SPARSITY_SCHEMA_VERSION:
            raise ValueError("unsupported E2 sparsity calibration identity")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SparsityCalibrationParameters":
        if not isinstance(value, dict):
            raise TypeError("sparsity calibration parameters must be a mapping")
        return cls(**value)


class SparsityAwareCalibrator:
    """Apply the frozen E2 logistic score and sparsity rule."""

    def __init__(self, parameters: SparsityCalibrationParameters):
        if not isinstance(parameters, SparsityCalibrationParameters):
            raise TypeError("parameters must be SparsityCalibrationParameters")
        self.parameters = parameters

    def predict(
        self,
        raw_scores: Sequence[float],
        predicted_box_keyframe_point_counts: Sequence[int | float],
    ) -> np.ndarray:
        scores = _finite_vector(raw_scores, "raw_scores")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("raw_scores must be in [0, 1]")
        counts = _point_counts(predicted_box_keyframe_point_counts, len(scores))
        logits = (
            self.parameters.intercept
            + self.parameters.score_weight * _logit(scores)
            + self.parameters.sparsity_weight * np.log1p(counts)
        )
        return _sigmoid(logits)

    def to_dict(self) -> dict[str, Any]:
        return self.parameters.to_dict()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SparsityAwareCalibrator":
        return cls(SparsityCalibrationParameters.from_dict(value))


@dataclass(frozen=True)
class SparsitySearchRecord:
    configuration_id: str
    family: str
    parameters: dict[str, Any]
    validation_metrics: dict[str, float | None]
    training_metrics: dict[str, float | None]
    validation_strategy: str
    fold_count: int
    valid: bool
    selected: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_sparsity_calibrator(
    raw_scores: Sequence[float],
    predicted_box_keyframe_point_counts: Sequence[int | float],
    labels: Sequence[int | bool],
    *,
    ridge: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-9,
) -> SparsityAwareCalibrator:
    """Fit E2 on mini-train TP/FP labels and prediction-only features."""

    scores = _finite_vector(raw_scores, "raw_scores")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("raw_scores must be in [0, 1]")
    counts = _point_counts(predicted_box_keyframe_point_counts, len(scores))
    targets = _finite_vector(labels, "labels")
    if len(scores) == 0 or len(targets) != len(scores):
        raise ValueError("raw_scores and labels must be non-empty and aligned")
    if np.any(targets != np.round(targets)):
        raise ValueError("labels must be binary")
    design = np.column_stack((np.ones(len(scores)), _logit(scores), np.log1p(counts)))
    coefficients = _fit_newton(design, targets, float(ridge), int(max_iter), float(tolerance))
    return SparsityAwareCalibrator(
        SparsityCalibrationParameters(
            intercept=coefficients[0],
            score_weight=coefficients[1],
            sparsity_weight=coefficients[2],
            ridge=float(ridge),
        )
    )


def calibrate_prediction_with_point_counts(
    prediction: PredictionBatch,
    predicted_box_keyframe_point_counts: Sequence[int | float],
    calibrator: SparsityAwareCalibrator,
) -> PredictionBatch:
    """Apply E2 to aligned inference features without accepting ground truth."""

    if not isinstance(prediction, PredictionBatch):
        raise TypeError("prediction must be a PredictionBatch")
    if not isinstance(calibrator, SparsityAwareCalibrator):
        raise TypeError("calibrator must be a SparsityAwareCalibrator")
    raw_scores = [box.score for box in prediction.boxes]
    if any(score is None for score in raw_scores):
        raise ValueError("all predicted boxes require a score for calibration")
    counts = _point_counts(predicted_box_keyframe_point_counts, len(prediction.boxes))
    calibrated_scores = calibrator.predict(raw_scores, counts)
    boxes = [
        Box3D(
            center=box.center.copy(),
            size=box.size.copy(),
            yaw=box.yaw,
            label=box.label,
            score=float(score),
            velocity=None if box.velocity is None else box.velocity.copy(),
            track_id=box.track_id,
        )
        for box, score in zip(prediction.boxes, calibrated_scores)
    ]
    metadata = dict(prediction.metadata)
    metadata.update({
        "score_policy": "e2_score_sparsity",
        "sparsity_calibration_parameters": calibrator.to_dict(),
        "point_count_source": "current_keyframe",
    })
    return PredictionBatch(prediction.frame_id, boxes, prediction.runtime_ms, metadata)


def apply_sparsity_policy(
    prediction: PredictionBatch,
    sensor_points: np.ndarray,
    calibrator: SparsityAwareCalibrator,
) -> PredictionBatch:
    """Compute predicted-box current-frame counts and apply E2 at inference."""

    counts = predicted_box_keyframe_point_counts(prediction, sensor_points)
    return calibrate_prediction_with_point_counts(prediction, counts, calibrator)


def _fit_family(
    family: str,
    scores: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    ridge: float,
) -> ScoreOnlyCalibrator | SparsityAwareCalibrator:
    if family == "score_only":
        return fit_logistic_calibrator(scores, None, labels, family="score_only", ridge=ridge)
    return fit_sparsity_calibrator(scores, counts, labels, ridge=ridge)


def _predict_family(
    model: ScoreOnlyCalibrator | SparsityAwareCalibrator,
    scores: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    if isinstance(model, ScoreOnlyCalibrator):
        return model.predict(scores)
    return model.predict(scores, counts)


def search_sparsity_calibrators(
    raw_scores: Sequence[float],
    predicted_box_keyframe_point_counts: Sequence[int | float],
    labels: Sequence[int | bool],
    scene_ids: Sequence[str],
    *,
    ridge_values: Sequence[float] = (0.1, 1.0, 10.0),
) -> tuple[SparsityAwareCalibrator, list[SparsitySearchRecord], ScoreOnlyCalibrator]:
    """Select E2 by deterministic leave-one-scene-out mini-train log loss."""

    scores = _finite_vector(raw_scores, "raw_scores")
    counts = _point_counts(predicted_box_keyframe_point_counts, len(scores))
    targets = _finite_vector(labels, "labels")
    scenes = np.asarray(list(scene_ids), dtype=object)
    if len(targets) != len(scores) or len(scenes) != len(scores) or len(scores) == 0:
        raise ValueError("scores, counts, labels, and scene_ids must be non-empty and aligned")
    unique_scenes = sorted({str(scene) for scene in scenes})
    if len(unique_scenes) < 2 or any(not scene for scene in unique_scenes):
        raise ValueError("leave-one-scene-out validation requires at least two named scenes")
    ridges = tuple(float(value) for value in ridge_values)
    if not ridges or any(not np.isfinite(value) or value < 0 for value in ridges):
        raise ValueError("ridge_values must contain finite non-negative values")

    records: list[SparsitySearchRecord] = []
    fitted: dict[str, ScoreOnlyCalibrator | SparsityAwareCalibrator] = {}
    for family in ("score_only", "score_sparsity"):
        for index, ridge in enumerate(ridges, start=1):
            configuration_id = f"{family}_{index:02d}"
            try:
                oof = np.empty(len(scores), dtype=np.float64)
                for held_out in unique_scenes:
                    validation_mask = scenes == held_out
                    fit_mask = ~validation_mask
                    fold_model = _fit_family(family, scores[fit_mask], counts[fit_mask], targets[fit_mask], ridge)
                    oof[validation_mask] = _predict_family(fold_model, scores[validation_mask], counts[validation_mask])
                model = _fit_family(family, scores, counts, targets, ridge)
                training_scores = _predict_family(model, scores, counts)
                records.append(SparsitySearchRecord(
                    configuration_id=configuration_id,
                    family=family,
                    parameters=model.to_dict(),
                    validation_metrics=calibration_metrics(oof, targets),
                    training_metrics=calibration_metrics(training_scores, targets),
                    validation_strategy="leave-one-scene-out",
                    fold_count=len(unique_scenes),
                    valid=True,
                    selected=False,
                    reason="tested",
                ))
                fitted[configuration_id] = model
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                records.append(SparsitySearchRecord(
                    configuration_id, family, {"ridge": ridge}, {}, {},
                    "leave-one-scene-out", len(unique_scenes), False, False, str(exc),
                ))

    valid_sparsity = [record for record in records if record.family == "score_sparsity" and record.valid]
    valid_controls = [record for record in records if record.family == "score_only" and record.valid]
    if not valid_sparsity or not valid_controls:
        raise ValueError("no valid E2 sparsity/control calibration configuration")
    winner = min(valid_sparsity, key=lambda record: (float(record.validation_metrics["log_loss"]), record.configuration_id))
    control_winner = min(valid_controls, key=lambda record: (float(record.validation_metrics["log_loss"]), record.configuration_id))
    updated: list[SparsitySearchRecord] = []
    for record in records:
        if record.configuration_id == winner.configuration_id:
            reason = "selected by minimum leave-one-scene-out mini_train log loss"
            selected = True
        elif record.configuration_id == control_winner.configuration_id:
            reason = "selected score-only internal control"
            selected = False
        elif record.valid:
            reason = "valid but not selected"
            selected = False
        else:
            reason = record.reason
            selected = False
        updated.append(SparsitySearchRecord(
            record.configuration_id, record.family, record.parameters,
            record.validation_metrics, record.training_metrics,
            record.validation_strategy, record.fold_count, record.valid, selected, reason,
        ))
    selected_model = fitted[winner.configuration_id]
    control_model = fitted[control_winner.configuration_id]
    if not isinstance(selected_model, SparsityAwareCalibrator) or not isinstance(control_model, ScoreOnlyCalibrator):
        raise RuntimeError("E2 search returned an unexpected model family")
    return selected_model, updated, control_model


__all__ = [
    "SPARSITY_SCHEMA_VERSION",
    "SparsityAwareCalibrator",
    "SparsityCalibrationParameters",
    "SparsitySearchRecord",
    "apply_sparsity_policy",
    "calibrate_prediction_with_point_counts",
    "fit_sparsity_calibrator",
    "search_sparsity_calibrators",
]
