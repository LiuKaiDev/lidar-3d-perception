"""Small, prediction-only score calibration policies for Phase 6 E1.

The calibrators in this module are deliberately boring.  They fit a logistic
map with a fixed, interpretable feature definition and apply it to a
``PredictionBatch`` without accepting ground truth.  Ground-truth labels are
accepted only by the explicit ``fit_*`` functions, which keeps the
fit/inference boundary visible in the API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.experiments.features import predicted_box_range_m
from lidar_perception.geometry.boxes3d import Box3D


CALIBRATION_SCHEMA_VERSION = "lidar_perception.e1_calibration.v1"
CALIBRATION_FAMILIES = {"score_only", "score_range"}
DEFAULT_RANGE_CENTER_M = 25.0
DEFAULT_RANGE_SCALE_M = 25.0


def _finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    return result


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    # This form avoids overflow for very large negative or positive logits.
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class CalibrationParameters:
    """Frozen coefficients and feature constants for one E1 calibrator."""

    family: str
    intercept: float
    score_weight: float
    range_weight: float = 0.0
    range_center_m: float = DEFAULT_RANGE_CENTER_M
    range_scale_m: float = DEFAULT_RANGE_SCALE_M
    ridge: float = 1.0
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.family not in CALIBRATION_FAMILIES:
            raise ValueError(f"family must be one of {sorted(CALIBRATION_FAMILIES)}")
        for name in ("intercept", "score_weight", "range_weight", "range_center_m", "range_scale_m", "ridge"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.range_scale_m <= 0:
            raise ValueError("range_scale_m must be positive")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative")
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported calibration schema version")
        if self.family == "score_only" and abs(self.range_weight) > 1e-12:
            raise ValueError("score_only calibration cannot contain a range coefficient")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibrationParameters":
        if not isinstance(value, dict):
            raise TypeError("calibration parameters must be a mapping")
        return cls(**value)


class LogisticCalibrator:
    """Apply a frozen logistic score map to predicted boxes."""

    def __init__(self, parameters: CalibrationParameters):
        if not isinstance(parameters, CalibrationParameters):
            raise TypeError("parameters must be CalibrationParameters")
        self.parameters = parameters

    @property
    def family(self) -> str:
        return self.parameters.family

    def predict(self, raw_scores: Sequence[float], predicted_ranges_m: Sequence[float] | None = None) -> np.ndarray:
        scores = _finite_vector(raw_scores, "raw_scores")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("raw_scores must be in [0, 1]")
        if self.family == "score_range":
            if predicted_ranges_m is None:
                raise ValueError("score_range calibration requires predicted_ranges_m")
            ranges = _finite_vector(predicted_ranges_m, "predicted_ranges_m")
            if len(ranges) != len(scores) or np.any(ranges < 0):
                raise ValueError("predicted_ranges_m must align with non-negative raw_scores")
        else:
            ranges = np.zeros_like(scores)
        normalized_range = (ranges - self.parameters.range_center_m) / self.parameters.range_scale_m
        logits = (
            self.parameters.intercept
            + self.parameters.score_weight * _logit(scores)
            + self.parameters.range_weight * normalized_range
        )
        return _sigmoid(logits)

    def to_dict(self) -> dict[str, Any]:
        return self.parameters.to_dict()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LogisticCalibrator":
        return cls(CalibrationParameters.from_dict(value))


class ScoreOnlyCalibrator(LogisticCalibrator):
    """Comparable logistic control using only the raw score feature."""

    def __init__(self, parameters: CalibrationParameters):
        if parameters.family != "score_only":
            raise ValueError("ScoreOnlyCalibrator requires score_only parameters")
        super().__init__(parameters)


class RangeAwareCalibrator(LogisticCalibrator):
    """Global score plus predicted-range logistic calibration used by E1."""

    def __init__(self, parameters: CalibrationParameters):
        if parameters.family != "score_range":
            raise ValueError("RangeAwareCalibrator requires score_range parameters")
        super().__init__(parameters)


@dataclass(frozen=True)
class CalibrationSearchRecord:
    """One attempted mini-train configuration retained in the search log."""

    configuration_id: str
    family: str
    parameters: dict[str, Any]
    metrics: dict[str, float | None]
    valid: bool
    selected: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _design_matrix(scores: np.ndarray, ranges: np.ndarray | None, family: str, center: float, scale: float) -> np.ndarray:
    columns = [np.ones(len(scores), dtype=np.float64), _logit(scores)]
    if family == "score_range":
        if ranges is None:
            raise ValueError("score_range fitting requires predicted ranges")
        columns.append((ranges - center) / scale)
    return np.column_stack(columns)


def _fit_newton(design: np.ndarray, labels: np.ndarray, ridge: float, max_iter: int, tolerance: float) -> np.ndarray:
    if len(labels) == 0:
        raise ValueError("at least one fit example is required")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must be binary")
    if np.all(labels == labels[0]):
        raise ValueError("logistic calibration requires both positive and negative labels")
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0  # Do not shrink the intercept.
    for _ in range(max_iter):
        probabilities = _sigmoid(design @ coefficients)
        gradient = design.T @ (probabilities - labels) + penalty @ coefficients
        weights = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * weights[:, None]) + penalty + np.eye(design.shape[1]) * 1e-8
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular calibration fit") from exc
        coefficients -= step
        if np.linalg.norm(step, ord=np.inf) <= tolerance:
            break
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("calibration fit produced non-finite parameters")
    return coefficients


def fit_logistic_calibrator(
    raw_scores: Sequence[float],
    predicted_ranges_m: Sequence[float] | None,
    labels: Sequence[int | bool],
    *,
    family: str = "score_range",
    ridge: float = 1.0,
    range_center_m: float = DEFAULT_RANGE_CENTER_M,
    range_scale_m: float = DEFAULT_RANGE_SCALE_M,
    max_iter: int = 100,
    tolerance: float = 1e-9,
) -> LogisticCalibrator:
    """Fit on TP/FP labels; this is the only API that accepts fit labels."""

    if family not in CALIBRATION_FAMILIES:
        raise ValueError(f"family must be one of {sorted(CALIBRATION_FAMILIES)}")
    scores = _finite_vector(raw_scores, "raw_scores")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("raw_scores must be in [0, 1]")
    ranges = None if predicted_ranges_m is None else _finite_vector(predicted_ranges_m, "predicted_ranges_m")
    if ranges is not None and len(ranges) != len(scores):
        raise ValueError("predicted_ranges_m must align with raw_scores")
    if ranges is not None and np.any(ranges < 0):
        raise ValueError("predicted_ranges_m must be non-negative")
    targets = _finite_vector(labels, "labels")
    if len(scores) != len(targets) or len(scores) == 0:
        raise ValueError("raw_scores and labels must be non-empty and aligned")
    if np.any(targets != np.round(targets)):
        raise ValueError("labels must be binary")
    if family == "score_only":
        ranges = None
    design = _design_matrix(scores, ranges, family, float(range_center_m), float(range_scale_m))
    coefficients = _fit_newton(design, targets, float(ridge), int(max_iter), float(tolerance))
    params = CalibrationParameters(
        family=family,
        intercept=coefficients[0],
        score_weight=coefficients[1],
        range_weight=0.0 if family == "score_only" else coefficients[2],
        range_center_m=float(range_center_m),
        range_scale_m=float(range_scale_m),
        ridge=float(ridge),
    )
    return ScoreOnlyCalibrator(params) if family == "score_only" else RangeAwareCalibrator(params)


def search_calibrators(
    raw_scores: Sequence[float],
    predicted_ranges_m: Sequence[float],
    labels: Sequence[int | bool],
    *,
    ridge_values: Sequence[float] = (0.1, 1.0, 10.0),
    range_center_m: float = DEFAULT_RANGE_CENTER_M,
    range_scale_m: float = DEFAULT_RANGE_SCALE_M,
) -> tuple[RangeAwareCalibrator, list[CalibrationSearchRecord], ScoreOnlyCalibrator]:
    """Run the small frozen E1 grid and return the selected range model.

    Both families use the same ridge grid and deterministic Newton solver. The
    score-only result is retained as an internal control; selection is made
    only among the range-aware candidates using mini-train log loss.
    """

    ridges = tuple(float(value) for value in ridge_values)
    if not ridges or any(not np.isfinite(value) or value < 0 for value in ridges):
        raise ValueError("ridge_values must contain finite non-negative values")
    records: list[CalibrationSearchRecord] = []
    fitted: dict[str, LogisticCalibrator] = {}
    for family in ("score_only", "score_range"):
        for index, ridge in enumerate(ridges, start=1):
            config_id = f"{family}_{index:02d}"
            try:
                model = fit_logistic_calibrator(
                    raw_scores,
                    None if family == "score_only" else predicted_ranges_m,
                    labels,
                    family=family,
                    ridge=ridge,
                    range_center_m=range_center_m,
                    range_scale_m=range_scale_m,
                )
                scores = model.predict(raw_scores, predicted_ranges_m)
                metrics = calibration_metrics(scores, labels)
                records.append(CalibrationSearchRecord(config_id, family, model.to_dict(), metrics, True, False, "tested"))
                fitted[config_id] = model
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                records.append(CalibrationSearchRecord(config_id, family, {"ridge": ridge}, {}, False, False, str(exc)))
    valid_range = [record for record in records if record.family == "score_range" and record.valid]
    if not valid_range:
        raise ValueError("no valid score_range calibration configuration")
    winner = min(valid_range, key=lambda record: (float(record.metrics["log_loss"]), record.configuration_id))
    updated: list[CalibrationSearchRecord] = []
    for record in records:
        if record.configuration_id == winner.configuration_id:
            updated.append(CalibrationSearchRecord(record.configuration_id, record.family, record.parameters, record.metrics, record.valid, True, "selected by minimum mini_train log loss"))
        elif record.family == "score_only" and record.valid:
            updated.append(CalibrationSearchRecord(record.configuration_id, record.family, record.parameters, record.metrics, record.valid, False, "score-only internal control"))
        elif record.valid:
            updated.append(CalibrationSearchRecord(record.configuration_id, record.family, record.parameters, record.metrics, record.valid, False, "valid but not selected"))
        else:
            updated.append(record)
    selected = fitted[winner.configuration_id]
    control_candidates = [record for record in updated if record.family == "score_only" and record.valid]
    control_record = min(control_candidates, key=lambda record: (float(record.metrics["log_loss"]), record.configuration_id))
    control = ScoreOnlyCalibrator(CalibrationParameters.from_dict(control_record.parameters))
    return selected if isinstance(selected, RangeAwareCalibrator) else RangeAwareCalibrator(selected.parameters), updated, control


def calibrate_prediction(prediction: PredictionBatch, calibrator: LogisticCalibrator) -> PredictionBatch:
    """Return calibrated scores while preserving every predicted box field.

    The signature intentionally has no ground-truth argument.  At inference
    the only inputs are the prediction's raw scores and predicted centers.
    """

    if not isinstance(prediction, PredictionBatch):
        raise TypeError("prediction must be a PredictionBatch")
    if not isinstance(calibrator, LogisticCalibrator):
        raise TypeError("calibrator must be a LogisticCalibrator")
    raw_scores = [box.score for box in prediction.boxes]
    if any(score is None for score in raw_scores):
        raise ValueError("all predicted boxes require a score for calibration")
    ranges = [predicted_box_range_m(box) for box in prediction.boxes]
    calibrated_scores = calibrator.predict(raw_scores, ranges)
    boxes: list[Box3D] = []
    for box, score in zip(prediction.boxes, calibrated_scores):
        boxes.append(
            Box3D(
                center=box.center.copy(),
                size=box.size.copy(),
                yaw=box.yaw,
                label=box.label,
                score=float(score),
                velocity=None if box.velocity is None else box.velocity.copy(),
                track_id=box.track_id,
            )
        )
    metadata = dict(prediction.metadata)
    metadata["score_policy"] = f"e1_{calibrator.family}"
    metadata["calibration_parameters"] = calibrator.to_dict()
    return PredictionBatch(prediction.frame_id, boxes, prediction.runtime_ms, metadata)


def calibration_metrics(predicted: Sequence[float], labels: Sequence[int | bool]) -> dict[str, float | None]:
    """Return lightweight supplemental Brier/log-loss diagnostics."""

    scores = _finite_vector(predicted, "predicted")
    targets = _finite_vector(labels, "labels")
    if len(scores) != len(targets) or len(scores) == 0:
        raise ValueError("predicted and labels must be non-empty and aligned")
    if np.any((scores < 0) | (scores > 1)) or np.any((targets < 0) | (targets > 1)):
        raise ValueError("predicted and labels must be in [0, 1]")
    clipped = np.clip(scores, 1e-12, 1.0 - 1e-12)
    return {
        "brier_score": float(np.mean((scores - targets) ** 2)),
        "log_loss": float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))),
    }


__all__ = [
    "CALIBRATION_FAMILIES",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationSearchRecord",
    "CalibrationParameters",
    "LogisticCalibrator",
    "RangeAwareCalibrator",
    "ScoreOnlyCalibrator",
    "calibrate_prediction",
    "calibration_metrics",
    "fit_logistic_calibrator",
    "search_calibrators",
]
