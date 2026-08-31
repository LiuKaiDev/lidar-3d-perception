"""Phase 6 experiment protocol and lightweight analysis infrastructure."""

from .bootstrap import (
    BootstrapInterval,
    PairedBootstrapInterval,
    SceneMetricCounts,
    SceneMetricRecord,
    group_scene_counts,
    paired_scene_bootstrap,
    scene_level_bootstrap,
)
from .cache import PredictionCache, PredictionCacheProvenance
from .calibration import (
    CALIBRATION_FAMILIES,
    CALIBRATION_SCHEMA_VERSION,
    CalibrationParameters,
    CalibrationSearchRecord,
    LogisticCalibrator,
    RangeAwareCalibrator,
    ScoreOnlyCalibrator,
    calibrate_prediction,
    calibration_metrics,
    fit_logistic_calibrator,
    search_calibrators,
)
from .features import PredictedBoxFeatures, extract_prediction_features, predicted_box_range_m

__all__ = [
    "BootstrapInterval",
    "CALIBRATION_FAMILIES",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationParameters",
    "CalibrationSearchRecord",
    "LogisticCalibrator",
    "PairedBootstrapInterval",
    "PredictedBoxFeatures",
    "PredictionCache",
    "PredictionCacheProvenance",
    "RangeAwareCalibrator",
    "ScoreOnlyCalibrator",
    "SceneMetricCounts",
    "SceneMetricRecord",
    "extract_prediction_features",
    "calibrate_prediction",
    "calibration_metrics",
    "fit_logistic_calibrator",
    "group_scene_counts",
    "paired_scene_bootstrap",
    "predicted_box_range_m",
    "scene_level_bootstrap",
    "search_calibrators",
]
