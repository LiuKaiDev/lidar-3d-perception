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
from .features import PredictedBoxFeatures, extract_prediction_features, predicted_box_range_m

__all__ = [
    "BootstrapInterval",
    "PairedBootstrapInterval",
    "PredictedBoxFeatures",
    "PredictionCache",
    "PredictionCacheProvenance",
    "SceneMetricCounts",
    "SceneMetricRecord",
    "extract_prediction_features",
    "group_scene_counts",
    "paired_scene_bootstrap",
    "predicted_box_range_m",
    "scene_level_bootstrap",
]
