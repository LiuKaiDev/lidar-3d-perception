import json

import numpy as np
import pytest

from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.experiments.fusion import (
    FusionConfig,
    analyze_complementarity,
    associate_predictions,
    fuse_predictions,
    load_frozen_config,
    naive_union,
    save_frozen_config,
    search_fusion_configs,
)
from lidar_perception.geometry.boxes3d import Box3D


def box(center, label="car", score=0.8, yaw=0.0, velocity=None):
    return Box3D(center, [2.0, 1.0, 1.5], yaw, label, score=score, velocity=velocity)


def batch(boxes, frame="sample"):
    return PredictionBatch(frame, list(boxes))


def test_association_is_class_aware_one_to_one_and_inclusive():
    cp = batch([box([0, 0, 0]), box([0.1, 0, 0], score=0.7)])
    vn = batch([box([1, 0, 0]), box([0, 0, 0], label="pedestrian")])
    pairs = associate_predictions(cp, vn, threshold_m=1.0)
    assert [(p.centerpoint_index, p.voxelnext_index) for p in pairs] == [(1, 0)]
    assert pairs[0].distance_m == pytest.approx(0.9)


def test_fusion_score_geometry_velocity_and_unmatched_candidates():
    cp = batch([box([0, 0, 0], score=0.8, yaw=3.0, velocity=[1, 2, 0]), box([8, 0, 0], score=0.2)])
    vn = batch([box([0.2, 0, 0], score=0.6, yaw=-3.0, velocity=[3, 4, 0]), box([20, 0, 0], label="pedestrian", score=0.9)])
    out = fuse_predictions(cp, vn, FusionConfig(association_threshold_m=0.5))
    assert len(out.boxes) == 3
    assert out.boxes[0].score == pytest.approx(1 - (1 - .8) * (1 - .6))
    assert out.boxes[0].yaw == pytest.approx(3.0)
    np.testing.assert_allclose(out.boxes[0].velocity, [1, 2, 0])
    assert out.metadata["candidate_count_before_limit"] == 3


def test_max_box_limit_is_deterministic_and_naive_union_keeps_complements():
    cp = batch([box([0, 0, 0], score=.5), box([1, 0, 0], score=.9)])
    vn = batch([box([10, 0, 0], label="pedestrian", score=.8)])
    out = fuse_predictions(cp, vn, FusionConfig(association_threshold_m=0, max_boxes=2))
    assert [b.score for b in out.boxes] == pytest.approx([.9, .8])
    union = naive_union(cp, vn, max_boxes=10)
    assert len(union.boxes) == 3


def test_complementarity_and_config_serialization(tmp_path):
    cp = [batch([box([0, 0, 0])])]
    vn = [batch([box([50, 0, 0], label="pedestrian")])]
    gt = [[box([0, 0, 0]), box([50, 0, 0], label="pedestrian")]]
    summary = analyze_complementarity(cp, vn, gt, gt_point_counts=[[2, 4]])
    assert summary["sections"]["overall"]["counts"] == {"detected_by_both": 0, "centerpoint_only": 1, "voxelnext_only": 1, "neither": 0}
    path = save_frozen_config(FusionConfig(association_threshold_m=.5), tmp_path / "frozen_config.json")
    assert load_frozen_config(path).association_threshold_m == .5
    assert json.loads(path.read_text())["schema_version"]


def test_search_records_all_configs_and_selects_without_validation_split():
    cp = [batch([box([50, 0, 0], score=.8)])]
    vn = [batch([box([50.4, 0, 0], score=.8)])]
    gt = [[box([50, 0, 0])]]
    winner, records = search_fusion_configs(cp, vn, gt, association_thresholds_m=(.5, 1.0), centerpoint_weights=(.8, 1.0))
    assert len(records) == 4
    assert sum(record["selected"] for record in records) == 1
    assert winner.association_threshold_m in (.5, 1.0)


def test_fusion_rejects_mismatched_frames():
    with pytest.raises(ValueError):
        fuse_predictions(batch([]), batch([], frame="other"))
