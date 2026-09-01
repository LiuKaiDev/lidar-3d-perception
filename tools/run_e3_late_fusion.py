#!/usr/bin/env python3
"""Populate frozen detector caches and complete the Phase 6.3 E3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from lidar_perception.benchmark.report import collect_environment
from lidar_perception.datasets.nuscenes_adapter import NUSCENES_DETECTION_CLASSES, NuScenesAdapter
from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.density_eval import DensityAwareEvaluator
from lidar_perception.evaluation.distance_eval import DistanceAwareEvaluator
from lidar_perception.evaluation.matching import SampleEvaluation, center_distance, match_prediction_to_ground_truth
from lidar_perception.evaluation.metrics import DEFAULT_DENSITY_BINS, DEFAULT_DISTANCE_BINS, safe_ratio
from lidar_perception.evaluation.nuscenes import evaluate_nuscenes, evaluation_sample_tokens
from lidar_perception.experiments.bootstrap import PHASE6_BOOTSTRAP_METRICS, SceneMetricCounts, paired_scene_bootstrap
from lidar_perception.experiments.cache import PredictionCache, PredictionCacheProvenance
from lidar_perception.experiments.fusion import (
    FusionConfig,
    analyze_complementarity,
    fuse_predictions,
    naive_union,
    search_fusion_configs,
)
from lidar_perception.geometry.boxes3d import Box3D, count_points_in_box
from lidar_perception.utils.io import save_json

from common import load_detector_config, make_backend


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments/e3_late_fusion"
CACHE_ROOT = ROOT / "outputs/phase6_prediction_cache"
OFFICIAL_ROOT = ROOT / "outputs/phase6_e3_official"
CASE_ROOT = ROOT / "outputs/phase6_e3_cases"
CP_CONFIG = ROOT / "configs/detectors/centerpoint/nuscenes_mini.yaml"
VN_CONFIG = ROOT / "configs/detectors/voxelnext/nuscenes_mini.yaml"
VN_CHECKPOINT = Path("~/checkpoints/openpcdet/voxelnext_nuscenes.pth").expanduser()
CP_CONFIG_SHA256 = "58b5b8b2e4303a4563b6635c7d9f75a41acc8714635914b4d2cfb95ba8e40fc0"
VN_CONFIG_SHA256 = "84f5ddcab780ef108af77412f9bad587c21299419530c8708121442449115a0d"
CP_CHECKPOINT_SHA256 = "955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a"
VN_CHECKPOINT_SHA256 = "9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b"
SCORE_POLICY = "fixed upstream OpenPCDet and project candidate threshold 0.1"
SELECTION_RULE = "maximize 50m+ recall then 0-5-point recall, lower FP, lower threshold, weight closest to 1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(detector: str, split: str, token: str) -> PredictionCacheProvenance:
    is_cp = detector == "centerpoint_pointpillar"
    return PredictionCacheProvenance(
        dataset="nuscenes",
        dataset_version="v1.0-mini",
        split=split,
        sample_token=token,
        detector=detector,
        detector_config=("configs/detectors/centerpoint/nuscenes_mini.yaml" if is_cp else "cbgs_voxel0075_voxelnext.yaml"),
        detector_config_sha256=CP_CONFIG_SHA256 if is_cp else VN_CONFIG_SHA256,
        checkpoint_sha256=CP_CHECKPOINT_SHA256 if is_cp else VN_CHECKPOINT_SHA256,
        sweeps=10,
        candidate_threshold=0.1,
        score_filtering_policy=SCORE_POLICY,
    )


def _validate_prediction(prediction: PredictionBatch, token: str) -> dict[str, Any]:
    if prediction.frame_id != token:
        raise RuntimeError(f"cache frame mismatch: {prediction.frame_id} != {token}")
    valid_classes = set(NUSCENES_DETECTION_CLASSES)
    for box in prediction.boxes:
        if box.label not in valid_classes:
            raise RuntimeError(f"invalid class {box.label!r} in {token}")
        if not np.all(np.isfinite(box.center)) or not np.all(np.isfinite(box.size)) or np.any(box.size <= 0):
            raise RuntimeError(f"invalid box geometry in {token}")
        if not np.isfinite(box.yaw) or box.score is None or not np.isfinite(box.score) or box.score < 0.1:
            raise RuntimeError(f"invalid yaw/score in {token}")
        if box.velocity is not None and not np.all(np.isfinite(box.velocity)):
            raise RuntimeError(f"invalid velocity in {token}")
    return {
        "sample_token": token,
        "prediction_count": len(prediction.boxes),
        "velocity_count": sum(box.velocity is not None for box in prediction.boxes),
        "minimum_score": min((float(box.score) for box in prediction.boxes), default=None),
        "classes": sorted({box.label for box in prediction.boxes}),
    }


def _load_cache(cache: PredictionCache, detector: str, split: str, tokens: list[str]) -> tuple[dict[str, PredictionBatch], list[str]]:
    predictions: dict[str, PredictionBatch] = {}
    missing: list[str] = []
    for token in tokens:
        prediction = cache.load(_provenance(detector, split, token))
        if prediction is None:
            missing.append(token)
        else:
            predictions[token] = prediction
    return predictions, missing


def load_or_infer_voxelnext(
    adapter: NuScenesAdapter,
    split: str,
    *,
    no_inference: bool,
) -> tuple[list[str], dict[str, PredictionBatch], dict[str, Any]]:
    tokens = evaluation_sample_tokens(adapter, eval_set=split)
    if len(tokens) != (323 if split == "mini_train" else 81) or len(tokens) != len(set(tokens)):
        raise RuntimeError(f"unexpected or duplicate official {split} tokens: {len(tokens)}")
    cache = PredictionCache(CACHE_ROOT)
    predictions, missing = _load_cache(cache, "voxelnext", split, tokens)
    load_report: dict[str, Any] | None = None
    peak_vram_mb: float | None = None
    if missing and no_inference:
        raise RuntimeError(f"missing or incompatible VoxelNeXt {split} entries: {len(missing)}")
    if missing:
        if _sha256(VN_CHECKPOINT) != VN_CHECKPOINT_SHA256:
            raise RuntimeError("VoxelNeXt checkpoint SHA-256 mismatch")
        config, opcdet_config = load_detector_config(VN_CONFIG)
        backend = make_backend(config, opcdet_config, VN_CHECKPOINT)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        backend.load()
        load_report = dict(backend.load_report)
        for index, token in enumerate(missing, start=1):
            with torch.inference_mode():
                prediction = backend.predict(adapter.load_sample(token, max_sweeps=10))
            _validate_prediction(prediction, token)
            cache.save(prediction, _provenance("voxelnext", split, token))
            predictions[token] = prediction
            if index % 10 == 0 or index == len(missing):
                print(f"{split} VoxelNeXt inference: {index}/{len(missing)}", flush=True)
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        del backend
        torch.cuda.empty_cache()
    ordered = {token: predictions[token] for token in tokens}
    cp_predictions, cp_missing = _load_cache(cache, "centerpoint_pointpillar", split, tokens)
    if cp_missing:
        raise RuntimeError(f"CenterPoint {split} cache missing/incompatible entries: {len(cp_missing)}")
    if set(ordered) != set(cp_predictions):
        raise RuntimeError(f"CP/VN {split} token alignment mismatch")
    validation_indices = sorted(set((0, len(tokens) // 2, len(tokens) - 1)))
    samples = [_validate_prediction(ordered[tokens[index]], tokens[index]) for index in validation_indices]
    return tokens, ordered, {
        "split": split,
        "entries": len(ordered),
        "cache_hits": len(tokens) - len(missing),
        "inference_count": len(missing),
        "no_duplicate_tokens": len(tokens) == len(set(tokens)),
        "centerpoint_aligned": list(ordered) == list(cp_predictions),
        "provenance": _provenance("voxelnext", split, tokens[0]).to_dict() | {"sample_token": "<per-entry>"},
        "representative_validation": samples,
        "load_report": load_report,
        "sequential_peak_vram_mb": peak_vram_mb,
    }


def _load_centerpoint(split: str, tokens: list[str]) -> dict[str, PredictionBatch]:
    predictions, missing = _load_cache(PredictionCache(CACHE_ROOT), "centerpoint_pointpillar", split, tokens)
    if missing:
        raise RuntimeError(f"missing CenterPoint {split} entries: {len(missing)}")
    return {token: predictions[token] for token in tokens}


def _prepare_ground_truth(adapter: NuScenesAdapter, tokens: list[str]) -> tuple[list[list[Box3D]], list[list[int]], list[str]]:
    ground_truths: list[list[Box3D]] = []
    point_counts: list[list[int]] = []
    scenes: list[str] = []
    for index, token in enumerate(tokens, start=1):
        frame = adapter.load_sample(token, max_sweeps=1)
        ground_truth = adapter.load_boxes(token)
        ground_truths.append(ground_truth)
        point_counts.append([count_points_in_box(frame.points[:, :3], box) for box in ground_truth])
        scenes.append(str(adapter.sample_record(token)["scene_token"]))
        if index % 50 == 0 or index == len(tokens):
            print(f"{len(tokens)}-sample GT preparation: {index}/{len(tokens)}", flush=True)
    return ground_truths, point_counts, scenes


def _samples(
    tokens: list[str],
    predictions: dict[str, PredictionBatch],
    ground_truths: list[list[Box3D]],
    point_counts: list[list[int]],
    scenes: list[str],
) -> list[SampleEvaluation]:
    result: list[SampleEvaluation] = []
    for token, truth, counts, scene in zip(tokens, ground_truths, point_counts, scenes):
        prediction = predictions[token]
        prediction.metadata.setdefault("sample_token", token)
        prediction.metadata.setdefault("scene_token", scene)
        match = match_prediction_to_ground_truth(prediction, truth, distance_threshold_m=2.0, gt_point_counts=counts)
        result.append(SampleEvaluation(prediction, truth, counts, match))
    return result


def _scene_counts(samples: list[SampleEvaluation]) -> list[SceneMetricCounts]:
    result: list[SceneMetricCounts] = []
    scene_ids = sorted({str(sample.prediction.metadata["scene_token"]) for sample in samples})
    for scene_id in scene_ids:
        subset = [sample for sample in samples if sample.prediction.metadata["scene_token"] == scene_id]
        distance = DistanceAwareEvaluator().evaluate(subset)["overall"]
        density = DensityAwareEvaluator().evaluate(subset)["overall"]
        by_distance = {row["bin"]: row for row in distance}
        by_density = {row["bin"]: row for row in density}
        result.append(SceneMetricCounts(scene_id, {
            "recall_50m_plus": (by_distance["50m+"]["matched_count"], by_distance["50m+"]["gt_count"]),
            "recall_0_5_points": (by_density["0-5"]["matched_count"], by_density["0-5"]["gt_count"]),
            "recall_40_50m": (by_distance["40-50m"]["matched_count"], by_distance["40-50m"]["gt_count"]),
            "overall_custom_recall": (sum(len(sample.match.matches) for sample in subset), sum(len(sample.ground_truth) for sample in subset)),
        }))
    return result


def _custom_metrics(samples: list[SampleEvaluation]) -> dict[str, Any]:
    distance = DistanceAwareEvaluator().evaluate(samples)
    density = DensityAwareEvaluator().evaluate(samples)
    matched = sum(len(sample.match.matches) for sample in samples)
    gt_count = sum(len(sample.ground_truth) for sample in samples)
    fp = sum(len(sample.match.false_positives) for sample in samples)
    errors = [match.localization_error_m for sample in samples for match in sample.match.matches]
    confidences = [float(match.prediction_score) for sample in samples for match in sample.match.matches if match.prediction_score is not None]
    distance_rows = {row["bin"]: row for row in distance["overall"]}
    density_rows = {row["bin"]: row for row in density["overall"]}
    return {
        "sample_count": len(samples),
        "prediction_count": sum(len(sample.prediction.boxes) for sample in samples),
        "gt_count": gt_count,
        "matched_count": matched,
        "fp_count": fp,
        "precision": safe_ratio(matched, matched + fp),
        "overall_custom_recall": safe_ratio(matched, gt_count),
        "recall_50m_plus": distance_rows["50m+"]["recall"],
        "recall_40_50m": distance_rows["40-50m"]["recall"],
        "recall_0_5_points": density_rows["0-5"]["recall"],
        "matched_localization_error_m": None if not errors else float(np.mean(errors)),
        "average_matched_confidence": None if not confidences else float(np.mean(confidences)),
        "distance": distance,
        "density": density,
    }


def _candidate_coverage(
    predictions: dict[str, PredictionBatch],
    tokens: list[str],
    ground_truths: list[list[Box3D]],
    point_counts: list[list[int]],
) -> dict[str, Any]:
    categories = {"overall": [0, 0], "50m_plus": [0, 0], "0_5_points": [0, 0]}
    covered_sets: dict[str, set[tuple[str, int]]] = {name: set() for name in categories}
    for token, truth, counts in zip(tokens, ground_truths, point_counts):
        for index, (box, count) in enumerate(zip(truth, counts)):
            names = ["overall"]
            if np.linalg.norm(box.center[:2]) >= 50:
                names.append("50m_plus")
            if count <= 5:
                names.append("0_5_points")
            viable = any(candidate.label == box.label and center_distance(candidate, box) <= 2.0 for candidate in predictions[token].boxes)
            for name in names:
                categories[name][1] += 1
                categories[name][0] += int(viable)
                if viable:
                    covered_sets[name].add((token, index))
    return {
        "sections": {name: {"covered": values[0], "gt_count": values[1], "coverage": safe_ratio(*values)} for name, values in categories.items()},
        "covered_keys": {name: sorted([f"{token}:{index}" for token, index in values]) for name, values in covered_sets.items()},
    }


def _union_ceiling(cp: dict[str, Any], vn: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("overall", "50m_plus", "0_5_points"):
        cp_keys = set(cp["covered_keys"][name])
        vn_keys = set(vn["covered_keys"][name])
        total = cp["sections"][name]["gt_count"]
        result[name] = {
            "centerpoint": cp["sections"][name],
            "voxelnext": vn["sections"][name],
            "union": {"covered": len(cp_keys | vn_keys), "gt_count": total, "coverage": safe_ratio(len(cp_keys | vn_keys), total)},
            "centerpoint_only": len(cp_keys - vn_keys),
            "voxelnext_only": len(vn_keys - cp_keys),
        }
    return result


def _recovery(
    cp_samples: list[SampleEvaluation], vn_samples: list[SampleEvaluation], e3_samples: list[SampleEvaluation]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for section, predicate in (
        ("overall", lambda box, count: True),
        ("50m_plus", lambda box, count: np.linalg.norm(box.center[:2]) >= 50),
        ("0_5_points", lambda box, count: count <= 5),
    ):
        counts = {"cp_misses_covered_by_vn": 0, "cp_misses_recovered_by_e3": 0, "vn_misses_covered_by_cp": 0, "vn_misses_recovered_by_e3": 0}
        for cp, vn, e3 in zip(cp_samples, vn_samples, e3_samples):
            cp_set = {item.gt_index for item in cp.match.matches}
            vn_set = {item.gt_index for item in vn.match.matches}
            e3_set = {item.gt_index for item in e3.match.matches}
            for index, (box, point_count) in enumerate(zip(cp.ground_truth, cp.gt_point_counts)):
                if not predicate(box, point_count):
                    continue
                if index not in cp_set and index in vn_set:
                    counts["cp_misses_covered_by_vn"] += 1
                    counts["cp_misses_recovered_by_e3"] += int(index in e3_set)
                if index not in vn_set and index in cp_set:
                    counts["vn_misses_covered_by_cp"] += 1
                    counts["vn_misses_recovered_by_e3"] += int(index in e3_set)
        output[section] = counts
    return output


def _benchmark_fusion(cp: list[PredictionBatch], vn: list[PredictionBatch], config: FusionConfig) -> dict[str, Any]:
    timings: list[dict[str, float]] = []
    tracemalloc.start()
    for _ in range(5):
        for cp_batch, vn_batch in zip(cp, vn):
            fuse_predictions(cp_batch, vn_batch, config)
    tracemalloc.reset_peak()
    for _ in range(20):
        for cp_batch, vn_batch in zip(cp, vn):
            _, diagnostics = fuse_predictions(cp_batch, vn_batch, config, return_diagnostics=True)
            timings.append(diagnostics.to_dict())
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    means = {key: float(np.mean([row[key] for row in timings])) for key in ("association_time_ms", "fusion_time_ms", "sorting_time_ms", "total_fusion_time_ms")}
    return {
        "fusion_only": means | {"iterations": 20, "samples_per_iteration": len(cp), "peak_python_memory_bytes": peak},
        "dual_detector_total": {
            "type": "estimated",
            "centerpoint_e2e_ms": 70.24,
            "voxelnext_e2e_ms": 111.69,
            "fusion_overhead_ms": means["total_fusion_time_ms"],
            "estimated_total_ms": 70.24 + 111.69 + means["total_fusion_time_ms"],
        },
        "execution_model": "detectors generated sequentially; fusion operates on cached predictions",
    }


def _representative_cases(
    adapter: NuScenesAdapter,
    cp_samples: list[SampleEvaluation],
    vn_samples: list[SampleEvaluation],
    e3_samples: list[SampleEvaluation],
) -> list[dict[str, Any]]:
    from visualize_nuscenes import _plot_bev

    selected: dict[str, int] = {}
    for sample_index, (cp, vn, e3) in enumerate(zip(cp_samples, vn_samples, e3_samples)):
        cp_set = {item.gt_index for item in cp.match.matches}
        vn_set = {item.gt_index for item in vn.match.matches}
        e3_set = {item.gt_index for item in e3.match.matches}
        for gt_index, (box, points) in enumerate(zip(cp.ground_truth, cp.gt_point_counts)):
            preferred = np.linalg.norm(box.center[:2]) >= 50 or points <= 5
            candidates = []
            if gt_index not in cp_set and gt_index in vn_set and gt_index in e3_set:
                candidates.append("cp_miss_recovered")
            if gt_index not in vn_set and gt_index in cp_set and gt_index in e3_set:
                candidates.append("vn_miss_recovered")
            if gt_index not in cp_set and gt_index not in vn_set:
                candidates.append("both_miss")
            for category in candidates:
                if category not in selected or preferred:
                    selected[category] = sample_index
        if e3.match.false_positives and "fusion_fp" not in selected:
            selected["fusion_fp"] = sample_index
    rows: list[dict[str, Any]] = []
    for category in ("cp_miss_recovered", "vn_miss_recovered", "both_miss", "fusion_fp"):
        if category not in selected:
            continue
        sample = e3_samples[selected[category]]
        token = sample.sample_id
        path = CASE_ROOT / f"{category}_{token}_bev.png"
        _plot_bev(adapter.load_sample(token, max_sweeps=1), sample.ground_truth, sample.prediction.boxes, path)
        rows.append({"category": category, "sample_token": token, "path": str(path.relative_to(ROOT))})
    return rows


def _figures(complementarity: dict[str, Any], metrics: dict[str, Any], official: dict[str, Any], output: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    overall = complementarity["sections"]["overall"]["counts"]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Both", "CP only", "VN only", "Neither"]
    keys = ["detected_by_both", "centerpoint_only", "voxelnext_only", "neither"]
    ax.bar(labels, [overall[key] for key in keys], color=["#2a9d8f", "#457b9d", "#e9c46a", "#6c757d"])
    ax.set_ylabel("mini_train GT count"); fig.tight_layout()
    path = output / "complementarity.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path.relative_to(ROOT)))
    for metric_name, bins, filename in (("distance", [item.name for item in DEFAULT_DISTANCE_BINS], "recall_by_distance.png"), ("density", [item.name for item in DEFAULT_DENSITY_BINS], "recall_by_density.png")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for model, color in (("centerpoint", "#457b9d"), ("voxelnext", "#e9c46a"), ("e3", "#2a9d8f")):
            rows = metrics["mini_val"][model][metric_name]["overall"]
            ax.plot(bins, [row["recall"] for row in rows], marker="o", label=model, color=color)
        ax.set_ylim(0, 1); ax.set_ylabel("Recall"); ax.legend(); ax.tick_params(axis="x", rotation=25); fig.tight_layout()
        path = output / filename; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path.relative_to(ROOT)))
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["CenterPoint", "VoxelNeXt", "Naive Union", "E3"]
    keys = ["centerpoint", "voxelnext", "naive_union", "e3"]
    x = np.arange(len(keys)); width = 0.36
    ax.bar(x - width / 2, [official[key]["mAP"] for key in keys], width, label="mAP")
    ax.bar(x + width / 2, [official[key]["NDS"] for key in keys], width, label="NDS")
    ax.set_xticks(x, names, rotation=20); ax.set_ylim(0, 0.7); ax.legend(); fig.tight_layout()
    path = output / "official_metrics.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path.relative_to(ROOT)))
    return paths


def _classification(metrics: dict[str, Any], official: dict[str, Any]) -> str:
    e3 = metrics["e3"]
    cp = metrics["centerpoint"]
    vn = metrics["voxelnext"]
    best_far = max(cp["recall_50m_plus"], vn["recall_50m_plus"])
    best_sparse = max(cp["recall_0_5_points"], vn["recall_0_5_points"])
    official_best = max(official["centerpoint"]["NDS"], official["voxelnext"]["NDS"])
    if e3["recall_50m_plus"] > best_far and e3["recall_0_5_points"] >= best_sparse and official["e3"]["NDS"] >= official_best:
        return "POSITIVE"
    if e3["recall_50m_plus"] > best_far or e3["recall_0_5_points"] > best_sparse:
        return "DIRECTIONAL" if official["e3"]["NDS"] >= official_best - 0.05 else "NEGATIVE"
    if e3["recall_50m_plus"] == best_far and e3["recall_0_5_points"] == best_sparse and official["e3"]["NDS"] >= official_best:
        return "UNCHANGED"
    return "NEGATIVE"


def _render_analysis(payload: dict[str, Any]) -> str:
    train = payload["complementarity"]["sections"]
    ceiling = payload["candidate_coverage"]["mini_val"]
    recovery = payload["recovery"]
    val = payload["metrics"]["mini_val"]
    official = payload["official"]
    runtime = payload["benchmark"]["dual_detector_total"]
    return f"""# Experiment

E3 - CenterPoint + VoxelNeXt late prediction fusion (`nuScenes v1.0-mini exploratory experiment`).

## Hypothesis

Frozen CenterPoint and VoxelNeXt candidates may be complementary for far-range and sparse objects, allowing deterministic prediction-only late fusion to recover misses.

## Baseline

Both pretrained OpenPCDet detectors, the candidate floor (0.1), ten-sweep input, and Phase 4 evaluation protocol remained frozen.

## Change

E3 uses class-aware one-to-one center-distance association, probabilistic-OR scores, winner-take-all geometry, retained unmatched candidates, deterministic sorting, and a top-500 limit. The selected mini_train parameters are `{payload['frozen_config']['selected_config']}`.

## Controlled Variables

Selection used mini_train only under the preregistered rule: {SELECTION_RULE}. The final configuration was written before mini_val predictions were loaded. Ground truth is evaluation-only and never enters fusion.

## Main Metrics

Mini_val 50m+ recall: CP `{val['centerpoint']['recall_50m_plus']}`, VN `{val['voxelnext']['recall_50m_plus']}`, naive union `{val['naive_union']['recall_50m_plus']}`, E3 `{val['e3']['recall_50m_plus']}`. Mini_val 0-5-point recall: CP `{val['centerpoint']['recall_0_5_points']}`, VN `{val['voxelnext']['recall_0_5_points']}`, naive union `{val['naive_union']['recall_0_5_points']}`, E3 `{val['e3']['recall_0_5_points']}`.

## Distance-aware Metrics

All six frozen bins for CP, VN, naive union, and E3 are recorded in `metrics.json` and plotted in `figures/recall_by_distance.png`.

## Density-aware Metrics

All five current-keyframe GT density bins are recorded in `metrics.json` and plotted in `figures/recall_by_density.png`. GT density is evaluation-only.

## Candidate Complementarity

Mini_train overall both/CP-only/VN-only/neither counts are `{train['overall']['counts']}`. At 50m+ they are `{train['50m_plus']['counts']}`; at 0-5 points they are `{train['0_5_points']['counts']}`.

On mini_val, the union ceiling is `{ceiling}`. Adding VN raises the ceiling beyond CP wherever `voxelnext_only` is nonzero; adding CP raises it beyond VN wherever `centerpoint_only` is nonzero.

## Recovery Analysis

Potential versus actual recovery is `{recovery}`. This separates complementary exported candidates from GT actually matched after fusion.

## Official Metrics

Official detection_cvpr_2019 mini_val mAP/NDS are CP `{official['centerpoint']['mAP']}/{official['centerpoint']['NDS']}`, VN `{official['voxelnext']['mAP']}/{official['voxelnext']['NDS']}`, naive union `{official['naive_union']['mAP']}/{official['naive_union']['NDS']}`, and E3 `{official['e3']['mAP']}/{official['e3']['NDS']}`.

## Runtime

Fusion-only timings and memory are in `benchmark.json`. The dual-detector total is explicitly estimated, not measured: `{runtime['estimated_total_ms']}` ms/sample from the Phase 5 detector references plus cached-prediction fusion overhead. Models were generated sequentially; peak VRAM values are reported per sequential generation run and are not added.

## Representative Cases

Success and failure examples are recorded under `outputs/phase6_e3_cases/` and listed in `metrics.json`.

## Uncertainty

Paired scene bootstrap uses 1000 repetitions, seed 42, and 95% intervals with identical resampled indices. Mini_val has only two scenes, so intervals may be wide or degenerate. This split was already exposed in earlier phases.

## Result

E3 classification: **{payload['classification']}**. It is judged against the strongest single model, with far/sparse recall prioritized and official mAP/NDS treated as guardrails.

## Conclusion

Detector complementarity, candidate-ceiling change, actual recovery, false-positive cost, strongest-single comparison, official metric trade-off, and runtime cost are quantified above and in the JSON artifacts. Retention in the final portfolio is `{payload['retain_recommendation']}`.

## Next Experiment

Repository is ready for Phase 6.4: Final Ablation / Repeat Validation / Phase 6 Closure. E4 is not implemented here.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="~/datasets/nuscenes")
    parser.add_argument("--no-inference", action="store_true")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter = NuScenesAdapter(Path(args.dataset_root).expanduser(), version="v1.0-mini", max_sweeps=10)

    # Anti-leakage order: populate and analyze mini_train, select, then freeze.
    train_tokens, vn_train, train_cache = load_or_infer_voxelnext(adapter, "mini_train", no_inference=args.no_inference)
    cp_train = _load_centerpoint("mini_train", train_tokens)
    train_gt, train_counts, train_scenes = _prepare_ground_truth(adapter, train_tokens)
    complementarity = analyze_complementarity(
        [cp_train[token] for token in train_tokens], [vn_train[token] for token in train_tokens], train_gt, gt_point_counts=train_counts
    )
    cp_train_coverage = _candidate_coverage(cp_train, train_tokens, train_gt, train_counts)
    vn_train_coverage = _candidate_coverage(vn_train, train_tokens, train_gt, train_counts)
    train_ceiling = _union_ceiling(cp_train_coverage, vn_train_coverage)
    selected, search_records = search_fusion_configs(
        [cp_train[token] for token in train_tokens],
        [vn_train[token] for token in train_tokens],
        train_gt,
        gt_point_counts=train_counts,
        association_thresholds_m=(0.5, 1.0, 1.5, 2.0),
        centerpoint_weights=(0.8, 1.0, 1.2),
        voxelnext_weight=1.0,
        candidate_floor=0.1,
        max_boxes=500,
    )
    for record in search_records:
        config = FusionConfig.from_dict(record["parameters"])
        predictions = {token: fuse_predictions(cp_train[token], vn_train[token], config) for token in train_tokens}
        record["candidate_count"] = sum(len(prediction.boxes) for prediction in predictions.values())
        record["official_metrics"] = None
        record["reason"] = record.pop("selection_reason")
    search_payload = {
        "schema_version": "lidar_perception.e3_search.v1",
        "fit_split": "mini_train",
        "mini_val_used": False,
        "selection_rule": SELECTION_RULE,
        "configurations": search_records,
        "selected_config": selected.to_dict(),
    }
    save_json(search_payload, output / "search_results.json")
    frozen_config = {
        "frozen": True,
        "frozen_before_split": "mini_val",
        "order_marker": "mini_train_cache->complementarity->search->freeze->mini_val_cache->confirmation",
        "selection_rule": SELECTION_RULE,
        "selected_config": selected.to_dict(),
        "mini_train_metrics": next(record["metrics"] for record in search_records if record["selected"]),
        "search_log": "search_results.json",
    }
    save_json(frozen_config, output / "frozen_config.json")
    save_json({"status": "completed", "cache": train_cache, "matching": complementarity["protocol"], "sections": complementarity["sections"], "candidate_ceiling": train_ceiling}, output / "complementarity.json")
    train_variants = {
        "centerpoint": cp_train,
        "voxelnext": vn_train,
        "naive_union": {token: naive_union(cp_train[token], vn_train[token], duplicate_threshold_m=0.01, candidate_floor=0.1, max_boxes=500) for token in train_tokens},
        "e3": {token: fuse_predictions(cp_train[token], vn_train[token], selected) for token in train_tokens},
    }
    train_metrics = {
        name: _custom_metrics(_samples(train_tokens, predictions, train_gt, train_counts, train_scenes))
        for name, predictions in train_variants.items()
    }

    # The mini_val cache and its metrics are touched only after the freeze above.
    val_tokens, vn_val, val_cache = load_or_infer_voxelnext(adapter, "mini_val", no_inference=args.no_inference)
    cp_val = _load_centerpoint("mini_val", val_tokens)
    val_gt, val_counts, val_scenes = _prepare_ground_truth(adapter, val_tokens)
    e3_val = {token: fuse_predictions(cp_val[token], vn_val[token], selected) for token in val_tokens}
    union_val = {token: naive_union(cp_val[token], vn_val[token], duplicate_threshold_m=0.01, candidate_floor=0.1, max_boxes=500) for token in val_tokens}
    variants = {"centerpoint": cp_val, "voxelnext": vn_val, "naive_union": union_val, "e3": e3_val}
    sample_sets = {name: _samples(val_tokens, predictions, val_gt, val_counts, val_scenes) for name, predictions in variants.items()}
    val_metrics = {name: _custom_metrics(samples) for name, samples in sample_sets.items()}
    cp_val_coverage = _candidate_coverage(cp_val, val_tokens, val_gt, val_counts)
    vn_val_coverage = _candidate_coverage(vn_val, val_tokens, val_gt, val_counts)
    e3_val_coverage = _candidate_coverage(e3_val, val_tokens, val_gt, val_counts)
    val_ceiling = _union_ceiling(cp_val_coverage, vn_val_coverage)
    for section in val_ceiling:
        val_ceiling[section]["e3"] = e3_val_coverage["sections"][section]
    recovery = _recovery(sample_sets["centerpoint"], sample_sets["voxelnext"], sample_sets["e3"])
    bootstrap = {
        baseline: {key: value.to_dict() for key, value in paired_scene_bootstrap(
            _scene_counts(sample_sets[baseline]), _scene_counts(sample_sets["e3"]), metrics=PHASE6_BOOTSTRAP_METRICS,
            repetitions=1000, seed=42, confidence_level=0.95,
        ).items()}
        for baseline in ("centerpoint", "voxelnext")
    }
    official: dict[str, Any] = {}
    if not args.skip_official:
        for name, predictions in variants.items():
            print(f"official mini_val evaluation: {name}", flush=True)
            official[name] = evaluate_nuscenes([predictions[token] for token in val_tokens], adapter, OFFICIAL_ROOT / name, eval_set="mini_val")
    else:
        official = {name: None for name in variants}
    benchmark = _benchmark_fusion([cp_val[token] for token in val_tokens], [vn_val[token] for token in val_tokens], selected)
    benchmark["sequential_voxelnext_peak_vram_mb"] = {"mini_train": train_cache["sequential_peak_vram_mb"], "mini_val": val_cache["sequential_peak_vram_mb"]}
    cases = _representative_cases(adapter, sample_sets["centerpoint"], sample_sets["voxelnext"], sample_sets["e3"])
    classification = "BLOCKED" if args.skip_official else _classification(val_metrics, official)
    metric_payload = {
        "label": "nuScenes v1.0-mini exploratory experiment",
        "experiment_id": "E3",
        "status": "PASS" if not args.skip_official else "BLOCKED",
        "classification": classification,
        "cache": {"mini_train": train_cache, "mini_val": val_cache},
        "metrics": {"mini_train": train_metrics | {"selected": frozen_config["mini_train_metrics"]}, "mini_val": val_metrics},
        "candidate_coverage": {"mini_train": train_ceiling, "mini_val": val_ceiling},
        "recovery": recovery,
        "bootstrap": bootstrap,
        "official": official,
        "representative_cases": cases,
    }
    figures = [] if args.skip_official else _figures(complementarity, metric_payload["metrics"], official, output / "figures")
    metric_payload["figures"] = figures
    save_json(metric_payload, output / "metrics.json")
    save_json(benchmark, output / "benchmark.json")
    environment = collect_environment(ROOT / "third_party/OpenPCDet")
    environment.update({"nuscenes_devkit": __import__("nuscenes").__version__ if hasattr(__import__("nuscenes"), "__version__") else "installed", "spconv": __import__("spconv").__version__})
    (output / "environment.txt").write_text("\n".join(f"{key}: {value}" for key, value in sorted(environment.items())) + "\n", encoding="utf-8")
    if not args.skip_official:
        full_payload = {
            "complementarity": complementarity,
            "candidate_coverage": metric_payload["candidate_coverage"],
            "recovery": recovery,
            "metrics": metric_payload["metrics"],
            "official": official,
            "benchmark": benchmark,
            "frozen_config": frozen_config,
            "classification": classification,
            "retain_recommendation": "retain as a documented late-fusion ablation" if classification in {"POSITIVE", "DIRECTIONAL"} else "do not retain as the preferred final detector; retain the negative ablation record",
        }
        (output / "analysis.md").write_text(_render_analysis(full_payload), encoding="utf-8")
    print({"status": metric_payload["status"], "classification": classification, "selected": selected.to_dict(), "output": str(output)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
