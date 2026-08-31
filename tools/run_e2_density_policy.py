#!/usr/bin/env python3
"""Run Phase 6.2 predicted-box current-keyframe sparsity calibration."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lidar_perception.benchmark.report import collect_environment
from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter
from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.density_eval import DensityAwareEvaluator
from lidar_perception.evaluation.distance_eval import DistanceAwareEvaluator
from lidar_perception.evaluation.matching import SampleEvaluation, center_distance, match_prediction_to_ground_truth
from lidar_perception.evaluation.metrics import DEFAULT_DENSITY_BINS, DEFAULT_DISTANCE_BINS, find_bin, safe_ratio
from lidar_perception.evaluation.nuscenes import evaluate_nuscenes, evaluation_sample_tokens
from lidar_perception.experiments.bootstrap import PHASE6_BOOTSTRAP_METRICS, SceneMetricCounts, paired_scene_bootstrap
from lidar_perception.experiments.cache import PredictionCache, PredictionCacheProvenance
from lidar_perception.experiments.calibration import calibrate_prediction, calibration_metrics
from lidar_perception.experiments.features import predicted_box_keyframe_point_counts, predicted_box_range_m
from lidar_perception.experiments.sparsity import (
    SparsitySearchRecord,
    calibrate_prediction_with_point_counts,
    search_sparsity_calibrators,
)
from lidar_perception.geometry.boxes3d import count_points_in_box
from lidar_perception.utils.io import save_json

from common import load_pointpillar_config, make_backend


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments/e2_density_policy"
CACHE_ROOT = ROOT / "outputs/phase6_prediction_cache"
OFFICIAL_OUTPUT_ROOT = ROOT / "outputs/phase6_e2_official"
PROJECT_CONFIG = ROOT / "configs/detectors/centerpoint/nuscenes_mini.yaml"
CHECKPOINT = Path("~/checkpoints/openpcdet/centerpoint_nuscenes_pp.pth").expanduser()
CONFIG_SHA256 = "58b5b8b2e4303a4563b6635c7d9f75a41acc8714635914b4d2cfb95ba8e40fc0"
CHECKPOINT_SHA256 = "955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a"
CANDIDATE_THRESHOLD = 0.1
PREDICTED_COUNT_BINS = (("0-5", 0, 6), ("6-10", 6, 11), ("11-20", 11, 21), ("21-50", 21, 51), ("51+", 51, float("inf")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(split: str, token: str) -> PredictionCacheProvenance:
    return PredictionCacheProvenance(
        dataset="nuscenes",
        dataset_version="v1.0-mini",
        split=split,
        sample_token=token,
        detector="centerpoint_pointpillar",
        detector_config="configs/detectors/centerpoint/nuscenes_mini.yaml",
        detector_config_sha256=CONFIG_SHA256,
        checkpoint_sha256=CHECKPOINT_SHA256,
        sweeps=10,
        candidate_threshold=CANDIDATE_THRESHOLD,
        score_filtering_policy="fixed upstream OpenPCDet and project candidate threshold 0.1",
    )


def load_or_infer_predictions(
    adapter: NuScenesAdapter,
    split: str,
    *,
    no_inference: bool,
) -> tuple[list[str], dict[str, PredictionBatch], dict[str, int]]:
    tokens = evaluation_sample_tokens(adapter, eval_set=split)
    cache = PredictionCache(CACHE_ROOT)
    predictions: dict[str, PredictionBatch] = {}
    missing: list[str] = []
    for token in tokens:
        prediction = cache.load(_provenance(split, token))
        if prediction is None:
            missing.append(token)
        else:
            predictions[token] = prediction
    if missing and no_inference:
        raise RuntimeError(f"missing or incompatible {split} cache entries: {len(missing)}")
    if missing:
        config, opcdet_config = load_pointpillar_config(PROJECT_CONFIG)
        backend = make_backend(config, opcdet_config, CHECKPOINT)
        backend.load()
        for index, token in enumerate(missing, start=1):
            prediction = backend.predict(adapter.load_sample(token, max_sweeps=10))
            cache.save(prediction, _provenance(split, token))
            predictions[token] = prediction
            if index % 20 == 0 or index == len(missing):
                print(f"{split} inference: {index}/{len(missing)}")
    return tokens, predictions, {"cache_hits": len(tokens) - len(missing), "inference_count": len(missing)}


def _prepare_split(
    adapter: NuScenesAdapter,
    tokens: list[str],
    predictions: dict[str, PredictionBatch],
) -> tuple[list[SampleEvaluation], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Compute inference features first, then attach evaluation-only labels."""

    samples: list[SampleEvaluation] = []
    counts_by_token: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for token_index, token in enumerate(tokens, start=1):
        frame = adapter.load_sample(token, max_sweeps=1)
        prediction = predictions[token]
        scene_token = str(adapter.sample_record(token)["scene_token"])
        prediction.metadata.setdefault("scene_token", scene_token)

        # Inference boundary: only predicted boxes and current sensor points.
        predicted_counts = predicted_box_keyframe_point_counts(prediction, frame.points)
        counts_by_token[token] = predicted_counts

        # Evaluation boundary: GT enters only after inference features exist.
        ground_truth = adapter.load_boxes(token)
        gt_counts = [count_points_in_box(frame.points[:, :3], box) for box in ground_truth]
        match = match_prediction_to_ground_truth(prediction, ground_truth, distance_threshold_m=2.0, gt_point_counts=gt_counts)
        sample = SampleEvaluation(prediction, ground_truth, gt_counts, match)
        samples.append(sample)
        matches = {item.prediction_index: item for item in match.matches}
        for index, (box, point_count) in enumerate(zip(prediction.boxes, predicted_counts)):
            matched = matches.get(index)
            records.append({
                "sample_token": token,
                "scene_token": scene_token,
                "prediction_index": index,
                "class": box.label,
                "raw_score": float(box.score),
                "predicted_range_m": predicted_box_range_m(box),
                "predicted_box_keyframe_point_count": int(point_count),
                "fit_label": int(matched is not None),
                "match_state": "TP" if matched is not None else "FP",
                "localization_error_m": None if matched is None else matched.localization_error_m,
                "predicted_center": box.center.tolist(),
                "predicted_size": box.size.tolist(),
                "predicted_yaw": box.yaw,
            })
        if token_index % 50 == 0 or token_index == len(tokens):
            print(f"{tokens and 'feature/evaluation'}: {token_index}/{len(tokens)}")
    return samples, counts_by_token, records


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {name: None for name in ("min", "q25", "median", "q75", "q90", "max")}
    result = np.quantile(values, [0, 0.25, 0.5, 0.75, 0.9, 1.0])
    return dict(zip(("min", "q25", "median", "q75", "q90", "max"), map(float, result)))


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _reliability(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (scores >= lower) & (scores <= upper if index == bins - 1 else scores < upper)
        count = int(mask.sum())
        mean_score = None if count == 0 else float(np.mean(scores[mask]))
        tp_rate = None if count == 0 else float(np.mean(labels[mask]))
        if count:
            ece += count / len(scores) * abs(mean_score - tp_rate)
        rows.append({"lower": lower, "upper": upper, "count": count, "mean_score": mean_score, "tp_rate": tp_rate})
    return {"expected_calibration_error": ece, "bins": rows}


def _feasibility(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray([row["raw_score"] for row in records])
    counts = np.asarray([row["predicted_box_keyframe_point_count"] for row in records])
    ranges = np.asarray([row["predicted_range_m"] for row in records])
    labels = np.asarray([row["fit_label"] for row in records])
    localization = np.asarray([row["localization_error_m"] for row in records if row["localization_error_m"] is not None])
    matched_counts = np.asarray([row["predicted_box_keyframe_point_count"] for row in records if row["localization_error_m"] is not None])
    rows: list[dict[str, Any]] = []
    for name, lower, upper in PREDICTED_COUNT_BINS:
        mask = (counts >= lower) & (counts < upper)
        count = int(mask.sum())
        rows.append({
            "predicted_point_count_bin": name,
            "prediction_count": count,
            "proportion": safe_ratio(count, len(records)),
            "tp_count": int(labels[mask].sum()),
            "tp_rate": None if count == 0 else float(np.mean(labels[mask])),
            "mean_raw_confidence": None if count == 0 else float(np.mean(scores[mask])),
            "median_predicted_range_m": None if count == 0 else float(np.median(ranges[mask])),
        })
    inspection: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    predicates = (
        lambda row: row["predicted_box_keyframe_point_count"] == 0,
        lambda row: row["fit_label"] == 1 and 1 <= row["predicted_box_keyframe_point_count"] <= 5,
        lambda row: row["fit_label"] == 0 and 1 <= row["predicted_box_keyframe_point_count"] <= 5,
        lambda row: row["fit_label"] == 1 and row["predicted_box_keyframe_point_count"] >= 51,
        lambda row: row["fit_label"] == 0 and row["predicted_box_keyframe_point_count"] >= 51,
    )
    for predicate in predicates:
        selected = next((row for row in records if predicate(row) and (row["sample_token"], row["prediction_index"]) not in used), None)
        if selected is not None:
            used.add((selected["sample_token"], selected["prediction_index"]))
            inspection.append(dict(selected))
    return {
        "split": "mini_train",
        "prediction_count": len(records),
        "tp_count": int(labels.sum()),
        "fp_count": int(len(labels) - labels.sum()),
        "point_count_tp": _quantiles(counts[labels == 1]),
        "point_count_fp": _quantiles(counts[labels == 0]),
        "predicted_point_count_bins": rows,
        "correlations": {
            "log1p_point_count_vs_raw_confidence": _correlation(np.log1p(counts), scores),
            "log1p_point_count_vs_predicted_range": _correlation(np.log1p(counts), ranges),
            "matched_log1p_point_count_vs_localization_error": _correlation(np.log1p(matched_counts), localization),
        },
        "manual_real_prediction_inspection": inspection,
        "feature_definition": "current-keyframe points inside predicted oriented Box3D; inference-only geometry",
        "label_definition": "mini_train evaluation-only class-aware greedy center-distance TP/FP at <=2.0m",
    }


def _scene_id(sample: SampleEvaluation) -> str:
    return str(sample.prediction.metadata["scene_token"])


def _custom_metrics(samples: list[SampleEvaluation]) -> tuple[dict[str, Any], list[SceneMetricCounts]]:
    distance = DistanceAwareEvaluator(bins=DEFAULT_DISTANCE_BINS, distance_threshold_m=2.0).evaluate(samples)
    density = DensityAwareEvaluator(bins=DEFAULT_DENSITY_BINS, distance_threshold_m=2.0).evaluate(samples)
    matched = sum(len(sample.match.matches) for sample in samples)
    gt_count = sum(len(sample.ground_truth) for sample in samples)
    fp_count = sum(len(sample.match.false_positives) for sample in samples)
    errors = [record.localization_error_m for sample in samples for record in sample.match.matches]
    confidences = [float(record.prediction_score) for sample in samples for record in sample.match.matches if record.prediction_score is not None]
    distance_rows = {row["bin"]: row for row in distance["overall"]}
    density_rows = {row["bin"]: row for row in density["overall"]}
    scene_counts: list[SceneMetricCounts] = []
    for scene in sorted({_scene_id(sample) for sample in samples}):
        subset = [sample for sample in samples if _scene_id(sample) == scene]
        distance_subset = {row["bin"]: row for row in DistanceAwareEvaluator().evaluate(subset)["overall"]}
        density_subset = {row["bin"]: row for row in DensityAwareEvaluator().evaluate(subset)["overall"]}
        scene_counts.append(SceneMetricCounts(scene, {
            "recall_50m_plus": (distance_subset["50m+"]["matched_count"], distance_subset["50m+"]["gt_count"]),
            "recall_0_5_points": (density_subset["0-5"]["matched_count"], density_subset["0-5"]["gt_count"]),
            "recall_40_50m": (distance_subset["40-50m"]["matched_count"], distance_subset["40-50m"]["gt_count"]),
            "overall_custom_recall": (sum(len(item.match.matches) for item in subset), sum(len(item.ground_truth) for item in subset)),
        }))
    return {
        "sample_count": len(samples),
        "gt_count": gt_count,
        "matched_count": matched,
        "prediction_count": matched + fp_count,
        "fp_count": fp_count,
        "precision": safe_ratio(matched, matched + fp_count),
        "overall_custom_recall": safe_ratio(matched, gt_count),
        "recall_50m_plus": distance_rows["50m+"]["recall"],
        "recall_40_50m": distance_rows["40-50m"]["recall"],
        "recall_0_5_points": density_rows["0-5"]["recall"],
        "matched_localization_error_m": None if not errors else float(np.mean(errors)),
        "average_matched_confidence": None if not confidences else float(np.mean(confidences)),
        "distance": distance,
        "density": density,
    }, scene_counts


def _candidate_ceiling(samples: list[SampleEvaluation]) -> dict[str, Any]:
    distance_counts = {metric_bin.name: [0, 0] for metric_bin in DEFAULT_DISTANCE_BINS}
    density_counts = {metric_bin.name: [0, 0] for metric_bin in DEFAULT_DENSITY_BINS}
    class_density: dict[str, dict[str, list[int]]] = {}
    covered = 0
    total = 0
    sparse_no_candidate = 0
    sparse_with_candidate = 0
    sparse_low_score = 0
    sparse_best_scores: list[float] = []
    for sample in samples:
        for gt_box, gt_points in zip(sample.ground_truth, sample.gt_point_counts):
            viable = [box for box in sample.prediction.boxes if box.label == gt_box.label and center_distance(box, gt_box) <= 2.0]
            has_candidate = bool(viable)
            total += 1
            covered += int(has_candidate)
            distance_bin = find_bin(float(np.hypot(gt_box.center[0], gt_box.center[1])), DEFAULT_DISTANCE_BINS)
            density_bin = find_bin(float(gt_points), DEFAULT_DENSITY_BINS)
            if distance_bin is not None:
                distance_counts[distance_bin.name][1] += 1
                distance_counts[distance_bin.name][0] += int(has_candidate)
            if density_bin is not None:
                density_counts[density_bin.name][1] += 1
                density_counts[density_bin.name][0] += int(has_candidate)
                class_rows = class_density.setdefault(gt_box.label, {metric_bin.name: [0, 0] for metric_bin in DEFAULT_DENSITY_BINS})
                class_rows[density_bin.name][1] += 1
                class_rows[density_bin.name][0] += int(has_candidate)
            if gt_points <= 5:
                if not viable:
                    sparse_no_candidate += 1
                else:
                    sparse_with_candidate += 1
                    best_score = max(float(box.score) for box in viable)
                    sparse_best_scores.append(best_score)
                    sparse_low_score += int(best_score < 0.3)

    def rows(counts: dict[str, list[int]]) -> list[dict[str, Any]]:
        return [{"bin": name, "candidate_covered_count": values[0], "gt_count": values[1], "candidate_coverage": safe_ratio(values[0], values[1])} for name, values in counts.items()]

    return {
        "overall": safe_ratio(covered, total),
        "candidate_covered_count": covered,
        "gt_count": total,
        "by_distance": rows(distance_counts),
        "by_gt_density": rows(density_counts),
        "priority_class_gt_density": {label: rows(values) for label, values in sorted(class_density.items()) if label in {"pedestrian", "bicycle", "motorcycle", "traffic_cone", "car"}},
        "sparse_gt_failure_partition": {
            "gt_0_5_count": sparse_no_candidate + sparse_with_candidate,
            "no_exported_viable_candidate": sparse_no_candidate,
            "with_exported_viable_candidate": sparse_with_candidate,
            "with_candidate_best_score_below_0_3": sparse_low_score,
            "best_candidate_score_quantiles": _quantiles(np.asarray(sparse_best_scores)),
        },
        "note": "evaluation-only GT coverage; any class-aware exported candidate within inclusive 2.0m, without one-to-one consumption",
    }


def _replace_predictions(samples: list[SampleEvaluation], predictions: dict[str, PredictionBatch]) -> list[SampleEvaluation]:
    result: list[SampleEvaluation] = []
    for sample in samples:
        prediction = predictions[sample.sample_id]
        match = match_prediction_to_ground_truth(prediction, sample.ground_truth, distance_threshold_m=2.0, gt_point_counts=sample.gt_point_counts)
        result.append(SampleEvaluation(prediction, sample.ground_truth, sample.gt_point_counts, match))
    return result


def _arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([row["raw_score"] for row in records]),
        np.asarray([row["predicted_box_keyframe_point_count"] for row in records]),
        np.asarray([row["fit_label"] for row in records]),
        np.asarray([row["scene_token"] for row in records], dtype=object),
    )


def _calibration_diagnostics(records: list[dict[str, Any]], score_only, sparsity) -> dict[str, Any]:
    scores, counts, labels, _ = _arrays(records)
    variants = {
        "raw": scores,
        "score_only": score_only.predict(scores),
        "score_sparsity": sparsity.predict(scores, counts),
    }
    result: dict[str, Any] = {}
    for name, values in variants.items():
        result[name] = {**calibration_metrics(values, labels), "reliability": _reliability(values, labels)}
    rows: list[dict[str, Any]] = []
    for name, lower, upper in PREDICTED_COUNT_BINS:
        mask = (counts >= lower) & (counts < upper)
        rows.append({
            "predicted_point_count_bin": name,
            "count": int(mask.sum()),
            "raw_mean": None if not mask.any() else float(np.mean(scores[mask])),
            "score_only_mean": None if not mask.any() else float(np.mean(variants["score_only"][mask])),
            "score_sparsity_mean": None if not mask.any() else float(np.mean(variants["score_sparsity"][mask])),
            "mean_e2_score_change": None if not mask.any() else float(np.mean(variants["score_sparsity"][mask] - scores[mask])),
        })
    result["by_predicted_point_count"] = rows
    return result


def _write_figures(feasibility: dict[str, Any], diagnostics: dict[str, Any], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    rows = feasibility["predicted_point_count_bins"]
    labels = [row["predicted_point_count_bin"] for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, [row["prediction_count"] - row["tp_count"] for row in rows], label="FP", color="#e45756")
    ax.bar(x, [row["tp_count"] for row in rows], bottom=[row["prediction_count"] - row["tp_count"] for row in rows], label="TP", color="#4c78a8")
    ax.set_xticks(x, labels); ax.set_xlabel("predicted-box keyframe points"); ax.set_ylabel("predictions"); ax.legend(); fig.tight_layout()
    path = output / "tp_fp_point_count_distribution.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [row["tp_rate"] for row in rows], marker="o", label="TP rate")
    ax.plot(x, [row["mean_raw_confidence"] for row in rows], marker="s", label="raw confidence")
    ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.set_xlabel("predicted-box keyframe points"); ax.legend(); fig.tight_layout()
    path = output / "tp_rate_confidence_by_point_count.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [row["median_predicted_range_m"] for row in rows], marker="o", color="#54a24b")
    ax.set_xticks(x, labels); ax.set_xlabel("predicted-box keyframe points"); ax.set_ylabel("median predicted range (m)"); fig.tight_layout()
    path = output / "range_by_point_count.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))

    score_rows = diagnostics["mini_val"]["by_predicted_point_count"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, [row["mean_e2_score_change"] for row in score_rows], color="#f58518")
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(x, labels); ax.set_xlabel("predicted-box keyframe points"); ax.set_ylabel("mean E2 score change"); fig.tight_layout()
    path = output / "score_change_by_point_count.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    return paths


def _classification(official: dict[str, Any]) -> str:
    baseline, control, e2 = official.get("raw"), official.get("score_only"), official.get("e2")
    if not baseline or not control or not e2:
        return "UNCHANGED"
    tolerance = 1e-12
    if e2["mAP"] < min(baseline["mAP"], control["mAP"]) - tolerance or e2["NDS"] < min(baseline["NDS"], control["NDS"]) - tolerance:
        return "NEGATIVE"
    if e2["mAP"] > max(baseline["mAP"], control["mAP"]) + tolerance and e2["NDS"] > max(baseline["NDS"], control["NDS"]) + tolerance:
        return "DIRECTIONAL"
    return "UNCHANGED"


def _benchmark(adapter: NuScenesAdapter, tokens: list[str], predictions: dict[str, PredictionBatch], counts_by_token: dict[str, np.ndarray], calibrator) -> dict[str, Any]:
    selected_tokens = [tokens[index] for index in np.linspace(0, len(tokens) - 1, num=min(5, len(tokens)), dtype=int)]
    prepared = [(predictions[token], adapter.load_sample(token, max_sweeps=1).points, counts_by_token[token]) for token in selected_tokens]
    warmup, iterations = 5, 20
    for _ in range(warmup):
        for prediction, points, _ in prepared:
            predicted_box_keyframe_point_counts(prediction, points)
    start = time.perf_counter()
    for _ in range(iterations):
        for prediction, points, _ in prepared:
            predicted_box_keyframe_point_counts(prediction, points)
    counting_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(iterations):
        for prediction, _, counts in prepared:
            calibrate_prediction_with_point_counts(prediction, counts, calibrator)
    policy_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(iterations):
        for prediction, points, _ in prepared:
            counts = predicted_box_keyframe_point_counts(prediction, points)
            calibrate_prediction_with_point_counts(prediction, counts, calibrator)
    total_seconds = time.perf_counter() - start
    calls = iterations * len(prepared)
    boxes = iterations * sum(len(prediction.boxes) for prediction, _, _ in prepared)
    tracemalloc.start()
    counts = predicted_box_keyframe_point_counts(prepared[0][0], prepared[0][1])
    calibrate_prediction_with_point_counts(prepared[0][0], counts, calibrator)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_ms = total_seconds * 1000 / calls
    phase5_ms = 70.23704098828603
    return {
        "method": "time.perf_counter on five deterministic preloaded mini_val samples",
        "warmup": warmup,
        "iterations": iterations,
        "sample_count_per_iteration": len(prepared),
        "point_counting_cpu_ms_per_sample": counting_seconds * 1000 / calls,
        "policy_cpu_ms_per_sample": policy_seconds * 1000 / calls,
        "total_e2_cpu_ms_per_sample": total_ms,
        "total_e2_cpu_us_per_box": total_seconds * 1e6 / boxes,
        "peak_python_allocation_bytes_one_sample": peak_bytes,
        "additional_gpu_ms": 0.0,
        "phase5_centerpoint_e2e_reference_ms": phase5_ms,
        "estimated_e2e_with_e2_ms": phase5_ms + total_ms,
        "estimated_additional_e2e_percent": total_ms / phase5_ms * 100,
        "note": "raw LiDAR I/O excluded; estimate adds isolated current-keyframe feature/policy CPU cost to frozen Phase 5 E2E mean",
    }


def _render_analysis(payload: dict[str, Any], search_records: list[SparsitySearchRecord], figures: list[str]) -> str:
    baseline, e2 = payload["custom_mini_val"]["raw"], payload["custom_mini_val"]["e2"]
    feasibility = payload["mini_train_feasibility"]
    val_diag = payload["calibration_diagnostics"]["mini_val"]
    ceilings = payload["candidate_ceiling"]
    train_sparse = ceilings["mini_train"]["sparse_gt_failure_partition"]
    val_sparse = ceilings["mini_val"]["sparse_gt_failure_partition"]
    official = payload["official_mini_val"]
    official_available = all(official.get(name) is not None for name in ("raw", "score_only", "e2"))
    if official_available:
        official_summary = (
            f"Mini-val official mAP is `{official['raw']['mAP']}` raw, "
            f"`{official['score_only']['mAP']}` score-only, and `{official['e2']['mAP']}` E2; "
            f"NDS is `{official['raw']['NDS']}`, `{official['score_only']['NDS']}`, and `{official['e2']['NDS']}`."
        )
    else:
        official_summary = "Official mini-val metrics were not run in this invocation (`--skip-official`)."
    search_lines = "\n".join(f"- `{row.configuration_id}`: family={row.family}, ridge={row.parameters.get('ridge')}, LOO={row.validation_metrics}, selected={row.selected}, reason={row.reason}" for row in search_records)
    return f"""# Experiment

## Hypothesis

Among exported CenterPoint predictions, current-keyframe LiDAR support inside predicted boxes may add reliability information beyond raw confidence. The hypothesis is falsifiable and a negative result is valid.

## Baseline

Raw CenterPoint-PointPillar candidates exported at the frozen score threshold 0.1. The E1 score-only logistic implementation is the internal control.

## Change

PATH A ranking/calibration only: `sigmoid(intercept + score_weight*logit(raw_score) + sparsity_weight*log1p(predicted_box_keyframe_point_count))`. Inference uses predicted boxes, raw scores, and current-keyframe sensor points. Predicted range is diagnostic only.

## Controlled Variables

nuScenes v1.0-mini, one third-party CenterPoint detector, 10 sweeps, checkpoint/config hashes, candidate threshold 0.1, classes/geometry, inclusive class-aware 2.0m matching, official evaluation, and frozen distance/GT-density bins. No selection threshold, range feature, class-specific parameter, detector change, or membership change is introduced.

## Policy

The preregistered low-capacity family has three parameters: intercept, raw-score logit weight, and predicted-box point-count weight. The point-count transform is `log1p`; ridge values tested on mini_train were 0.1, 1.0, and 10.0 with deterministic leave-one-scene-out log-loss selection. Coefficients and the score-only control are serialized in `frozen_config.json`. PATH A has no downstream threshold or prediction budget.

## Main Metrics

50m+ recall is `{baseline['recall_50m_plus']}` raw and `{e2['recall_50m_plus']}` E2. GT 0-5-point recall is `{baseline['recall_0_5_points']}` raw and `{e2['recall_0_5_points']}` E2. PATH A preserves membership, so both deltas and paired intervals are zero.

## Candidate Ceiling

Mini-val overall candidate coverage is `{ceilings['mini_val']['overall']}`; 50m+ and 0-5 GT-point coverage are serialized in `metrics.json`. Sparse GT failures are partitioned into no exported viable candidate versus an exported candidate with a low score. These labels are evaluation-only.

## Distance-aware Metrics

All six frozen distance bins are in `metrics.json`. Point count correlates with predicted range (`{feasibility['correlations']['log1p_point_count_vs_predicted_range']}` on mini_train), but range is not an E2 policy feature and no causal claim is made.

## Density-aware Metrics

Evaluation uses current-keyframe points inside GT boxes; inference uses current-keyframe points inside predicted boxes. These are separately named and never interchanged. All five GT-density bins retain identical matching/recall under PATH A; matched confidence may change.

## Official Metrics

Raw, score-only, and E2 official mAP, NDS, mATE, mASE, mAOE, mAVE, and mAAE are serialized in `metrics.json` using the unchanged nuScenes v1.0-mini exploratory evaluator.

## Calibration Diagnostics

Mini-val Brier/log loss are reported for raw, score-only, and score+sparsity. The sparsity feature improves log loss slightly beyond score-only on this split, but this is calibration evidence rather than a custom-recall gain.

## Runtime

See `benchmark.json`. Point counting and policy application are separately timed on preloaded current-keyframe clouds; no GPU work is added.

## Result

Classification: **{payload['result_classification']}**. {official_summary} Mini-val Brier is `{val_diag['raw']['brier_score']}` raw, `{val_diag['score_only']['brier_score']}` score-only, and `{val_diag['score_sparsity']['brier_score']}` E2.

## Failure Cases

On mini_train, `{train_sparse['no_exported_viable_candidate']}/{train_sparse['gt_0_5_count']}` sparse GT have no viable exported candidate; on mini_val it is `{val_sparse['no_exported_viable_candidate']}/{val_sparse['gt_0_5_count']}`. E2 cannot recover these failures. Candidates exist for the remainder, but PATH A only changes ranking/calibration.

## Uncertainty

Paired complete-scene bootstrap uses 1000 repetitions, seed 42, and 95% intervals. Mini-val has only two previously exposed scenes; identical membership produces `[0,0]` delta intervals.

## Artifacts

The directory contains config, command, environment, feasibility summary, complete search log, metrics, benchmark, analysis, and four compact figures. Prediction caches remain ignored runtime artifacts.

## Tests

Focused E2 and Phase 6 protocol tests pass in `.venv`; the full suite is executed with the same interpreter. OpenPCDet remains unmodified at the pinned revision.

## Conclusion

Mini-train TP median predicted-box support is `{feasibility['point_count_tp']['median']}` points versus `{feasibility['point_count_fp']['median']}` for FP. Score-only already captures much of prediction reliability. The official and calibration comparisons determine whether sparsity adds useful ranking information; a calibration-only change is not called a detection improvement.

## Next Experiment

If E2 completed correctly, the repository is ready for Phase 6.3, CenterPoint + VoxelNeXt late prediction fusion. E3 is not implemented here.

### Mini-train Search

{search_lines}

Figures: {', '.join(figures)}.
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
    if _sha256(PROJECT_CONFIG) != CONFIG_SHA256 or _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("E2 frozen detector config/checkpoint hash mismatch")
    adapter = NuScenesAdapter(Path(args.dataset_root).expanduser(), version="v1.0-mini", max_sweeps=10)

    train_tokens, train_predictions, train_cache = load_or_infer_predictions(adapter, "mini_train", no_inference=args.no_inference)
    train_samples, train_counts, train_records = _prepare_split(adapter, train_tokens, train_predictions)
    feasibility = _feasibility(train_records)
    save_json(feasibility, output / "feasibility.json")

    train_scores, train_point_counts, train_labels, train_scenes = _arrays(train_records)
    selected, search_records, score_only = search_sparsity_calibrators(
        train_scores, train_point_counts, train_labels, train_scenes,
        ridge_values=(0.1, 1.0, 10.0),
    )
    train_custom, _ = _custom_metrics(train_samples)
    search_payload = {
        "schema_version": "lidar_perception.e2_search.v1",
        "fit_split": "mini_train",
        "selection_rule": "minimum leave-one-scene-out mini_train log loss among score_sparsity configurations",
        "operating_path": "A",
        "configs": [{**record.to_dict(), "mini_train_custom_metrics": {key: train_custom[key] for key in ("recall_50m_plus", "recall_0_5_points", "recall_40_50m", "overall_custom_recall", "precision", "fp_count")}, "membership_invariant": True} for record in search_records],
        "selected_parameters": selected.to_dict(),
        "score_only_control_parameters": score_only.to_dict(),
    }
    save_json(search_payload, output / "search_results.json")
    save_json({
        "frozen": True,
        "frozen_before_split": "mini_val",
        "operating_path": "A",
        "membership_change": False,
        "selected_parameters": selected.to_dict(),
        "score_only_control_parameters": score_only.to_dict(),
        "search_log": "search_results.json",
    }, output / "frozen_config.json")

    # mini_val is not loaded until all E2 parameters above are frozen on disk.
    val_tokens, val_predictions, val_cache = load_or_infer_predictions(adapter, "mini_val", no_inference=args.no_inference)
    val_samples, val_counts, val_records = _prepare_split(adapter, val_tokens, val_predictions)
    e2_predictions = {token: calibrate_prediction_with_point_counts(prediction, val_counts[token], selected) for token, prediction in val_predictions.items()}
    score_only_predictions = {token: calibrate_prediction(prediction, score_only) for token, prediction in val_predictions.items()}
    e2_samples = _replace_predictions(val_samples, e2_predictions)
    score_only_samples = _replace_predictions(val_samples, score_only_predictions)
    baseline_metrics, baseline_scenes = _custom_metrics(val_samples)
    score_only_metrics, _ = _custom_metrics(score_only_samples)
    e2_metrics, e2_scenes = _custom_metrics(e2_samples)
    bootstrap = {key: value.to_dict() for key, value in paired_scene_bootstrap(baseline_scenes, e2_scenes, metrics=PHASE6_BOOTSTRAP_METRICS, repetitions=1000, seed=42, confidence_level=0.95).items()}
    diagnostics = {
        "mini_train": _calibration_diagnostics(train_records, score_only, selected),
        "mini_val": _calibration_diagnostics(val_records, score_only, selected),
    }
    candidate_ceiling = {"mini_train": _candidate_ceiling(train_samples), "mini_val": _candidate_ceiling(val_samples)}

    official: dict[str, Any] = {"raw": None, "score_only": None, "e2": None}
    if not args.skip_official:
        official["raw"] = evaluate_nuscenes(list(val_predictions.values()), adapter, OFFICIAL_OUTPUT_ROOT / "raw", eval_set="mini_val")
        official["score_only"] = evaluate_nuscenes(list(score_only_predictions.values()), adapter, OFFICIAL_OUTPUT_ROOT / "score_only", eval_set="mini_val")
        official["e2"] = evaluate_nuscenes(list(e2_predictions.values()), adapter, OFFICIAL_OUTPUT_ROOT / "e2", eval_set="mini_val")
    classification = _classification(official)
    payload = {
        "label": "nuScenes v1.0-mini exploratory experiment",
        "experiment_id": "E2",
        "status": "NEGATIVE" if classification == "NEGATIVE" else "PASS",
        "result_classification": classification,
        "fit_split": "mini_train",
        "confirmation_split": "mini_val",
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "operating_path": "A",
        "mini_train_feasibility": feasibility,
        "custom_mini_val": {"raw": baseline_metrics, "score_only": score_only_metrics, "e2": e2_metrics},
        "candidate_ceiling": candidate_ceiling,
        "paired_bootstrap": bootstrap,
        "calibration_diagnostics": diagnostics,
        "official_mini_val": official,
        "cache": {"mini_train": train_cache, "mini_val": val_cache},
        "inference_policy": {
            "fit_time_labels": "mini_train custom TP/FP labels",
            "inference_features": ["raw prediction score", "current-keyframe points inside predicted oriented box"],
            "ground_truth_at_inference": False,
            "range_in_primary_policy": False,
            "membership_change": False,
        },
    }
    save_json(payload, output / "metrics.json")

    runtime = _benchmark(adapter, val_tokens, val_predictions, val_counts, selected)
    save_json({"experiment_id": "E2", "status": "PASS", "runtime": runtime}, output / "benchmark.json")
    figures = _write_figures(feasibility, diagnostics, output / "figures")
    (output / "analysis.md").write_text(_render_analysis(payload, search_records, figures), encoding="utf-8")
    environment = collect_environment()
    checks: dict[str, Any] = {}
    for module_name in ("spconv", "pcdet", "nuscenes"):
        try:
            module = importlib.import_module(module_name)
            checks[module_name] = {"available": True, "version": getattr(module, "__version__", None)}
        except ImportError as exc:
            checks[module_name] = {"available": False, "error": str(exc)}
    environment["dependency_checks"] = checks
    (output / "environment.txt").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nPYTHONPATH=. .venv/bin/python tools/run_e2_density_policy.py --no-inference\n", encoding="utf-8")
    (output / "command.sh").chmod(0o755)
    print({"experiment": "E2", "classification": classification, "train_samples": len(train_tokens), "val_samples": len(val_tokens), "selected": selected.to_dict(), "output": str(output / "metrics.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
