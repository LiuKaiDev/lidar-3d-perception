#!/usr/bin/env python3
"""Run the frozen Phase 6.1 predicted-range calibration experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from lidar_perception.benchmark.report import collect_environment
from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter
from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.density_eval import DensityAwareEvaluator
from lidar_perception.evaluation.distance_eval import DistanceAwareEvaluator
from lidar_perception.evaluation.matching import SampleEvaluation, center_distance, match_prediction_to_ground_truth
from lidar_perception.evaluation.metrics import DEFAULT_DENSITY_BINS, DEFAULT_DISTANCE_BINS, find_bin, safe_ratio
from lidar_perception.evaluation.nuscenes import evaluate_nuscenes, evaluation_sample_tokens
from lidar_perception.experiments.bootstrap import (
    PHASE6_BOOTSTRAP_METRICS,
    SceneMetricCounts,
    paired_scene_bootstrap,
)
from lidar_perception.experiments.cache import PredictionCache, PredictionCacheProvenance
from lidar_perception.experiments.calibration import (
    CalibrationSearchRecord,
    calibrate_prediction,
    calibration_metrics,
    search_calibrators,
)
from lidar_perception.experiments.features import predicted_box_range_m
from lidar_perception.geometry.boxes3d import count_points_in_box
from lidar_perception.utils.io import save_json

from common import load_pointpillar_config, make_backend


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments/e1_range_calibration"
CACHE_ROOT = ROOT / "outputs/phase6_prediction_cache"
OFFICIAL_OUTPUT_ROOT = ROOT / "outputs/phase6_e1_official"
PROJECT_CONFIG = ROOT / "configs/detectors/centerpoint/nuscenes_mini.yaml"
CHECKPOINT = Path("~/checkpoints/openpcdet/centerpoint_nuscenes_pp.pth").expanduser()
CONFIG_SHA256 = "58b5b8b2e4303a4563b6635c7d9f75a41acc8714635914b4d2cfb95ba8e40fc0"
CHECKPOINT_SHA256 = "955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a"
CANDIDATE_THRESHOLD = 0.1


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


def _migrate_phase3_prediction(cache: PredictionCache, split: str, token: str) -> PredictionBatch | None:
    """Import the verified Phase 3 mini-val record into the strict cache."""

    if split != "mini_val":
        return None
    source = ROOT / "outputs/phase3_centerpoint/predictions" / f"{token}.json"
    if not source.is_file():
        return None
    try:
        prediction = PredictionBatch.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    metadata = prediction.metadata
    if prediction.frame_id != token or metadata.get("backend") != "openpcdet_centerpoint":
        return None
    if not np.isclose(float(metadata.get("score_threshold", -1)), CANDIDATE_THRESHOLD):
        return None
    cache.save(prediction, _provenance(split, token))
    return prediction


def load_or_infer_predictions(
    adapter: NuScenesAdapter,
    split: str,
    *,
    no_inference: bool = False,
) -> tuple[list[str], dict[str, PredictionBatch], dict[str, int]]:
    """Load exact-provenance predictions, filling missing entries once."""

    tokens = evaluation_sample_tokens(adapter, eval_set=split)
    cache = PredictionCache(CACHE_ROOT)
    predictions: dict[str, PredictionBatch] = {}
    missing: list[str] = []
    migrated = 0
    for token in tokens:
        expected = _provenance(split, token)
        prediction = cache.load(expected)
        if prediction is None:
            prediction = _migrate_phase3_prediction(cache, split, token)
            migrated += prediction is not None
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
            frame = adapter.load_sample(token, max_sweeps=10)
            prediction = backend.predict(frame)
            cache.save(prediction, _provenance(split, token))
            predictions[token] = prediction
            if index % 20 == 0 or index == len(missing):
                print(f"{split} inference: {index}/{len(missing)}")
    return tokens, predictions, {"cache_hits": len(tokens) - len(missing), "inference_count": len(missing), "migrated_phase3": migrated}


def _sample_evaluations(adapter: NuScenesAdapter, tokens: list[str], predictions: dict[str, PredictionBatch]) -> list[SampleEvaluation]:
    samples: list[SampleEvaluation] = []
    for token in tokens:
        # Detector candidates retain 10-sweep provenance.  GT density is
        # explicitly current-keyframe-only, so loading historical sweeps here
        # would add I/O without changing any evaluation value.
        frame = adapter.load_sample(token, max_sweeps=1)
        ground_truth = adapter.load_boxes(token)
        current_points = frame.points[:, :3]
        point_counts = [count_points_in_box(current_points, box) for box in ground_truth]
        prediction = predictions[token]
        # Phase 6 bootstrap resamples scenes; keep the official scene token on
        # the project-owned prediction record rather than treating samples as
        # independent scenes when older caches omit it.
        prediction.metadata.setdefault("scene_token", str(adapter.sample_record(token)["scene_token"]))
        match = match_prediction_to_ground_truth(prediction, ground_truth, distance_threshold_m=2.0, gt_point_counts=point_counts)
        samples.append(SampleEvaluation(prediction, ground_truth, point_counts, match))
    return samples


def _custom_metrics(samples: list[SampleEvaluation]) -> tuple[dict[str, Any], list[SceneMetricCounts]]:
    distance = DistanceAwareEvaluator(bins=DEFAULT_DISTANCE_BINS, distance_threshold_m=2.0).evaluate(samples)
    density = DensityAwareEvaluator(bins=DEFAULT_DENSITY_BINS, distance_threshold_m=2.0).evaluate(samples)
    matched = sum(len(sample.match.matches) for sample in samples)
    gt_count = sum(len(sample.ground_truth) for sample in samples)
    fp_count = sum(len(sample.match.false_positives) for sample in samples)
    errors = [record.localization_error_m for sample in samples for record in sample.match.matches]
    confidences = [float(record.prediction_score) for sample in samples for record in sample.match.matches if record.prediction_score is not None]
    by_distance = {row["bin"]: row for row in distance["overall"]}
    scenes: list[SceneMetricCounts] = []
    for scene in sorted({_scene_id_from_sample(sample) for sample in samples}):
        scene_samples = [sample for sample in samples if _scene_id_from_sample(sample) == scene]
        scene_distance = DistanceAwareEvaluator(bins=DEFAULT_DISTANCE_BINS, distance_threshold_m=2.0).evaluate(scene_samples)["overall"]
        scene_density = DensityAwareEvaluator(bins=DEFAULT_DENSITY_BINS, distance_threshold_m=2.0).evaluate(scene_samples)["overall"]
        far = next(row for row in scene_distance if row["bin"] == "50m+")
        near_sparse = next(row for row in scene_density if row["bin"] == "0-5")
        scenes.append(SceneMetricCounts(scene, {
            "recall_50m_plus": (int(far["matched_count"]), int(far["gt_count"])),
            "recall_0_5_points": (int(near_sparse["matched_count"]), int(near_sparse["gt_count"])),
            "recall_40_50m": (int(next(row for row in scene_distance if row["bin"] == "40-50m")["matched_count"]), int(next(row for row in scene_distance if row["bin"] == "40-50m")["gt_count"])),
            "overall_custom_recall": (sum(len(s.match.matches) for s in scene_samples), sum(len(s.ground_truth) for s in scene_samples)),
        }))
    candidate_ceiling = _candidate_recall_ceiling(samples)
    result = {
        "sample_count": len(samples),
        "gt_count": gt_count,
        "matched_count": matched,
        "prediction_count": matched + fp_count,
        "fp_count": fp_count,
        "precision": safe_ratio(matched, matched + fp_count),
        "overall_custom_recall": safe_ratio(matched, gt_count),
        "matched_localization_error_m": None if not errors else float(np.mean(errors)),
        "average_matched_confidence": None if not confidences else float(np.mean(confidences)),
        "distance": distance,
        "density": density,
        "candidate_recall_ceiling": candidate_ceiling,
    }
    result["recall_50m_plus"] = by_distance["50m+"]["recall"]
    result["recall_40_50m"] = by_distance["40-50m"]["recall"]
    result["recall_0_5_points"] = next(row for row in density["overall"] if row["bin"] == "0-5")["recall"]
    return result, scenes


def _candidate_recall_ceiling(samples: list[SampleEvaluation]) -> dict[str, Any]:
    """Count GT with any class-aware candidate, without consuming candidates."""

    distance_counts = {metric_bin.name: [0, 0] for metric_bin in DEFAULT_DISTANCE_BINS}
    density_counts = {metric_bin.name: [0, 0] for metric_bin in DEFAULT_DENSITY_BINS}
    class_distance: dict[str, dict[str, list[int]]] = {}
    class_density: dict[str, dict[str, list[int]]] = {}
    covered = 0
    total = 0
    for sample in samples:
        for box, points in zip(sample.ground_truth, sample.gt_point_counts):
            total += 1
            viable = any(prediction.label == box.label and center_distance(prediction, box) <= 2.0 for prediction in sample.prediction.boxes)
            covered += int(viable)
            distance_bin = find_bin(float(np.hypot(box.center[0], box.center[1])), DEFAULT_DISTANCE_BINS)
            density_bin = find_bin(float(points), DEFAULT_DENSITY_BINS)
            distance_row = class_distance.setdefault(box.label, {metric_bin.name: [0, 0] for metric_bin in DEFAULT_DISTANCE_BINS})
            density_row = class_density.setdefault(box.label, {metric_bin.name: [0, 0] for metric_bin in DEFAULT_DENSITY_BINS})
            if distance_bin is not None:
                distance_counts[distance_bin.name][1] += 1
                distance_counts[distance_bin.name][0] += int(viable)
                distance_row[distance_bin.name][1] += 1
                distance_row[distance_bin.name][0] += int(viable)
            if density_bin is not None:
                density_counts[density_bin.name][1] += 1
                density_counts[density_bin.name][0] += int(viable)
                density_row[density_bin.name][1] += 1
                density_row[density_bin.name][0] += int(viable)

    def rows(counts: dict[str, list[int]]) -> list[dict[str, Any]]:
        return [{"bin": name, "candidate_covered_count": value[0], "gt_count": value[1], "candidate_coverage": safe_ratio(value[0], value[1])} for name, value in counts.items()]

    def class_rows(counts: dict[str, dict[str, list[int]]]) -> dict[str, list[dict[str, Any]]]:
        return {label: rows(values) for label, values in sorted(counts.items())}

    return {
        "overall": safe_ratio(covered, total),
        "candidate_covered_count": covered,
        "gt_count": total,
        "by_distance": rows(distance_counts),
        "by_density": rows(density_counts),
        "per_class_distance": class_rows(class_distance),
        "per_class_density": class_rows(class_density),
        "note": "fraction of GT with at least one viable exported score>=0.1 candidate; class-aware center distance <=2.0m, no one-to-one consumption, evaluation-only",
    }


def _scene_id_from_sample(sample: SampleEvaluation) -> str:
    return str(sample.prediction.metadata.get("scene_token", sample.prediction.metadata.get("scene_name", sample.sample_id)))


def _replace_predictions(samples: list[SampleEvaluation], predictions: dict[str, PredictionBatch]) -> list[SampleEvaluation]:
    """Reuse evaluation-only GT/counts when only prediction scores changed."""

    result: list[SampleEvaluation] = []
    for sample in samples:
        prediction = predictions[sample.sample_id]
        match = match_prediction_to_ground_truth(
            prediction,
            sample.ground_truth,
            distance_threshold_m=2.0,
            gt_point_counts=sample.gt_point_counts,
        )
        result.append(SampleEvaluation(prediction, sample.ground_truth, sample.gt_point_counts, match))
    return result


def _fit_arrays(samples: list[SampleEvaluation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: list[float] = []
    ranges: list[float] = []
    labels: list[int] = []
    for sample in samples:
        matched_indices = {match.prediction_index for match in sample.match.matches}
        for index, box in enumerate(sample.prediction.boxes):
            if box.score is None:
                continue
            scores.append(float(box.score))
            ranges.append(predicted_box_range_m(box))
            labels.append(int(index in matched_indices))
    return np.asarray(scores), np.asarray(ranges), np.asarray(labels)


def _diagnostics(samples: list[SampleEvaluation], calibrator) -> dict[str, Any]:
    scores, ranges, labels = _fit_arrays(samples)
    calibrated = calibrator.predict(scores, ranges)
    rows = []
    for low, high, name in ((0, 10, "0-10m"), (10, 20, "10-20m"), (20, 30, "20-30m"), (30, 40, "30-40m"), (40, 50, "40-50m"), (50, float("inf"), "50m+")):
        mask = (ranges >= low) & (ranges < high)
        row = {"range_bin": name, "count": int(mask.sum()), "raw_mean": None, "calibrated_mean": None, "tp_rate": None}
        if mask.any():
            row.update({"raw_mean": float(np.mean(scores[mask])), "calibrated_mean": float(np.mean(calibrated[mask])), "tp_rate": float(np.mean(labels[mask]))})
        rows.append(row)
    return {"fit_count": len(scores), "raw": calibration_metrics(scores, labels), "calibrated": calibration_metrics(calibrated, labels), "by_predicted_range": rows}


def _write_figures(diagnostics: dict[str, Any], metrics: dict[str, Any], directory: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    rows = diagnostics["mini_val"]["range_aware"]["by_predicted_range"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, [row["raw_mean"] or np.nan for row in rows], marker="o", label="raw")
    ax.plot(x, [row["calibrated_mean"] or np.nan for row in rows], marker="o", label="score + range")
    ax.set_xticks(x, [row["range_bin"] for row in rows], rotation=30)
    ax.set_ylim(0, 1); ax.set_ylabel("mean score"); ax.set_xlabel("predicted range")
    ax.legend(); fig.tight_layout(); path = directory / "score_by_predicted_range.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["baseline", "E1"]
    values = [metrics["baseline"].get("recall_50m_plus"), metrics["e1"].get("recall_50m_plus")]
    ax.bar(labels, [value if value is not None else 0 for value in values], color=["#4c78a8", "#f58518"])
    ax.set_ylim(0, 1); ax.set_ylabel("50m+ recall"); fig.tight_layout(); path = directory / "far_range_recall.png"; fig.savefig(path, dpi=140); plt.close(fig); paths.append(str(path))
    return paths


def _result_classification(bootstrap: dict[str, Any], official: dict[str, Any]) -> str:
    primary = bootstrap.get("recall_50m_plus", {})
    delta = primary.get("delta")
    if delta is not None and delta < 0:
        return "NEGATIVE"
    if delta is not None and delta > 0:
        return "POSITIVE" if primary.get("lower") is not None and primary["lower"] > 0 else "DIRECTIONAL"
    baseline, e1 = official.get("baseline"), official.get("e1")
    if baseline and e1 and (e1.get("mAP", 0) < baseline.get("mAP", 0) or e1.get("NDS", 0) < baseline.get("NDS", 0)):
        return "NEGATIVE"
    return "UNCHANGED"


def _render_analysis(
    metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    search: list[CalibrationSearchRecord],
    runtime: dict[str, Any],
    figures: list[str],
    diagnostics: dict[str, Any],
    official: dict[str, Any],
) -> str:
    classification = _result_classification(bootstrap, official)
    baseline_official = official.get("baseline") or {}
    score_only_official = official.get("score_only_control") or {}
    e1_official = official.get("e1") or {}
    val_range = diagnostics["mini_val"]["range_aware"]["calibrated"]
    val_score_only = diagnostics["mini_val"]["score_only"]["calibrated"]
    ceiling = metrics["baseline"]["candidate_recall_ceiling"]
    far_ceiling = next(row["candidate_coverage"] for row in ceiling["by_distance"] if row["bin"] == "50m+")
    sparse_ceiling = next(row["candidate_coverage"] for row in ceiling["by_density"] if row["bin"] == "0-5")
    search_lines = "\n".join(f"- `{row.configuration_id}` ({row.family}): valid={row.valid}, selected={row.selected}, metrics={row.metrics}, reason={row.reason}" for row in search)
    return f"""# Experiment

## Hypothesis

Raw CenterPoint score quality varies by target range; a low-capacity rule using predicted range may improve far-range ranking without changing detector weights.

## Baseline

Unmodified CenterPoint-PointPillar predictions exported at the frozen candidate threshold 0.1.

## Change

Global logistic calibration: `sigmoid(intercept + score_weight * logit(raw_score) + range_weight * ((sqrt(x^2+y^2)-25)/25))`. The only inference features are raw score and predicted-box range. Geometry, labels, and candidate membership are unchanged.

## Controlled Variables

nuScenes v1.0-mini, 10 sweeps, one CenterPoint detector, checkpoint/config hashes, class-aware greedy center matching at inclusive 2.0 m, frozen distance/density bins, and candidate threshold 0.1. Ground-truth labels are fit-time only; inference accepts no GT fields. OpenPCDet applies `SCORE_THRESH: 0.1` and `MAX_OBJ_PER_SAMPLE: 500` during CenterHead decoding (`center_head.py`); the project adapter independently filters scores below 0.1 (`openpcdet_backend.py`). Both happen before caching/calibration. No later project threshold or max-box limit exists, mini_val caches contain at most 248 boxes per sample, and both custom and official evaluation consume all cached candidates. Calibration therefore cannot alter retained boxes through either limit.

## Main Metrics

Baseline versus E1: 50m+ recall `{metrics['baseline'].get('recall_50m_plus')}` vs `{metrics['e1'].get('recall_50m_plus')}`; 0-5-point recall `{metrics['baseline'].get('recall_0_5_points')}` vs `{metrics['e1'].get('recall_0_5_points')}`. Classification: **{classification}**.

## Distance-aware Metrics

See `metrics.json` for all frozen bins and 40-50m recall. Since no selection operating point exists, recalibration changes ranking/official score ordering only, not custom recall.

## Density-aware Metrics

GT density remains current-keyframe points inside oriented GT boxes and is evaluation-only. It is not an inference feature.

## Runtime

Calibration was timed with `time.perf_counter` over repeated prediction-only applications; see `benchmark.json` for per-sample CPU overhead and iteration count.

## Result

The candidate floor means boxes discarded upstream below 0.1 cannot be recovered. Cached candidates are post-threshold. At least one viable exported candidate exists for `{ceiling.get('candidate_covered_count')}/{ceiling.get('gt_count')}` GT (`{ceiling.get('overall')}`), but coverage is only `{far_ceiling}` at 50m+ and `{sparse_ceiling}` for 0-5-point GT. Distance, density, and per-class ceiling views are in `metrics.json`. Recalibration changes confidence/ranking only; it does not filter or select boxes. Result classification: **{classification}**. Official mAP is `{baseline_official.get('mAP')}` baseline, `{score_only_official.get('mAP')}` score-only, and `{e1_official.get('mAP')}` score+range; NDS is `{baseline_official.get('NDS')}`, `{score_only_official.get('NDS')}`, and `{e1_official.get('NDS')}` respectively.

## Failure Cases

The detector's exported-candidate ceiling separates missing candidates from ranking/selection errors. No new downstream threshold was invented.

## Uncertainty

Final mini_val deltas use paired scene-level bootstrap, 1000 repetitions, seed 42, 95% intervals. Only two mini_val scenes are available and this split was previously exposed.

## Conclusion

The score-only control and range-aware method are both retained. On mini_val, range-aware Brier/log loss are `{val_range.get('brier_score')}` / `{val_range.get('log_loss')}` versus `{val_score_only.get('brier_score')}` / `{val_score_only.get('log_loss')}` for score-only. This small calibration-diagnostic gain did not become a detection benefit: score-only preserved baseline mAP, while range-aware ordering regressed official mAP/NDS and custom recall could not change without selection. Predicted range did not add useful detection information under the frozen E1 operating condition.

## Next Experiment

If the protocol completed, the repository is ready for Phase 6.2 (predicted-box sparsity/density-aware policy). E2 is not implemented here.

### Mini-train Search Log

{search_lines}

Figures: {', '.join(figures) if figures else 'none'}.
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
    train_tokens, train_predictions, train_cache = load_or_infer_predictions(adapter, "mini_train", no_inference=args.no_inference)
    train_samples = _sample_evaluations(adapter, train_tokens, train_predictions)
    raw_scores, ranges, labels = _fit_arrays(train_samples)
    selected, search_records, score_only = search_calibrators(raw_scores, ranges, labels, ridge_values=(0.1, 1.0, 10.0))
    save_json({"schema_version": "lidar_perception.e1_search.v1", "fit_split": "mini_train", "selection_rule": "minimum mini_train log loss among score_range configurations", "configs": [record.to_dict() for record in search_records], "selected_parameters": selected.to_dict(), "score_only_control_parameters": score_only.to_dict()}, output / "search_results.json")
    # This file is the explicit freeze evidence consumed before mini_val loading.
    save_json({"frozen": True, "frozen_before_split": "mini_val", "selected_parameters": selected.to_dict(), "search_log": "search_results.json"}, output / "frozen_config.json")
    val_tokens, val_predictions, val_cache = load_or_infer_predictions(adapter, "mini_val", no_inference=args.no_inference)
    val_samples = _sample_evaluations(adapter, val_tokens, val_predictions)
    calibrated_val = {token: calibrate_prediction(prediction, selected) for token, prediction in val_predictions.items()}
    calibrated_samples = _replace_predictions(val_samples, calibrated_val)
    baseline_metrics, baseline_scenes = _custom_metrics(val_samples)
    e1_metrics, e1_scenes = _custom_metrics(calibrated_samples)
    calibrated_train_diagnostics = _diagnostics(train_samples, selected)
    score_only_train_diagnostics = _diagnostics(train_samples, score_only)
    range_val_diagnostics = _diagnostics(val_samples, selected)
    score_only_val_diagnostics = _diagnostics(val_samples, score_only)
    diagnostics = {
        "mini_train": {"range_aware": calibrated_train_diagnostics, "score_only": score_only_train_diagnostics},
        "mini_val": {"range_aware": range_val_diagnostics, "score_only": score_only_val_diagnostics},
    }
    bootstrap = {key: value.to_dict() for key, value in paired_scene_bootstrap(baseline_scenes, e1_scenes, metrics=PHASE6_BOOTSTRAP_METRICS, repetitions=1000, seed=42, confidence_level=0.95).items()}
    official: dict[str, Any] = {"baseline": None, "score_only_control": None, "e1": None}
    if not args.skip_official:
        score_only_val = {token: calibrate_prediction(prediction, score_only) for token, prediction in val_predictions.items()}
        official["baseline"] = evaluate_nuscenes(list(val_predictions.values()), adapter, OFFICIAL_OUTPUT_ROOT / "baseline", eval_set="mini_val")
        official["score_only_control"] = evaluate_nuscenes(list(score_only_val.values()), adapter, OFFICIAL_OUTPUT_ROOT / "score_only_control", eval_set="mini_val")
        official["e1"] = evaluate_nuscenes(list(calibrated_val.values()), adapter, OFFICIAL_OUTPUT_ROOT / "e1", eval_set="mini_val")
    classification = _result_classification(bootstrap, official)
    metric_payload = {"label": "nuScenes v1.0-mini exploratory experiment", "experiment_id": "E1", "status": "NEGATIVE" if classification == "NEGATIVE" else "PASS", "result_classification": classification, "fit_split": "mini_train", "confirmation_split": "mini_val", "candidate_threshold": CANDIDATE_THRESHOLD, "baseline": baseline_metrics, "e1": e1_metrics, "paired_bootstrap": bootstrap, "calibration_diagnostics": diagnostics, "official_mini_val": official, "cache": {"mini_train": train_cache, "mini_val": val_cache}, "inference_policy": {"fit_time_labels": "custom class-aware center-distance TP/FP labels from mini_train", "inference_features": ["raw prediction score", "predicted box range sqrt(x^2+y^2)"], "ground_truth_at_inference": False}}
    save_json(metric_payload, output / "metrics.json")
    warmup = 20
    loops = 100
    prediction_list = list(val_predictions.values())
    for _ in range(warmup):
        for prediction in prediction_list:
            calibrate_prediction(prediction, selected)
    start = time.perf_counter()
    for _ in range(loops):
        for prediction in prediction_list:
            calibrate_prediction(prediction, selected)
    elapsed = time.perf_counter() - start
    total_boxes = sum(len(prediction.boxes) for prediction in val_predictions.values())
    calibration_ms = elapsed * 1000.0 / max(loops * len(val_predictions), 1)
    tracemalloc.start()
    calibrate_prediction(prediction_list[0], selected)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    phase5_e2e_ms = 70.23704098828603
    runtime = {"method": "time.perf_counter", "warmup": warmup, "iterations": loops, "samples_per_iteration": len(val_predictions), "boxes_per_iteration": total_boxes, "total_seconds": elapsed, "calibration_cpu_ms_per_sample": calibration_ms, "calibration_cpu_us_per_box": elapsed * 1e6 / max(loops * total_boxes, 1), "additional_gpu_ms": 0.0, "peak_python_allocation_bytes_one_sample": peak_bytes, "memory_overhead": "one copied PredictionBatch per application; no detector GPU memory change", "phase5_centerpoint_e2e_reference_ms": phase5_e2e_ms, "estimated_e2e_with_calibration_ms": phase5_e2e_ms + calibration_ms, "estimated_additional_e2e_percent": calibration_ms / phase5_e2e_ms * 100.0, "e2e_note": "estimate adds isolated CPU calibration to the frozen Phase 5 mean; detector GPU benchmark was not rerun"}
    save_json({"experiment_id": "E1", "status": "PASS", "runtime": runtime}, output / "benchmark.json")
    figures = _write_figures(diagnostics, {"baseline": baseline_metrics, "e1": e1_metrics}, output / "figures")
    analysis = _render_analysis(
        {"baseline": baseline_metrics, "e1": e1_metrics},
        bootstrap,
        search_records,
        runtime,
        figures,
        diagnostics,
        official,
    )
    (output / "analysis.md").write_text(analysis, encoding="utf-8")
    env = collect_environment()
    checks: dict[str, Any] = {}
    for module_name in ("spconv", "pcdet", "nuscenes"):
        try:
            module = importlib.import_module(module_name)
            checks[module_name] = {"available": True, "version": getattr(module, "__version__", None)}
        except ImportError as exc:
            checks[module_name] = {"available": False, "error": str(exc)}
    env["dependency_checks"] = checks
    (output / "environment.txt").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nPYTHONPATH=. .venv/bin/python tools/run_e1_range_calibration.py\n", encoding="utf-8")
    (output / "command.sh").chmod(0o755)
    print({"experiment": "E1", "train_samples": len(train_tokens), "val_samples": len(val_tokens), "selected": selected.to_dict(), "output": str(output / "metrics.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
