#!/usr/bin/env python3
"""Run the project-owned Phase 4 distance/density and bad-case analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from lidar_perception.analysis.badcase_mining import mine_bad_cases
from lidar_perception.analysis.report import flatten_metrics, generate_figures, write_csv, write_json
from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter
from lidar_perception.detection.schemas import PredictionBatch
from lidar_perception.evaluation.density_eval import DensityAwareEvaluator
from lidar_perception.evaluation.distance_eval import DistanceAwareEvaluator
from lidar_perception.evaluation.matching import SampleEvaluation, box_to_dict, match_prediction_to_ground_truth
from lidar_perception.evaluation.metrics import DEFAULT_DENSITY_BINS, DEFAULT_DISTANCE_BINS, coerce_bins, find_bin
from lidar_perception.geometry.boxes3d import count_points_in_box
from lidar_perception.utils.config import load_yaml_config
from lidar_perception.utils.io import save_json

from common import load_pointpillar_config, make_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/phase4_nuscenes.yaml")
    parser.add_argument("--no-inference", action="store_true", help="Fail instead of filling missing cached predictions")
    return parser


def _path(value: str | Path) -> Path:
    target = Path(value).expanduser()
    return target if target.is_absolute() else (Path.cwd() / target).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_cached_prediction(path: Path, token: str, backend_name: str, threshold: float) -> PredictionBatch | None:
    if not path.is_file():
        return None
    try:
        prediction = PredictionBatch.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    metadata = prediction.metadata
    if prediction.frame_id != token or metadata.get("sample_token", token) != token:
        return None
    if metadata.get("backend") != backend_name:
        return None
    cached_threshold = metadata.get("score_threshold")
    if cached_threshold is not None and not np.isclose(float(cached_threshold), threshold):
        return None
    return prediction


def _current_keyframe_points(points: np.ndarray) -> np.ndarray:
    if points.shape[1] < 5:
        return points[:, :3]
    return points[np.isclose(points[:, 4], 0.0, atol=1e-7), :3]


def _relationship_rows(samples: list[SampleEvaluation], distance_bins: tuple, density_bins: tuple) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        matches = {item.gt_index: item for item in sample.match.matches}
        for gt_index, (box, point_count) in enumerate(zip(sample.ground_truth, sample.gt_point_counts)):
            distance_m = float(np.linalg.norm(box.center[:2]))
            distance_bin = find_bin(distance_m, distance_bins)
            density_bin = find_bin(float(point_count), density_bins)
            match = matches.get(gt_index)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "gt_index": gt_index,
                    "class": box.label,
                    "distance_m": distance_m,
                    "distance_bin": None if distance_bin is None else distance_bin.name,
                    "gt_point_count": point_count,
                    "density_bin": None if density_bin is None else density_bin.name,
                    "match_state": "TP" if match is not None else "FN",
                    "prediction_index": None if match is None else match.prediction_index,
                    "confidence": None if match is None else match.prediction_score,
                    "localization_error_m": None if match is None else match.localization_error_m,
                }
            )
    return rows


def _relationship_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    distances = np.asarray([row["distance_m"] for row in rows], dtype=np.float64)
    points = np.asarray([row["gt_point_count"] for row in rows], dtype=np.float64)
    correlation = None
    if len(rows) >= 2 and np.std(distances) > 0 and np.std(points) > 0:
        correlation = float(np.corrcoef(distances, points)[0, 1])
    return {
        "gt_count": len(rows),
        "matched_gt_count": sum(row["match_state"] == "TP" for row in rows),
        "missed_gt_count": sum(row["match_state"] == "FN" for row in rows),
        "pearson_distance_vs_current_keyframe_points": correlation,
        "interpretation": "exploratory association only; mini data does not establish causality",
    }


def _render_bad_cases(cases, samples_by_id: dict[str, SampleEvaluation], adapter: NuScenesAdapter, output_dir: Path) -> list[dict[str, Any]]:
    from tools.visualize_nuscenes import _plot_3d, _plot_bev

    rows: list[dict[str, Any]] = []
    category_ranks: dict[str, int] = {}
    for case in cases:
        category_ranks[case.category] = category_ranks.get(case.category, 0) + 1
        rank = category_ranks[case.category]
        sample = samples_by_id[case.sample_id]
        frame = adapter.load_sample(case.sample_id, max_sweeps=adapter.max_sweeps)
        stem = f"{case.category}_{rank:02d}_{case.sample_id}"
        bev_path = output_dir / "bad_cases" / f"{stem}_bev.png"
        _plot_bev(frame, sample.ground_truth, sample.prediction.boxes, bev_path)
        three_d_path = None
        if rank == 1:
            three_d_path = output_dir / "bad_cases" / f"{stem}_3d.png"
            _plot_3d(frame, sample.ground_truth, sample.prediction.boxes, three_d_path, max_points=50000)
        row = case.to_dict()
        row["rank"] = rank
        row["snapshot_bev"] = str(bev_path)
        row["snapshot_3d"] = None if three_d_path is None else str(three_d_path)
        rows.append(row)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config)
    dataset_cfg = config["dataset"]
    detector_cfg = config["detector"]
    analysis_cfg = config["analysis"]
    detector_project_config = _path(detector_cfg["config"])
    project_config, opcdet_config = load_pointpillar_config(detector_project_config)
    backend_cfg = project_config["backend"]
    threshold = float(analysis_cfg.get("matching_threshold_m", 2.0))
    distance_bins = coerce_bins(analysis_cfg.get("distance_bins"), DEFAULT_DISTANCE_BINS)
    density_bins = coerce_bins(analysis_cfg.get("density_bins"), DEFAULT_DENSITY_BINS)
    adapter = NuScenesAdapter(_path(dataset_cfg["root"]), version=dataset_cfg.get("version", "v1.0-mini"), max_sweeps=int(dataset_cfg.get("max_sweeps", 10)))
    from lidar_perception.evaluation.nuscenes import evaluation_sample_tokens

    tokens = evaluation_sample_tokens(adapter, eval_set=dataset_cfg.get("split", "mini_val"))
    predictions_dir = _path(detector_cfg.get("predictions_dir", "outputs/phase3_centerpoint/predictions"))
    output_dir = _path(analysis_cfg.get("output_dir", "outputs/phase4_analysis"))
    predictions_dir.mkdir(parents=True, exist_ok=True)
    expected_backend = "openpcdet_centerpoint"
    expected_threshold = float(backend_cfg.get("score_threshold", 0.1))
    predictions: dict[str, PredictionBatch] = {}
    missing_tokens: list[str] = []
    for token in tokens:
        prediction = _valid_cached_prediction(predictions_dir / f"{token}.json", token, expected_backend, expected_threshold)
        if prediction is None:
            missing_tokens.append(token)
        else:
            predictions[token] = prediction
    if missing_tokens and args.no_inference:
        raise RuntimeError(f"missing or incompatible cached predictions: {len(missing_tokens)}")
    if missing_tokens:
        backend = make_backend(project_config, opcdet_config)
        backend.load()
        for index, token in enumerate(missing_tokens, start=1):
            frame = adapter.load_sample(token, max_sweeps=adapter.max_sweeps)
            prediction = backend.predict(frame)
            save_json(prediction.to_dict(), predictions_dir / f"{token}.json")
            predictions[token] = prediction
            if index % 10 == 0 or index == len(missing_tokens):
                print(f"inference: {index}/{len(missing_tokens)}")

    samples: list[SampleEvaluation] = []
    for index, token in enumerate(tokens, start=1):
        frame = adapter.load_sample(token, max_sweeps=adapter.max_sweeps)
        gt_boxes = adapter.load_boxes(token)
        current_points = _current_keyframe_points(frame.points)
        point_counts = [count_points_in_box(current_points, box) for box in gt_boxes]
        prediction = predictions[token]
        match = match_prediction_to_ground_truth(prediction, gt_boxes, distance_threshold_m=threshold, gt_point_counts=point_counts)
        samples.append(SampleEvaluation(prediction, gt_boxes, point_counts, match))
        if index % 20 == 0 or index == len(tokens):
            print(f"analysis: {index}/{len(tokens)}")

    distance_metrics = DistanceAwareEvaluator(bins=distance_bins, distance_threshold_m=threshold).evaluate(samples)
    density_metrics = DensityAwareEvaluator(bins=density_bins, distance_threshold_m=threshold).evaluate(samples)
    relationship_rows = _relationship_rows(samples, distance_bins, density_bins)
    for row in relationship_rows:
        row["matching_strategy"] = "center_distance"
        row["matching_threshold_m"] = threshold
        row["density_policy"] = analysis_cfg.get("density_policy", "current keyframe only (time_lag == 0)")
    relationship_summary = _relationship_summary(relationship_rows)
    cases, bad_case_counts = mine_bad_cases(
        samples,
        max_per_category=int(analysis_cfg.get("max_cases_per_category", 5)),
        low_confidence_threshold=float(analysis_cfg.get("low_confidence_threshold", 0.3)),
        high_localization_error_m=float(analysis_cfg.get("high_localization_error_m", 1.0)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bad_case_rows = _render_bad_cases(cases, {sample.sample_id: sample for sample in samples}, adapter, output_dir)
    for row in bad_case_rows:
        row["matching_strategy"] = "center_distance"
        row["matching_threshold_m"] = threshold
    write_csv(flatten_metrics(distance_metrics), output_dir / "distance_metrics.csv")
    write_csv(flatten_metrics(density_metrics), output_dir / "density_metrics.csv")
    write_csv(relationship_rows, output_dir / "distance_density.csv")
    write_csv(bad_case_rows, output_dir / "bad_cases.csv")
    matching_summary = {
        "protocol": samples[0].match.protocol if samples else {},
        "sample_count": len(samples),
        "matched_count": sum(len(sample.match.matches) for sample in samples),
        "false_positive_count": sum(len(sample.match.false_positives) for sample in samples),
        "false_negative_count": sum(len(sample.match.false_negatives) for sample in samples),
        "samples": [sample.match.to_dict() for sample in samples],
    }
    write_json(matching_summary, output_dir / "matching_summary.json")
    figure_paths = generate_figures(distance_metrics, density_metrics, relationship_rows, output_dir / "figures")
    provenance = {
        "dataset": {"root": str(adapter.root), "version": adapter.version, "split": dataset_cfg.get("split", "mini_val"), "sample_count": len(tokens), "max_sweeps": adapter.max_sweeps},
        "detector": {"name": expected_backend, "config": str(opcdet_config), "checkpoint": str(Path(backend_cfg["checkpoint"]).expanduser().resolve()), "checkpoint_sha256": _sha256(Path(backend_cfg["checkpoint"]).expanduser().resolve()), "score_threshold": expected_threshold, "score_filtering_policy": analysis_cfg.get("score_filtering_policy", f"backend score_threshold ({expected_threshold:g})")},
        "density_policy": analysis_cfg.get("density_policy", "current keyframe only (time_lag == 0)"),
        "random_seed": analysis_cfg.get("random_seed"),
        "prediction_cache": str(predictions_dir),
    }
    write_json(provenance, output_dir / "provenance.json")
    manual_examples = []
    for sample in samples:
        if sample.match.matches and not any(item["state"] == "TP" for item in manual_examples):
            item = sample.match.matches[0]
            manual_examples.append({"state": "TP", **item.to_dict(), "gt_box": box_to_dict(sample.ground_truth[item.gt_index]), "prediction_box": box_to_dict(sample.prediction.boxes[item.prediction_index])})
        if sample.match.false_positives and not any(item["state"] == "FP" for item in manual_examples):
            item = sample.match.false_positives[0]
            manual_examples.append({"state": "FP", **item.to_dict(), "gt_box": None, "prediction_box": box_to_dict(sample.prediction.boxes[item.prediction_index])})
        if sample.match.false_negatives and not any(item["state"] == "FN" for item in manual_examples):
            item = sample.match.false_negatives[0]
            manual_examples.append({"state": "FN", **item.to_dict(), "gt_box": box_to_dict(sample.ground_truth[item.gt_index]), "prediction_box": None})
    write_json({"protocol": matching_summary["protocol"], "examples": manual_examples, "inspection_scope": "real nuScenes v1.0-mini predictions and generated BEV snapshots"}, output_dir / "manual_matching_validation.json")
    summary = {
        "label": "nuScenes v1.0-mini exploratory analysis",
        "provenance": provenance,
        "matching": matching_summary["protocol"],
        "distance": distance_metrics,
        "density": density_metrics,
        "distance_density": relationship_summary,
        "bad_case_counts": bad_case_counts,
        "outputs": {"distance_metrics": str(output_dir / "distance_metrics.csv"), "density_metrics": str(output_dir / "density_metrics.csv"), "distance_density": str(output_dir / "distance_density.csv"), "matching_summary": str(output_dir / "matching_summary.json"), "bad_cases": str(output_dir / "bad_cases.csv"), "figures": [str(path) for path in figure_paths]},
    }
    write_json(summary, output_dir / "summary.json")
    print({"samples": len(samples), "gt": sum(len(sample.ground_truth) for sample in samples), "matched": matching_summary["matched_count"], "fp": matching_summary["false_positive_count"], "fn": matching_summary["false_negative_count"], "bad_cases": bad_case_counts, "output": str(output_dir / "summary.json")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
