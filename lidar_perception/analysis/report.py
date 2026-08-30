"""CSV, JSON, and deterministic figures for Phase 4 reports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def write_json(value: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    fieldnames = sorted({key for row in rows_list for key in row})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_list)
    return target


def flatten_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = metrics.get("protocol", {})
    common = {
        "analysis_metric": protocol.get("metric"),
        "matching_strategy": "center_distance",
        "matching_threshold_m": protocol.get("matching_threshold_m"),
    }
    rows: list[dict[str, Any]] = []
    for row in metrics["overall"]:
        rows.append({**common, "scope": "overall", "class": "", **row})
    for label, class_rows in metrics["per_class"].items():
        rows.extend({**common, "scope": "class", "class": label, **row} for row in class_rows)
    return rows


def _plot(x: list[float], y: list[float], labels: list[str], xlabel: str, ylabel: str, title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x, y, marker="o", linewidth=1.8)
    axis.set_xticks(x, labels)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def generate_figures(distance_metrics: dict[str, Any], density_metrics: dict[str, Any], relationship_rows: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    distance_rows = distance_metrics["overall"]
    density_rows = density_metrics["overall"]
    distance_x = [row["lower_m"] + (0.5 if row["upper_m"] == float("inf") else (row["upper_m"] - row["lower_m"]) / 2) for row in distance_rows]
    distance_labels = [row["bin"] for row in distance_rows]
    density_x = [row["lower_points"] + (0.5 if row["upper_points_exclusive"] is None else (row["upper_points_exclusive"] - row["lower_points"]) / 2) for row in density_rows]
    density_labels = [row["bin"] for row in density_rows]
    paths = []
    path = output / "recall_vs_distance.png"
    _plot(distance_x, [row["recall"] or 0.0 for row in distance_rows], distance_labels, "GT center distance bin (m)", "Recall", "Recall vs target distance", path)
    paths.append(path)
    path = output / "localization_error_vs_distance.png"
    _plot(distance_x, [row["matched_localization_error_m"] or 0.0 for row in distance_rows], distance_labels, "GT center distance bin (m)", "Matched localization error (m)", "Localization error vs target distance", path)
    paths.append(path)
    path = output / "recall_vs_point_count.png"
    _plot(density_x, [row["recall"] or 0.0 for row in density_rows], density_labels, "GT points in current keyframe", "Recall", "Recall vs GT point-count bin", path)
    paths.append(path)
    path = output / "average_confidence_vs_point_count.png"
    _plot(density_x, [row["average_matched_confidence"] or 0.0 for row in density_rows], density_labels, "GT points in current keyframe", "Average matched confidence", "Matched confidence vs GT point-count bin", path)
    paths.append(path)
    relationship_distance = {}
    for row in relationship_rows:
        bucket = relationship_distance.setdefault(row["distance_bin"], [0, 0])
        bucket[0] += row["gt_point_count"]
        bucket[1] += 1
    labels = [row["bin"] for row in distance_rows]
    means = [relationship_distance.get(label, [0, 0])[0] / relationship_distance.get(label, [1, 1])[1] for label in labels]
    path = output / "gt_point_count_vs_distance.png"
    _plot(distance_x, means, labels, "GT center distance bin (m)", "Mean current-keyframe GT points", "GT point count vs target distance", path)
    paths.append(path)
    return paths


__all__ = ["flatten_metrics", "generate_figures", "write_csv", "write_json"]
