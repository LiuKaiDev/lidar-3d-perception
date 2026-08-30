#!/usr/bin/env python3
"""Save nuScenes scene/sample BEV or 3D GT-vs-prediction visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter, NuScenesError
from lidar_perception.detection.schemas import PredictionBatch

from common import load_pointpillar_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detectors/centerpoint/nuscenes_mini.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--scene", help="Scene name or token; saves consecutive samples")
    parser.add_argument("--sample-token", help="Single sample token")
    parser.add_argument("--predictions-dir", default="outputs/phase3_centerpoint/predictions")
    parser.add_argument("--output-dir", default="outputs/phase3_centerpoint/visualizations")
    parser.add_argument("--view", choices=("bev", "3d"), default="bev")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-sweeps", type=int)
    parser.add_argument("--max-points", type=int, default=100000)
    return parser


def _color(index: int, predicted: bool) -> tuple[float, float, float]:
    palette = ((0.9, 0.2, 0.15), (0.1, 0.65, 0.25), (0.15, 0.35, 0.9), (0.9, 0.55, 0.05))
    value = palette[index % len(palette)]
    return tuple(min(1.0, channel * 0.8 + 0.15) for channel in value) if predicted else value


def _draw_bev_box(axis, box, color, linestyle: str, prefix: str) -> None:
    corners = np.vstack((box.bev, box.bev[0]))
    axis.plot(corners[:, 0], corners[:, 1], color=color, linestyle=linestyle, linewidth=1.5)
    score = "" if box.score is None else f" {box.score:.2f}"
    axis.text(box.center[0], box.center[1], f"{prefix}{box.label}{score}", color=color, fontsize=7)
    if box.velocity is not None and np.linalg.norm(box.velocity[:2]) > 1e-3:
        axis.quiver(box.center[0], box.center[1], box.velocity[0], box.velocity[1], color=color, angles="xy", scale_units="xy", scale=1.0, width=0.002)


def _plot_bev(frame, gt_boxes, predicted_boxes, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 10))
    points = frame.points
    axis.scatter(points[:, 0], points[:, 1], s=0.12, c=points[:, 2], cmap="viridis", alpha=0.4, linewidths=0)
    for index, box in enumerate(gt_boxes):
        _draw_bev_box(axis, box, _color(index, False), "-", "G:")
    for index, box in enumerate(predicted_boxes):
        _draw_bev_box(axis, box, _color(index, True), "--", "P:")
    axis.set_title(f"nuScenes {frame.metadata.get('scene_name', '')} | {frame.frame_id} | BEV")
    axis.set_xlabel("LiDAR x (forward, m)")
    axis.set_ylabel("LiDAR y (left, m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_3d(frame, gt_boxes, predicted_boxes, output: Path, max_points: int) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")
    points = frame.points[:, :3]
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=np.int64)]
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.15, c=points[:, 2], cmap="viridis", alpha=0.3)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for boxes, predicted in ((gt_boxes, False), (predicted_boxes, True)):
        for index, box in enumerate(boxes):
            corners = box.corners
            color = _color(index, predicted)
            for start, end in edges:
                axis.plot(*zip(corners[start], corners[end]), color=color, linestyle="--" if predicted else "-", linewidth=1.2)
            axis.text(*box.center, f"{'P:' if predicted else 'G:'}{box.label}", color=color, fontsize=7)
    axis.set_title(f"nuScenes {frame.metadata.get('scene_name', '')} | {frame.frame_id} | 3D")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = build_parser().parse_args()
    if bool(args.scene) == bool(args.sample_token):
        raise SystemExit("provide exactly one of --scene or --sample-token")
    if args.max_samples <= 0 or args.max_points <= 0:
        raise SystemExit("--max-samples and --max-points must be positive")
    try:
        config, _ = load_pointpillar_config(args.config)
        dataset_cfg = config["dataset"]
        max_sweeps = args.max_sweeps if args.max_sweeps is not None else int(dataset_cfg.get("max_sweeps", 10))
        adapter = NuScenesAdapter(args.dataset_root or dataset_cfg["root"], version=dataset_cfg.get("version", "v1.0-mini"), max_sweeps=max_sweeps)
        tokens = [args.sample_token] if args.sample_token else adapter.sample_tokens(args.scene)[: args.max_samples]
        for token in tokens:
            frame = adapter.load_sample(token, max_sweeps=max_sweeps)
            gt_boxes = adapter.load_boxes(token)
            prediction_path = Path(args.predictions_dir).expanduser() / f"{token}.json"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"prediction JSON not found: {prediction_path}")
            predicted_boxes = PredictionBatch.from_dict(json.loads(prediction_path.read_text())).boxes
            suffix = "bev" if args.view == "bev" else "3d"
            output = Path(args.output_dir).expanduser() / f"{token}_gt_pred_{suffix}.png"
            if args.view == "bev":
                _plot_bev(frame, gt_boxes, predicted_boxes, output)
            else:
                _plot_3d(frame, gt_boxes, predicted_boxes, output, args.max_points)
            print(f"saved: {output}")
    except (NuScenesError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"nuScenes visualization failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
