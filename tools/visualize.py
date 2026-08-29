#!/usr/bin/env python3
"""Visualize one KITTI frame in BEV, 3D, or camera image space."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lidar_perception.datasets.kitti_adapter import KittiAdapter, KittiError
from lidar_perception.geometry.boxes3d import Box3D
from lidar_perception.geometry.projection import project_box_to_image, project_points_to_image
from lidar_perception.utils.config import dataset_config, load_yaml_config


BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets/kitti.yaml", help="YAML dataset config")
    parser.add_argument("--dataset-root", help="Override dataset.root")
    parser.add_argument("--split", help="Override dataset.split")
    parser.add_argument("--frame-id", required=True, help="KITTI frame stem, for example 000123")
    parser.add_argument("--view", choices=("bev", "3d", "image"), default="bev")
    parser.add_argument("--output", help="Save a figure instead of opening an interactive window")
    parser.add_argument("--max-points", type=int, default=150000, help="Maximum points drawn in a 3D view")
    return parser


def _box_color(index: int) -> tuple[float, float, float]:
    palette = ((0.95, 0.25, 0.2), (0.2, 0.75, 0.3), (0.2, 0.45, 0.95), (0.95, 0.65, 0.15))
    return palette[index % len(palette)]


def _finish_figure(figure, output: str | None) -> None:
    if output:
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"saved: {output_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


def _plot_bev(frame, boxes: list[Box3D], output: str | None) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 9))
    points = frame.points
    axis.scatter(points[:, 0], points[:, 1], s=0.15, c=points[:, 2], cmap="viridis", alpha=0.55, linewidths=0)
    for index, box in enumerate(boxes):
        corners = box.bev
        closed = np.vstack((corners, corners[0]))
        color = _box_color(index)
        axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.8)
        axis.text(box.center[0], box.center[1], box.label, color=color, fontsize=8)
    axis.set_title(f"KITTI {frame.frame_id} | BEV")
    axis.set_xlabel("LiDAR x (forward, m)")
    axis.set_ylabel("LiDAR y (left, m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    _finish_figure(figure, output)


def _plot_3d_matplotlib(frame, boxes: list[Box3D], output: str, max_points: int) -> None:
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")
    points = frame.points
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2, c=points[:, 2], cmap="viridis", alpha=0.35)
    for index, box in enumerate(boxes):
        corners = box.corners
        color = _box_color(index)
        for start, end in BOX_EDGES:
            axis.plot(*zip(corners[start], corners[end]), color=color, linewidth=1.4)
        axis.text(*box.center, box.label, color=color, fontsize=8)
    axis.set_title(f"KITTI {frame.frame_id} | 3D")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    _finish_figure(figure, output)


def _plot_3d_open3d(frame, boxes: list[Box3D]) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is not installed; pass --output to use the matplotlib 3D fallback") from exc
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(frame.points[:, :3])
    point_cloud.paint_uniform_color((0.65, 0.65, 0.65))
    geometries = [point_cloud]
    for index, box in enumerate(boxes):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box.corners)
        line_set.lines = o3d.utility.Vector2iVector(BOX_EDGES)
        line_set.colors = o3d.utility.Vector3dVector([_box_color(index)] * len(BOX_EDGES))
        geometries.append(line_set)
    o3d.visualization.draw_geometries(geometries, window_name=f"KITTI {frame.frame_id} | 3D")


def _plot_image(adapter: KittiAdapter, frame, boxes: list[Box3D], output: str | None) -> None:
    import cv2
    import matplotlib.pyplot as plt

    image = cv2.cvtColor(adapter.load_image(frame.frame_id), cv2.COLOR_BGR2RGB)
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.imshow(image)
    calibration = adapter.load_calibration(frame.frame_id)
    points_result = project_points_to_image(frame.points[:, :3], calibration, image_shape=image.shape[:2])
    mask = points_result.inside_image if points_result.inside_image is not None else points_result.valid_mask
    axis.scatter(points_result.pixels[mask, 0], points_result.pixels[mask, 1], s=0.4, c=points_result.depth[mask], cmap="turbo", alpha=0.55)
    for index, box in enumerate(boxes):
        result = project_box_to_image(box, calibration, image_shape=image.shape[:2])
        color = _box_color(index)
        for start, end in BOX_EDGES:
            if result.valid_mask[start] and result.valid_mask[end]:
                axis.plot([result.pixels[start, 0], result.pixels[end, 0]], [result.pixels[start, 1], result.pixels[end, 1]], color=color, linewidth=1.6)
        if result.bbox is not None:
            axis.text(result.bbox[0], result.bbox[1], box.label, color=color, fontsize=8)
    axis.set_title(f"KITTI {frame.frame_id} | image_2 projection")
    axis.set_axis_off()
    _finish_figure(figure, output)


def main() -> int:
    args = build_parser().parse_args()
    if args.max_points <= 0:
        raise SystemExit("KITTI visualization failed: --max-points must be positive")
    try:
        section = dataset_config(load_yaml_config(args.config))
        adapter = KittiAdapter(args.dataset_root or section["root"], split=args.split or section.get("split", "training"))
        frame = adapter.load_frame(args.frame_id)
        classes = set(section.get("classes", [])) or None
        boxes = adapter.load_boxes(args.frame_id, classes=classes)
    except (KittiError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"KITTI visualization failed: {exc}") from exc
    if args.view == "bev":
        _plot_bev(frame, boxes, args.output)
    elif args.view == "image":
        _plot_image(adapter, frame, boxes, args.output)
    elif args.output:
        _plot_3d_matplotlib(frame, boxes, args.output, args.max_points)
    else:
        _plot_3d_open3d(frame, boxes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
