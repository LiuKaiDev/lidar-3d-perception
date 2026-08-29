#!/usr/bin/env python3
"""Compute KITTI ground-truth statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from lidar_perception.datasets.kitti_adapter import KittiAdapter, KittiError
from lidar_perception.datasets.kitti_stats import compute_kitti_statistics
from lidar_perception.utils.config import dataset_config, load_yaml_config
from lidar_perception.utils.io import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets/kitti.yaml", help="YAML dataset config")
    parser.add_argument("--dataset-root", help="Override dataset.root")
    parser.add_argument("--split", help="Override dataset.split")
    parser.add_argument("--max-frames", type=int, help="Analyze only the first N frame IDs")
    parser.add_argument("--output", help="JSON output path; defaults to outputs/kitti_stats_<split>.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        section = dataset_config(load_yaml_config(args.config))
        root = args.dataset_root or section["root"]
        split = args.split or section.get("split", "training")
        classes = set(section.get("classes", [])) or None
        adapter = KittiAdapter(root, split=split)
        frame_ids = adapter.frame_ids()
        if args.max_frames is not None:
            if args.max_frames <= 0:
                raise ValueError("--max-frames must be positive")
            frame_ids = frame_ids[: args.max_frames]
        statistics = compute_kitti_statistics(adapter, frame_ids=frame_ids, classes=classes)
    except (KittiError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"KITTI analysis failed: {exc}") from exc
    output = Path(args.output).expanduser() if args.output else Path("outputs") / f"kitti_stats_{adapter.split}.json"
    save_json(statistics, output)
    print(f"frames: {statistics['frame_count']}")
    print(f"classes: {statistics['class_counts']}")
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
