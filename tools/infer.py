#!/usr/bin/env python3
"""Run one real KITTI frame through the PointPillars backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from lidar_perception.datasets.kitti_adapter import KittiAdapter, KittiError
from lidar_perception.utils.config import load_yaml_config

from common import load_pointpillar_config, make_backend, write_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detectors/pointpillar/kitti.yaml")
    parser.add_argument("--checkpoint", help="Override backend.checkpoint")
    parser.add_argument("--dataset-root", help="Override dataset.root")
    parser.add_argument("--split", default="training", help="KITTI split used to locate the frame")
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output", help="Prediction JSON; defaults to outputs/predictions/<frame>.json")
    parser.add_argument("--score-threshold", type=float, help="Override visualization/serialization threshold")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config, opcdet_config = load_pointpillar_config(args.config)
        dataset_cfg = config["dataset"]
        root = args.dataset_root or dataset_cfg["root"]
        adapter = KittiAdapter(root, split=args.split or dataset_cfg.get("split", "training"))
        frame = adapter.load_frame(args.frame_id)
        backend = make_backend(config, opcdet_config, args.checkpoint)
        if args.score_threshold is not None:
            backend.score_threshold = backend._validate_score_threshold(args.score_threshold)
        backend.load()
        prediction = backend.predict(frame)
    except (KittiError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"PointPillars inference failed: {exc}") from exc
    output = Path(args.output).expanduser() if args.output else Path("outputs/predictions") / f"{args.frame_id}.json"
    write_prediction(prediction, output)
    print(f"frame: {prediction.frame_id}")
    print(f"predictions: {len(prediction.boxes)}")
    for box in prediction.boxes[:10]:
        print(f"{box.label} score={box.score:.4f} center={box.center.round(3).tolist()} size={box.size.round(3).tolist()} yaw={box.yaw:.4f}")
    print(f"runtime_ms: {prediction.runtime_ms:.3f}")
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
