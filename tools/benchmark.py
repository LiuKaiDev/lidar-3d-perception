#!/usr/bin/env python3
"""Benchmark PointPillars latency and peak CUDA memory on one KITTI frame."""

from __future__ import annotations

import argparse
from pathlib import Path

from lidar_perception.benchmark.latency import benchmark_pointpillar
from lidar_perception.datasets.kitti_adapter import KittiAdapter, KittiError
from lidar_perception.utils.io import save_json

from common import load_pointpillar_config, make_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detectors/pointpillar/kitti.yaml")
    parser.add_argument("--checkpoint", help="Override backend.checkpoint")
    parser.add_argument("--dataset-root", help="Override dataset.root")
    parser.add_argument("--split", default="training")
    parser.add_argument("--frame-id", default="004139")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output", default="outputs/metrics/pointpillar_benchmark.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config, opcdet_config = load_pointpillar_config(args.config)
        backend = make_backend(config, opcdet_config, args.checkpoint)
        backend.load()
        adapter = KittiAdapter(args.dataset_root or config["dataset"]["root"], split=args.split)
        frame = adapter.load_frame(args.frame_id)
        benchmark_cfg = config.get("benchmark", {})
        result = benchmark_pointpillar(
            backend,
            frame,
            warmup=args.warmup if args.warmup is not None else int(benchmark_cfg.get("warmup", 20)),
            iterations=args.iterations if args.iterations is not None else int(benchmark_cfg.get("iterations", 100)),
        )
    except (KittiError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"PointPillars benchmark failed: {exc}") from exc
    output = save_json(result, args.output)
    print(result)
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
