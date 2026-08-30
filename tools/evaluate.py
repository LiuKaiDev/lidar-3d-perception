#!/usr/bin/env python3
"""Run official OpenPCDet KITTI evaluation for PointPillars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lidar_perception.evaluation.official import evaluate_kitti

from common import load_pointpillar_config, make_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detectors/pointpillar/kitti.yaml")
    parser.add_argument("--checkpoint", help="Override backend.checkpoint")
    parser.add_argument("--dataset-root", help="Override dataset.root")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", default="outputs/metrics/pointpillar_kitti")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config, opcdet_config = load_pointpillar_config(args.config)
        backend = make_backend(config, opcdet_config, args.checkpoint)
        backend.load()
        result = evaluate_kitti(
            backend,
            args.dataset_root or config["dataset"]["root"],
            split=args.split,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"KITTI evaluation failed: {exc}") from exc
    output = Path(args.output_dir).expanduser() / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"protocol: {result['protocol']}")
    print(result["ap_result"] or "No evaluation result")
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
