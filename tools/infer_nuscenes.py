#!/usr/bin/env python3
"""Run CenterPoint on one nuScenes sample through the project adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter, NuScenesError
from lidar_perception.utils.io import save_json

from common import load_pointpillar_config, make_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detectors/centerpoint/nuscenes_mini.yaml")
    parser.add_argument("--checkpoint", help="Override backend.checkpoint")
    parser.add_argument("--dataset-root", help="Override dataset.root")
    parser.add_argument("--version", help="Override dataset.version")
    parser.add_argument("--sample-token", required=True)
    parser.add_argument("--max-sweeps", type=int)
    parser.add_argument("--output", help="Prediction JSON; defaults under outputs/phase3_centerpoint")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config, opcdet_config = load_pointpillar_config(args.config)
        dataset_cfg = config["dataset"]
        version = args.version or dataset_cfg.get("version", "v1.0-mini")
        max_sweeps = args.max_sweeps if args.max_sweeps is not None else int(dataset_cfg.get("max_sweeps", 10))
        adapter = NuScenesAdapter(args.dataset_root or dataset_cfg["root"], version=version, max_sweeps=max_sweeps)
        frame = adapter.load_sample(args.sample_token, max_sweeps=max_sweeps)
        backend = make_backend(config, opcdet_config, args.checkpoint)
        backend.load()
        prediction = backend.predict(frame)
    except (NuScenesError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"nuScenes CenterPoint inference failed: {exc}") from exc
    output = Path(args.output).expanduser() if args.output else Path("outputs/phase3_centerpoint/predictions") / f"{args.sample_token}.json"
    save_json(prediction.to_dict(), output)
    classes = sorted({box.label for box in prediction.boxes})
    top_scores = [round(float(box.score), 4) for box in sorted(prediction.boxes, key=lambda item: item.score or 0.0, reverse=True)[:5]]
    velocity_count = sum(box.velocity is not None for box in prediction.boxes)
    print({"sample_token": prediction.frame_id, "prediction_count": len(prediction.boxes), "classes": classes, "top_scores": top_scores, "velocity_count": velocity_count})
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
