#!/usr/bin/env python3
"""Run CenterPoint inference and official nuScenes mini evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter, NuScenesError
from lidar_perception.evaluation.nuscenes import evaluate_nuscenes, evaluation_sample_tokens

from common import load_pointpillar_config, make_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detectors/centerpoint/nuscenes_mini.yaml")
    parser.add_argument("--checkpoint", help="Override backend.checkpoint")
    parser.add_argument("--dataset-root", help="Override dataset.root")
    parser.add_argument("--output-dir", default="outputs/phase3_centerpoint/evaluation")
    parser.add_argument("--max-sweeps", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config, opcdet_config = load_pointpillar_config(args.config)
        dataset_cfg = config["dataset"]
        max_sweeps = args.max_sweeps if args.max_sweeps is not None else int(dataset_cfg.get("max_sweeps", 10))
        adapter = NuScenesAdapter(args.dataset_root or dataset_cfg["root"], version=dataset_cfg.get("version", "v1.0-mini"), max_sweeps=max_sweeps)
        backend = make_backend(config, opcdet_config, args.checkpoint)
        backend.load()
        tokens = evaluation_sample_tokens(adapter, eval_set=config.get("evaluation", {}).get("split", "mini_val"))
        predictions = []
        for index, token in enumerate(tokens, start=1):
            frame = adapter.load_sample(token, max_sweeps=max_sweeps)
            predictions.append(backend.predict(frame))
            if index % 10 == 0 or index == len(tokens):
                print(f"inference: {index}/{len(tokens)}")
        result = evaluate_nuscenes(predictions, adapter, args.output_dir, eval_set=config.get("evaluation", {}).get("split", "mini_val"))
    except (NuScenesError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"nuScenes evaluation failed: {exc}") from exc
    print({key: result[key] for key in ("label", "sample_count", "protocol", "mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE")})
    print(f"saved: {Path(args.output_dir).expanduser() / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
