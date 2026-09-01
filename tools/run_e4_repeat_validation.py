#!/usr/bin/env python3
"""Repeat frozen E3 confirmation and final four-way ablation for Phase 6.4."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lidar_perception.benchmark.report import collect_environment
from lidar_perception.datasets.nuscenes_adapter import NuScenesAdapter
from lidar_perception.evaluation.nuscenes import evaluate_nuscenes, evaluation_sample_tokens
from lidar_perception.experiments.cache import PredictionCache
from lidar_perception.experiments.fusion import FusionConfig, fuse_predictions, naive_union
from lidar_perception.utils.io import save_json

from run_e3_late_fusion import (
    CACHE_ROOT,
    CP_CHECKPOINT_SHA256,
    VN_CHECKPOINT_SHA256,
    _custom_metrics,
    _load_cache,
    _load_centerpoint,
    _prepare_ground_truth,
    _provenance,
    _samples,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments/e4_repeat_validation"
OFFICIAL_ROOT = ROOT / "outputs/phase6_e4_official"
E3_DIR = ROOT / "experiments/e3_late_fusion"


def _cache_hash(detector: str, split: str, tokens: list[str]) -> dict[str, Any]:
    cache = PredictionCache(CACHE_ROOT)
    entries: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for token in tokens:
        path = cache.path_for(_provenance(detector, split, token))
        if not path.is_file():
            raise RuntimeError(f"missing cache file: {path}")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"sample_token": token, "sha256": file_hash})
        digest.update(token.encode("utf-8")); digest.update(b"\0"); digest.update(file_hash.encode("ascii")); digest.update(b"\n")
    return {"detector": detector, "split": split, "entries": len(entries), "aggregate_sha256": digest.hexdigest(), "entry_hashes": entries}


def _round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("recall_50m_plus", "recall_0_5_points", "recall_40_50m", "overall_custom_recall", "precision", "fp_count", "matched_count", "gt_count", "matched_localization_error_m", "average_matched_confidence")
    return {key: metrics.get(key) for key in keys}


def _metric_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("recall_50m_plus", "recall_0_5_points", "recall_40_50m", "overall_custom_recall", "precision", "matched_localization_error_m", "average_matched_confidence"):
        left, right = current.get(key), previous.get(key)
        result[key] = None if left is None or right is None else float(left) - float(right)
    result["fp_count"] = int(current["fp_count"]) - int(previous["fp_count"])
    result["matched_count"] = int(current["matched_count"]) - int(previous["matched_count"])
    return result


def _official_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in ("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "protocol", "sample_count")}


def _render_analysis(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    official = payload["official"]
    comparisons = payload["comparisons"]
    return f"""# E4 - Repeat Validation and Final Ablation

## Scope

This is the final controlled Phase 6.4 repeat of the frozen E3 experiment on the `nuScenes v1.0-mini` `mini_val` split. No parameter search, threshold change, candidate regeneration, or mini_val tuning was performed.

## Frozen Source

The E3 configuration was read from `experiments/e3_late_fusion/frozen_config.json`: `{payload['frozen_config']}`. The four-way ablation is CenterPoint, VoxelNeXt, Naive Union, and the frozen E3 policy.

## Reproducibility

The repeat found `{payload['cache_alignment']['entries']}` aligned mini_val tokens with no duplicates or missing entries. Cache aggregate hashes are recorded in `metrics.json`; the E3 source result and this repeat are compared field-by-field.

## Custom Metrics

| Variant | 50m+ recall | 0-5 recall | Overall recall | Precision | FP |
|---|---:|---:|---:|---:|---:|
| CenterPoint | {metrics['centerpoint']['recall_50m_plus']:.6f} | {metrics['centerpoint']['recall_0_5_points']:.6f} | {metrics['centerpoint']['overall_custom_recall']:.6f} | {metrics['centerpoint']['precision']:.6f} | {metrics['centerpoint']['fp_count']} |
| VoxelNeXt | {metrics['voxelnext']['recall_50m_plus']:.6f} | {metrics['voxelnext']['recall_0_5_points']:.6f} | {metrics['voxelnext']['overall_custom_recall']:.6f} | {metrics['voxelnext']['precision']:.6f} | {metrics['voxelnext']['fp_count']} |
| Naive Union | {metrics['naive_union']['recall_50m_plus']:.6f} | {metrics['naive_union']['recall_0_5_points']:.6f} | {metrics['naive_union']['overall_custom_recall']:.6f} | {metrics['naive_union']['precision']:.6f} | {metrics['naive_union']['fp_count']} |
| E3 | {metrics['e3']['recall_50m_plus']:.6f} | {metrics['e3']['recall_0_5_points']:.6f} | {metrics['e3']['overall_custom_recall']:.6f} | {metrics['e3']['precision']:.6f} | {metrics['e3']['fp_count']} |

The E3-vs-original comparison is `{comparisons['e3_vs_e3_source']}`. The frozen E3 result remains directional: it improves far/sparse custom recall over both singles, but VoxelNeXt remains stronger on official metrics and E3 incurs more false positives.

## Official Metrics

All official results use `detection_cvpr_2019` and are labeled `nuScenes v1.0-mini exploratory experiment`:

| Variant | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CenterPoint | {official['centerpoint']['mAP']:.6f} | {official['centerpoint']['NDS']:.6f} | {official['centerpoint']['mATE']:.6f} | {official['centerpoint']['mASE']:.6f} | {official['centerpoint']['mAOE']:.6f} | {official['centerpoint']['mAVE']:.6f} | {official['centerpoint']['mAAE']:.6f} |
| VoxelNeXt | {official['voxelnext']['mAP']:.6f} | {official['voxelnext']['NDS']:.6f} | {official['voxelnext']['mATE']:.6f} | {official['voxelnext']['mASE']:.6f} | {official['voxelnext']['mAOE']:.6f} | {official['voxelnext']['mAVE']:.6f} | {official['voxelnext']['mAAE']:.6f} |
| Naive Union | {official['naive_union']['mAP']:.6f} | {official['naive_union']['NDS']:.6f} | {official['naive_union']['mATE']:.6f} | {official['naive_union']['mASE']:.6f} | {official['naive_union']['mAOE']:.6f} | {official['naive_union']['mAVE']:.6f} | {official['naive_union']['mAAE']:.6f} |
| E3 | {official['e3']['mAP']:.6f} | {official['e3']['NDS']:.6f} | {official['e3']['mATE']:.6f} | {official['e3']['mASE']:.6f} | {official['e3']['mAOE']:.6f} | {official['e3']['mAVE']:.6f} | {official['e3']['mAAE']:.6f} |

## Closure Decision

Phase 6 is closed as a reproducible exploratory study. E3 should remain in the project as a documented directional ablation and should not replace VoxelNeXt as the default detector. Future full-nuScenes or learned-fusion work must be separately scoped and preregistered.
"""


def main() -> int:
    adapter = NuScenesAdapter(Path("~/datasets/nuscenes").expanduser(), version="v1.0-mini", max_sweeps=10)
    tokens = evaluation_sample_tokens(adapter, eval_set="mini_val")
    if len(tokens) != 81 or len(tokens) != len(set(tokens)):
        raise RuntimeError(f"expected 81 unique mini_val tokens, got {len(tokens)}")
    cp = _load_centerpoint("mini_val", tokens)
    vn, vn_missing = _load_cache(PredictionCache(CACHE_ROOT), "voxelnext", "mini_val", tokens)
    if vn_missing:
        raise RuntimeError(f"missing VoxelNeXt cache entries: {len(vn_missing)}")
    cache_hashes = {"centerpoint": _cache_hash("centerpoint_pointpillar", "mini_val", tokens), "voxelnext": _cache_hash("voxelnext", "mini_val", tokens)}
    ground_truths, point_counts, scenes = _prepare_ground_truth(adapter, tokens)
    frozen = json.loads((E3_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    config = FusionConfig.from_dict(frozen["selected_config"])
    variants = {
        "centerpoint": cp,
        "voxelnext": vn,
        "naive_union": {token: naive_union(cp[token], vn[token], duplicate_threshold_m=0.01, candidate_floor=0.1, max_boxes=500) for token in tokens},
        "e3": {token: fuse_predictions(cp[token], vn[token], config) for token in tokens},
    }
    samples = {name: _samples(tokens, predictions, ground_truths, point_counts, scenes) for name, predictions in variants.items()}
    custom = {name: _custom_metrics(samples[name]) for name in variants}
    official: dict[str, Any] = {}
    for name, predictions in variants.items():
        print(f"E4 official mini_val evaluation: {name}", flush=True)
        official[name] = _official_summary(evaluate_nuscenes([predictions[token] for token in tokens], adapter, OFFICIAL_ROOT / name, eval_set="mini_val"))
    source_metrics = json.loads((E3_DIR / "metrics.json").read_text(encoding="utf-8"))
    source_custom = source_metrics["metrics"]["mini_val"]
    fields = ("recall_50m_plus", "recall_0_5_points", "recall_40_50m", "overall_custom_recall", "precision", "fp_count", "matched_count", "gt_count", "matched_localization_error_m", "average_matched_confidence")
    source_e3 = {key: source_custom["e3"].get(key) for key in fields}
    repeat_e3 = _round_metrics(custom["e3"])
    comparison = {
        "e3_vs_e3_source": {"delta": _metric_delta(repeat_e3, source_e3), "exact_match": repeat_e3 == source_e3},
        "official_vs_e3_source": {"delta": {key: official["e3"][key] - source_metrics["official"]["e3"][key] for key in ("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE")}, "exact_match": all(np.isclose(official["e3"][key], source_metrics["official"]["e3"][key]) for key in ("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"))},
    }
    if not comparison["e3_vs_e3_source"]["exact_match"] or not comparison["official_vs_e3_source"]["exact_match"]:
        raise RuntimeError(f"frozen E3 repeat mismatch: {comparison}")
    payload = {
        "label": "nuScenes v1.0-mini exploratory experiment",
        "experiment_id": "E4",
        "status": "PASS",
        "classification": "DIRECTIONAL",
        "frozen_config": config.to_dict(),
        "cache_alignment": {"entries": len(tokens), "unique_tokens": len(tokens) == len(set(tokens)), "cache_hashes": cache_hashes, "checkpoint_sha256": {"centerpoint": CP_CHECKPOINT_SHA256, "voxelnext": VN_CHECKPOINT_SHA256}},
        "metrics": custom,
        "official": official,
        "comparisons": comparison,
        "repeat_protocol": {"tuning": False, "mini_val_used_for_tuning": False, "official_protocol": "detection_cvpr_2019", "variants": list(variants)},
    }
    save_json(payload, EXPERIMENT_DIR / "metrics.json")
    save_json({"experiment": "E4", "status": "PASS", "frozen_source": "experiments/e3_late_fusion/frozen_config.json", "selected_config": config.to_dict(), "e3_repeat_exact_match": True, "official_repeat_exact_match": True, "variants": list(variants)}, EXPERIMENT_DIR / "ablation.json")
    environment = collect_environment(ROOT / "third_party/OpenPCDet")
    environment["repeat_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (EXPERIMENT_DIR / "environment.txt").write_text("\n".join(f"{key}: {value}" for key, value in sorted(environment.items())) + "\n", encoding="utf-8")
    (EXPERIMENT_DIR / "analysis.md").write_text(_render_analysis({"frozen_config": config.to_dict(), "cache_alignment": {"entries": len(tokens)}, "metrics": custom, "official": official, "comparisons": comparison}), encoding="utf-8")
    print({"status": "PASS", "classification": "DIRECTIONAL", "e3_exact_match": True, "official_exact_match": True, "output": str(EXPERIMENT_DIR)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
