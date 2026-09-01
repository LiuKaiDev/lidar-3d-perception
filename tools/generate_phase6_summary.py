#!/usr/bin/env python3
"""Generate and validate the deterministic Phase 6 results summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E3_METRICS = ROOT / "experiments/e3_late_fusion/metrics.json"
E4_METRICS = ROOT / "experiments/e4_repeat_validation/metrics.json"
E3_BENCHMARK = ROOT / "experiments/e3_late_fusion/benchmark.json"
OUTPUT = ROOT / "reports/phase6_summary.md"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_summary(e3: dict, e4: dict, benchmark: dict) -> str:
    metrics = e3["metrics"]["mini_val"]
    official = e3["official"]
    sample_count = int(official["e3"]["sample_count"])
    names = {"centerpoint": "CenterPoint", "voxelnext": "VoxelNeXt", "naive_union": "Naive Union", "e3": "E3 late fusion"}
    detector_runtime = benchmark["dual_detector_total"]
    runtime = {
        "centerpoint": f"{detector_runtime['centerpoint_e2e_ms']:.2f} ms*",
        "voxelnext": f"{detector_runtime['voxelnext_e2e_ms']:.2f} ms*",
        "naive_union": "N/A",
        "e3": f"{detector_runtime['estimated_total_ms']:.2f} ms est.",
    }
    rows = []
    for key, name in names.items():
        value = metrics[key]
        official_value = official[key]
        rows.append(
            f"| {name} | {_pct(value['recall_50m_plus'])} | {_pct(value['recall_0_5_points'])} | "
            f"{_pct(value['overall_custom_recall'])} | {_pct(value['precision'])} | {value['fp_count']:,} | "
            f"{official_value['mAP']:.4f} | {official_value['NDS']:.4f} | {runtime[key]} |"
        )
    centerpoint_runtime = detector_runtime["centerpoint_e2e_ms"]
    voxelnext_runtime = detector_runtime["voxelnext_e2e_ms"]
    e3_runtime = detector_runtime["estimated_total_ms"]
    return f"""# Phase 6 Results Summary

This report is generated from the committed E3/E4 machine-readable artifacts;
run `PYTHONPATH=. .venv/bin/python tools/generate_phase6_summary.py --check`
to detect drift. It describes the **nuScenes v1.0-mini exploratory experiment**
only: `mini_val` has {sample_count} samples from two scenes, was previously exposed, and
these results do not represent full-nuScenes generalization or a SOTA claim.

## Confirmatory mini-val results

Custom metrics use class-aware one-to-one center matching at 2 m. Recall bins
are defined by GT range or predicted point count as documented in the Phase 6
protocol. Official columns are from the nuScenes detection evaluator.

| Variant | 50m+ recall | 0-5 points recall | Overall recall | Precision | FP | mAP | NDS | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

E3 is the frozen sequential late-fusion ablation. It raises long-range and
sparse recall relative to either detector, but its false-positive count and
lower precision are material costs. Naive Union is retained only as a
high-recall, high-FP comparison.

## Runtime semantics

- CenterPoint and VoxelNeXt single-detector reference values are historical
  Phase 5 detector E2E measurements: {centerpoint_runtime:.2f} ms/sample and
  {voxelnext_runtime:.2f} ms/sample,
  respectively. They include preprocessing, transfer, inference, and schema
  conversion under the Phase 5 scope.
- E3's **{e3_runtime:.2f} ms/sample** is an *estimated* sequential total from
  those two detector references plus measured cached-prediction CPU fusion.
  It is not a newly measured end-to-end detector latency.
- Phase 7A VoxelNeXt demo timings (about 785 s cold CLI wall time and 41.4 s
  warm CLI wall time) include process/model startup and asset I/O. They must
  not be mixed with detector E2E benchmark columns.
- E4 repeat validation status is **{e4['status']}** for both custom and official
  metrics; it repeats frozen cached-prediction evaluation and does not retune.

## Recommendation

Use **VoxelNeXt** as the default model because it has the strongest official
mini-val mAP/NDS and the best precision among the compared variants. Keep
**CenterPoint** as the baseline, **E3** as a directional complementarity
ablation, and **Naive Union** only as a diagnostic control.

## Source artifacts and figures

- [`experiments/e3_late_fusion/metrics.json`](../experiments/e3_late_fusion/metrics.json)
- [`experiments/e4_repeat_validation/metrics.json`](../experiments/e4_repeat_validation/metrics.json)
- [`experiments/e3_late_fusion/benchmark.json`](../experiments/e3_late_fusion/benchmark.json)

Distance recall shows where complementarity is concentrated, including the
small 50m+ E3 gain over VoxelNeXt.

![Recall by distance](../experiments/e3_late_fusion/figures/recall_by_distance.png)

Density recall exposes the sparse 0-5-point slice rather than hiding it inside
the overall score.

![Recall by point density](../experiments/e3_late_fusion/figures/recall_by_density.png)

Official metrics show why VoxelNeXt remains the default despite E3's custom
recall gains.

![Official mAP and NDS](../experiments/e3_late_fusion/figures/official_metrics.png)

Complementarity separates shared detections from model-only recoveries and
provides the rationale for retaining E3 as an ablation.

![Detector complementarity](../experiments/e3_late_fusion/figures/complementarity.png)
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the committed report is stale")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rendered = render_summary(_load(E3_METRICS), _load(E4_METRICS), _load(E3_BENCHMARK))
    if args.check:
        actual = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if actual != rendered:
            print(f"stale summary: {args.output}")
            return 1
        print(f"summary check passed: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
