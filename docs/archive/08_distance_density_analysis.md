# Distance, Density, and Bad-Case Analysis

Phase 4 is a project-owned exploratory analysis of the official CenterPoint-
PointPillar predictions on `nuScenes v1.0-mini`. It is separate from the
official nuScenes mAP/NDS protocol used in Phase 3.

## Reproducible Protocol

The one-click command is:

```bash
PYTHONPATH=. python tools/analyze_phase4.py \
  --config configs/analysis/phase4_nuscenes.yaml
```

The runner validates prediction frame tokens, backend metadata, and score
filtering before reusing the Phase 3 cache. Missing samples are inferred with
the configured CenterPoint backend. The report records dataset/version, split,
10-sweep setting, detector config, checkpoint hash, and all analysis settings.

## Matching

Each sample uses class-aware, one-to-one greedy matching. Candidate pairs are
sorted by `(center distance, GT index, prediction index)` and accepted when
the horizontal LiDAR center distance is `<= 2.0 m`. The threshold is inclusive
and configurable. This matcher produces custom TP/FP/FN records for analysis;
it is not nuScenes official recall or mAP.

## Distance

Target range is `sqrt(x^2 + y^2)` from the GT box center in the reference
LiDAR frame (`x` forward, `y` left). Bins are half-open:

```text
[0,10), [10,20), [20,30), [30,40), [40,50), [50,+inf)
```

Rows include explicit zero-count bins, GT and prediction counts, matches,
false negatives/positives, recall, precision, and mean matched center error.
Matched predictions inherit their GT range bin; unmatched predictions use their
own range bin, so each precision denominator is target-conditioned and equals
matches plus false positives.

## Density

Density is the number of current keyframe LiDAR points inside each oriented GT
3D box. Accumulated sweeps are deliberately excluded from this primary
statistic so it represents physical observability at the target timestamp.
The exact integer bins are `0-5`, `6-10`, `11-20`, `21-50`, and `51+`,
implemented as `[0,6)`, `[6,11)`, `[11,21)`, `[21,51)`, and `[51,+inf)`.
Rows include GT count, matches, false negatives, recall, average matched
confidence, and matched localization error.

## Bad-Case Taxonomy

The miner exports deterministic top-ranked examples for false negatives,
false positives, low-confidence true positives (`score < 0.3`), high-error
true positives (`center error >= 1.0 m`), far-range misses (`distance >= 50 m`),
and low-density misses (`<= 5` current-keyframe points). Ranking is category
specific: distance descending for far/FN, point count ascending for low
density, localization error descending for high-error TP, confidence ascending
for low-confidence TP, and confidence descending for FP. Each case retains
sample id, class, geometry, confidence, distance, point count, match indices,
reason, and BEV snapshot; the first case in each category also gets a 3D view.

## Mini-Dataset Findings

The generated `summary.json` contains 4,441 GT boxes, 3,571 matches, 870 FNs,
and 12,841 FPs under this custom protocol. Observations from the complete
81-sample `mini_val` run:

- Recall falls from 95.9% in `[0,10)` to 14.0% in `50m+`; matched center error rises from 0.145 m to 0.683 m.
- The strongest range degradation is in car (96.8% at 10-20 m to 10.4% at 50 m+), pedestrian (97.5% to 20.3%), and motorcycle (81.5% to 12.5%); bicycle has only six GTs in `50m+` and recalls zero there.
- Recall is 63.3% for the `0-5` point bin, then 96.3%, 97.3%, 97.7%, and 99.7% for denser bins. Matched confidence rises from 0.497 to 0.794 while localization error falls from 0.365 m to 0.148 m.
- Low-density bins contain 821 of 870 FNs. Of 430 far-range misses, 421 (97.9%) are also `0-5` point misses, an association rather than proof of causality.
- The largest raw FN class is car (566), followed by pedestrian (162) and motorcycle (73). FP volume is high because the score threshold is 0.1 and the custom matcher is intentionally independent of the official evaluator.

Manual real-data validation selected a TP, FP, and FN from sample
`3e8750f331d7499e9b5123e9eb70f2e2`; their LiDAR-frame box geometry and match
indices are recorded in `manual_matching_validation.json`. The inspected BEV
snapshots include a far-range FN (`false_negative_01_c5f58c19249d4137ae063b0e9ecd8b8e_bev.png`), a high-confidence FP (`false_positive_01_f40544fd4f5d42abbcfa948eeaf86850_bev.png`), and a low-confidence TP (`low_confidence_true_positive_01_5b7cb170eee6468aa1fdbd3abcf63c5a_bev.png`).

The exact TP/FP/FN sample also has a dedicated overlay at
`manual_matching/3e8750f331d7499e9b5123e9eb70f2e2_gt_pred_bev.png`.

The distance/point-count Pearson association is -0.326 in this mini run. This
supports an exploratory relationship between range and observability, but the
small, scene-correlated mini split cannot establish that density causes the
recall change or generalize to the full train/val benchmark.

## Outputs

The runner writes `distance_metrics.csv`, `density_metrics.csv`,
`distance_density.csv`, `matching_summary.json`, `manual_matching_validation.json`,
`bad_cases.csv`, `provenance.json`, `summary.json`, and five deterministic
figures under `outputs/phase4_analysis/`. Representative BEV and selected 3D
snapshots are under `outputs/phase4_analysis/bad_cases/`.
