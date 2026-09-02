# Phase 6 V1.1 Experiment Protocol

## Scope And Existing Evidence

Phase 6 is compute-constrained exploratory research on `nuScenes v1.0-mini`.
The Phase 4 CenterPoint analysis observed recall decreasing from 0.9587 at
0-10 m to 0.1400 at 50 m+, and recall of 0.6330 for GT boxes containing 0-5
current-keyframe points. Of 430 far-range misses, 421 were also 0-5-point
misses. Distance and point count had Pearson association -0.326. These are
exploratory associations on correlated mini scenes, not causal findings.

The frozen pretrained baselines are:

| Model | Checkpoint SHA-256 | mini_val mAP | mini_val NDS |
|---|---|---:|---:|
| CenterPoint-PointPillar | `955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a` | 0.4371 | 0.4919 |
| VoxelNeXt | `9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b` | 0.5218 | 0.5446 |

These detectors and checkpoints are third-party OpenPCDet baselines. No Phase
6 optimization rule is implemented in Phase 6.0.

## Official Split Roles

The installed nuScenes devkit was queried locally rather than relying on an
assumed count:

| Split | Local scenes | Local samples | Role |
|---|---:|---:|---|
| `mini_train` | 8 | 323 | development, parameter selection, calibration, optional scene-level internal validation |
| `mini_val` | 2 | 81 | confirmatory comparison after settings are frozen |

`mini_val` has already been examined extensively in Phases 3-5. It is not an
untouched final test set. Phase 6 reduces additional overfitting by searching
parameters only on `mini_train`, freezing settings before `mini_val`, avoiding
repeated tuning from `mini_val` outcomes, and recording every attempt. The two
`mini_val` scenes also make uncertainty intervals fragile and likely wide.

## Preregistered Metrics

Primary custom metrics, in priority order:

1. 50m+ recall.
2. 0-5-point GT recall.

Secondary metrics are 40-50 m recall, overall custom recall, precision, FP
count, matched center error, average matched confidence, and per-class
distance/density recall. Priority classes are pedestrian, bicycle,
motorcycle, and traffic_cone; car is retained where its larger support is
useful.

Distance bins remain `[0,10)`, `[10,20)`, `[20,30)`, `[30,40)`, `[40,50)`,
and `[50,+inf)`. GT density bins remain 0-5, 6-10, 11-20, 21-50, and 51+.
GT density means current-keyframe LiDAR points inside the oriented GT box.
This definition must not be silently replaced by multi-sweep density.

Custom matching remains deterministic, class-aware, greedy one-to-one center
distance with the inclusive condition `distance <= 2.0 m`. Changing this
threshold would confound comparisons and requires a separate sensitivity
study.

Every final `mini_val` experiment must also report official nuScenes mAP,
NDS, mATE, mASE, mAOE, mAVE, and mAAE through the existing official wrapper.
Custom recall is not an official nuScenes metric, official metrics receive no
fabricated bootstrap interval, and no composite score is introduced.

## Ground-Truth Leakage Boundary

Ground truth is allowed only for mini_train metric computation/tuning,
official/custom evaluation, and offline error analysis. It is forbidden as an
inference-time input.

Allowed inference information includes predicted boxes/classes/scores,
predicted-box range `sqrt(x^2+y^2)`, current-keyframe points inside a predicted
box, separately labeled multi-sweep points inside a predicted box, and
associations between CenterPoint/VoxelNeXt predictions. GT distance, GT point
count, GT class/geometry, GT-derived score changes, and GT-assisted fusion are
forbidden.

`lidar_perception.experiments.features` exposes only `PredictionBatch` and
sensor-point inputs. It computes features but contains no scoring policy.
Current-keyframe and multi-sweep point counts are separate labeled calls.

## Prediction Cache And Candidate Policy

The Phase 6 cache envelope identifies dataset, version, split, sample token,
detector, detector config and SHA-256, checkpoint SHA-256, sweep count,
candidate threshold, score policy, and prediction schema version. All fields
must match before reuse; malformed and stale entries are misses. The standard
root is `outputs/phase6_prediction_cache/`, already excluded from Git. It can
represent CenterPoint/VoxelNeXt on both official mini splits without changing
the schema.

Both current detector exports use the fixed upstream and project threshold
0.1. It is accepted as the initial calibration/fusion candidate pool because
it retained the low-confidence detections used by Phase 4 analysis (including
417 matched predictions below 0.3) while keeping the pool bounded. Candidates
discarded upstream below 0.1 cannot be recovered or studied. Phase 6 freezes
0.1 for the planned sequence unless a different export threshold is separately
preregistered on `mini_train`. Such a change must regenerate compatible caches
for all compared models/splits and cannot silently reuse existing files.

## Scene-Level Bootstrap

Custom recall is represented by additive `(matched_count, gt_count)` pairs per
scene. The default interval resamples complete scenes with replacement for
1000 repetitions using seed 42 and a percentile 95% confidence interval.
Frames are not treated as independent. Empty metric bins return no point or
interval, and one-scene inputs return a degenerate interval rather than an
invented variance estimate.

Baseline/experiment comparison uses paired scene bootstrap: the scene sets
and metric denominators must match, and identical sampled scene indices are
used for both methods before calculating the recall delta.

Result language is frozen:

- positive delta and paired CI entirely above zero: `bootstrap-supported
  improvement on mini`;
- positive delta with CI crossing zero: `directional improvement; uncertainty
  overlaps zero`;
- negative delta: report `regression on mini`;
- zero or insufficient data: report that state directly.

The phrase statistically significant is not used without a separately
designed hypothesis test, and these intervals do not imply full-nuScenes
generalization.

## Future Experiment Boundaries

- E1 may calibrate/select by predicted-box range only. No rule or parameter is
  selected in Phase 6.0.
- E2 may use current-keyframe predicted-box point count; multi-sweep count is
  optional only when separately labeled. GT boxes/counts are unavailable to
  the policy.
- E3 may deterministically associate same-class predictions using a
  preregistered center-distance, BEV-IoU, or 3D-IoU option. The choice and
  fusion rule are selected on `mini_train`, never from `mini_val` feedback.
- E4 repeats the frozen final commands and reports the complete ablation. It
  does not reopen parameter search.

Fusion reports the compute cost of both detectors plus fusion. Any method that
changes inference behavior reports added CPU/GPU cost, end-to-end latency,
FPS, and materially changed VRAM using the Phase 5 methodology.

## Registry And Repeatability

Each experiment directory contains `config.yaml`, `command.sh`,
`environment.txt`, `metrics.json`, `benchmark.json`, `analysis.md`, and
`figures/`. The validated manifest freezes hypotheses, identities, bins,
metrics, candidate threshold, bootstrap, tunables, controlled variables,
leakage policy, and runtime protocol before execution.

The deterministic post-processing experiments do not require arbitrary neural
training seeds. They require fixed inputs/configs/seeds, exact hashes, fixed
bootstrap resampling, an independent final-command rerun, and retained
negative results. A later stochastic training experiment must preregister and
report multiple seeds without cherry-picking.

The experiment index is `experiments/README.md`, the reusable skeleton is
`experiments/_template/`, and E0 is
`experiments/e0_baseline_protocol/`. E0 copies only verified Phase 4/5 values;
VoxelNeXt custom distance/density metrics remain explicitly not evaluated.
