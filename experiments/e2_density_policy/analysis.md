# Experiment

## Hypothesis

Among exported CenterPoint predictions, current-keyframe LiDAR support inside predicted boxes may add reliability information beyond raw confidence. The hypothesis is falsifiable and a negative result is valid.

## Baseline

Raw CenterPoint-PointPillar candidates exported at the frozen score threshold 0.1. The E1 score-only logistic implementation is the internal control.

## Change

PATH A ranking/calibration only: `sigmoid(intercept + score_weight*logit(raw_score) + sparsity_weight*log1p(predicted_box_keyframe_point_count))`. Inference uses predicted boxes, raw scores, and current-keyframe sensor points. Predicted range is diagnostic only.

## Controlled Variables

nuScenes v1.0-mini, one third-party CenterPoint detector, 10 sweeps, checkpoint/config hashes, candidate threshold 0.1, classes/geometry, inclusive class-aware 2.0m matching, official evaluation, and frozen distance/GT-density bins. No selection threshold, range feature, class-specific parameter, detector change, or membership change is introduced.

## Policy

The preregistered low-capacity family has three parameters: intercept, raw-score logit weight, and predicted-box point-count weight. The point-count transform is `log1p`; ridge values tested on mini_train were 0.1, 1.0, and 10.0 with deterministic leave-one-scene-out log-loss selection. Coefficients and the score-only control are serialized in `frozen_config.json`. PATH A has no downstream threshold or prediction budget.

## Main Metrics

50m+ recall is `0.14` raw and `0.14` E2. GT 0-5-point recall is `0.6329906124273581` raw and `0.6329906124273581` E2. PATH A preserves membership, so both deltas and paired intervals are zero.

## Candidate Ceiling

Mini-val overall candidate coverage is `0.8115289349245666`; 50m+ and 0-5 GT-point coverage are serialized in `metrics.json`. Sparse GT failures are partitioned into no exported viable candidate versus an exported candidate with a low score. These labels are evaluation-only.

## Distance-aware Metrics

All six frozen distance bins are in `metrics.json`. Point count correlates with predicted range (`-0.4265281929875091` on mini_train), but range is not an E2 policy feature and no causal claim is made.

## Density-aware Metrics

Evaluation uses current-keyframe points inside GT boxes; inference uses current-keyframe points inside predicted boxes. These are separately named and never interchanged. All five GT-density bins retain identical matching/recall under PATH A; matched confidence may change.

## Official Metrics

Raw, score-only, and E2 official mAP, NDS, mATE, mASE, mAOE, mAVE, and mAAE are serialized in `metrics.json` using the unchanged nuScenes v1.0-mini exploratory evaluator.

## Calibration Diagnostics

Mini-val Brier/log loss are reported for raw, score-only, and score+sparsity. The sparsity feature improves log loss slightly beyond score-only on this split, but this is calibration evidence rather than a custom-recall gain.

## Runtime

See `benchmark.json`. Point counting and policy application are separately timed on preloaded current-keyframe clouds; no GPU work is added.

## Result

Classification: **DIRECTIONAL**. Mini-val official mAP is `0.4370867499749207` raw, `0.4370867499749207` score-only, and `0.43733807076717107` E2; NDS is `0.4918972812308196`, `0.4918971050195758`, and `0.4920781158348097`. Mini-val Brier is `0.08006991174939278` raw, `0.06310959100855174` score-only, and `0.06306877419122621` E2.

## Failure Cases

On mini_train, `3443/8372` sparse GT have no viable exported candidate; on mini_val it is `798/2237`. E2 cannot recover these failures. Candidates exist for the remainder, but PATH A only changes ranking/calibration.

## Uncertainty

Paired complete-scene bootstrap uses 1000 repetitions, seed 42, and 95% intervals. Mini-val has only two previously exposed scenes; identical membership produces `[0,0]` delta intervals.

## Artifacts

The directory contains config, command, environment, feasibility summary, complete search log, metrics, benchmark, analysis, and four compact figures. Prediction caches remain ignored runtime artifacts.

## Tests

Focused E2 and Phase 6 protocol tests pass in `.venv`; the full suite is executed with the same interpreter. OpenPCDet remains unmodified at the pinned revision.

## Conclusion

Mini-train TP median predicted-box support is `6.0` points versus `2.0` for FP. Score-only already captures much of prediction reliability. The official and calibration comparisons determine whether sparsity adds useful ranking information; a calibration-only change is not called a detection improvement.

## Next Experiment

If E2 completed correctly, the repository is ready for Phase 6.3, CenterPoint + VoxelNeXt late prediction fusion. E3 is not implemented here.

### Mini-train Search

- `score_only_01`: family=score_only, ridge=0.1, LOO={'brier_score': 0.058845458281409704, 'log_loss': 0.21717120815294136}, selected=False, reason=valid but not selected
- `score_only_02`: family=score_only, ridge=1.0, LOO={'brier_score': 0.05884631460126251, 'log_loss': 0.21716908647487}, selected=False, reason=valid but not selected
- `score_only_03`: family=score_only, ridge=10.0, LOO={'brier_score': 0.058855255690268234, 'log_loss': 0.2171510421909932}, selected=False, reason=selected score-only internal control
- `score_sparsity_01`: family=score_sparsity, ridge=0.1, LOO={'brier_score': 0.05952035419112421, 'log_loss': 0.22165294376321218}, selected=False, reason=valid but not selected
- `score_sparsity_02`: family=score_sparsity, ridge=1.0, LOO={'brier_score': 0.05952172510710957, 'log_loss': 0.2216469805031685}, selected=False, reason=valid but not selected
- `score_sparsity_03`: family=score_sparsity, ridge=10.0, LOO={'brier_score': 0.059535837189921316, 'log_loss': 0.2215917447053783}, selected=True, reason=selected by minimum leave-one-scene-out mini_train log loss

Figures: /home/chaos/workspace/lidar-3d-perception/experiments/e2_density_policy/figures/tp_fp_point_count_distribution.png, /home/chaos/workspace/lidar-3d-perception/experiments/e2_density_policy/figures/tp_rate_confidence_by_point_count.png, /home/chaos/workspace/lidar-3d-perception/experiments/e2_density_policy/figures/range_by_point_count.png, /home/chaos/workspace/lidar-3d-perception/experiments/e2_density_policy/figures/score_change_by_point_count.png.
