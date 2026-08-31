# Experiment

## Hypothesis

Raw CenterPoint score quality varies by target range; a low-capacity rule using predicted range may improve far-range ranking without changing detector weights.

## Baseline

Unmodified CenterPoint-PointPillar predictions exported at the frozen candidate threshold 0.1.

## Change

Global logistic calibration: `sigmoid(intercept + score_weight * logit(raw_score) + range_weight * ((sqrt(x^2+y^2)-25)/25))`. The only inference features are raw score and predicted-box range. Geometry, labels, and candidate membership are unchanged.

## Controlled Variables

nuScenes v1.0-mini, 10 sweeps, one CenterPoint detector, checkpoint/config hashes, class-aware greedy center matching at inclusive 2.0 m, frozen distance/density bins, and candidate threshold 0.1. Ground-truth labels are fit-time only; inference accepts no GT fields. OpenPCDet applies `SCORE_THRESH: 0.1` and `MAX_OBJ_PER_SAMPLE: 500` during CenterHead decoding (`center_head.py`); the project adapter independently filters scores below 0.1 (`openpcdet_backend.py`). Both happen before caching/calibration. No later project threshold or max-box limit exists, mini_val caches contain at most 248 boxes per sample, and both custom and official evaluation consume all cached candidates. Calibration therefore cannot alter retained boxes through either limit.

## Main Metrics

Baseline versus E1: 50m+ recall `0.14` vs `0.14`; 0-5-point recall `0.6329906124273581` vs `0.6329906124273581`. Classification: **NEGATIVE**.

## Distance-aware Metrics

See `metrics.json` for all frozen bins and 40-50m recall. Since no selection operating point exists, recalibration changes ranking/official score ordering only, not custom recall.

## Density-aware Metrics

GT density remains current-keyframe points inside oriented GT boxes and is evaluation-only. It is not an inference feature.

## Runtime

Calibration was timed with `time.perf_counter` over repeated prediction-only applications; see `benchmark.json` for per-sample CPU overhead and iteration count.

## Result

The candidate floor means boxes discarded upstream below 0.1 cannot be recovered. Cached candidates are post-threshold. At least one viable exported candidate exists for `3604/4441` GT (`0.8115289349245666`), but coverage is only `0.144` at 50m+ and `0.6432722396066161` for 0-5-point GT. Distance, density, and per-class ceiling views are in `metrics.json`. Recalibration changes confidence/ranking only; it does not filter or select boxes. Result classification: **NEGATIVE**. Official mAP is `0.4370867499749207` baseline, `0.4370867499749207` score-only, and `0.43428050968674237` score+range; NDS is `0.4918972812308196`, `0.4918971039613552`, and `0.4899303070866585` respectively.

## Failure Cases

The detector's exported-candidate ceiling separates missing candidates from ranking/selection errors. No new downstream threshold was invented.

## Uncertainty

Final mini_val deltas use paired scene-level bootstrap, 1000 repetitions, seed 42, 95% intervals. Only two mini_val scenes are available and this split was previously exposed.

## Conclusion

The score-only control and range-aware method are both retained. On mini_val, range-aware Brier/log loss are `0.0629532048372537` / `0.22634951286783755` versus `0.06313147888933605` / `0.2269410931629314` for score-only. This small calibration-diagnostic gain did not become a detection benefit: score-only preserved baseline mAP, while range-aware ordering regressed official mAP/NDS and custom recall could not change without selection. Predicted range did not add useful detection information under the frozen E1 operating condition.

## Next Experiment

If the protocol completed, the repository is ready for Phase 6.2 (predicted-box sparsity/density-aware policy). E2 is not implemented here.

### Mini-train Search Log

- `score_only_01` (score_only): valid=True, selected=False, metrics={'brier_score': 0.0580307184935977, 'log_loss': 0.21180058907344534}, reason=score-only internal control
- `score_only_02` (score_only): valid=True, selected=False, metrics={'brier_score': 0.05803095207556685, 'log_loss': 0.21180060933884773}, reason=score-only internal control
- `score_only_03` (score_only): valid=True, selected=False, metrics={'brier_score': 0.058033563229467885, 'log_loss': 0.2118026082666955}, reason=score-only internal control
- `score_range_01` (score_range): valid=True, selected=True, metrics={'brier_score': 0.05756507835897017, 'log_loss': 0.2109576037384344}, reason=selected by minimum mini_train log loss
- `score_range_02` (score_range): valid=True, selected=False, metrics={'brier_score': 0.05756581560479257, 'log_loss': 0.21095762951196861}, reason=valid but not selected
- `score_range_03` (score_range): valid=True, selected=False, metrics={'brier_score': 0.05757347991106621, 'log_loss': 0.2109601606912133}, reason=valid but not selected

Figures: /home/chaos/workspace/lidar-3d-perception/experiments/e1_range_calibration/figures/score_by_predicted_range.png, /home/chaos/workspace/lidar-3d-perception/experiments/e1_range_calibration/figures/far_range_recall.png.
