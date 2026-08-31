# Experiment

E3 — CenterPoint + VoxelNeXt late prediction fusion (`nuScenes v1.0-mini`).

## Hypothesis

Frozen CenterPoint and VoxelNeXt candidates may be complementary for far-range
and sparse objects. This remains untested until both detector cache families are
available.

## Baseline

CenterPoint and VoxelNeXt remain frozen third-party OpenPCDet models. Existing
Phase 5 references are 70.24 ms and 111.69 ms end-to-end respectively.

## Change

Added prediction-only, class-aware one-to-one center-distance association,
probabilistic-OR matched scores, winner-take-all geometry, retained unmatched
candidates, deterministic sorting/top-500, and a naive-union control.

## Controlled Variables

Ten sweeps, candidate floor 0.1, inclusive 2 m GT matching, frozen Phase 4
distance/density bins, mini_train-only tuning, and no E1/E2 policy composition.
The association grid (0.5/1.0/1.5/2.0 m) and CenterPoint weights
(0.8/1.0/1.2) were declared before search.

## Main Metrics

Blocked: VoxelNeXt mini_train and mini_val prediction caches are absent, so no
complementarity, search, or custom metrics are reported.

## Distance-aware Metrics

Frozen six-bin protocol is declared in `config.yaml`; evaluation pending caches.

## Density-aware Metrics

Frozen five-bin current-keyframe GT-point protocol is declared in `config.yaml`;
evaluation pending caches.

## Runtime

Fusion API exposes association, fusion, and sorting timings. Detector references
remain separate; dual-detector runtime and VRAM were not measured.

## Result

BLOCKED before mini_train complementarity gate. It would be invalid to label E3
positive, directional, negative, or unchanged without aligned predictions.

## Failure Cases

No cases can be mined without VoxelNeXt predictions.

## Uncertainty

No bootstrap was run. The configured protocol is 1000 paired scene resamples,
seed 42, 95% CI.

## Conclusion

The project-owned fusion implementation and deterministic tests are complete,
but the experiment result is blocked by missing compatible VoxelNeXt caches.

## Next Experiment

Populate/verify VoxelNeXt mini_train and mini_val caches with the declared
provenance, then run complementarity analysis, frozen search, and confirmation.
Do not tune on mini_val and do not start E4 in this experiment.
