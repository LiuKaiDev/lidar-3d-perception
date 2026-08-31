# Experiment

## Hypothesis

The existing pretrained results can be recorded under one frozen protocol.
E0 tests no optimization hypothesis.

## Baseline

CenterPoint-PointPillar and VoxelNeXt official mini metrics are sourced from
Phase 5. CenterPoint custom distance/density metrics are sourced from Phase 4.

## Change

None. This is the preregistered baseline/protocol record.

## Controlled Variables

Dataset version, 10 sweeps, candidate threshold 0.1, class-aware inclusive
2.0 m matching, bins, official evaluator, and runtime protocol are frozen in
`config.yaml`.

## Main Metrics

CenterPoint: 50m+ recall 0.1400 and 0-5-point recall 0.6330. VoxelNeXt custom
metrics have not been evaluated and are not inferred from official metrics.

## Distance-aware Metrics

The verified CenterPoint mini_val recalls are 0.9587, 0.9548, 0.9103,
0.8301, 0.7007, and 0.1400 across the six frozen distance bins.

## Density-aware Metrics

The verified CenterPoint mini_val recalls are 0.6330, 0.9630, 0.9733,
0.9774, and 0.9966 across the five frozen GT density bins.

## Runtime

Existing Phase 5 measurements are recorded in `benchmark.json`; no runtime was
rerun for Phase 6.0.

## Result

Baseline provenance recorded. No positive or negative optimization result.

## Failure Cases

Phase 4 recorded 430 far-range misses; 421 were also in the 0-5 point GT bin.
This is an exploratory association, not a causal result.

## Uncertainty

No interval is backfilled from aggregate-only values. Scene-level and paired
bootstrap utilities are frozen for subsequent per-scene comparisons.

## Conclusion

E0 is complete as a protocol record.

## Next Experiment

E1, after its mini_train parameters are preregistered. No E1 work is included
in Phase 6.0.
