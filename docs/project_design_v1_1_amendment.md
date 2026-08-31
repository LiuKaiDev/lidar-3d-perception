# Project Design V1.1 Amendment

Status: Accepted for Phase 6.0 protocol freeze

This document amends only the Phase 6 experiment scope in
`docs/project_design_v1.md`. The V1.0 document remains unchanged as the
historical design and continues to govern Phases 0-5 and all unaffected
project boundaries.

## Context

V1.0 designated nuScenes mini for pipeline development and expected full
nuScenes train/val for formal optimization experiments. It also expected at
least three ablations and one positive optimization result.

The available system is an RTX 2060 with 6 GB VRAM, and the available primary
dataset is nuScenes v1.0-mini. Full train/val acquisition and large-scale 3D
detector retraining are outside the current data and compute budget. Claiming
that those experiments were completed would be inaccurate.

## Decision

Phase 6 is redefined as analysis-driven, lightweight exploratory optimization
on nuScenes v1.0-mini around fixed pretrained CenterPoint-PointPillar and
VoxelNeXt predictions. Its research question is:

> How can far-range and sparse LiDAR target detection be improved under
> constrained compute without retraining large 3D detectors?

The primary planned methods are project-owned calibration, prediction-time
sparsity features, and late prediction fusion. Full-detector training,
architecture-scale modification, full nuScenes, and a larger GPU are not
mandatory. Small training or head-only fine-tuning remains optional only if it
later proves practical and is separately preregistered.

Every Phase 6 result must be labeled `nuScenes v1.0-mini exploratory
experiment` or equivalent. Mini results are not full nuScenes benchmark
improvements, statistically definitive full-dataset results, or state of the
art claims.

## Experimental Compensation

The smaller scale is compensated by stricter research discipline:

- official devkit splits with `mini_train` for parameter selection and frozen
  confirmation on `mini_val`;
- explicit acknowledgement that `mini_val` was already inspected in Phases
  3-5 and is not a pristine test set;
- preregistered primary, secondary, and official guardrail metrics;
- exact reuse of Phase 4 matching, distance, and density definitions;
- inference-time ground-truth leakage prevention;
- exact prediction-cache provenance and stale-cache rejection;
- scene-level bootstrap and paired scene-level delta intervals for custom
  recall metrics;
- fixed seeds, complete experiment manifests, runtime accounting, and
  independent final-command reruns;
- retention of all attempted, blocked, and negative experiments.

## Result Policy Change

V1.1 removes the requirement to manufacture at least one positive result. A
reproducible positive result or a reproducible negative result with a credible
explanation is valid research-engineering output. Thresholds must not be
searched until something improves, failed attempts must not be deleted, and a
method may be called an improvement only when measured evidence supports that
wording.

## Third-Party And Claim Boundary

OpenPCDet remains pinned at
`233f849829b6ac19afb8af8837a0246890908755`. CenterPoint, VoxelNeXt,
PointPillars, their checkpoints, CUDA ops, and training framework remain
third-party. Phase 6 project ownership covers protocol, cache/provenance,
prediction-only features, calibration, fusion, custom evaluation, bootstrap,
orchestration, reporting, tests, and documentation.

No V1.1 result may claim causal proof from the Phase 4 associations,
generalization to full nuScenes, untouched-test performance, SOTA status, or a
network architecture developed by this project.

## Planned Sequence

| ID | Scope |
|---|---|
| E0 | Frozen baseline protocol and verified prior records |
| E1 | Predicted-range-aware calibration/selection |
| E2 | Predicted-box density/sparsity-aware policy |
| E3 | CenterPoint + VoxelNeXt late prediction fusion |
| E4 | Repeat validation and final controlled ablation |

This Phase 6.0 amendment implements protocol infrastructure only. It does not
execute E1-E4 or select any optimization parameters.
