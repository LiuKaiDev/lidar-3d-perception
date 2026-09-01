# Phase 6 Closure

Status: PASS after E4 repeat validation

Phase 6 is a controlled `nuScenes v1.0-mini exploratory experiment`. It does
not establish full-nuScenes generalization, state-of-the-art performance, or
an untouched-test result.

## Experiment Outcomes

| Experiment | Result | Final interpretation |
|---|---|---|
| E0 | PASS | Protocol, splits, metrics, cache provenance, and leakage boundaries frozen. |
| E1 | NEGATIVE | Predicted-range calibration did not improve detection under the frozen operating condition. |
| E2 | DIRECTIONAL | Predicted-box sparsity policy provided limited directional evidence, retained as an ablation. |
| E3 | DIRECTIONAL | Late fusion improved far/sparse custom recall but increased false positives and trailed VoxelNeXt on official mAP/NDS. |
| E4 | PASS | Frozen E3 custom and official metrics reproduced exactly with no tuning. |

## Final Controlled Ablation

| Variant | 50m+ recall | 0-5-point recall | Precision | FP | mAP | NDS |
|---|---:|---:|---:|---:|---:|---:|
| CenterPoint | 0.1400 | 0.6330 | 0.2176 | 12,841 | 0.4371 | 0.4919 |
| VoxelNeXt | 0.2340 | 0.6871 | 0.2795 | 9,591 | 0.5209 | 0.5442 |
| Naive Union | 0.2460 | 0.7139 | 0.1277 | 25,891 | 0.3900 | 0.4726 |
| Frozen E3 | 0.2460 | 0.7117 | 0.1613 | 19,678 | 0.4996 | 0.5210 |

E3 converts most exported candidate complementarity into recall. Relative to
VoxelNeXt, it gains 1.2 percentage points at 50 m+ and 2.46 points for 0-5
point GT, but adds 10,087 false positives and loses 2.13 mAP and 2.32 NDS
points. Naive Union shows that unstructured candidate addition is worse on
both precision and official metrics.

## Repeat Validation

E4 read the frozen E3 configuration and exact Phase 6 caches. It did not run a
search or regenerate candidates. All ten recorded E3 custom summary fields
matched exactly, and all seven official metrics matched the original E3
artifact. The 81 CenterPoint and VoxelNeXt mini_val cache entries were aligned
by sample token; aggregate and per-entry hashes are retained in
`experiments/e4_repeat_validation/metrics.json`.

## Final Recommendation

VoxelNeXt remains the default detector because it has the strongest official
accuracy, highest precision, lowest false-positive count among the useful
variants, and lower deployment complexity than dual-detector fusion. E3 is
retained as a reproducible directional research ablation for far-range and
sparse recall, not as the default production policy.

The next mainline phase is Phase 7 engineering and portfolio packaging. Any
full-nuScenes validation, learned fusion, or new detector training is a new
experiment and must not reopen the frozen Phase 6 search.
