# E4 - Repeat Validation and Final Ablation

## Scope

This is the final controlled Phase 6.4 repeat of the frozen E3 experiment on the `nuScenes v1.0-mini` `mini_val` split. No parameter search, threshold change, candidate regeneration, or mini_val tuning was performed.

## Frozen Source

The E3 configuration was read from `experiments/e3_late_fusion/frozen_config.json`: `{'association_threshold_m': 0.5, 'centerpoint_weight': 1.2, 'voxelnext_weight': 1.0, 'score_rule': 'probabilistic_or', 'geometry_policy': 'winner_take_all', 'candidate_floor': 0.1, 'max_boxes': 500, 'schema_version': 'lidar_perception.e3_late_fusion.v1'}`. The four-way ablation is CenterPoint, VoxelNeXt, Naive Union, and the frozen E3 policy.

## Reproducibility

The repeat found `81` aligned mini_val tokens with no duplicates or missing entries. Cache aggregate hashes are recorded in `metrics.json`; the E3 source result and this repeat are compared field-by-field.

## Custom Metrics

| Variant | 50m+ recall | 0-5 recall | Overall recall | Precision | FP |
|---|---:|---:|---:|---:|---:|
| CenterPoint | 0.140000 | 0.632991 | 0.804098 | 0.217585 | 12841 |
| VoxelNeXt | 0.234000 | 0.687081 | 0.837874 | 0.279522 | 9591 |
| Naive Union | 0.246000 | 0.713903 | 0.853862 | 0.127750 | 25891 |
| E3 | 0.246000 | 0.711667 | 0.852286 | 0.161318 | 19678 |

The E3-vs-original comparison is `{'delta': {'recall_50m_plus': 0.0, 'recall_0_5_points': 0.0, 'recall_40_50m': 0.0, 'overall_custom_recall': 0.0, 'precision': 0.0, 'matched_localization_error_m': 0.0, 'average_matched_confidence': 0.0, 'fp_count': 0, 'matched_count': 0}, 'exact_match': True}`. The frozen E3 result remains directional: it improves far/sparse custom recall over both singles, but VoxelNeXt remains stronger on official metrics and E3 incurs more false positives.

## Official Metrics

All official results use `detection_cvpr_2019` and are labeled `nuScenes v1.0-mini exploratory experiment`:

| Variant | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CenterPoint | 0.437087 | 0.491897 | 0.442069 | 0.456985 | 0.581061 | 0.384972 | 0.401374 |
| VoxelNeXt | 0.520915 | 0.544193 | 0.418053 | 0.442120 | 0.542696 | 0.376463 | 0.383316 |
| Naive Union | 0.390050 | 0.472606 | 0.442375 | 0.444247 | 0.571412 | 0.383631 | 0.382528 |
| E3 | 0.499640 | 0.521009 | 0.431892 | 0.451179 | 0.612177 | 0.370787 | 0.422071 |

## Closure Decision

Phase 6 is closed as a reproducible exploratory study. E3 should remain in the project as a documented directional ablation and should not replace VoxelNeXt as the default detector. Future full-nuScenes or learned-fusion work must be separately scoped and preregistered.
