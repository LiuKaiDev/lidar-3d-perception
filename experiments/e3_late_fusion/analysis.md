# Experiment

E3 - CenterPoint + VoxelNeXt late prediction fusion (`nuScenes v1.0-mini exploratory experiment`).

## Hypothesis

Frozen CenterPoint and VoxelNeXt candidates may be complementary for far-range and sparse objects, allowing deterministic prediction-only late fusion to recover misses.

## Baseline

Both pretrained OpenPCDet detectors, the candidate floor (0.1), ten-sweep input, and Phase 4 evaluation protocol remained frozen.

## Change

E3 uses class-aware one-to-one center-distance association, probabilistic-OR scores, winner-take-all geometry, retained unmatched candidates, deterministic sorting, and a top-500 limit. The selected mini_train parameters are `{'association_threshold_m': 0.5, 'centerpoint_weight': 1.2, 'voxelnext_weight': 1.0, 'score_rule': 'probabilistic_or', 'geometry_policy': 'winner_take_all', 'candidate_floor': 0.1, 'max_boxes': 500, 'schema_version': 'lidar_perception.e3_late_fusion.v1'}`.

## Controlled Variables

Selection used mini_train only under the preregistered rule: maximize 50m+ recall then 0-5-point recall, lower FP, lower threshold, weight closest to 1. The final configuration was written before mini_val predictions were loaded. Ground truth is evaluation-only and never enters fusion.

## Main Metrics

Mini_val 50m+ recall: CP `0.14`, VN `0.234`, naive union `0.246`, E3 `0.246`. Mini_val 0-5-point recall: CP `0.6329906124273581`, VN `0.687080911935628`, naive union `0.7139025480554314`, E3 `0.7116674117121145`.

## Distance-aware Metrics

All six frozen bins for CP, VN, naive union, and E3 are recorded in `metrics.json` and plotted in `figures/recall_by_distance.png`.

## Density-aware Metrics

All five current-keyframe GT density bins are recorded in `metrics.json` and plotted in `figures/recall_by_density.png`. GT density is evaluation-only.

## Candidate Complementarity

Mini_train overall both/CP-only/VN-only/neither counts are `{'detected_by_both': 9896, 'centerpoint_only': 281, 'voxelnext_only': 591, 'neither': 3155}`. At 50m+ they are `{'detected_by_both': 527, 'centerpoint_only': 48, 'voxelnext_only': 238, 'neither': 2041}`; at 0-5 points they are `{'detected_by_both': 4555, 'centerpoint_only': 253, 'voxelnext_only': 508, 'neither': 3056}`.

On mini_val, the union ceiling is `{'overall': {'centerpoint': {'covered': 3604, 'gt_count': 4441, 'coverage': 0.8115289349245666}, 'voxelnext': {'covered': 3744, 'gt_count': 4441, 'coverage': 0.8430533663589281}, 'union': {'covered': 3800, 'gt_count': 4441, 'coverage': 0.8556631389326729}, 'centerpoint_only': 56, 'voxelnext_only': 196, 'e3': {'covered': 3800, 'gt_count': 4441, 'coverage': 0.8556631389326729}}, '50m_plus': {'centerpoint': {'covered': 72, 'gt_count': 500, 'coverage': 0.144}, 'voxelnext': {'covered': 120, 'gt_count': 500, 'coverage': 0.24}, 'union': {'covered': 126, 'gt_count': 500, 'coverage': 0.252}, 'centerpoint_only': 6, 'voxelnext_only': 54, 'e3': {'covered': 126, 'gt_count': 500, 'coverage': 0.252}}, '0_5_points': {'centerpoint': {'covered': 1439, 'gt_count': 2237, 'coverage': 0.6432722396066161}, 'voxelnext': {'covered': 1556, 'gt_count': 2237, 'coverage': 0.6955744300402324}, 'union': {'covered': 1605, 'gt_count': 2237, 'coverage': 0.7174787662047385}, 'centerpoint_only': 49, 'voxelnext_only': 166, 'e3': {'covered': 1605, 'gt_count': 2237, 'coverage': 0.7174787662047385}}}`. Adding VN raises the ceiling beyond CP wherever `voxelnext_only` is nonzero; adding CP raises it beyond VN wherever `centerpoint_only` is nonzero.

## Recovery Analysis

Potential versus actual recovery is `{'overall': {'cp_misses_covered_by_vn': 214, 'cp_misses_recovered_by_e3': 213, 'vn_misses_covered_by_cp': 64, 'vn_misses_recovered_by_e3': 63}, '50m_plus': {'cp_misses_covered_by_vn': 53, 'cp_misses_recovered_by_e3': 53, 'vn_misses_covered_by_cp': 6, 'vn_misses_recovered_by_e3': 6}, '0_5_points': {'cp_misses_covered_by_vn': 176, 'cp_misses_recovered_by_e3': 176, 'vn_misses_covered_by_cp': 55, 'vn_misses_recovered_by_e3': 54}}`. This separates complementary exported candidates from GT actually matched after fusion.

## Official Metrics

Official detection_cvpr_2019 mini_val mAP/NDS are CP `0.4370867499749207/0.4918972812308196`, VN `0.5209154714332773/0.5441930215262918`, naive union `0.390049659695518/0.47260551086085456`, and E3 `0.49963968241175916/0.5210093465417833`.

## Runtime

Fusion-only timings and memory are in `benchmark.json`. The dual-detector total is explicitly estimated, not measured: `240.8962852498917` ms/sample from the Phase 5 detector references plus cached-prediction fusion overhead. Models were generated sequentially; peak VRAM values are reported per sequential generation run and are not added.

## Representative Cases

Success and failure examples are recorded under `outputs/phase6_e3_cases/` and listed in `metrics.json`.

## Uncertainty

Paired scene bootstrap uses 1000 repetitions, seed 42, and 95% intervals with identical resampled indices. Mini_val has only two scenes, so intervals may be wide or degenerate. This split was already exposed in earlier phases.

## Result

E3 classification: **DIRECTIONAL**. It is judged against the strongest single model, with far/sparse recall prioritized and official mAP/NDS treated as guardrails.

## Conclusion

Detector complementarity, candidate-ceiling change, actual recovery, false-positive cost, strongest-single comparison, official metric trade-off, and runtime cost are quantified above and in the JSON artifacts. Retention in the final portfolio is `retain as a documented late-fusion ablation`.

## Next Experiment

Repository is ready for Phase 6.4: Final Ablation / Repeat Validation / Phase 6 Closure. E4 is not implemented here.
