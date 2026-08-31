# Phase 5 Multi-Model Benchmark

## Candidate Inventory

The inventory below comes from OpenPCDet revision
`233f849829b6ac19afb8af8837a0246890908755`. All candidates use the 10
nuScenes detection classes and inherit `MAX_SWEEPS: 10` from
`nuscenes_dataset.yaml`.

| Model | OpenPCDet config | Family / representation | Official checkpoint | Expected local path | Current availability |
|---|---|---|---|---|---|
| CenterPoint-PointPillar | `cbgs_dyn_pp_centerpoint.yaml` | Anchor-free center head; dynamic pillars and 2D BEV backbone | Model Zoo `model-23M`, ID `1UvGm6mROMyJzeSRu7OD1leU_YWoAZG7v` | `~/checkpoints/openpcdet/centerpoint_nuscenes_pp.pth` | Available and validated |
| VoxelNeXt 0.075 | `cbgs_voxel0075_voxelnext.yaml` | Fully sparse center-based detector; 0.075 x 0.075 x 0.2 m voxels | Model Zoo `model-31M`, ID `1IV7e7G9X-61KXSjMGtQo579pzDNbhwvf` | `~/checkpoints/openpcdet/voxelnext_nuscenes.pth` | Unavailable locally |
| VoxelNeXt 0.075 double-flip | `cbgs_voxel0075_voxelnext_doubleflip.yaml` | Same sparse-voxel family with test-time double flip | No distinct checkpoint reference in the pinned Model Zoo | Not selected | Config available; no matching checkpoint selected |
| PointPillar-MultiHead | `cbgs_pp_multihead.yaml` | Anchor-based multi-head detector; 0.2 x 0.2 x 8.0 m pillars | Model Zoo `model-23M`, ID `1p-501mTWsq0G9RzroTWSXreIMyTUUpBM` | `~/checkpoints/openpcdet/pointpillar_nuscenes.pth` | Unavailable locally |

The standard VoxelNeXt config is selected because it is the exact config
paired with the official checkpoint in the pinned Model Zoo. The KITTI
PointPillars checkpoint is incompatible with the nuScenes config and is never
used for this comparison.

## Protocol

The comparable accuracy dataset is `nuScenes v1.0-mini / mini_val`, labeled
as pipeline validation. Accuracy uses the official nuScenes detection devkit.
Runtime uses one model at a time on the local RTX 2060, batch size 1, FP32, 20
warmup iterations, and 100 measured iterations. CUDA events measure the
model-only forward/decode/NMS scope. Synchronized wall-clock time measures the
end-to-end project call, including preprocessing and postprocessing. Peak
allocated and reserved CUDA memory are reset and measured for each scope.

The runner frees the previous backend, runs garbage collection, empties the
CUDA cache, and resets peak statistics before the next model. It does not
change voxel size, range, depth, heads, classes, or checkpoint architecture to
fit the 6GB GPU.

## Current Result

The cached CenterPoint result exactly matches the wrapper config, official
checkpoint source, SHA-256, 10-sweep input, `mini_val` split, and official
evaluator provenance. Its checkpoint is 24,151,945 bytes with SHA-256
`955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a`.
Strict loading matched 458/458 state entries with no missing, unexpected, or
shape-mismatched keys.

| Model | mAP | NDS | Model-only mean / median / P95 | Model-only FPS | E2E mean / median / P95 | E2E FPS | Peak allocated / reserved |
|---|---:|---:|---|---:|---|---:|---|
| CenterPoint-PointPillar | 0.4371 | 0.4919 | 63.46 / 62.22 / 72.09 ms | 15.76 | 83.54 / 77.35 / 124.07 ms | 11.97 | 286,804,992 / 446,693,376 bytes |

These are local measurements, not Model Zoo runtime numbers.

Phase 5 is currently **BLOCKED** because only one comparable model has local
accuracy and runtime results. A single bounded attempt to retrieve the exact
official VoxelNeXt checkpoint timed out after 60 seconds on 2026-08-31 and
created no artifact. No alternate or random checkpoint was substituted, and
the download was not retried.

KITTI PointPillars AP_R40 remains a historical reference in the generated
report. It is not placed in the nuScenes accuracy ranking.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python tools/benchmark_phase5.py \
  --config configs/benchmark/phase5_nuscenes.yaml
```

Use `--skip-runtime` to assemble provenance only. Once a matching official
checkpoint is present at its wrapper path, use `--evaluate-missing` to run
official mini accuracy before regenerating the complete report. Generated
JSON, CSV, environment, and Markdown views are under
`outputs/phase5_benchmark/` and remain ignored runtime artifacts.
