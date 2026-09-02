# Phase 5 Multi-Model Benchmark

## Candidate Inventory

The inventory below comes from OpenPCDet revision
`233f849829b6ac19afb8af8837a0246890908755`. All candidates use the 10
nuScenes detection classes and inherit `MAX_SWEEPS: 10` from
`nuscenes_dataset.yaml`.

| Model | OpenPCDet config | Family / representation | Official checkpoint | Expected local path | Current availability |
|---|---|---|---|---|---|
| CenterPoint-PointPillar | `cbgs_dyn_pp_centerpoint.yaml` | Anchor-free center head; dynamic pillars and 2D BEV backbone | Model Zoo `model-23M`, ID `1UvGm6mROMyJzeSRu7OD1leU_YWoAZG7v` | `~/checkpoints/openpcdet/centerpoint_nuscenes_pp.pth` | Available and validated |
| VoxelNeXt 0.075 | `cbgs_voxel0075_voxelnext.yaml` | Fully sparse center-based detector; 0.075 x 0.075 x 0.2 m voxels | Model Zoo `model-31M`, ID `1IV7e7G9X-61KXSjMGtQo579pzDNbhwvf` | `~/checkpoints/openpcdet/voxelnext_nuscenes.pth` | Available and validated |
| VoxelNeXt 0.075 double-flip | `cbgs_voxel0075_voxelnext_doubleflip.yaml` | Same sparse-voxel family with test-time double flip | No distinct checkpoint reference in the pinned Model Zoo | Not selected | Config available; no matching checkpoint selected |
| PointPillar-MultiHead | `cbgs_pp_multihead.yaml` | Anchor-based multi-head detector; 0.2 x 0.2 x 8.0 m pillars | Model Zoo `model-23M`, ID `1p-501mTWsq0G9RzroTWSXreIMyTUUpBM` | `~/checkpoints/openpcdet/pointpillar_nuscenes.pth` | Unavailable locally |

The standard VoxelNeXt config is selected because it is the exact config
paired with the official checkpoint in the pinned Model Zoo. The KITTI
PointPillars checkpoint is incompatible with the nuScenes config and is never
used for this comparison.

VoxelNeXt consumes the inherited five point features
`[x,y,z,intensity,timestamp]` across 10 sweeps. `MeanVFE` feeds the fully
sparse `VoxelResBackBone8xVoxelNeXt`; `VoxelNeXtHead` predicts center offset,
center height, dimensions, rotation, and planar velocity for all 10 classes.
It is not an OpenPCDet `CenterHead`. The fixed post-processing uses score
threshold 0.1, GPU NMS threshold 0.2, and at most 500 decoded objects without
changing the official point-cloud range or class set.

## Protocol

The comparable accuracy dataset is `nuScenes v1.0-mini / mini_val`, labeled
as pipeline validation. Accuracy uses the official nuScenes detection devkit.
Runtime uses one model at a time on the local RTX 2060, batch size 1, FP32, 20
warmup iterations, and 100 measured iterations. CUDA events measure the
model-only forward/decode/NMS scope from a prepared, device-resident batch.
Synchronized wall-clock time measures CPU point preprocessing/voxelization,
host-to-device transfer, network forward, decode/NMS, and project
`PredictionBatch` conversion. Raw LiDAR loading and 10-sweep assembly are
excluded from both scopes because the same preloaded frame is reused. Dynamic
pillarization remains inside CenterPoint's model forward, while VoxelNeXt's
fixed voxelization occurs in CPU preprocessing; both model-only and end-to-end
values are reported so that boundary remains visible.

The runner frees the previous backend, runs garbage collection, empties the
CUDA cache, and resets peak statistics before the next model. It does not
change voxel size, range, depth, heads, classes, or checkpoint architecture to
fit the 6GB GPU.

## Checkpoint Validation

The cached CenterPoint result exactly matches the wrapper config, official
checkpoint source, SHA-256, 10-sweep input, `mini_val` split, and official
evaluator provenance. Its checkpoint is 24,151,945 bytes with SHA-256
`955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a`.
Strict loading matched 458/458 state entries with no missing, unexpected, or
shape-mismatched keys.

The official VoxelNeXt checkpoint is 32,157,961 bytes with SHA-256
`9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b`.
It contains only `model_state` metadata. Strict loading matched 542/542 state
entries with zero missing, unexpected, or shape-mismatched keys. No weights
were force-loaded and no OpenPCDet architecture was changed.

## Real VoxelNeXt Inference

Three real 10-sweep samples passed finite center/yaw/score/velocity checks,
positive `[length,width,height]` size checks, class mapping, and unified
`PredictionBatch` conversion:

| Sample token | Predictions | Classes | Top five scores |
|---|---:|---|---|
| `ca9a282c9e77460f8360f564131a8af5` | 304 | all 10 classes | 0.8696, 0.8293, 0.7970, 0.7689, 0.7641 |
| `39586f9d59004284a7114a68825e8eec` | 195 | all 10 classes | 0.8997, 0.8908, 0.8587, 0.8353, 0.8319 |
| `356d81f38dd9473ba590f39e266f54e5` | 196 | 9 classes (no construction_vehicle) | 0.9433, 0.9052, 0.8814, 0.8698, 0.8629 |

The first standalone forward incurred 417 seconds of one-time sparse-kernel
initialization. Later forwards were approximately 0.45-0.52 seconds in that
smoke process. Steady-state benchmark values below exclude initialization by
using the same 20 warmup iterations for both models.

## Comparable Accuracy

All values below are local `nuScenes v1.0-mini / mini_val` results from the
official `detection_cvpr_2019` evaluator. They are pipeline validation values,
not full nuScenes train/val results.

| Model | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CenterPoint-PointPillar | 0.4371 | 0.4919 | 0.4421 | 0.4570 | 0.5811 | 0.3850 | 0.4014 |
| VoxelNeXt | 0.5218 | 0.5446 | 0.4180 | 0.4422 | 0.5426 | 0.3765 | 0.3833 |

## Runtime And Memory

| Model | Scope | Mean | Median | P95 | FPS | Peak allocated | Peak reserved |
|---|---|---:|---:|---:|---:|---:|---:|
| CenterPoint-PointPillar | Model-only | 59.76 ms | 58.96 ms | 67.87 ms | 16.73 | 268,455,936 B | 446,693,376 B |
| CenterPoint-PointPillar | End-to-end | 70.24 ms | 68.95 ms | 77.62 ms | 14.24 | 286,804,992 B | 446,693,376 B |
| VoxelNeXt | Model-only | 84.79 ms | 83.62 ms | 95.81 ms | 11.79 | 145,545,216 B | 178,257,920 B |
| VoxelNeXt | End-to-end | 111.69 ms | 108.35 ms | 131.07 ms | 8.95 | 146,302,464 B | 180,355,072 B |

## Trade-offs

- VoxelNeXt improves mini mAP by 0.0847 and NDS by 0.0527. All five aggregate
  TP error metrics are lower in this mini run.
- VoxelNeXt costs 25.02 ms more model-only mean latency and 41.46 ms more
  end-to-end mean latency. Its end-to-end throughput is 8.95 FPS versus 14.24
  FPS for CenterPoint-PointPillar.
- End-to-end overhead beyond model-only is 10.48 ms for CenterPoint and 26.91
  ms for VoxelNeXt, reflecting their different preprocessing boundaries.
- VoxelNeXt uses less measured runtime memory: 146.3 MB peak allocated versus
  286.8 MB for CenterPoint end-to-end, despite its slower sparse-voxel path.
- These mini results do not prove that VoxelNeXt fixes the Phase 4 far-range or
  low-density degradation. That question remains for controlled Phase 6 work.

These are local measurements, not Model Zoo runtime numbers.

Phase 5 is **PASS**: CenterPoint-PointPillar and VoxelNeXt both completed the
same mini dataset, official accuracy protocol, hardware, and runtime protocol.

KITTI PointPillars AP_R40 remains a historical reference in the generated
report. It is not placed in the nuScenes accuracy ranking.

## Limitations

- Accuracy uses the 81-sample `mini_val` split for pipeline validation; full
  nuScenes train/val remains deferred.
- Runtime is specific to the local RTX 2060 6GB WSL environment. VoxelNeXt has
  a material one-time sparse-kernel initialization cost before steady state.
- The optional nuScenes PointPillars checkpoint is not local, so it remains a
  separate unavailable candidate rather than delaying the valid two-model
  acceptance.
- `pip check` retains the pre-existing optional metadata gaps for `typeguard`,
  `opencv-python-headless`, and `parameterized`; CUDA, OpenPCDet, spconv,
  nuScenes devkit, real data, and compiled-op validation all pass.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python tools/benchmark_phase5.py \
  --config configs/benchmark/phase5_nuscenes.yaml
```

To reproduce VoxelNeXt official accuracy from scratch:

```bash
PYTHONPATH=. .venv/bin/python tools/evaluate_nuscenes.py \
  --config configs/detectors/voxelnext/nuscenes_mini.yaml \
  --output-dir outputs/phase5_benchmark/voxelnext/evaluation
```

The Phase 5 runner safely reuses official accuracy only when its provenance
sidecar matches the current dataset, config hashes, checkpoint hash, sweep
count, and evaluator protocol. Use `--evaluate-missing` to fill an absent or
mismatched cache, or `--skip-runtime` to assemble provenance only. Generated
JSON, CSV, environment, and Markdown views remain under
`outputs/phase5_benchmark/` as ignored runtime artifacts.
