# nuScenes and CenterPoint

Phase 3 uses the official `v1.0-mini` nuScenes tables for pipeline validation.
The mini archive has 10 scenes and 404 keyframe samples. It is deliberately
not a formal full-train/val benchmark; full data and formal experiments remain
deferred.

## Dataset Structure

`scene` groups a driving sequence. `sample` is a synchronized keyframe and
links the sensor channels. `sample_data` identifies one recorded file such as
`LIDAR_TOP`, its timestamp, and the previous/next sweep. `calibrated_sensor`
stores the sensor-to-ego extrinsic, while `ego_pose` stores the ego-to-global
pose. `sample_annotation` stores a global box, instance token, category
instance, and point counts. `NuScenesAdapter` follows these links directly
and keeps sample/instance tokens in project metadata.

## Coordinates and Sweeps

The project convention stays LiDAR-local `x` forward, `y` left, `z` up, with
geometric box centers, `[length,width,height]` sizes, and yaw about `+z`.
nuScenes records use `[w,x,y,z]` quaternions. The adapter builds homogeneous
transforms:

```text
T_global_sensor = T_global_ego @ T_ego_sensor
T_reference_sensor_from_sweep_sensor
    = inverse(T_global_reference_sensor) @ T_global_sweep_sensor
```

Past `LIDAR_TOP` sweeps are transformed into the reference sensor frame before
concatenation. Points are emitted as `[x,y,z,intensity,time_lag]`, where
`time_lag = reference_timestamp - sweep_timestamp` in seconds and the current
keyframe has zero lag. Raw sweep coordinates are never concatenated directly.

## CenterPoint Pipeline

The pinned OpenPCDet config is the official CenterPoint-PointPillar baseline
`cbgs_dyn_pp_centerpoint.yaml`:

```text
multi-sweep points
    -> dynamic pillarization
    -> DynPillarVFE
    -> PointPillar scatter
    -> 2D BEV backbone
    -> CenterHead heatmaps and regressions
    -> decode, score threshold, class-agnostic NMS
    -> project Box3D / PredictionBatch
```

CenterPoint is anchor-free even with this PointPillar input representation. A Gaussian
heatmap marks object centers; regression channels recover sub-voxel center
offset, height, log dimensions, sine/cosine yaw, and planar velocity. The
CenterPoint backend preserves the native 9D box output as project size/yaw and
`velocity=[vx,vy,0]` in the reference LiDAR frame.

This replaces the originally selected voxel-size-0.075 CenterPoint model
because its official pretrained checkpoint is no longer available. This run
uses the official OpenPCDet CenterPoint-PointPillar pretrained baseline; it is
not the voxel-based CenterPoint model.

Multi-sweep context supplies more observations for moving and distant objects,
while motion compensation through ego/global poses prevents scene smear. The
tradeoff is more points, voxel memory, and preprocessing time.

## Official Metrics

`evaluate_nuscenes` delegates scoring to the official devkit. Current results
must be labeled `nuScenes v1.0-mini / pipeline validation`:

- mAP: mean per-class average precision over official center-distance thresholds.
- NDS: the official aggregate combining mAP and true-positive errors.
- mATE: translation error; mASE: scale error; mAOE: orientation error.
- mAVE: velocity error; mAAE: attribute error.

Mini metrics validate data flow, coordinate conversion, inference, velocity,
and evaluator integration only. They must not be presented as full nuScenes
train/val benchmark numbers.

## Phase 3 Validation Results

This run is labeled `nuScenes v1.0-mini / pipeline validation` and uses the
official OpenPCDet CenterPoint-PointPillar config
`cbgs_dyn_pp_centerpoint.yaml` with 10-sweep input. The checkpoint is
`~/checkpoints/openpcdet/centerpoint_nuscenes_pp.pth`, 24,151,945 bytes,
SHA-256 `955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a`,
from the Model Zoo `model-23M` link recorded in the project config. Strict
loading matched 458/458 state entries with zero missing, unexpected, or
shape-mismatched keys.

Representative real inference results:

| Sample token | Predictions | Classes | Top scores |
|---|---:|---|---|
| `ca9a282c9e77460f8360f564131a8af5` | 266 | all 10 detection classes | 0.7607, 0.7598, 0.7295, 0.7264, 0.7245 |
| `39586f9d59004284a7114a68825e8eec` | 217 | all 10 detection classes | 0.8794, 0.8773, 0.8449, 0.8151, 0.8118 |
| `356d81f38dd9473ba590f39e266f54e5` | 228 | all 10 detection classes | 0.9239, 0.8303, 0.8057, 0.7938, 0.7860 |

The official devkit evaluation covered 81 `mini_val` samples:

| Metric | Value |
|---|---:|
| mAP | 0.4371 |
| NDS | 0.4919 |
| mATE | 0.4421 |
| mASE | 0.4570 |
| mAOE | 0.5811 |
| mAVE | 0.3850 |
| mAAE | 0.4014 |

Per-class AP (mean over the official 0.5/1/2/4 m center-distance thresholds):

| Class | AP |
|---|---:|
| car | 0.8792 |
| truck | 0.6909 |
| bus | 0.9710 |
| trailer | 0.0000 |
| construction_vehicle | 0.0000 |
| pedestrian | 0.8745 |
| motorcycle | 0.5474 |
| bicycle | 0.2684 |
| traffic_cone | 0.1395 |
| barrier | 0.0000 |

Saved scene outputs for five consecutive samples in `scene-0061` are under
`outputs/phase3_centerpoint/visualizations/`, with both `_gt_pred_bev.png` and
`_gt_pred_3d.png` variants. Each prediction includes finite planar velocity;
the numerical audit also verified the project-to-global result export.

## Checkpoint

The matching official Model Zoo artifact is the `model-23M` checkpoint linked
beside `cbgs_dyn_pp_centerpoint.yaml` in the fixed OpenPCDet README. The local
file, source URL, and config are recorded in
`configs/detectors/centerpoint/nuscenes_mini.yaml`. The originally selected
voxel-size-0.075 checkpoint is no longer available and is not used here.
Random weights may be used for interface smoke tests only and are never
baseline results.
