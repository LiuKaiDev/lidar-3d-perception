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

The pinned OpenPCDet config is
`cbgs_voxel0075_res3d_centerpoint.yaml`:

```text
multi-sweep points
    -> voxelization
    -> MeanVFE
    -> sparse VoxelResBackBone8x
    -> HeightCompression / BEV features
    -> 2D BEV backbone
    -> CenterHead heatmaps and regressions
    -> decode, score threshold, class-agnostic NMS
    -> project Box3D / PredictionBatch
```

Unlike PointPillars' anchor head, CenterPoint is anchor-free. A Gaussian
heatmap marks object centers; regression channels recover sub-voxel center
offset, height, log dimensions, sine/cosine yaw, and planar velocity. The
CenterPoint backend preserves the native 9D box output as project size/yaw and
`velocity=[vx,vy,0]` in the reference LiDAR frame.

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

## Checkpoint

The matching Model Zoo artifact is the `model-34M` checkpoint linked beside
the pinned voxel-size-0.075 config in the fixed OpenPCDet README. Its URL and
expected local filename are recorded in
`configs/detectors/centerpoint/nuscenes_mini.yaml`. Random weights may be used
for interface smoke tests only and are never baseline results. The matching
checkpoint is not present locally in this Phase 3 run; one bounded Model Zoo
download attempt timed out. Consequently, pretrained CenterPoint inference,
mini metrics, and scene GT-vs-prediction outputs remain blocked until that
exact file is supplied.
