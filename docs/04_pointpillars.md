# PointPillars Baseline

The project uses PointPillars from the fixed OpenPCDet submodule as a
third-party baseline. The project-owned code does not copy the network; it
adapts its outputs into `Box3D` and `PredictionBatch`.

## Pipeline

```text
KITTI point cloud [x,y,z,intensity]
    -> PillarVFE
    -> PointPillarScatter (BEV pseudo-image)
    -> BaseBEVBackbone (2D CNN)
    -> AnchorHeadSingle
    -> box decode + direction correction
    -> GPU NMS
    -> project Box3D / PredictionBatch
```

The actual config is
`third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml`. Its
architecture names are `PillarVFE`, `PointPillarScatter`, `BaseBEVBackbone`,
and `AnchorHeadSingle`; the class order is `Car, Pedestrian, Cyclist`. The
fixed revision builds this model with 4,834,888 parameters and the KITTI val
infos contain 3769 frames.

## Why It Is Fast

Pillars avoid expensive 3D convolution along the vertical axis. Each pillar is
encoded once, scattered into a 2D grid, and processed by mature 2D CNN
operators. This trades some vertical geometric detail for lower latency and
memory use.

## Anchors and Outputs

`AnchorHeadSingle` is anchor-based: each BEV location has predefined anchor
sizes and rotations for each class. Training assigns ground-truth boxes to
anchors using the configured IoU thresholds. The head predicts classification
scores, seven box residuals (center, dimensions, yaw) and a two-bin direction
classifier. Decode and NMS convert these predictions to final boxes.

## Project Boundary

`OpenPCDetBackend` owns config/checkpoint/model lifecycle and converts native
`[x,y,z,dx,dy,dz,heading]` tensors to the project convention
`center=[x,y,z]`, `size=[length,width,height]`, `yaw=heading`. Official KITTI
evaluation remains OpenPCDet's implementation behind the project wrapper. The
fixed revision reports the official KITTI AP_R40 protocol for BEV and 3D AP;
the wrapper records this protocol alongside the returned result.

## Benchmark Protocol

`tools/benchmark.py` reports both model-only and end-to-end measurements. The
model-only path prepares one frame once, then times decode and NMS with CUDA
events (or `perf_counter` on CPU). End-to-end timing includes project frame
preparation, device transfer, model inference, decode, NMS, and an explicit
CUDA synchronization. Both summaries include mean, median, P95 latency, and
batch-1 FPS. Warmups and measured iterations are recorded in the JSON output.
Peak allocated and reserved CUDA memory are captured with PyTorch's memory API
after warmup for both timing scopes; the top-level peak fields refer to the
end-to-end scope.

The official checkpoint source is the `model-18M` Google Drive link in the
fixed revision's README:

```text
https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing
```

The wrapper config records this URL as `backend.checkpoint_source`, alongside
the local checkpoint filename and the associated OpenPCDet YAML. A successful
load report records the checkpoint path, source, config, key counts, and
`load_result: loaded`; incompatible key sets are rejected before inference.

The checkpoint was subsequently supplied locally at
`~/checkpoints/openpcdet/pointpillar_kitti.pth`.

## Local Phase 2 Results

The supplied checkpoint is 19,383,576 bytes with SHA-256
`4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2`. It
loaded against the fixed config with 127/127 model keys and zero missing,
unexpected, or shape-mismatch keys.

Real inference used GPU preprocessing and the project convention
`center=[x,y,z]`, `size=[length,width,height]`, `yaw=heading`:

| Frame | Predictions | Classes | Top scores |
|---|---:|---|---|
| 000000 | 29 | Car, Pedestrian, Cyclist | 0.8790, 0.7860, 0.7413 |
| 004139 | 62 | Car, Pedestrian, Cyclist | 0.9456, 0.9393, 0.9334 |
| 007480 | 50 | Car, Pedestrian, Cyclist | 0.9797, 0.9497, 0.9440 |

The native-to-`Box3D` comparison on `004139` had zero absolute error for
center, size, and yaw after thresholding. Prediction JSON files are under
`outputs/phase2_pointpillar/predictions/`; GT-vs-prediction BEV files cover all
three frames, with a 3D file for `004139`, under
`outputs/phase2_pointpillar/visualizations/`.

Official evaluation on the 3769-frame val split used OpenPCDet AP_R40:

| Class | BEV Easy | BEV Moderate | BEV Hard | 3D Easy | 3D Moderate | 3D Hard |
|---|---:|---:|---:|---:|---:|---:|
| Car | 74.4632 | 73.2043 | 72.1110 | 71.4167 | 65.8038 | 63.2164 |
| Pedestrian | 46.1495 | 43.0493 | 39.6601 | 42.9543 | 39.7137 | 36.1833 |
| Cyclist | 68.8149 | 53.0500 | 49.9794 | 65.9770 | 50.4967 | 47.0470 |

RTX 2060 benchmark, batch 1, FP32, 20 warmups, 100 measured iterations:

| Scope | Mean ms | Median ms | P95 ms | FPS |
|---|---:|---:|---:|---:|
| Model-only | 32.8620 | 32.8268 | 33.7736 | 30.4303 |
| End-to-end | 52.0698 | 51.1510 | 57.7668 | 19.2050 |

Peak allocated VRAM was 511,781,888 bytes and peak reserved VRAM was
803,209,216 bytes in the end-to-end scope. Model-only timing uses CUDA Events;
end-to-end timing uses `perf_counter` plus explicit CUDA synchronization.
Results are saved in `outputs/phase2_pointpillar/evaluation/summary.json` and
`outputs/phase2_pointpillar/benchmark.json`.
