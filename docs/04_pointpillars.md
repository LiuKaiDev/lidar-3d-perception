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

The official Model Zoo checkpoint could not be downloaded in the current
network environment, so real PointPillars inference, evaluation and benchmark
results remain pending. No synthetic detector output is presented as a
baseline result.

The official checkpoint source is the `model-18M` Google Drive link in the
fixed revision's README:

```text
https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing
```

The wrapper config records this URL as `backend.checkpoint_source`, alongside
the local checkpoint filename and the associated OpenPCDet YAML. A successful
load report records the checkpoint path, source, config, key counts, and
`load_result: loaded`; incompatible key sets are rejected before inference.

Both direct `curl`/Drive endpoints and `gdown 6.1.0` were attempted; the
current environment returned `Network is unreachable`. This is the only
remaining Phase 2 blocker.
