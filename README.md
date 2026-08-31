# LiDAR 3D Perception

LiDAR 3D perception and object detection system for autonomous driving and mobile robotics.

## Current Status

**Phase 6.0 - V1.1 Experiment Protocol Freeze (PASS)**

Phase 0 is frozen in [`docs/environment.lock.md`](docs/environment.lock.md).
Phase 1 real KITTI geometry validation passed on frames `000000`, `004139`, and
`007480`; Phase 2 covers the PointPillars detector boundary and KITTI
evaluation. Phase 3 validated the pretrained nuScenes CenterPoint-PointPillar
pipeline, Phase 4 adds project-owned matching and bad-case analysis, and Phase
5 compares CenterPoint-PointPillar with VoxelNeXt under one official mini
accuracy and RTX 2060 runtime protocol. Results are recorded under `outputs/`.

## Project Strategy

OpenPCDet is used as the third-party 3D detection backend. The project will independently build the surrounding engineering and research system, including:

- unified project configuration and prediction schemas
- 3D geometry utilities
- dataset analysis
- distance-aware evaluation
- point-density-aware evaluation
- bad-case mining
- runtime / VRAM benchmarking
- BEV / 3D / scene visualization
- reproducible experiment reporting
- tests and documentation

See [`docs/project_design_v1.md`](docs/project_design_v1.md) for the frozen V1.0 project design.

## Planned Mainline

1. Phase 0 — Environment and repository initialization
2. Phase 1 — KITTI data and 3D geometry
3. Phase 2 — PointPillars baseline
4. Phase 3 — nuScenes + CenterPoint-PointPillar (validated mini pipeline)
5. Phase 4 — Layered evaluation and bad-case mining (validated mini analysis)
6. Phase 5 — Multi-model benchmark
7. Phase 6 — Long-range sparse-object optimization
8. Phase 7 — Engineering and portfolio packaging
9. Phase 8 — 3D tracking (advanced, only after the mainline)

## Repository Bootstrap

After copying this folder into WSL:

```bash
cd ~/workspace
mv /path/to/lidar-3d-perception-starter lidar-3d-perception
cd lidar-3d-perception

git init -b main
git add .
git commit -m "chore: initialize lidar perception repository"
```

Then create the GitHub repository, add `origin`, and push `main`. Phase 0 and
the normal project mainline stay on `main`; do not create `develop` or
`phase/*` branches for this personal project.

OpenPCDet is already integrated as the fixed Phase 0 submodule at
`third_party/OpenPCDet`; Phase 1 geometry does not call it for core parsing or
coordinate conversion.

## Phase 1 Validation

The validated KITTI Object Detection root is configured as
`~/datasets/kitti`. The training split contains 7481 aligned Velodyne,
calibration, label, and image files. Full ground-truth statistics cover 7481
frames and 40570 non-`DontCare` annotations.

Run the tests and tools with the Phase 0 virtual environment:

```bash
source .venv/bin/activate
PYTHONPATH= pytest -q

python tools/analyze_dataset.py \
  --config configs/datasets/kitti.yaml \
  --output outputs/phase1_validation/kitti_training_stats.json

python tools/visualize.py \
  --config configs/datasets/kitti.yaml \
  --frame-id 004139 \
  --view bev \
  --output outputs/phase1_validation/004139_bev.png
```

The fixed real-data validation frames and generated image paths are recorded
in [`docs/02_coordinate_systems.md`](docs/02_coordinate_systems.md).

## Phase 2 PointPillars

The fixed OpenPCDet revision supplies the PointPillars network and CUDA ops;
the project adapter lives in `lidar_perception/detection/`. The wrapper config
is [`configs/detectors/pointpillar/kitti.yaml`](configs/detectors/pointpillar/kitti.yaml).
OpenPCDet KITTI data is linked without copying:

```bash
ln -sfn ~/datasets/kitti/training third_party/OpenPCDet/data/kitti/training
ln -sfn ~/datasets/kitti/testing third_party/OpenPCDet/data/kitti/testing
```

After downloading the official Model Zoo `model-18M` checkpoint to the path in
the wrapper config, the project commands are:

```bash
python tools/infer.py --config configs/detectors/pointpillar/kitti.yaml \
  --frame-id 004139 --output outputs/phase2_pointpillar/predictions/004139.json

python tools/visualize.py --config configs/datasets/kitti.yaml \
  --frame-id 004139 --predictions outputs/phase2_pointpillar/predictions/004139.json \
  --view bev --output outputs/phase2_pointpillar/visualizations/004139_gt_pred_bev.png

CUDA_HOME="$PWD/.cuda-home-12.0" \
LD_LIBRARY_PATH="$PWD/.cuda-home-12.0/nvvm/lib64:$PWD/.cuda-home-12.0/lib64:$LD_LIBRARY_PATH" \
python tools/evaluate.py --config configs/detectors/pointpillar/kitti.yaml \
  --split val --output-dir outputs/phase2_pointpillar/evaluation

python tools/benchmark.py --config configs/detectors/pointpillar/kitti.yaml \
  --frame-id 004139 --output outputs/phase2_pointpillar/benchmark.json
```

OpenPCDet's fixed-revision KITTI evaluator uses AP_R40; evaluation and
benchmark JSON files record the protocol and timing scope so paper/model-zoo
numbers cannot be confused with local measurements. The measured results are
summarized in [`docs/04_pointpillars.md`](docs/04_pointpillars.md).

## Phase 3 nuScenes + CenterPoint

The project-owned nuScenes adapter, multi-sweep transforms, CenterPoint
boundary, velocity-preserving schema conversion, official mini evaluator, and
scene visualizer are documented in
[`docs/05_nuscenes_centerpoint.md`](docs/05_nuscenes_centerpoint.md). The
available data is `~/datasets/nuscenes/v1.0-mini` and is pipeline validation
only. Phase 3 uses the official CenterPoint-PointPillar baseline
`cbgs_dyn_pp_centerpoint.yaml`; the originally selected voxel-size-0.075
checkpoint is no longer available. The exact config/source/path are recorded in
[`configs/detectors/centerpoint/nuscenes_mini.yaml`](configs/detectors/centerpoint/nuscenes_mini.yaml).

Run a sample with:

```bash
python tools/infer_nuscenes.py --sample-token <sample-token>
python tools/visualize_nuscenes.py --sample-token <sample-token>
python tools/evaluate_nuscenes.py
```

Prepare OpenPCDet's ignored KITTI metadata once (raw data is symlinked, not
copied):

```bash
ln -sfn ~/datasets/kitti/training third_party/OpenPCDet/data/kitti/training
ln -sfn ~/datasets/kitti/testing third_party/OpenPCDet/data/kitti/testing
cd third_party/OpenPCDet
PYTHONPATH= ../../.venv/bin/python -m pcdet.datasets.kitti.kitti_dataset \
  create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## Phase 4 Distance / Density Analysis

The project-owned matcher, distance/density evaluators, bad-case miner, and
reporting pipeline are documented in
[`docs/08_distance_density_analysis.md`](docs/08_distance_density_analysis.md).
Run the complete mini-dataset analysis with cached Phase 3 predictions (the
runner fills only missing valid samples):

```bash
PYTHONPATH=. python tools/analyze_phase4.py \
  --config configs/analysis/phase4_nuscenes.yaml
```

Reports and representative snapshots are written under
`outputs/phase4_analysis/`. Results are exploratory `nuScenes v1.0-mini`
analysis and do not replace official nuScenes metrics.

## Phase 5 Multi-Model Benchmark

Phase 5 uses one config-driven OpenPCDet boundary for the comparable
nuScenes candidates:

- CenterPoint-PointPillar: `configs/detectors/centerpoint/nuscenes_mini.yaml`
- VoxelNeXt: `configs/detectors/voxelnext/nuscenes_mini.yaml`
- PointPillars MultiHead (optional): `configs/detectors/pointpillar/nuscenes_mini.yaml`

The exact configs and official checkpoint sources come from the pinned
OpenPCDet revision. The validated checkpoint hashes are:

- CenterPoint: `955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a`
- VoxelNeXt: `9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b`

Never reuse the KITTI PointPillars checkpoint for nuScenes.

All comparable accuracy below is local `nuScenes v1.0-mini / mini_val`
pipeline validation from the official devkit, not full train/val results:

| Model | mAP | NDS | Mean E2E | P95 E2E | FPS | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|
| CenterPoint-PointPillar | 0.4371 | 0.4919 | 70.24 ms | 77.62 ms | 14.24 | 286,804,992 / 446,693,376 B |
| VoxelNeXt | 0.5218 | 0.5446 | 111.69 ms | 131.07 ms | 8.95 | 146,302,464 / 180,355,072 B |

Run the sequential benchmark and report generation with the fixed protocol
(batch 1, FP32, 20 warmups, 100 measured iterations, synchronized CUDA
timing, preprocessing included only in end-to-end timing):

```bash
PYTHONPATH=. .venv/bin/python tools/benchmark_phase5.py \
  --config configs/benchmark/phase5_nuscenes.yaml
```

To reproduce VoxelNeXt accuracy from scratch:

```bash
PYTHONPATH=. .venv/bin/python tools/evaluate_nuscenes.py \
  --config configs/detectors/voxelnext/nuscenes_mini.yaml \
  --output-dir outputs/phase5_benchmark/voxelnext/evaluation
```

Raw file loading and multi-sweep assembly are outside both timing scopes.
Model-only uses a prepared device-resident batch; end-to-end additionally
includes CPU point preprocessing/voxelization, host-to-device transfer, and
project schema conversion. Add `--skip-runtime` for report assembly only or
`--evaluate-missing` when an official accuracy cache is absent or mismatched.
Reports are written to
`outputs/phase5_benchmark/` (`benchmark.json`, `accuracy.json`,
`benchmark.csv`, `environment.json`, and `README.md`). The main table only
contains local results from `nuScenes v1.0-mini / mini_val`; the historical
KITTI PointPillars AP_R40 result remains a separate reference and is never
ranked with nuScenes mAP/NDS.

## Phase 6 V1.1 Scope

Due to the RTX 2060 6GB and available-data constraints, Phase 6 is explicitly
controlled exploratory optimization on `nuScenes v1.0-mini`, not a full
nuScenes benchmark or SOTA claim. Parameters are selected on official
`mini_train`, frozen before confirmatory `mini_val`, and every result is
labeled `nuScenes v1.0-mini exploratory experiment`. The protocol includes
GT-leakage prevention, frozen distance/density metrics, paired scene-level
bootstrap, exact prediction-cache provenance, runtime accounting, and retained
negative results. No optimization experiment was executed in Phase 6.0.

See [`docs/project_design_v1_1_amendment.md`](docs/project_design_v1_1_amendment.md),
[`docs/10_phase6_experiment_protocol.md`](docs/10_phase6_experiment_protocol.md),
and [`experiments/README.md`](experiments/README.md).
