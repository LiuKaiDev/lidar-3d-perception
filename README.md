# LiDAR 3D Perception

LiDAR 3D perception and object detection system for autonomous driving and mobile robotics.

## Current Status

**Phase 2 — PointPillars baseline (implementation complete; checkpoint validation pending)**

Phase 0 is frozen in [`docs/environment.lock.md`](docs/environment.lock.md).
This phase adds project-owned KITTI parsing, coordinate transforms, oriented
3D boxes, projection, statistics, visualization, and deterministic tests.
Phase 1 real KITTI geometry validation passed on frames `000000`, `004139`, and
`007480`. Phase 2 now includes the project-owned detector boundary, unified
prediction schema, PointPillars adapter, official evaluation wrapper, and
CUDA-synchronized benchmark tooling. The official PointPillars checkpoint is
not yet available in the current network environment, so inference/evaluation
remain explicitly blocked.

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
4. Phase 3 — nuScenes + CenterPoint
5. Phase 4 — Layered evaluation and bad-case mining
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
  --frame-id 004139 --output outputs/predictions/004139.json

python tools/visualize.py --config configs/datasets/kitti.yaml \
  --frame-id 004139 --predictions outputs/predictions/004139.json \
  --view bev --output outputs/figures/004139_gt_pred_bev.png

python tools/evaluate.py --config configs/detectors/pointpillar/kitti.yaml \
  --split val --output-dir outputs/metrics/pointpillar_kitti

python tools/benchmark.py --config configs/detectors/pointpillar/kitti.yaml \
  --frame-id 004139 --output outputs/metrics/pointpillar_benchmark.json
```

The project currently validates these commands' argument handling and clear
missing-checkpoint errors, but does not claim detector results until the
official checkpoint is loaded. OpenPCDet's fixed-revision KITTI evaluator uses
AP_R40; evaluation and benchmark JSON files record the protocol and timing
scope so paper/model-zoo numbers cannot be confused with local measurements.

Prepare OpenPCDet's ignored KITTI metadata once (raw data is symlinked, not
copied):

```bash
ln -sfn ~/datasets/kitti/training third_party/OpenPCDet/data/kitti/training
ln -sfn ~/datasets/kitti/testing third_party/OpenPCDet/data/kitti/testing
cd third_party/OpenPCDet
PYTHONPATH= ../../.venv/bin/python -m pcdet.datasets.kitti.kitti_dataset \
  create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
