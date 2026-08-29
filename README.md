# LiDAR 3D Perception

LiDAR 3D perception and object detection system for autonomous driving and mobile robotics.

## Current Status

**Phase 0 — Environment and repository initialization**

This starter repository intentionally contains **no perception business logic**. The first development task is to validate and freeze the WSL2 / Ubuntu / NVIDIA / PyTorch / spconv / OpenPCDet environment before Phase 1 begins.

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

Then create the GitHub repository, add `origin`, push `main`, and create `develop`.

Do **not** add OpenPCDet manually before Phase 0. Let Codex inspect the local environment first, then select and freeze a compatible OpenPCDet commit and dependency combination.
