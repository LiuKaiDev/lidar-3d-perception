# Start Here

This package is intentionally minimal.

## 1. Copy into WSL

Recommended target:

```text
~/workspace/lidar-3d-perception
```

Keep the repository and, where practical, datasets in the WSL Linux filesystem rather than under `/mnt/c` or `/mnt/d`.

## 2. Initialize Git

```bash
cd ~/workspace/lidar-3d-perception

git init -b main
git add .
git commit -m "chore: initialize lidar perception repository"
```

## 3. Push to GitHub

Create an empty GitHub repository named:

```text
lidar-3d-perception
```

Do not ask GitHub to generate a README, `.gitignore`, or license because this package already contains the initial repository contents.

Then add the remote and push `main`.

Example:

```bash
git remote add origin git@github.com:<YOUR_USERNAME>/lidar-3d-perception.git
git push -u origin main
```

The current personal-project policy keeps Phase 0 and the normal mainline on
`main`. Do not create `develop` or `phase/*` branches. Reserve `exp/*` for an
experiment that genuinely needs isolation.

## 4. Current Development State

Phase 0 is complete and frozen in:

```text
docs/environment.lock.md
```

Phase 1 implementation is present under `lidar_perception/`, `tools/`, and
`tests/`. Configure a local KITTI Object Detection dataset through:

```text
configs/datasets/kitti.yaml
```

Phase 2 PointPillars project adapters are present under
`lidar_perception/detection/`, `lidar_perception/evaluation/`, and
`lidar_perception/benchmark/`. Phase 2 is PASS with the official checkpoint,
real KITTI inference/evaluation, GT-vs-prediction visualizations, and RTX 2060
benchmark recorded in `docs/04_pointpillars.md`.

Phase 3 nuScenes/CenterPoint pipeline code is present under
`lidar_perception/datasets/nuscenes_adapter.py`,
`lidar_perception/evaluation/nuscenes.py`, and the `tools/*nuscenes*.py`
commands. It currently validates the `v1.0-mini` data path and contracts; the
matching CenterPoint checkpoint is still required for real detector results.

## 5. Stop condition

Do not enter Phase 4 until the CenterPoint checkpoint, real mini inference,
official mini evaluation, velocity visualization, and regression tests have
passed. Full nuScenes train/val metrics remain deferred until the full dataset
is available.
