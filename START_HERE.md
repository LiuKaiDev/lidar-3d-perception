# Start Here

This repository has completed Phase 6.4 and is in Phase 7 engineering and
reproducibility packaging. VoxelNeXt is the default detector; CenterPoint is
the baseline and frozen E3 fusion is an optional directional ablation.

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

Phase 0 is frozen in:

```text
docs/environment.lock.md
```

Phases 1-6 are complete. Phase 6 closure and exact E4 repeat validation are
recorded in `docs/11_phase6_closure.md` and `experiments/e4_repeat_validation/`.
The local nuScenes mini data and both fixed checkpoints are external assets;
they are validated without downloading or modifying them:

```bash
PYTHONPATH=.:tools .venv/bin/python tools/validate_environment.py --profile gpu
PYTHONPATH=.:tools .venv/bin/python tools/validate_assets.py --detector voxelnext
PYTHONPATH=.:tools .venv/bin/python tools/demo_nuscenes.py --sample-token <token>
```

All reported Phase 6 results are `nuScenes v1.0-mini exploratory experiment`
results, not full-nuScenes benchmark or SOTA claims. Mini-val contains two
scenes and was already used in earlier phases.
