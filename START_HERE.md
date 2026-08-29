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

## 4. Start Codex

Open Codex with the repository root as its working directory and give it the contents of:

```text
prompts/phase0/00_start.md
```

The design document it must read is already included at:

```text
docs/project_design_v1.md
```

## 5. Stop condition

Do not enter Phase 1 until Phase 0 acceptance criteria are satisfied and the environment is frozen in `docs/environment.lock.md`.
