# Codex Prompt — Phase 0 Start

You are assisting with the project **LiDAR 3D Perception and Object Detection System**.

The complete frozen project design is:

`docs/project_design_v1.md`

Current stage:

**Phase 0 — Environment and repository initialization**

Read the complete design document first.

## Hard constraints

- Do not implement Phase 1 or later business functionality.
- Do not generate the whole project at once.
- Do not treat OpenPCDet code as self-developed code.
- Do not modify OpenPCDet internals in this bootstrap task.
- Do not install or upgrade CUDA / PyTorch / spconv before inspecting the actual machine.
- Do not install a Linux NVIDIA display driver inside WSL.
- Distinguish the Windows NVIDIA driver / WSL-exposed CUDA capability from a locally installed CUDA Toolkit / `nvcc`.
- Do not guess environment versions.

## First task

Before changing files or installing packages, inspect and report:

1. project goal and phase order;
2. OpenPCDet vs self-developed responsibility boundary;
3. current Git branch, status and remotes;
4. current repository tree;
5. WSL / Ubuntu / kernel / architecture;
6. GPU model, NVIDIA driver and `nvidia-smi`;
7. Python / pip / Conda or Mamba status;
8. PyTorch version, torch CUDA version, CUDA availability, GPU name and cuDNN if available;
9. GCC / G++ / CMake / Ninja / `nvcc` status;
10. whether spconv and pcdet are already installed/importable.

Then propose a **Phase 0 execution plan** covering:

- compatibility matrix decision;
- Python environment strategy;
- PyTorch / spconv strategy;
- OpenPCDet fixed revision strategy;
- `third_party/OpenPCDet` integration;
- CUDA ops build and smoke test;
- `docs/environment.lock.md` generation;
- reproducible commands;
- Git commits.

## Required pre-development output

Before implementation, output:

1. Phase objective
2. Files you expect to add or modify
3. Existing interfaces/dependencies used
4. Third-party files you will not modify
5. Risks
6. Acceptance commands

## Phase 0 acceptance criteria

Phase 0 is complete only when all applicable checks pass and exact versions are recorded:

- `nvidia-smi` works in WSL;
- `torch.cuda.is_available() == True`;
- spconv imports successfully;
- pcdet imports successfully;
- required OpenPCDet CUDA ops smoke test succeeds;
- OpenPCDet commit is fixed;
- environment versions are recorded;
- Git state is clean and reproducible.

For this first interaction, **do not perform large installations yet**. Inspect first, report facts, propose the compatibility-aware plan, and stop for review before making environment-changing decisions.
