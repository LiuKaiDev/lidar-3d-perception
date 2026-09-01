# Phase 7A Reproducibility Entrypoints

Phase 7A adds engineering entrypoints around the frozen Phase 6 result. It
does not train, tune, regenerate experiment candidates, or modify E0-E4.
VoxelNeXt is the default detector. CenterPoint remains the baseline and E3 is
an optional frozen late-fusion ablation.

## Environment Validation

```bash
PYTHONPATH=.:tools .venv/bin/python tools/validate_environment.py --profile cpu
PYTHONPATH=.:tools .venv/bin/python tools/validate_environment.py --profile gpu
```

The CPU profile checks Python, Torch, NumPy, PyYAML, project structure, and
configuration parsing. It does not require CUDA and does not claim detector
inference. The GPU profile additionally checks CUDA, spconv, nuScenes devkit,
the fixed OpenPCDet revision, and required compiled CUDA op imports. Both are
read-only; `--output report.json` is the only way they write a report.

## Asset Validation

```bash
PYTHONPATH=.:tools .venv/bin/python tools/validate_assets.py --detector voxelnext
PYTHONPATH=.:tools .venv/bin/python tools/validate_assets.py \
  --detector e3 --split mini_train --split mini_val --require-cache
```

The first command checks assets required for a single-sample demo: detector
config, checkpoint presence/hash, nuScenes root/version, and minimal metadata.
Experiment caches are optional and are never generated. Passing `--split`
checks cache counts/provenance; `--require-cache` makes missing or incompatible
caches fatal. Path precedence is CLI override, environment variable, system
config, then detector config. Supported variables are `NUSCENES_ROOT`,
`CENTERPOINT_CHECKPOINT`, `VOXELNEXT_CHECKPOINT`, and `PHASE6_CACHE_ROOT`.

Validators never install packages, download assets, run inference, create a
prediction cache, or modify input files.

## Single-Sample Demo

```bash
PYTHONPATH=.:tools .venv/bin/python tools/demo_nuscenes.py \
  --sample-token <mini-sample-token>
PYTHONPATH=.:tools .venv/bin/python tools/demo_nuscenes.py \
  --detector centerpoint --sample-token <mini-sample-token>
PYTHONPATH=.:tools .venv/bin/python tools/demo_nuscenes.py \
  --detector e3 --sample-token <mini-sample-token>
```

The omitted detector selects VoxelNeXt through `configs/system/portfolio.yaml`.
The demo writes to `outputs/demo/<detector>/<sample-token>.json` by default and uses the
project `PredictionBatch` box schema: class, score, center, size, yaw, and
velocity. No GT is loaded or used, and no Phase 6 cache is written.

`PredictionBatch.runtime_ms` retains its existing meaning: synchronized model
forward, decode, and NMS only. It is not end-to-end latency. Separate wall-time
fields record frame loading, model loading, prediction calls, and fusion. E3
runs CenterPoint and VoxelNeXt sequentially, releases each backend between
models, and reads only `experiments/e3_late_fusion/frozen_config.json`.

## Claims and Limits

All accuracy results remain `nuScenes v1.0-mini exploratory experiment`
results. They are not full-nuScenes benchmark or SOTA claims. Mini-val has two
scenes and was already exposed in earlier phases. E3 improves far/sparse custom
recall but increases false positives and remains below VoxelNeXt on official
mAP/NDS, so VoxelNeXt is the engineering default.
