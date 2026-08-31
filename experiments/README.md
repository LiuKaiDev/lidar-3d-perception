# Phase 6 Experiment Index

Phase 6 V1.1 is controlled exploratory work on `nuScenes v1.0-mini` under the
compute constraints recorded in `docs/project_design_v1_1_amendment.md`.
Negative results remain in this index and are not deleted.

| ID | Experiment | Status | Parameter selection | Confirmation |
|---|---|---|---|---|
| E0 | Frozen baseline protocol | PASS | Existing verified records | Existing `mini_val` records |
| E1 | Predicted-range-aware calibration | NEGATIVE | `mini_train` only | Frozen settings on `mini_val` |
| E2 | Predicted-box sparsity-aware policy | PASS | `mini_train` only | Frozen settings on `mini_val` |
| E3 | CenterPoint + VoxelNeXt late fusion | PLANNED | `mini_train` only | Frozen settings on `mini_val` |
| E4 | Repeat validation and final ablation | PLANNED | No new `mini_val` tuning | `mini_val` repeat/report |

Allowed statuses are `PLANNED`, `RUNNING`, `PASS`, `NEGATIVE`, and `BLOCKED`.
Copy `experiments/_template/` for each new experiment, fill the complete
manifest before execution, and validate it with:

```bash
PYTHONPATH=. .venv/bin/python -m lidar_perception.experiments.manifest \
  experiments/<experiment>/config.yaml
```

Large prediction caches, logs, checkpoints, and generated figures remain
ignored runtime artifacts. Small manifests and result summaries should be
committed so attempted and negative experiments stay auditable.
