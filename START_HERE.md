# Start Here

This repository is in Phase 7 engineering packaging. Read the top-level
[`README.md`](README.md), then run the CPU validator and focused tests:

```bash
PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile cpu
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_boxes3d.py tests/test_geometry.py tests/test_matching.py tests/test_e3_fusion.py tests/test_phase6_protocol.py tests/test_phase7_entrypoints.py tests/test_phase7_docs.py
```

When nuScenes mini data and a checkpoint are available, validate assets and run
one sample with the default VoxelNeXt detector. CenterPoint and E3 are explicit
optional modes; see the README for commands and timing semantics.

Phase 6 artifacts are frozen under `experiments/`. Do not tune on `mini_val`,
modify E1-E4 configs, add model assets to Git, or treat E3 as the default.
