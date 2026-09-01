# Portfolio Walkthrough

This project is a reproducibility-first LiDAR detection study shaped by an RTX
2060 6 GB constraint and access to the official nuScenes mini dataset. The goal
was a defensible engineering comparison, not a leaderboard claim.

## Research decisions

Full nuScenes training was out of scope for the available hardware and time, so
the protocol uses `mini_train` for calibration and a frozen, previously exposed
`mini_val` (two scenes) for confirmation. E1 selected a range policy, E2 tested
sparsity-aware scoring, E3 measured detector complementarity, and E4 repeated
the frozen result exactly. Configs, hashes, seeds, and negative controls are
stored under `experiments/`; no parameter is selected from confirmatory labels.

E1 confirmed a candidate-coverage ceiling and a recall weakness beyond 50 m, but
its range-aware recalibration did not change custom recall and regressed official
metrics; it is therefore a negative result, not a selected detector policy.
E2's score-plus-point-count calibration improved confidence diagnostics only;
membership and custom recall stayed fixed, so it is retained as directional
evidence rather than a new detector. E3 recovered complementary objects: on mini-val, 50m+
recall is 24.60% for E3 versus 23.40% VoxelNeXt and 14.00% CenterPoint, while
0-5-point recall is 71.17% versus 68.71% and 63.30%. The cost is 19,678 FP and
16.13% precision, versus VoxelNeXt's 9,591 FP and 27.95% precision. Naive Union
is a deliberately high-FP control. Official mAP/NDS still favor VoxelNeXt
(0.5209/0.5442) over E3 (0.4996/0.5210). E4 reports exact custom and official
repeat matches, supporting reproducibility rather than generalization.

Those results explain the product decision: VoxelNeXt is the default, CenterPoint
is the baseline, E3 is a directional ablation, and Naive Union is diagnostic.

## Engineering packaging

Phase 7 turns the research scripts into explicit config-driven validators, a
single-sample demo, stable schemas and cache provenance, a generated summary,
architecture documentation, and a focused CPU CI target. The CI path exercises
geometry, matching, bootstrap, fusion, provenance, config parsing, and report
checks without a GPU or detector initialization.

## Live demonstration commands

```bash
PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile cpu
PYTHONPATH=. .venv/bin/python tools/generate_phase6_summary.py --check
PYTHONPATH=. .venv/bin/python tools/validate_assets.py --detector voxelnext
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --sample-token <token>
PYTHONPATH=. .venv/bin/python -m pytest -q
```

## Code worth discussing

Start with `lidar_perception/detection/schemas.py` and
`lidar_perception/experiments/fusion.py` for stable contracts and deterministic
E3 behavior; then show `lidar_perception/evaluation/matching.py`,
`experiments/cache.py`, and `tools/demo_nuscenes.py` for evaluation boundaries,
provenance, lazy imports, and runtime accounting.

## Limits and next steps

Mini-val exposure, two scenes, external assets, and scope-specific timing limit
the claims. A release candidate needs an owner-selected project license, an
unseen/full-split repeat, a successful remote CI run, and a review of asset
distribution terms. Tracking, training, and a Phase 7 release/tag are outside
this phase.
