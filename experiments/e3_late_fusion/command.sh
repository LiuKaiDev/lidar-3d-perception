#!/usr/bin/env bash
set -euo pipefail
# E3 orchestration is intentionally cache-first.  Populate compatible
# CenterPoint/VoxelNeXt PredictionCache entries before running search/eval.
PYTHONPATH=. .venv/bin/python -m lidar_perception.experiments.manifest experiments/e3_late_fusion/config.yaml
