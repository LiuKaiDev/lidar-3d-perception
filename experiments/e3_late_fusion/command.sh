#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=. .venv/bin/python -m lidar_perception.experiments.manifest experiments/e3_late_fusion/config.yaml
PYTHONPATH=.:tools .venv/bin/python tools/run_e3_late_fusion.py "$@"
