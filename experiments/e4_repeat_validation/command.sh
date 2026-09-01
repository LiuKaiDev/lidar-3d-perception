#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=. .venv/bin/python -m lidar_perception.experiments.manifest experiments/e4_repeat_validation/config.yaml
PYTHONPATH=.:tools .venv/bin/python tools/run_e4_repeat_validation.py "$@"
