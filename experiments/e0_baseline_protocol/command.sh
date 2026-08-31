#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. .venv/bin/python -m lidar_perception.experiments.manifest \
  experiments/e0_baseline_protocol/config.yaml
