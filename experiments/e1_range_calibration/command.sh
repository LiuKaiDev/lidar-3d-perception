#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=. .venv/bin/python tools/run_e1_range_calibration.py
