#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=. .venv/bin/python tools/run_e2_density_policy.py --no-inference
