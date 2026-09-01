.PHONY: status tree validate-cpu validate-gpu validate-assets demo-help phase6-summary cpu-tests

status:
	@echo "LiDAR 3D Perception — Phase 7B portfolio packaging"
	@echo "Default detector: VoxelNeXt"
	@echo "Read: README.md, docs/13_system_architecture.md, docs/14_portfolio_walkthrough.md"

tree:
	@find . -maxdepth 3 -not -path './.git/*' | sort

validate-cpu:
	@PYTHONPATH=.:tools .venv/bin/python tools/validate_environment.py --profile cpu

validate-gpu:
	@PYTHONPATH=.:tools .venv/bin/python tools/validate_environment.py --profile gpu

validate-assets:
	@PYTHONPATH=.:tools .venv/bin/python tools/validate_assets.py --detector voxelnext

demo-help:
	@PYTHONPATH=.:tools .venv/bin/python tools/demo_nuscenes.py --help

phase6-summary:
	@PYTHONPATH=.:tools .venv/bin/python tools/generate_phase6_summary.py --check

cpu-tests:
	@PYTHONPATH=.:tools .venv/bin/python -m pytest -q tests/test_boxes3d.py tests/test_detection_schemas.py tests/test_geometry.py tests/test_matching.py tests/test_e3_fusion.py tests/test_phase6_protocol.py tests/test_phase7_entrypoints.py tests/test_phase7_docs.py
