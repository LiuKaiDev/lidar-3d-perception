.PHONY: status tree validate-cpu validate-gpu validate-assets demo-help phase6-summary cpu-tests

status:
	@echo "LiDAR 三维感知 — 文档与实验工具"
	@echo "默认模型: VoxelNeXt"
	@echo "文档入口: README.md, docs/README.md"

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
