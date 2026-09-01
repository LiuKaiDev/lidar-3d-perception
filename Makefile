.PHONY: status tree validate-cpu validate-gpu validate-assets demo-help

status:
	@echo "LiDAR 3D Perception — Phase 7A reproducibility entrypoints"
	@echo "Default detector: VoxelNeXt"
	@echo "Read: docs/12_phase7_reproducibility_entrypoints.md"

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
