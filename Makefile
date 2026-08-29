.PHONY: status tree

status:
	@echo "LiDAR 3D Perception — Phase 0 bootstrap"
	@echo "Read: docs/project_design_v1.md"
	@echo "Codex prompt: prompts/phase0/00_start.md"

tree:
	@find . -maxdepth 3 -not -path './.git/*' | sort
