from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_phase6_summary_is_generated_and_current() -> None:
    result = subprocess.run(
        [sys.executable, "tools/generate_phase6_summary.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_user_facing_markdown_links_resolve() -> None:
    paths = [ROOT / "README.md", ROOT / "START_HERE.md", *sorted((ROOT / "docs").rglob("*.md")), *sorted((ROOT / "reports").glob("*.md"))]
    pattern = re.compile(r"\[[^\]]+\]\(([^)#]+)\)")
    for path in paths:
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (path.parent / target).resolve().exists(), f"{path}: {target}"


def test_phase6_summary_figures_are_committed_and_compact() -> None:
    report = (ROOT / "reports/phase6_summary.md").read_text(encoding="utf-8")
    targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert targets
    for target in targets:
        figure = (ROOT / "reports" / target).resolve()
        assert figure.exists(), figure
        assert figure.stat().st_size < 1_000_000, figure


def test_committed_json_and_yaml_artifacts_parse() -> None:
    for path in [ROOT / "reports/phase6_provenance.json", ROOT / "experiments/e3_late_fusion/metrics.json", ROOT / "experiments/e4_repeat_validation/metrics.json"]:
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    for path in [ROOT / "configs/system/portfolio.yaml", ROOT / "configs/detectors/centerpoint/nuscenes_mini.yaml", ROOT / "configs/detectors/voxelnext/nuscenes_mini.yaml", ROOT / ".github/workflows/cpu-tests.yml"]:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None


def test_repository_does_not_track_runtime_assets_or_modify_openpcdet() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    forbidden = (".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".engine")
    assert not [path for path in tracked if path.endswith(forbidden) or "/cache/" in path or path.startswith("cache/")]
    assert not [path for path in tracked if path.startswith("outputs/") and path != "outputs/.gitkeep"]
    # Compare the gitlink itself so this check also works in a CPU checkout
    # where the optional OpenPCDet submodule is deliberately not initialized.
    assert subprocess.run(["git", "diff", "--quiet", "--submodule=log", "--", "third_party/OpenPCDet"], cwd=ROOT).returncode == 0
    submodule = ROOT / "third_party/OpenPCDet"
    if (submodule / ".git").exists() or (submodule / "HEAD").exists():
        assert subprocess.run(["git", "-C", str(submodule), "diff", "--quiet"], cwd=ROOT).returncode == 0


def test_new_user_facing_docs_have_no_machine_absolute_paths() -> None:
    paths = [ROOT / "README.md", ROOT / "START_HERE.md", ROOT / "docs/13_system_architecture.md", ROOT / "docs/14_portfolio_walkthrough.md", ROOT / "docs/15_third_party_and_assets.md", ROOT / "docs/16_phase7_release_readiness.md", ROOT / "docs/releases/v0.7.0-rc1.md", ROOT / "reports/phase6_summary.md", ROOT / ".github/workflows/cpu-tests.yml"]
    forbidden = ("/home/chaos", "/workspace/", "C:\\Users\\")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path
