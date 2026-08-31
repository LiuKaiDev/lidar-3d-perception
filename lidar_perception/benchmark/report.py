"""Deterministic Phase 5 benchmark aggregation and report writers."""

from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch

from lidar_perception.utils.io import save_json


REPORT_SCHEMA = "lidar_perception.phase5_benchmark.v1"


def sha256_file(path: str | Path) -> str | None:
    target = Path(path).expanduser()
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment(opcdet_root: str | Path = "third_party/OpenPCDet") -> dict[str, Any]:
    """Capture runtime identifiers used to reproduce a local measurement."""

    cuda_available = bool(torch.cuda.is_available())
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu": platform.machine(),
        "cpu_model": _cpu_model(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "openpcdet_revision": None,
        "project_revision": git_revision(),
    }
    try:
        environment["openpcdet_revision"] = subprocess.check_output(
            ["git", "-C", str(Path(opcdet_root)), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return environment


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return platform.processor() or None
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _accuracy_row(model: dict[str, Any]) -> dict[str, Any]:
    accuracy = model.get("accuracy") or {}
    return {
        "model": model.get("name"),
        "input": f"{model.get('sweeps', '?')} sweeps",
        "mAP": accuracy.get("mAP"),
        "NDS": accuracy.get("NDS"),
        "mATE": accuracy.get("mATE"),
        "mASE": accuracy.get("mASE"),
        "mAOE": accuracy.get("mAOE"),
        "mAVE": accuracy.get("mAVE"),
        "mAAE": accuracy.get("mAAE"),
        "status": model.get("status"),
    }


def build_report(
    *,
    dataset: dict[str, Any],
    protocol: dict[str, Any],
    models: Iterable[dict[str, Any]],
    environment: dict[str, Any],
    historical_reference: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    model_list = [dict(model) for model in models]
    comparable_accuracy = [
        model for model in model_list
        if model.get("status") in {"completed", "cached_accuracy", "accuracy_completed"}
        and model.get("accuracy", {}).get("dataset") == dataset.get("name")
        and model.get("accuracy", {}).get("version") == dataset.get("version")
        and model.get("accuracy", {}).get("split") == dataset.get("split")
    ]
    completed = [model for model in comparable_accuracy if model.get("benchmark")]
    report_status = "PASS" if len(completed) >= 2 else "BLOCKED"
    return {
        "schema_version": REPORT_SCHEMA,
        "status": report_status,
        "comparable_dataset": {
            **dataset,
            "label": f"{dataset.get('name')} {dataset.get('version')} / pipeline validation",
            "protocol": "official nuScenes detection devkit",
        },
        "protocol": dict(protocol),
        "models": model_list,
        "accuracy_comparison": [_accuracy_row(model) for model in comparable_accuracy],
        "runtime_comparison": [
            {
                "model": model.get("name"),
                "input": f"{model.get('sweeps', '?')} sweeps",
                "model_only": model.get("benchmark", {}).get("model_only"),
                "end_to_end": model.get("benchmark", {}).get("end_to_end"),
                "peak_memory_allocated_bytes": model.get("benchmark", {}).get("peak_memory_allocated_bytes"),
                "peak_memory_reserved_bytes": model.get("benchmark", {}).get("peak_memory_reserved_bytes"),
                "model_only_peak_memory_allocated_bytes": model.get("benchmark", {}).get("model_only_peak_memory_allocated_bytes"),
                "model_only_peak_memory_reserved_bytes": model.get("benchmark", {}).get("model_only_peak_memory_reserved_bytes"),
                "end_to_end_peak_memory_allocated_bytes": model.get("benchmark", {}).get("end_to_end_peak_memory_allocated_bytes"),
                "end_to_end_peak_memory_reserved_bytes": model.get("benchmark", {}).get("end_to_end_peak_memory_reserved_bytes"),
                "status": model.get("status"),
            }
            for model in model_list
            if model.get("benchmark")
        ],
        "historical_reference": historical_reference,
        "environment": environment,
        "limitations": limitations or [],
    }


def write_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown views of a Phase 5 report."""

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark": output / "benchmark.json",
        "accuracy": output / "accuracy.json",
        "environment": output / "environment.json",
        "csv": output / "benchmark.csv",
        "markdown": output / "README.md",
    }
    save_json(report, paths["benchmark"])
    save_json({
        "schema_version": REPORT_SCHEMA,
        "status": report["status"],
        "dataset": report["comparable_dataset"],
        "models": report["accuracy_comparison"],
        "historical_reference": report.get("historical_reference"),
    }, paths["accuracy"])
    save_json(report["environment"], paths["environment"])

    columns = [
        "model", "input", "mAP", "NDS", "mean_latency_ms", "p95_latency_ms", "FPS",
        "peak_vram_allocated_bytes", "peak_vram_reserved_bytes", "status",
    ]
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["accuracy_comparison"]:
            runtime = next((
                item for item in report["runtime_comparison"] if item["model"] == row["model"]
            ), {})
            end_to_end = runtime.get("end_to_end") or {}
            writer.writerow({
                "model": row["model"],
                "input": row["input"],
                "mAP": row["mAP"],
                "NDS": row["NDS"],
                "mean_latency_ms": end_to_end.get("mean_ms"),
                "p95_latency_ms": end_to_end.get("p95_ms"),
                "FPS": end_to_end.get("fps_batch1"),
                "peak_vram_allocated_bytes": runtime.get("peak_memory_allocated_bytes"),
                "peak_vram_reserved_bytes": runtime.get("peak_memory_reserved_bytes"),
                "status": row["status"],
            })

    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    return paths


def render_markdown(report: dict[str, Any]) -> str:
    dataset = report["comparable_dataset"]
    lines = [
        "# Phase 5 Multi-Model Benchmark",
        "",
        f"**Status:** {report['status']}",
        f"**Comparable dataset:** {dataset['name']} {dataset['version']} / {dataset['split']}",
        f"**Accuracy protocol:** {dataset['protocol']}",
        "",
        "## Comparable Accuracy",
        "",
        "Only local results using the same dataset, split, and official protocol are listed here.",
        "",
        "| Model | Input | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["accuracy_comparison"]:
        lines.append("| {model} | {input} | {mAP} | {NDS} | {mATE} | {mASE} | {mAOE} | {mAVE} | {mAAE} | {status} |".format(**row))
    if not report["accuracy_comparison"]:
        lines.append("| No comparable local results yet | | | | | | | | | BLOCKED |")
    lines.extend([
        "",
        "## Runtime",
        "",
        "| Model | Scope | Mean (ms) | Median (ms) | P95 (ms) | FPS | Peak allocated (bytes) | Peak reserved (bytes) | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in report["runtime_comparison"]:
        for scope, label in (("model_only", "model-only"), ("end_to_end", "end-to-end")):
            timing = row.get(scope) or {}
            allocated = row.get(f"{scope}_peak_memory_allocated_bytes")
            reserved = row.get(f"{scope}_peak_memory_reserved_bytes")
            lines.append(
                f"| {row['model']} | {label} | {timing.get('mean_ms')} | {timing.get('median_ms')} | "
                f"{timing.get('p95_ms')} | {timing.get('fps_batch1')} | "
                f"{allocated} | {reserved} | {row['status']} |"
            )
    if not report["runtime_comparison"]:
        lines.append("| No local runtime measurements yet | | | | | | | | BLOCKED |")
    lines.extend(["", "## Model Provenance", ""])
    for model in report["models"]:
        provenance = model.get("provenance", {})
        lines.append(
            f"- **{model.get('name')}**: {model.get('status')}; config `{provenance.get('config')}`; "
            f"checkpoint `{provenance.get('checkpoint')}`; sha256 `{provenance.get('checkpoint_sha256')}`."
        )
    lines.extend(["", "## Historical Reference", ""])
    lines.append("KITTI AP_R40 results are intentionally kept separate and are not ranked with nuScenes mAP/NDS.")
    if report.get("historical_reference"):
        reference = report["historical_reference"]
        runtime = reference.get("runtime") or {}
        accuracy = reference.get("accuracy") or {}
        lines.extend([
            "",
            "| Model | Dataset / protocol | Car 3D AP_R40 E/M/H | Mean E2E (ms) | FPS | Peak allocated (bytes) |",
            "|---|---|---|---:|---:|---:|",
            f"| PointPillars | {reference.get('dataset')} / {reference.get('protocol')} | "
            f"{accuracy.get('Car_3d/easy_R40')} / {accuracy.get('Car_3d/moderate_R40')} / "
            f"{accuracy.get('Car_3d/hard_R40')} | {runtime.get('mean_ms')} | {runtime.get('fps_batch1')} | "
            f"{runtime.get('peak_memory_allocated_bytes')} |",
        ])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "REPORT_SCHEMA",
    "build_report",
    "collect_environment",
    "render_markdown",
    "sha256_file",
    "write_reports",
]
