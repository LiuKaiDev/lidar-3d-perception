"""Validated, serializable Phase 6 experiment manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import yaml

from lidar_perception.evaluation.metrics import DEFAULT_DENSITY_BINS, DEFAULT_DISTANCE_BINS


EXPERIMENT_STATUSES = {"PLANNED", "RUNNING", "PASS", "NEGATIVE", "BLOCKED"}
REQUIRED_PRIMARY_METRICS = {"recall_50m_plus", "recall_0_5_points"}
REQUIRED_OFFICIAL_METRICS = {"mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"}
REQUIRED_RUNTIME_FIELDS = {"batch_size", "precision", "warmup", "iterations", "timing_method"}
REQUIRED_TOP_LEVEL = {
    "experiment_id",
    "status",
    "hypothesis",
    "dataset",
    "models",
    "prediction_candidate_threshold",
    "score_filtering_policy",
    "matching",
    "distance_bins",
    "density_bins",
    "target_metrics",
    "secondary_metrics",
    "official_metrics",
    "bootstrap",
    "tunable_parameters",
    "controlled_variables",
    "runtime_protocol",
    "inference_policy",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_bins(actual: Any, expected: tuple, name: str) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ValueError(f"{name} must preserve the frozen bin count")
    for raw, metric_bin in zip(actual, expected):
        if not isinstance(raw, dict):
            raise ValueError(f"{name} entries must be mappings")
        upper = float(raw["upper"])
        same_upper = math.isinf(upper) and math.isinf(metric_bin.upper) or upper == metric_bin.upper
        if raw.get("name") != metric_bin.name or float(raw["lower"]) != metric_bin.lower or not same_upper:
            raise ValueError(f"{name} differs from the frozen Phase 4 definitions")


def validate_manifest(value: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(value))
    if missing:
        raise ValueError(f"experiment manifest is missing fields: {missing}")
    if not isinstance(value["experiment_id"], str) or not value["experiment_id"].strip():
        raise ValueError("experiment_id must be non-empty")
    if value["status"] not in EXPERIMENT_STATUSES:
        raise ValueError(f"status must be one of {sorted(EXPERIMENT_STATUSES)}")
    if not isinstance(value["hypothesis"], str) or not value["hypothesis"].strip():
        raise ValueError("hypothesis must be non-empty")

    dataset = _require_mapping(value["dataset"], "dataset")
    for key in ("name", "version", "split", "role"):
        if not dataset.get(key):
            raise ValueError(f"dataset.{key} is required")
    if dataset["version"] != "v1.0-mini" or dataset["split"] not in {"mini_train", "mini_val"}:
        raise ValueError("Phase 6 V1.1 requires nuScenes v1.0-mini official splits")

    models = value["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    for model in models:
        model = _require_mapping(model, "model")
        for key in ("name", "config", "config_sha256", "checkpoint_sha256", "sweeps"):
            if model.get(key) in (None, ""):
                raise ValueError(f"model.{key} is required")
        if not isinstance(model["sweeps"], int) or model["sweeps"] < 1:
            raise ValueError("model.sweeps must be a positive integer")
        if not _SHA256.fullmatch(str(model["config_sha256"])) or not _SHA256.fullmatch(str(model["checkpoint_sha256"])):
            raise ValueError("model config/checkpoint hashes must be lowercase SHA-256 digests")

    threshold = float(value["prediction_candidate_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("prediction_candidate_threshold must be in [0, 1]")
    if not value["score_filtering_policy"]:
        raise ValueError("score_filtering_policy is required")
    if value["experiment_id"] in {"E1", "E2"}:
        if threshold != 0.1:
            raise ValueError(f"{value['experiment_id']} freezes prediction_candidate_threshold at 0.1")
        if len(models) != 1:
            raise ValueError(f"{value['experiment_id']} must use exactly one detector")
        if str(models[0].get("name", "")).lower() != "centerpoint_pointpillar":
            raise ValueError(f"{value['experiment_id']} must use the CenterPoint-PointPillar detector")
        if models[0].get("sweeps") != 10:
            raise ValueError(f"{value['experiment_id']} freezes the detector input at 10 sweeps")
        tuning = value.get("tuning")
        if isinstance(tuning, dict) and tuning.get("mini_val_used_for_tuning") is not False:
            raise ValueError(f"{value['experiment_id']} tuning must be mini_train-only")
    if value["experiment_id"] == "E2":
        inference = _require_mapping(value.get("inference_policy"), "inference_policy")
        if inference.get("point_count_source") != "current_keyframe":
            raise ValueError("E2 freezes point_count_source as current_keyframe")
        operating = _require_mapping(value.get("operating_policy"), "operating_policy")
        if operating.get("path") != "A" or operating.get("membership_change") is not False:
            raise ValueError("E2 freezes operating PATH A with invariant membership")

    matching = _require_mapping(value["matching"], "matching")
    frozen_matching = {
        "strategy": "center_distance",
        "class_aware": True,
        "one_to_one": True,
        "threshold_m": 2.0,
        "comparison": "distance <= threshold",
    }
    if any(matching.get(key) != expected for key, expected in frozen_matching.items()):
        raise ValueError("matching must preserve the frozen class-aware inclusive 2.0 m protocol")

    _validate_bins(value["distance_bins"], DEFAULT_DISTANCE_BINS, "distance_bins")
    _validate_bins(value["density_bins"], DEFAULT_DENSITY_BINS, "density_bins")
    if not REQUIRED_PRIMARY_METRICS <= set(value["target_metrics"]):
        raise ValueError("target_metrics must include both preregistered primary metrics")
    if not REQUIRED_OFFICIAL_METRICS <= set(value["official_metrics"]):
        raise ValueError("official_metrics must include all nuScenes guardrail metrics")
    if not isinstance(value["secondary_metrics"], list):
        raise ValueError("secondary_metrics must be a list")

    bootstrap = _require_mapping(value["bootstrap"], "bootstrap")
    if bootstrap.get("resampling_unit") != "scene" or bootstrap.get("paired_comparison") is not True:
        raise ValueError("bootstrap must use scene resampling and paired comparison")
    if not isinstance(bootstrap.get("repetitions"), int) or bootstrap["repetitions"] < 1:
        raise ValueError("bootstrap.repetitions must be positive")
    if not isinstance(bootstrap.get("seed"), int):
        raise ValueError("bootstrap.seed must be an integer")
    if not 0 < float(bootstrap.get("confidence_level", 0)) < 1:
        raise ValueError("bootstrap.confidence_level must be in (0, 1)")

    if not isinstance(value["tunable_parameters"], dict) or not isinstance(value["controlled_variables"], dict):
        raise ValueError("tunable_parameters and controlled_variables must be mappings")
    runtime = _require_mapping(value["runtime_protocol"], "runtime_protocol")
    if not REQUIRED_RUNTIME_FIELDS <= set(runtime):
        raise ValueError("runtime_protocol is incomplete")
    inference = _require_mapping(value["inference_policy"], "inference_policy")
    if inference.get("ground_truth_at_inference") is not False:
        raise ValueError("ground truth must be forbidden at inference")
    if not isinstance(inference.get("allowed_features"), list):
        raise ValueError("inference_policy.allowed_features must be a list")


class ExperimentManifest:
    """Immutable-by-copy wrapper around a validated experiment declaration."""

    def __init__(self, value: dict[str, Any]) -> None:
        data = copy.deepcopy(value)
        validate_manifest(data)
        self._data = data

    @property
    def experiment_id(self) -> str:
        return self._data["experiment_id"]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self._data, sort_keys=False), encoding="utf-8")
        return target

    def verify_model_configs(self, project_root: str | Path = ".") -> None:
        """Reject model config drift from the hashes preregistered in the manifest."""

        root = Path(project_root).expanduser().resolve()
        for model in self._data["models"]:
            config_path = Path(model["config"]).expanduser()
            if not config_path.is_absolute():
                config_path = root / config_path
            if not config_path.is_file():
                raise FileNotFoundError(f"model config not found: {config_path}")
            digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
            if digest != model["config_sha256"]:
                raise ValueError(f"model config hash mismatch: {model['name']}")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentManifest":
        source = Path(path).expanduser()
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("experiment config must contain a YAML mapping")
        return cls(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a frozen Phase 6 experiment manifest")
    parser.add_argument("config")
    args = parser.parse_args()
    manifest = ExperimentManifest.load(args.config)
    manifest.verify_model_configs()
    print(f"valid experiment manifest: {manifest.experiment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EXPERIMENT_STATUSES", "ExperimentManifest", "validate_manifest"]
