"""Shared bins and safe metric helpers for Phase 4 reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class MetricBin:
    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name or not np.isfinite(self.lower) or self.lower < 0:
            raise ValueError("metric bins require a non-empty name and finite non-negative lower bound")
        if not (self.upper > self.lower or np.isinf(self.upper)):
            raise ValueError("metric bin upper bound must be greater than lower bound")

    def contains(self, value: float) -> bool:
        return self.lower <= float(value) < self.upper

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lower": self.lower, "upper": self.upper}


DEFAULT_DISTANCE_BINS = (
    MetricBin("0-10m", 0.0, 10.0),
    MetricBin("10-20m", 10.0, 20.0),
    MetricBin("20-30m", 20.0, 30.0),
    MetricBin("30-40m", 30.0, 40.0),
    MetricBin("40-50m", 40.0, 50.0),
    MetricBin("50m+", 50.0, float("inf")),
)

DEFAULT_DENSITY_BINS = (
    MetricBin("0-5", 0.0, 6.0),
    MetricBin("6-10", 6.0, 11.0),
    MetricBin("11-20", 11.0, 21.0),
    MetricBin("21-50", 21.0, 51.0),
    MetricBin("51+", 51.0, float("inf")),
)


def coerce_bins(values: Iterable[MetricBin | dict[str, Any]], default: tuple[MetricBin, ...]) -> tuple[MetricBin, ...]:
    """Parse config-provided bins while retaining explicit defaults."""

    if values is None:
        return default
    result = tuple(
        value if isinstance(value, MetricBin) else MetricBin(str(value["name"]), float(value["lower"]), float(value["upper"]))
        for value in values
    )
    if not result:
        raise ValueError("at least one metric bin is required")
    for previous, current in zip(result, result[1:]):
        if not np.isclose(previous.upper, current.lower):
            raise ValueError("metric bins must be contiguous and non-overlapping")
    return result


def find_bin(value: float, bins: tuple[MetricBin, ...]) -> MetricBin | None:
    for metric_bin in bins:
        if metric_bin.contains(value):
            return metric_bin
    return None


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


__all__ = [
    "DEFAULT_DENSITY_BINS",
    "DEFAULT_DISTANCE_BINS",
    "MetricBin",
    "coerce_bins",
    "find_bin",
    "mean_or_none",
    "safe_ratio",
]
