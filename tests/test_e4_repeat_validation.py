from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_e4_repeat_validation import _metric_delta, _round_metrics  # noqa: E402


def _metrics(**changes):
    values = {
        "recall_50m_plus": 0.25,
        "recall_0_5_points": 0.7,
        "recall_40_50m": 0.8,
        "overall_custom_recall": 0.85,
        "precision": 0.16,
        "fp_count": 100,
        "matched_count": 80,
        "gt_count": 90,
        "matched_localization_error_m": 0.25,
        "average_matched_confidence": 0.82,
        "distance": {"large": "payload"},
    }
    values.update(changes)
    return values


def test_e4_repeat_summary_and_zero_delta_are_exact() -> None:
    summary = _round_metrics(_metrics())
    assert "distance" not in summary
    assert summary == _round_metrics(_metrics())
    assert set(_metric_delta(summary, summary).values()) == {0.0}


def test_e4_repeat_delta_exposes_metric_drift() -> None:
    baseline = _round_metrics(_metrics())
    changed = _round_metrics(_metrics(recall_50m_plus=0.26, fp_count=103, matched_count=81))
    delta = _metric_delta(changed, baseline)
    assert delta["recall_50m_plus"] == pytest.approx(0.01)
    assert delta["fp_count"] == 3
    assert delta["matched_count"] == 1
