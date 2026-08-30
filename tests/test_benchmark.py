import numpy as np

from lidar_perception.benchmark.latency import benchmark_pointpillar
from lidar_perception.datasets.schemas import PointCloudFrame


class FakeBackend:
    name = lambda self: "fake"
    device = type("Device", (), {"type": "cpu"})()

    def prepare_frame(self, frame):
        return {"points": frame.points}

    def _predict_prepared(self, batch):
        return {}, 0.1

    def predict(self, frame):
        return type("Prediction", (), {})()


def test_benchmark_rejects_invalid_iteration_count() -> None:
    frame = PointCloudFrame("frame", np.zeros((1, 4), dtype=np.float32))
    try:
        benchmark_pointpillar(FakeBackend(), frame, warmup=0, iterations=0)
    except ValueError:
        return
    raise AssertionError("expected invalid iterations to raise")


def test_benchmark_summary_uses_upper_tail_for_small_samples() -> None:
    frame = PointCloudFrame("frame", np.zeros((1, 4), dtype=np.float32))
    result = benchmark_pointpillar(FakeBackend(), frame, warmup=0, iterations=2)
    assert result["model_only"]["p95_ms"] >= result["model_only"]["median_ms"]
    assert result["timing_method"]["model_only"] == "perf_counter_including_prepare"
