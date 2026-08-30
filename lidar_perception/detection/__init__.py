"""Project-owned detector interfaces and third-party backend adapters."""

from .base import DetectorBackend
from .schemas import PredictionBatch

__all__ = ["DetectorBackend", "PredictionBatch"]
