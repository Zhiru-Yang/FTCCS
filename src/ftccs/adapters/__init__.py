"""Task adapters convert episode evidence into generic task contrasts."""

from .base import PreparedTask, TaskAdapter
from .hyperspectral_target_detection import (
    HTDAdapterConfig,
    HTDPreparedTask,
    HyperspectralTargetDetectionAdapter,
)
__all__ = [
    "HTDAdapterConfig",
    "HTDPreparedTask",
    "HyperspectralTargetDetectionAdapter",
    "PreparedTask",
    "TaskAdapter",
]
