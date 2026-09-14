"""Minimal task-adapter boundary for future task families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PreparedTask:
    """The only task-specific information consumed by the generic core."""

    task_contrasts: np.ndarray
    nuisance_covariance: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TaskAdapter(ABC):
    """Translate a task episode into contrasts and nuisance covariance."""

    @abstractmethod
    def build_support_representation(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_nuisance_representation(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def estimate_nuisance_covariance(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_task_contrasts(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> PreparedTask:
        raise NotImplementedError
