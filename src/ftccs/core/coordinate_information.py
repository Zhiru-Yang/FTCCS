"""Per-coordinate singleton task information diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftccs.core.task_signal_model import TaskSignalModel


@dataclass(frozen=True)
class CoordinateInformation:
    importance: np.ndarray
    normalized_weight: np.ndarray
    variance_floor: float


def coordinate_information(
    signal_model: TaskSignalModel | np.ndarray,
    nuisance_covariance: np.ndarray,
    *,
    variance_floor_ratio: float = 1e-8,
    eps: float = 1e-12,
) -> CoordinateInformation:
    """Compute ``c_d = ||A[d,:]||^2 / Sigma_N[d,d]`` as in V3."""

    matrix = (
        signal_model.matrix
        if isinstance(signal_model, TaskSignalModel)
        else np.asarray(signal_model, dtype=np.float64)
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    covariance = np.asarray(nuisance_covariance, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("signal model must be D x r")
    if covariance.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("nuisance covariance does not match signal model")

    variances = np.asarray(np.diag(covariance), dtype=np.float64)
    positive = variances[variances > eps]
    reference = float(np.median(positive)) if positive.size else 1.0
    floor = max(float(variance_floor_ratio) * reference, eps)
    safe_variances = np.maximum(variances, floor)
    importance = np.sum(matrix**2, axis=1) / safe_variances
    maximum = float(np.max(importance)) if importance.size else 0.0
    normalized_weight = (
        np.sqrt(importance / maximum)
        if maximum > eps
        else np.zeros_like(importance)
    )
    return CoordinateInformation(
        importance=importance,
        normalized_weight=normalized_weight,
        variance_floor=floor,
    )

