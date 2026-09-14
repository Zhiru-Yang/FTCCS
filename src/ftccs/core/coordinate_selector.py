"""Covariance-aware original-coordinate design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ftccs.core.task_signal_model import TaskSignalModel


@dataclass(frozen=True)
class CoordinateSelection:
    indices: np.ndarray
    marginal_gains: np.ndarray
    objective_values: np.ndarray
    covariance_ridge: float
    information_scale: float


def _ridge_value(covariance: np.ndarray, ratio: float) -> float:
    diagonal = np.asarray(np.diag(covariance), dtype=np.float64)
    positive = diagonal[diagonal > 0]
    reference = float(np.median(positive)) if positive.size else 1.0
    return max(float(ratio) * reference, 1e-12)


def coordinate_objective(
    indices: Sequence[int],
    signal_matrix: np.ndarray,
    nuisance_covariance: np.ndarray,
    *,
    covariance_ridge: float,
    information_scale: float = 1.0,
) -> float:
    """Evaluate ``log det(I + gamma A_S^T Sigma_S^-1 A_S)``."""

    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return 0.0
    matrix = np.asarray(signal_matrix, dtype=np.float64)
    covariance = np.asarray(nuisance_covariance, dtype=np.float64)
    selected_matrix = matrix[idx, :]
    selected_covariance = covariance[np.ix_(idx, idx)]
    selected_covariance = selected_covariance + float(covariance_ridge) * np.eye(
        idx.size, dtype=np.float64
    )
    try:
        solved = np.linalg.solve(selected_covariance, selected_matrix)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(selected_covariance) @ selected_matrix
    information = selected_matrix.T @ solved
    information = 0.5 * (information + information.T)
    objective_matrix = np.eye(matrix.shape[1], dtype=np.float64) + max(
        float(information_scale), 0.0
    ) * information
    sign, logdet = np.linalg.slogdet(objective_matrix)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    return float(logdet)


def select_coordinates(
    signal_model: TaskSignalModel | np.ndarray,
    nuisance_covariance: np.ndarray,
    budget: int,
    *,
    covariance_ridge_ratio: float = 1e-6,
    information_scale: float = 1.0,
    coordinate_values: np.ndarray | None = None,
    min_spacing: float = 0.0,
) -> CoordinateSelection:
    """Greedy V3-equivalent coordinate selection with deterministic ties."""

    matrix = (
        signal_model.matrix
        if isinstance(signal_model, TaskSignalModel)
        else np.asarray(signal_model, dtype=np.float64)
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    covariance = np.asarray(nuisance_covariance, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("signal model must be D x r")
    dimension = matrix.shape[0]
    if covariance.shape != (dimension, dimension):
        raise ValueError("nuisance covariance must be D x D")
    if int(budget) < 1:
        raise ValueError("budget must be at least one")
    budget = min(int(budget), dimension)

    values = None if coordinate_values is None else np.asarray(coordinate_values).reshape(-1)
    if values is not None and values.size != dimension:
        raise ValueError("coordinate_values length does not match dimension")
    if min_spacing > 0 and values is None:
        raise ValueError("positive min_spacing requires coordinate_values")

    ridge = _ridge_value(covariance, covariance_ridge_ratio)
    selected: list[int] = []
    marginal_gains: list[float] = []
    objective_values: list[float] = []
    available = np.ones(dimension, dtype=bool)
    current_objective = 0.0

    for _ in range(budget):
        best_index = -1
        best_objective = -np.inf
        for coordinate in np.flatnonzero(available):
            if values is not None and min_spacing > 0 and selected:
                if np.any(np.abs(values[coordinate] - values[selected]) < min_spacing):
                    continue
            objective = coordinate_objective(
                selected + [int(coordinate)],
                matrix,
                covariance,
                covariance_ridge=ridge,
                information_scale=information_scale,
            )
            if objective > best_objective:
                best_index = int(coordinate)
                best_objective = objective
        if best_index < 0:
            break
        selected.append(best_index)
        available[best_index] = False
        marginal_gains.append(float(best_objective - current_objective))
        objective_values.append(float(best_objective))
        current_objective = float(best_objective)

    return CoordinateSelection(
        indices=np.asarray(selected, dtype=np.int32),
        marginal_gains=np.asarray(marginal_gains, dtype=np.float64),
        objective_values=np.asarray(objective_values, dtype=np.float64),
        covariance_ridge=ridge,
        information_scale=max(float(information_scale), 0.0),
    )

