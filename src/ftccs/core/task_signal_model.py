"""Map task geometry back to the original coordinate system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftccs.core.task_geometry import TaskGeometry
from ftccs.nuisance.covariance import psd_square_roots


@dataclass(frozen=True)
class TaskSignalModel:
    """Original-coordinate low-rank model ``A_tau`` for a task episode."""

    matrix: np.ndarray
    direction_energy: np.ndarray
    nuisance_square_root: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def rank(self) -> int:
        return int(self.matrix.shape[1])


def build_task_signal_model(
    geometry: TaskGeometry,
    nuisance_covariance: np.ndarray,
    *,
    eig_floor_ratio: float = 1e-6,
    renormalize_retained_energy: bool = True,
) -> TaskSignalModel:
    """Construct ``A_tau = Sigma_N^(1/2) U_tau Pi_tau^(1/2)``."""

    covariance = np.asarray(nuisance_covariance, dtype=np.float64)
    if covariance.shape != (geometry.basis.shape[0], geometry.basis.shape[0]):
        raise ValueError("nuisance covariance does not match task geometry")
    retained_energy = geometry.retained_direction_energy(
        renormalize=renormalize_retained_energy
    )
    roots = psd_square_roots(covariance, eig_floor_ratio)
    matrix = roots.square_root @ geometry.basis
    matrix = matrix * np.sqrt(retained_energy)[None, :]
    return TaskSignalModel(
        matrix=matrix,
        direction_energy=retained_energy,
        nuisance_square_root=roots.square_root,
    )

