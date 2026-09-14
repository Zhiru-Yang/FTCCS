"""Nuisance-normalized low-rank geometry for a few-shot task episode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftccs.nuisance.covariance import psd_square_roots


@dataclass(frozen=True)
class TaskGeometry:
    """Low-rank task geometry learned only from support-derived contrasts."""

    basis: np.ndarray
    direction_energy: np.ndarray
    singular_values: np.ndarray
    effective_rank: int
    cumulative_energy: np.ndarray
    effective_rank_entropy: float
    effective_rank_participation: float
    whitened_contrasts: np.ndarray
    normalized_whitened_contrasts: np.ndarray
    whitening: np.ndarray
    projection_matrix: np.ndarray
    original_coordinate_basis: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def retained_direction_energy(self, renormalize: bool = True) -> np.ndarray:
        retained = np.maximum(self.direction_energy[: self.rank], 0.0).copy()
        if renormalize:
            total = float(np.sum(retained))
            if total > 1e-12:
                retained /= total
            else:
                retained[:] = 1.0 / max(self.rank, 1)
        return retained


def learn_task_geometry(
    task_contrasts: np.ndarray,
    nuisance_covariance: np.ndarray,
    rank: int,
    *,
    eig_floor_ratio: float = 1e-6,
    eps: float = 1e-12,
) -> TaskGeometry:
    """Whiten, direction-normalize, and decompose the episode contrasts.

    This function is the generic counterpart of V3's
    ``target_isolated_subspace_multi_bg``. It deliberately preserves V3's SVD,
    rank threshold, energy normalization, projection, and QR diagnostics.
    """

    contrasts = np.asarray(task_contrasts, dtype=np.float64)
    covariance = np.asarray(nuisance_covariance, dtype=np.float64)
    if contrasts.ndim != 2 or contrasts.shape[0] < 1:
        raise ValueError("task_contrasts must be a non-empty K x D matrix")
    k, dimension = contrasts.shape
    if covariance.shape != (dimension, dimension):
        raise ValueError("nuisance_covariance must be D x D")
    if int(rank) < 1:
        raise ValueError("rank must be at least one")
    if not np.all(np.isfinite(contrasts)):
        raise ValueError("task_contrasts contain non-finite values")

    roots = psd_square_roots(covariance, eig_floor_ratio)
    whitening = roots.inverse_square_root
    whitened = contrasts @ whitening.T
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    normalized = whitened / np.maximum(norms, eps)

    basis_all, singular_values, _ = np.linalg.svd(
        normalized.T, full_matrices=False
    )
    energy = singular_values**2
    direction_energy = (
        energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
    )
    cumulative_energy = np.cumsum(direction_energy)

    threshold_reference = singular_values[0] if singular_values.size else 0.0
    effective_rank = int(
        np.sum(singular_values > max(threshold_reference, eps) * 1e-10)
    )
    retained_rank = max(
        1,
        min(int(rank), basis_all.shape[1], max(effective_rank, 1)),
    )
    basis = basis_all[:, :retained_rank]
    projection = basis.T @ whitening

    original_basis, _ = np.linalg.qr(projection.T)
    original_basis = original_basis[:, :retained_rank]

    positive_energy = direction_energy[direction_energy > 0]
    entropy_rank = (
        float(np.exp(-np.sum(positive_energy * np.log(positive_energy))))
        if positive_energy.size
        else 0.0
    )
    energy_square_sum = float(np.sum(energy**2))
    participation_rank = (
        float((energy.sum() ** 2) / energy_square_sum)
        if energy_square_sum > 0
        else 0.0
    )

    return TaskGeometry(
        basis=basis,
        direction_energy=direction_energy,
        singular_values=singular_values,
        effective_rank=effective_rank,
        cumulative_energy=cumulative_energy,
        effective_rank_entropy=entropy_rank,
        effective_rank_participation=participation_rank,
        whitened_contrasts=whitened,
        normalized_whitened_contrasts=normalized,
        whitening=whitening,
        projection_matrix=projection,
        original_coordinate_basis=original_basis,
    )

