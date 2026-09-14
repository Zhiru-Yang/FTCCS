"""Nuisance covariance estimation and stable PSD square roots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class NuisanceStatistics:
    """Robust location and Ledoit--Wolf covariance of nuisance samples."""

    center: np.ndarray
    covariance: np.ndarray
    shrinkage: float


@dataclass(frozen=True)
class PSDSquareRoots:
    """Stable square root and inverse square root of a PSD matrix."""

    square_root: np.ndarray
    inverse_square_root: np.ndarray
    eigenvalues: np.ndarray
    floored_eigenvalues: np.ndarray


def estimate_nuisance_statistics(samples: np.ndarray) -> NuisanceStatistics:
    """Match V3: coordinate-wise median plus assume-centered Ledoit--Wolf."""

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("samples must be a non-empty N x D matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples contain non-finite values")

    center = np.median(x, axis=0)
    estimator = LedoitWolf(assume_centered=True)
    estimator.fit(x - center)
    return NuisanceStatistics(
        center=np.asarray(center, dtype=np.float64),
        covariance=np.asarray(estimator.covariance_, dtype=np.float64),
        shrinkage=float(estimator.shrinkage_),
    )


def psd_square_roots(
    covariance: np.ndarray,
    eig_floor_ratio: float = 1e-6,
) -> PSDSquareRoots:
    """Return V3-compatible symmetric PSD square roots after eigenvalue flooring."""

    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    if eig_floor_ratio < 0:
        raise ValueError("eig_floor_ratio must be non-negative")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite values")

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    maximum = max(float(np.max(eigenvalues)), np.finfo(np.float64).eps)
    floor = float(eig_floor_ratio) * maximum
    floored = np.maximum(eigenvalues, floor)
    square_root = (eigenvectors * np.sqrt(floored)[None, :]) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors * (1.0 / np.sqrt(floored))[None, :]
    ) @ eigenvectors.T
    return PSDSquareRoots(
        square_root=square_root,
        inverse_square_root=inverse_square_root,
        eigenvalues=eigenvalues,
        floored_eigenvalues=floored,
    )


def inverse_sqrt_psd(
    covariance: np.ndarray,
    eig_floor_ratio: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper with the tuple returned by legacy V3."""

    roots = psd_square_roots(covariance, eig_floor_ratio)
    return roots.inverse_square_root, roots.eigenvalues, roots.floored_eigenvalues

