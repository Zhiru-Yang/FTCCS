"""Low-complexity nuisance prototypes and episode-conditioned references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True)
class NuisancePrototypeModel:
    centers: np.ndarray
    whitened_centers: np.ndarray
    counts: np.ndarray
    original_cluster_ids: np.ndarray
    inertia: float


@dataclass(frozen=True)
class ConditionedNuisanceReference:
    references: np.ndarray
    contrasts: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    nearest_indices: np.ndarray


def _soft_topk_weights(
    distances: np.ndarray,
    top_k: int,
    temperature: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    squared_distances = np.asarray(distances, dtype=np.float64)
    if squared_distances.ndim != 2 or squared_distances.shape[1] < 1:
        raise ValueError("distances must be N x M with M >= 1")
    count, prototypes = squared_distances.shape
    retained = max(1, min(int(top_k), prototypes))
    nearest = np.argmin(squared_distances, axis=1).astype(np.int32)
    weights = np.zeros((count, prototypes), dtype=np.float64)
    if retained == 1:
        weights[np.arange(count), nearest] = 1.0
        return weights, nearest

    top_indices = np.argpartition(
        squared_distances, kth=retained - 1, axis=1
    )[:, :retained]
    for row in range(count):
        indices = top_indices[row]
        row_distances = squared_distances[row, indices]
        delta = row_distances - float(np.min(row_distances))
        positive = delta[delta > eps]
        scale = float(np.median(positive)) if positive.size else 1.0
        tau = max(float(temperature), eps)
        logits = -delta / max(tau * scale, eps)
        logits -= np.max(logits)
        local = np.exp(logits)
        local /= max(float(np.sum(local)), eps)
        weights[row, indices] = local
    return weights, nearest


def fit_nuisance_prototypes(
    nuisance_samples: np.ndarray,
    nuisance_center: np.ndarray,
    whitening: np.ndarray,
    *,
    num_prototypes: int = 6,
    random_state: int = 0,
    batch_size: int = 2048,
    max_iter: int = 200,
    min_cluster_samples: int = 20,
    robust_center: str = "median",
) -> NuisancePrototypeModel:
    """Match V3 MiniBatchKMeans clustering in globally whitened coordinates."""

    samples = np.asarray(nuisance_samples, dtype=np.float64)
    center = np.asarray(nuisance_center, dtype=np.float64).reshape(-1)
    whitening = np.asarray(whitening, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("nuisance_samples must be N x D")
    count, _ = samples.shape
    if count < 1:
        raise ValueError("at least one nuisance sample is required")
    number = max(1, min(int(num_prototypes), count))
    whitened_samples = (samples - center[None, :]) @ whitening.T
    clustering = MiniBatchKMeans(
        n_clusters=number,
        random_state=int(random_state),
        batch_size=min(max(int(batch_size), number * 4), max(count, number)),
        max_iter=int(max_iter),
        n_init="auto",
        reassignment_ratio=0.01,
    )
    labels = clustering.fit_predict(whitened_samples)

    centers: list[np.ndarray] = []
    counts: list[int] = []
    cluster_ids: list[int] = []
    for cluster in range(number):
        member_mask = labels == cluster
        member_count = int(np.sum(member_mask))
        if member_count < int(min_cluster_samples):
            continue
        members = samples[member_mask]
        prototype = (
            np.mean(members, axis=0)
            if robust_center.lower() == "mean"
            else np.median(members, axis=0)
        )
        centers.append(prototype)
        counts.append(member_count)
        cluster_ids.append(cluster)

    if len(centers) < min(2, number):
        centers = []
        counts = []
        cluster_ids = []
        for cluster in range(number):
            member_mask = labels == cluster
            member_count = int(np.sum(member_mask))
            if member_count == 0:
                continue
            members = samples[member_mask]
            prototype = (
                np.mean(members, axis=0)
                if robust_center.lower() == "mean"
                else np.median(members, axis=0)
            )
            centers.append(prototype)
            counts.append(member_count)
            cluster_ids.append(cluster)

    center_matrix = np.asarray(centers, dtype=np.float64)
    return NuisancePrototypeModel(
        centers=center_matrix,
        whitened_centers=(center_matrix - center[None, :]) @ whitening.T,
        counts=np.asarray(counts, dtype=np.int32),
        original_cluster_ids=np.asarray(cluster_ids, dtype=np.int32),
        inertia=float(clustering.inertia_),
    )


def conditioned_nuisance_reference(
    support_representations: np.ndarray,
    prototype_centers: np.ndarray,
    nuisance_center: np.ndarray,
    whitening: np.ndarray,
    *,
    mode: str = "soft_topk",
    top_k: int = 2,
    temperature: float = 1.0,
) -> ConditionedNuisanceReference:
    """Form one confusing nuisance reference for each support representation."""

    support = np.asarray(support_representations, dtype=np.float64)
    prototypes = np.asarray(prototype_centers, dtype=np.float64)
    center = np.asarray(nuisance_center, dtype=np.float64).reshape(-1)
    whitening = np.asarray(whitening, dtype=np.float64)
    support_white = (support - center[None, :]) @ whitening.T
    prototype_white = (prototypes - center[None, :]) @ whitening.T
    support_norm = np.sum(support_white**2, axis=1, keepdims=True)
    prototype_norm = np.sum(prototype_white**2, axis=1, keepdims=True).T
    distances = np.maximum(
        support_norm
        + prototype_norm
        - 2.0 * support_white @ prototype_white.T,
        0.0,
    )
    if mode.lower() == "nearest":
        nearest = np.argmin(distances, axis=1).astype(np.int32)
        weights = np.zeros_like(distances)
        weights[np.arange(support.shape[0]), nearest] = 1.0
    elif mode.lower() == "soft_topk":
        weights, nearest = _soft_topk_weights(distances, top_k, temperature)
    else:
        raise ValueError("mode must be 'nearest' or 'soft_topk'")
    references = weights @ prototypes
    return ConditionedNuisanceReference(
        references=references,
        contrasts=support - references,
        distances=distances,
        weights=weights,
        nearest_indices=nearest,
    )

