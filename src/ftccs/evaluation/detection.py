"""HTD evaluation isolated from task-geometry and coordinate selection fitting."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _support_signature(
    data: np.ndarray,
    support_mask: np.ndarray,
    statistic: str,
    scale: np.ndarray | None,
    selected_indices: np.ndarray | None,
) -> np.ndarray:
    samples = np.asarray(data[support_mask], dtype=np.float64)
    if selected_indices is not None:
        samples = samples[:, np.asarray(selected_indices, dtype=int)]
    if scale is not None:
        gains = np.asarray(scale, dtype=np.float64).reshape(-1)
        if selected_indices is not None:
            gains = gains[np.asarray(selected_indices, dtype=int)]
        samples = samples * gains.reshape(1, -1)
    if samples.shape[0] == 0:
        raise ValueError("support mask contains no target pixels")
    return (
        np.median(samples, axis=0)
        if statistic.lower() == "median"
        else np.mean(samples, axis=0)
    )


def _second_moment(
    data: np.ndarray,
    valid_mask: np.ndarray,
    block_rows: int,
    scale: np.ndarray | None,
    selected_indices: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    height, _, dimension = data.shape
    if selected_indices is None:
        indices = None
        selected_dimension = dimension
    else:
        indices = np.asarray(selected_indices, dtype=int)
        selected_dimension = len(indices)
    gains = None
    if scale is not None:
        gains = np.asarray(scale, dtype=np.float64).reshape(-1)
        if indices is not None:
            gains = gains[indices]
    second_moment = np.zeros(
        (selected_dimension, selected_dimension), dtype=np.float64
    )
    count = 0
    for row_start in range(0, height, block_rows):
        row_stop = min(height, row_start + block_rows)
        block_mask = valid_mask[row_start:row_stop].reshape(-1)
        if not np.any(block_mask):
            continue
        block = np.asarray(data[row_start:row_stop], dtype=np.float64).reshape(
            -1, dimension
        )
        samples = block[block_mask]
        if indices is not None:
            samples = samples[:, indices]
        if gains is not None:
            samples = samples * gains.reshape(1, -1)
        samples = samples[np.all(np.isfinite(samples), axis=1)]
        if samples.shape[0] == 0:
            continue
        second_moment += samples.T @ samples
        count += samples.shape[0]
    if count == 0:
        raise ValueError("no valid pixels available for CEM")
    return second_moment / count, count


def cem_score_map(
    data: np.ndarray,
    support_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    regularization: float = 1e-6,
    block_rows: int = 64,
    signature_statistic: str = "mean",
    scale: np.ndarray | None = None,
    selected_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical CEM with V3's second-moment and regularization convention."""

    cube = np.asarray(data)
    height, width, dimension = cube.shape
    signature = _support_signature(
        cube,
        support_mask,
        signature_statistic,
        scale,
        selected_indices,
    )
    second_moment, _ = _second_moment(
        cube, valid_mask, block_rows, scale, selected_indices
    )
    selected_dimension = second_moment.shape[0]
    ridge = float(regularization) * max(
        float(np.trace(second_moment)) / max(selected_dimension, 1), 1e-12
    )
    regularized = second_moment + ridge * np.eye(
        selected_dimension, dtype=np.float64
    )
    inverse_signature = np.linalg.solve(regularized, signature)
    denominator = float(signature.T @ inverse_signature)
    if abs(denominator) < 1e-15:
        raise np.linalg.LinAlgError("CEM denominator is numerically zero")
    filter_weights = inverse_signature / denominator

    indices = (
        None
        if selected_indices is None
        else np.asarray(selected_indices, dtype=int)
    )
    gains = None
    if scale is not None:
        gains = np.asarray(scale, dtype=np.float64).reshape(-1)
        if indices is not None:
            gains = gains[indices]
    scores = np.full((height, width), np.nan, dtype=np.float32)
    for row_start in range(0, height, block_rows):
        row_stop = min(height, row_start + block_rows)
        block = np.asarray(cube[row_start:row_stop], dtype=np.float64).reshape(
            -1, dimension
        )
        if indices is not None:
            block = block[:, indices]
        if gains is not None:
            block = block * gains.reshape(1, -1)
        block_scores = block @ filter_weights
        scores[row_start:row_stop] = block_scores.reshape(
            row_stop - row_start, width
        ).astype(np.float32)
    scores[~valid_mask] = np.nan
    return scores, filter_weights, signature


def evaluate_detection_scores(
    score_map: np.ndarray,
    query_positive_mask: np.ndarray,
    background_mask: np.ndarray,
    *,
    far: float = 1e-3,
) -> dict[str, float | int]:
    """Compute V3 AP, AUC, Pd, normalized low-FAR pAUC, and diagnostics."""

    positive = np.asarray(score_map[query_positive_mask], dtype=np.float64)
    negative = np.asarray(score_map[background_mask], dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0:
        raise ValueError("no held-out query positive pixels")
    if negative.size == 0:
        raise ValueError("no background pixels")
    labels = np.concatenate(
        [
            np.ones(positive.size, dtype=np.uint8),
            np.zeros(negative.size, dtype=np.uint8),
        ]
    )
    scores = np.concatenate([positive, negative])
    ap = float(average_precision_score(labels, scores))
    auc = float(roc_auc_score(labels, scores))
    threshold = float(np.quantile(negative, 1.0 - far))
    pd_at_far = float(np.mean(positive > threshold))
    actual_far = float(np.mean(negative > threshold))
    background_std = float(np.std(negative))
    scr = float(
        (np.mean(positive) - np.mean(negative)) / max(background_std, 1e-12)
    )

    fpr, tpr, _ = roc_curve(labels, scores)
    safe_far = max(float(far), 1e-12)
    below = fpr < safe_far
    x_values = np.concatenate(
        [fpr[below], np.array([safe_far], dtype=np.float64)]
    )
    y_values = np.concatenate(
        [tpr[below], np.array([float(np.interp(safe_far, fpr, tpr))])]
    )
    if x_values.size == 0 or x_values[0] > 0.0:
        x_values = np.concatenate([np.array([0.0]), x_values])
        y_values = np.concatenate([np.array([0.0]), y_values])
    area = (
        np.trapezoid(y_values, x_values)
        if hasattr(np, "trapezoid")
        else np.trapz(y_values, x_values)
    )
    return {
        "AP": ap,
        "ROC_AUC": auc,
        f"Pd@FAR={far:g}": pd_at_far,
        "actual_FAR": actual_far,
        f"pAUC@FAR<={far:g}": float(area / safe_far),
        "threshold": threshold,
        "target_mean": float(np.mean(positive)),
        "target_std": float(np.std(positive)),
        "background_mean": float(np.mean(negative)),
        "background_std": background_std,
        "target_background_SCR": scr,
        "num_query_positive_pixels": int(positive.size),
        "num_background_pixels": int(negative.size),
    }


def metric_subset(metrics: Mapping[str, float | int], far: float) -> dict[str, float]:
    """Return the equivalence-report metrics requested by the migration protocol."""

    keys = ["AP", "ROC_AUC", f"Pd@FAR={far:g}", f"pAUC@FAR<={far:g}"]
    return {key: float(metrics[key]) for key in keys}
