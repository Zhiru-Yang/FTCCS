"""Public end-to-end API for the task-agnostic FTCCS core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinate_information import CoordinateInformation, coordinate_information
from .coordinate_selector import CoordinateSelection, select_coordinates
from .task_geometry import TaskGeometry, learn_task_geometry
from .task_signal_model import TaskSignalModel, build_task_signal_model


@dataclass(frozen=True)
class FTCCSResult:
    geometry: TaskGeometry
    signal_model: TaskSignalModel
    coordinate_information: CoordinateInformation
    selection: CoordinateSelection


def fit_and_select_coordinates(
    task_contrasts: np.ndarray,
    nuisance_covariance: np.ndarray,
    *,
    rank: int,
    budget: int,
    eig_floor_ratio: float = 1e-6,
    renormalize_retained_energy: bool = True,
    variance_floor_ratio: float = 1e-8,
    covariance_ridge_ratio: float = 1e-6,
    information_scale: float = 1.0,
    coordinate_values: np.ndarray | None = None,
    min_spacing: float = 0.0,
) -> FTCCSResult:
    """Fit all four generic stages after an adapter has built task contrasts."""

    geometry = learn_task_geometry(
        task_contrasts,
        nuisance_covariance,
        rank,
        eig_floor_ratio=eig_floor_ratio,
    )
    signal_model = build_task_signal_model(
        geometry,
        nuisance_covariance,
        eig_floor_ratio=eig_floor_ratio,
        renormalize_retained_energy=renormalize_retained_energy,
    )
    information = coordinate_information(
        signal_model,
        nuisance_covariance,
        variance_floor_ratio=variance_floor_ratio,
    )
    selection = select_coordinates(
        signal_model,
        nuisance_covariance,
        budget,
        covariance_ridge_ratio=covariance_ridge_ratio,
        information_scale=information_scale,
        coordinate_values=coordinate_values,
        min_spacing=min_spacing,
    )
    return FTCCSResult(
        geometry=geometry,
        signal_model=signal_model,
        coordinate_information=information,
        selection=selection,
    )
