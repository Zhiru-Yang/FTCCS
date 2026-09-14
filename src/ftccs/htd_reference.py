"""Small-scene orchestration for the HTD adapter and generic FTCCS core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.io as sio
import yaml

from ftccs.adapters.hyperspectral_target_detection import (
    HTDAdapterConfig,
    HTDPreparedTask,
    HyperspectralTargetDetectionAdapter,
)
from ftccs.core.api import FTCCSResult, fit_and_select_coordinates
from ftccs.evaluation.detection import cem_score_map, evaluate_detection_scores


@dataclass(frozen=True)
class HTDReferenceResult:
    prepared_task: HTDPreparedTask
    core_result: FTCCSResult
    selected_data: np.ndarray
    cem_score_maps: Mapping[str, np.ndarray]
    cem_metrics: Mapping[str, Mapping[str, float | int]]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    return loaded


def resolve_scene_path(config: Mapping[str, Any]) -> Path:
    data = config["data"]
    scene = Path(str(data["scene"]))
    return scene if scene.is_absolute() else Path(str(data["htd_root"])) / scene


def adapter_config_from_mapping(config: Mapping[str, Any]) -> HTDAdapterConfig:
    data = config["data"]
    support = config["support"]
    nuisance = config["nuisance"]
    ignore_values = tuple(int(value) for value in support.get("ignore_values", []))
    positive = support.get("positive_values")
    return HTDAdapterConfig(
        data_key=str(data.get("data_key", "data")),
        map_key=str(data.get("map_key", "map")),
        wavelength_key=data.get("wavelength_key"),
        spectral_axis=data.get("spectral_axis"),
        support_percent=support.get("percent"),
        random_seed=int(support.get("random_seed", 0)),
        ignore_values=ignore_values,
        positive_values=(
            None if positive is None else tuple(int(value) for value in positive)
        ),
        preserve_class_values=bool(support.get("preserve_class_values", True)),
        min_component_pixels=int(support.get("min_component_pixels", 1)),
        background_value=int(support.get("background_value", 0)),
        max_nuisance_samples=int(nuisance.get("max_samples", 50000)),
        eig_floor_ratio=float(nuisance.get("eig_floor_ratio", 1e-6)),
        num_nuisance_prototypes=int(nuisance.get("num_prototypes", 6)),
        min_prototype_samples=int(nuisance.get("min_prototype_samples", 20)),
        prototype_center=str(nuisance.get("prototype_center", "median")),
        reference_mode=str(nuisance.get("reference_mode", "soft_topk")),
        confusing_top_k=int(nuisance.get("confusing_top_k", 2)),
        reference_temperature=float(nuisance.get("reference_temperature", 1.0)),
    )


def run_reference(config: Mapping[str, Any]) -> HTDReferenceResult:
    adapter_config = adapter_config_from_mapping(config)
    prepared = HyperspectralTargetDetectionAdapter(adapter_config).prepare(
        resolve_scene_path(config)
    )
    geometry = config["geometry"]
    selection = config["selection"]
    core_result = fit_and_select_coordinates(
        prepared.task_contrasts,
        prepared.nuisance_covariance,
        rank=int(geometry["rank"]),
        budget=int(selection["budget"]),
        eig_floor_ratio=adapter_config.eig_floor_ratio,
        renormalize_retained_energy=bool(
            geometry.get("renormalize_retained_energy", True)
        ),
        variance_floor_ratio=float(selection.get("variance_floor_ratio", 1e-8)),
        covariance_ridge_ratio=float(
            selection.get("covariance_ridge_ratio", 1e-6)
        ),
        information_scale=float(selection.get("information_scale", 1.0)),
        coordinate_values=(prepared.scene.wavelengths if prepared.scene else None),
        min_spacing=float(selection.get("min_wavelength_spacing", 0.0)),
    )
    if prepared.scene is None or prepared.support is None:
        raise RuntimeError("HTD adapter did not preserve scene/support diagnostics")
    indices = core_result.selection.indices
    selected_data = np.asarray(
        prepared.scene.data[:, :, indices], dtype=np.float32
    )
    evaluation = config["evaluation"]
    valid_mask = ~prepared.support.ignore_mask
    query_mask = (
        None
        if prepared.support.full_positive_mask is None
        else prepared.support.full_positive_mask & (~prepared.support.support_mask)
    )
    if query_mask is None or not np.any(query_mask):
        raise ValueError("reference evaluation requires held-out positive pixels")
    background_mask = (
        prepared.support.original_map == adapter_config.background_value
    ) & valid_mask
    common = dict(
        data=prepared.scene.data,
        support_mask=prepared.support.support_mask,
        valid_mask=valid_mask,
        regularization=float(evaluation.get("cem_regularization", 1e-6)),
        block_rows=int(evaluation.get("block_rows", 64)),
        signature_statistic=str(evaluation.get("signature_statistic", "mean")),
    )
    original_scores, _, _ = cem_score_map(**common)
    weighted_scores, _, _ = cem_score_map(
        **common,
        scale=core_result.coordinate_information.normalized_weight,
    )
    selected_scores, _, _ = cem_score_map(
        **common,
        selected_indices=indices,
    )
    score_maps = {
        "Original CEM": original_scores,
        "Weighted CEM": weighted_scores,
        "Selected-band CEM": selected_scores,
    }
    far = float(evaluation.get("far", 1e-3))
    metrics = {
        name: evaluate_detection_scores(
            scores, query_mask, background_mask, far=far
        )
        for name, scores in score_maps.items()
    }
    return HTDReferenceResult(
        prepared_task=prepared,
        core_result=core_result,
        selected_data=selected_data,
        cem_score_maps=score_maps,
        cem_metrics=metrics,
    )


def result_payload(result: HTDReferenceResult) -> dict[str, Any]:
    prepared = result.prepared_task
    support = prepared.support
    stats = prepared.nuisance_statistics
    prototypes = prepared.prototype_model
    reference = prepared.conditioned_reference
    core = result.core_result
    if any(value is None for value in (support, stats, prototypes, reference)):
        raise RuntimeError("HTD diagnostic payload is incomplete")
    assert support is not None
    assert stats is not None
    assert prototypes is not None
    assert reference is not None
    return {
        "generated_support_map": support.generated_support_map,
        "support_mask": support.support_mask.astype(np.uint8),
        "support_object_labels": support.object_labels.astype(np.int32),
        "support_object_sizes": support.object_sizes.astype(np.int32),
        "support_object_spectra": support.representative_spectra,
        "background_mean": stats.center,
        "background_covariance": stats.covariance,
        "ledoit_wolf_shrinkage": np.array([[stats.shrinkage]]),
        "background_prototype_spectra": prototypes.centers,
        "background_prototype_counts": prototypes.counts,
        "support_background_references": reference.references,
        "support_to_prototype_distances": reference.distances,
        "support_prototype_weights": reference.weights,
        "target_contrasts": prepared.task_contrasts,
        "whitened_target_vectors": core.geometry.whitened_contrasts,
        "normalized_whitened_target_vectors": (
            core.geometry.normalized_whitened_contrasts
        ),
        "singular_values": core.geometry.singular_values,
        "normalized_singular_energy": core.geometry.direction_energy,
        "cumulative_singular_energy": core.geometry.cumulative_energy,
        "whitened_target_basis": core.geometry.basis,
        "projection_matrix": core.geometry.projection_matrix,
        "original_coordinate_subspace_basis": (
            core.geometry.original_coordinate_basis
        ),
        "retained_task_energy": core.signal_model.direction_energy,
        "task_signal_basis": core.signal_model.matrix,
        "band_task_importance": core.coordinate_information.importance,
        "band_weights": core.coordinate_information.normalized_weight,
        "selected_band_indices_0based": core.selection.indices,
        "dopt_marginal_gains": core.selection.marginal_gains,
        "dopt_objective_values": core.selection.objective_values,
        "selected_data": result.selected_data,
        "cem_original_score_map": result.cem_score_maps["Original CEM"],
        "cem_weighted_score_map": result.cem_score_maps["Weighted CEM"],
        "cem_selected_score_map": result.cem_score_maps["Selected-band CEM"],
    }


def save_reference_result(
    result: HTDReferenceResult,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output, result_payload(result), long_field_names=True)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(result.cem_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output
