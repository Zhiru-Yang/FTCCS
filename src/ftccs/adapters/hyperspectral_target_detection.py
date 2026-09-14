"""Hyperspectral target-detection adapter for the legacy V3 episode protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage

from ftccs.adapters.base import PreparedTask, TaskAdapter
from ftccs.io.hsi_mat import HSIMatScene, load_hsi_mat
from ftccs.nuisance.covariance import (
    NuisanceStatistics,
    estimate_nuisance_statistics,
    psd_square_roots,
)
from ftccs.nuisance.prototypes import (
    ConditionedNuisanceReference,
    NuisancePrototypeModel,
    conditioned_nuisance_reference,
    fit_nuisance_prototypes,
)


@dataclass(frozen=True)
class HTDAdapterConfig:
    data_key: str = "data"
    map_key: str = "map"
    wavelength_key: str | None = None
    spectral_axis: int | None = None
    support_percent: float | None = 10.0
    random_seed: int = 0
    ignore_values: tuple[int, ...] = (255,)
    positive_values: tuple[int, ...] | None = None
    support_values: tuple[int, ...] | None = None
    preserve_class_values: bool = True
    min_component_pixels: int = 1
    background_value: int = 0
    max_nuisance_samples: int = 50000
    eig_floor_ratio: float = 1e-6
    num_nuisance_prototypes: int = 6
    min_prototype_samples: int = 20
    prototype_center: str = "median"
    reference_mode: str = "soft_topk"
    confusing_top_k: int = 2
    reference_temperature: float = 1.0


@dataclass(frozen=True)
class HTDSupportRepresentation:
    original_map: np.ndarray
    generated_support_map: np.ndarray
    support_mask: np.ndarray
    full_positive_mask: np.ndarray | None
    ignore_mask: np.ndarray
    object_labels: np.ndarray
    object_ids: tuple[int, ...]
    object_sizes: np.ndarray
    representative_spectra: np.ndarray


@dataclass(frozen=True)
class HTDNuisanceRepresentation:
    candidate_mask: np.ndarray
    samples: np.ndarray


@dataclass(frozen=True)
class HTDPreparedTask(PreparedTask):
    scene: HSIMatScene | None = None
    support: HTDSupportRepresentation | None = None
    nuisance: HTDNuisanceRepresentation | None = None
    nuisance_statistics: NuisanceStatistics | None = None
    whitening: np.ndarray | None = None
    covariance_eigenvalues: np.ndarray | None = None
    covariance_floored_eigenvalues: np.ndarray | None = None
    prototype_model: NuisancePrototypeModel | None = None
    conditioned_reference: ConditionedNuisanceReference | None = None


def _ignore_mask(label_map: np.ndarray, ignore_values: Sequence[int]) -> np.ndarray:
    mask = np.zeros_like(label_map, dtype=bool)
    for value in ignore_values:
        mask |= label_map == value
    return mask


def _build_support_mask(
    support_map: np.ndarray,
    ignore_values: Sequence[int],
    support_values: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    ignored = _ignore_mask(support_map, ignore_values)
    if support_values:
        support = np.zeros_like(support_map, dtype=bool)
        for value in support_values:
            support |= support_map == value
        support &= ~ignored
    else:
        support = (support_map > 0) & (~ignored)
    return support, ignored


def _random_support_from_full_map(
    full_map: np.ndarray,
    support_percent: float,
    rng: np.random.Generator,
    ignore_values: Sequence[int],
    positive_values: Sequence[int] | None,
    preserve_class_values: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_map = np.asarray(full_map).squeeze()
    ignored = _ignore_mask(full_map, ignore_values)
    if positive_values:
        full_positive = np.zeros_like(full_map, dtype=bool)
        for value in positive_values:
            full_positive |= full_map == value
        full_positive &= ~ignored
    else:
        full_positive = (full_map > 0) & (~ignored)
    positive_indices = np.flatnonzero(full_positive.ravel())
    if positive_indices.size == 0:
        raise ValueError("no positive target pixels found in the original map")
    if not 0.0 < float(support_percent) <= 100.0:
        raise ValueError("support_percent must be in (0, 100]")
    count = max(
        1,
        int(np.ceil(positive_indices.size * float(support_percent) / 100.0)),
    )
    count = min(count, positive_indices.size)
    chosen = rng.choice(positive_indices, size=count, replace=False)
    support_mask = np.zeros(full_map.size, dtype=bool)
    support_mask[chosen] = True
    support_mask = support_mask.reshape(full_map.shape)
    generated = np.zeros_like(full_map)
    for value in ignore_values:
        generated[full_map == value] = value
    generated[support_mask] = (
        full_map[support_mask] if preserve_class_values else 1
    )
    return generated, support_mask, full_positive


def _connected_support_objects(
    support_mask: np.ndarray,
    min_pixels: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, number = ndimage.label(support_mask, structure=structure)
    relabeled = np.zeros_like(labeled, dtype=np.int32)
    kept: list[int] = []
    next_id = 1
    for object_id in range(1, number + 1):
        count = int(np.sum(labeled == object_id))
        if count >= int(min_pixels):
            relabeled[labeled == object_id] = next_id
            kept.append(next_id)
            next_id += 1
    return relabeled, tuple(kept)


def _object_median_spectra(
    data: np.ndarray,
    labels: np.ndarray,
    object_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    spectra: list[np.ndarray] = []
    sizes: list[int] = []
    for object_id in object_ids:
        pixels = data[labels == object_id]
        if pixels.size == 0:
            continue
        spectra.append(np.median(pixels, axis=0))
        sizes.append(pixels.shape[0])
    if not spectra:
        raise ValueError("no valid support objects found")
    return (
        np.asarray(spectra, dtype=np.float64),
        np.asarray(sizes, dtype=np.int32),
    )


def _sample_nuisance(
    data: np.ndarray,
    candidate_mask: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_indices = np.flatnonzero(candidate_mask.ravel())
    if flat_indices.size == 0:
        raise ValueError("no nuisance candidate pixels remain")
    count = min(int(max_samples), flat_indices.size)
    chosen = (
        rng.choice(flat_indices, size=count, replace=False)
        if count < flat_indices.size
        else flat_indices
    )
    flat_data = data.reshape(-1, data.shape[-1])
    return np.asarray(flat_data[chosen], dtype=np.float64)


class HyperspectralTargetDetectionAdapter(TaskAdapter):
    """Build V3 task contrasts from an HSI scene and a support protocol."""

    def __init__(self, config: HTDAdapterConfig | None = None) -> None:
        self.config = config or HTDAdapterConfig()

    def load_scene(self, path: str | Path) -> HSIMatScene:
        return load_hsi_mat(
            path,
            data_key=self.config.data_key,
            map_key=self.config.map_key,
            wavelength_key=self.config.wavelength_key,
            spectral_axis=self.config.spectral_axis,
        )

    def build_support_representation(
        self,
        scene: HSIMatScene,
        rng: np.random.Generator,
    ) -> HTDSupportRepresentation:
        original_map = scene.label_map.copy()
        if self.config.support_percent is None:
            generated = original_map.copy()
            support_mask, ignore_mask = _build_support_mask(
                generated,
                self.config.ignore_values,
                self.config.support_values,
            )
            full_positive = None
        else:
            generated, support_mask, full_positive = _random_support_from_full_map(
                original_map,
                self.config.support_percent,
                rng,
                self.config.ignore_values,
                self.config.positive_values,
                self.config.preserve_class_values,
            )
            ignore_mask = _ignore_mask(generated, self.config.ignore_values)
        object_labels, object_ids = _connected_support_objects(
            support_mask, self.config.min_component_pixels
        )
        representative_spectra, object_sizes = _object_median_spectra(
            scene.data, object_labels, object_ids
        )
        return HTDSupportRepresentation(
            original_map=original_map,
            generated_support_map=generated,
            support_mask=support_mask,
            full_positive_mask=full_positive,
            ignore_mask=ignore_mask,
            object_labels=object_labels,
            object_ids=object_ids,
            object_sizes=object_sizes,
            representative_spectra=representative_spectra,
        )

    def build_nuisance_representation(
        self,
        scene: HSIMatScene,
        support: HTDSupportRepresentation,
        rng: np.random.Generator,
    ) -> HTDNuisanceRepresentation:
        candidate_mask = (
            (support.generated_support_map == self.config.background_value)
            & (~support.ignore_mask)
        )
        samples = _sample_nuisance(
            scene.data,
            candidate_mask,
            self.config.max_nuisance_samples,
            rng,
        )
        return HTDNuisanceRepresentation(candidate_mask=candidate_mask, samples=samples)

    def estimate_nuisance_covariance(
        self,
        nuisance: HTDNuisanceRepresentation,
    ) -> NuisanceStatistics:
        return estimate_nuisance_statistics(nuisance.samples)

    def build_task_contrasts(
        self,
        support: HTDSupportRepresentation,
        nuisance_statistics: NuisanceStatistics,
        whitening: np.ndarray,
        prototype_model: NuisancePrototypeModel,
    ) -> ConditionedNuisanceReference:
        return conditioned_nuisance_reference(
            support.representative_spectra,
            prototype_model.centers,
            nuisance_statistics.center,
            whitening,
            mode=self.config.reference_mode,
            top_k=self.config.confusing_top_k,
            temperature=self.config.reference_temperature,
        )

    def prepare(self, path: str | Path) -> HTDPreparedTask:
        rng = np.random.default_rng(self.config.random_seed)
        scene = self.load_scene(path)
        support = self.build_support_representation(scene, rng)
        nuisance = self.build_nuisance_representation(scene, support, rng)
        nuisance_statistics = self.estimate_nuisance_covariance(nuisance)
        roots = psd_square_roots(
            nuisance_statistics.covariance, self.config.eig_floor_ratio
        )
        prototype_model = fit_nuisance_prototypes(
            nuisance.samples,
            nuisance_statistics.center,
            roots.inverse_square_root,
            num_prototypes=self.config.num_nuisance_prototypes,
            random_state=self.config.random_seed,
            min_cluster_samples=self.config.min_prototype_samples,
            robust_center=self.config.prototype_center,
        )
        reference = self.build_task_contrasts(
            support,
            nuisance_statistics,
            roots.inverse_square_root,
            prototype_model,
        )
        metadata = {
            "adapter": type(self).__name__,
            "source_path": str(scene.source_path),
            "random_seed": self.config.random_seed,
            "support_percent": self.config.support_percent,
            "support_components": len(support.object_ids),
            "support_pixels": int(np.sum(support.support_mask)),
            "nuisance_samples": int(nuisance.samples.shape[0]),
        }
        return HTDPreparedTask(
            task_contrasts=reference.contrasts,
            nuisance_covariance=nuisance_statistics.covariance,
            metadata=metadata,
            scene=scene,
            support=support,
            nuisance=nuisance,
            nuisance_statistics=nuisance_statistics,
            whitening=roots.inverse_square_root,
            covariance_eigenvalues=roots.eigenvalues,
            covariance_floored_eigenvalues=roots.floored_eigenvalues,
            prototype_model=prototype_model,
            conditioned_reference=reference,
        )
