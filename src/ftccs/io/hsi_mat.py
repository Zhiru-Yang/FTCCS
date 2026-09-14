"""Read-only MATLAB HSI loading and spatial/spectral axis alignment."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class HSIMatScene:
    data: np.ndarray
    label_map: np.ndarray
    wavelengths: np.ndarray | None
    source_path: Path
    alignment: str


def _load_v73(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    import h5py

    result: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for key in keys:
            if key not in handle:
                raise KeyError(f"Key '{key}' not found in MATLAB v7.3 file")
            result[key] = np.array(handle[key])
    return result


def _is_hdf5_mat(path: Path) -> bool:
    """Detect MATLAB-v7.3/HDF5 files before scipy tries the v5 reader."""

    try:
        import h5py
    except ImportError:
        return False
    return bool(h5py.is_hdf5(path))


def align_cube_and_map(
    data: np.ndarray,
    label_map: np.ndarray,
    spectral_axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the cube as H x W x D and map as H x W, matching V3."""

    cube = np.asarray(data)
    map_array = np.asarray(label_map).squeeze()
    if cube.ndim != 3:
        raise ValueError(f"data must be 3-D, got {cube.shape}")
    if map_array.ndim != 2:
        raise ValueError(f"map must be 2-D after squeeze, got {map_array.shape}")
    if spectral_axis is not None:
        axis = int(spectral_axis) % 3
        aligned = np.moveaxis(cube, axis, -1)
        if aligned.shape[:2] == map_array.shape:
            return aligned, map_array, f"explicit spectral_axis={axis}"
        if aligned.shape[:2] == map_array.T.shape:
            return aligned, map_array.T, f"explicit spectral_axis={axis}, map transposed"
        raise ValueError("explicit spectral axis cannot align cube and map")

    candidates: list[tuple[float, np.ndarray, np.ndarray, tuple[int, ...], bool]] = []
    for permutation in itertools.permutations(range(3)):
        aligned = np.transpose(cube, permutation)
        for transposed, candidate_map in ((False, map_array), (True, map_array.T)):
            if aligned.shape[:2] != candidate_map.shape:
                continue
            dimension = aligned.shape[2]
            score = 0.0
            if 4 <= dimension <= 2048:
                score += 10.0
            if dimension <= max(aligned.shape[0], aligned.shape[1]):
                score += 2.0
            if permutation == (0, 1, 2):
                score += 0.5
            if not transposed:
                score += 0.25
            candidates.append(
                (score, aligned, candidate_map, permutation, transposed)
            )
    if not candidates:
        raise ValueError(f"cannot align data {cube.shape} with map {map_array.shape}")
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    score, aligned, aligned_map, permutation, transposed = candidates[0]
    message = (
        f"auto permutation={permutation}, map_transposed={transposed}, "
        f"score={score:.2f}"
    )
    return aligned, aligned_map, message


def load_hsi_mat(
    path: str | Path,
    *,
    data_key: str = "data",
    map_key: str = "map",
    wavelength_key: str | None = None,
    spectral_axis: int | None = None,
) -> HSIMatScene:
    """Load a classic or v7.3 MAT file without modifying its source directory."""

    source = Path(path).resolve()
    keys = [data_key, map_key] + ([wavelength_key] if wavelength_key else [])
    if _is_hdf5_mat(source):
        loaded = _load_v73(source, keys)
    else:
        try:
            loaded = sio.loadmat(source)
            missing = [key for key in keys if key not in loaded]
            if missing:
                raise KeyError(f"Missing MAT keys: {missing}")
        except NotImplementedError:
            loaded = _load_v73(source, keys)
    data = np.asarray(loaded[data_key])
    label_map = np.asarray(loaded[map_key]).squeeze()
    wavelengths = (
        np.asarray(loaded[wavelength_key]).squeeze().astype(np.float64)
        if wavelength_key
        else None
    )
    data, label_map, alignment = align_cube_and_map(
        data, label_map, spectral_axis
    )
    if wavelengths is not None and wavelengths.size != data.shape[-1]:
        raise ValueError("wavelength count does not match coordinate dimension")
    return HSIMatScene(
        data=data,
        label_map=label_map,
        wavelengths=wavelengths,
        source_path=source,
        alignment=alignment,
    )
