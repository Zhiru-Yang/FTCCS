"""Run FTCCS from a YAML configuration and save representations and detection maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ftccs.adapters.hyperspectral_target_detection import HyperspectralTargetDetectionAdapter
from ftccs.core.api import fit_and_select_coordinates
from ftccs.evaluation.detection import cem_score_map, evaluate_detection_scores
from ftccs.htd_reference import adapter_config_from_mapping, load_yaml_config


def _scene_path(config: dict) -> Path:
    data = config["data"]
    scene = Path(str(data["scene"]))
    return scene if scene.is_absolute() else PROJECT_ROOT / str(data["htd_root"]) / scene


def _project(cube: np.ndarray, center: np.ndarray, projection: np.ndarray, block_rows: int) -> np.ndarray:
    height, width, bands = cube.shape
    output = np.empty((height, width, projection.shape[0]), dtype=np.float32)
    for start in range(0, height, block_rows):
        stop = min(start + block_rows, height)
        block = cube[start:stop].reshape(-1, bands).astype(np.float64)
        output[start:stop] = ((block - center) @ projection.T).reshape(stop - start, width, -1)
    return output


def _save_figure(score_maps: dict[str, np.ndarray], support: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(1, len(score_maps), figsize=(6 * len(score_maps), 5), squeeze=False)
    for axis, (name, scores) in zip(axes[0], score_maps.items()):
        finite = scores[np.isfinite(scores)]
        low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        image = axis.imshow(scores, cmap="inferno", vmin=low, vmax=high)
        rows, columns = np.nonzero(support)
        axis.scatter(columns, rows, s=3, c="cyan", marker=".", label="support")
        axis.set_title(name)
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration path")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--mode", choices=("both", "latent", "bands"), default="both")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_yaml_config(config_path)
    prepared = HyperspectralTargetDetectionAdapter(adapter_config_from_mapping(config)).prepare(_scene_path(config))
    if prepared.scene is None or prepared.support is None or prepared.nuisance_statistics is None or prepared.whitening is None:
        raise RuntimeError("FTCCS preparation did not produce the required scene statistics")

    adapter_config = adapter_config_from_mapping(config)
    cube = np.asarray(prepared.scene.data, dtype=np.float32)
    core = fit_and_select_coordinates(
        prepared.task_contrasts,
        prepared.nuisance_covariance,
        rank=int(config["geometry"]["rank"]),
        budget=int(config["selection"]["budget"]),
        eig_floor_ratio=adapter_config.eig_floor_ratio,
        renormalize_retained_energy=bool(config["geometry"].get("renormalize_retained_energy", True)),
        variance_floor_ratio=float(config["selection"].get("variance_floor_ratio", 1e-8)),
        covariance_ridge_ratio=float(config["selection"].get("covariance_ridge_ratio", 1e-6)),
        information_scale=float(config["selection"].get("information_scale", 1.0)),
        coordinate_values=prepared.scene.wavelengths,
        min_spacing=float(config["selection"].get("min_wavelength_spacing", 0.0)),
    )
    scene_name = _scene_path(config).stem
    output = args.output or PROJECT_ROOT / "outputs" / scene_name
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    selected = core.selection.indices
    wavelengths = prepared.scene.wavelengths
    selected_payload = {
        "indices_0based": selected.tolist(),
        "indices_1based": (selected + 1).tolist(),
        "wavelengths": None if wavelengths is None else np.asarray(wavelengths)[selected].tolist(),
    }
    (output / "selected_bands.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    score_maps: dict[str, np.ndarray] = {}
    support = prepared.support
    valid = ~support.ignore_mask
    evaluation = config["evaluation"]
    common = {"support_mask": support.support_mask, "valid_mask": valid, "regularization": float(evaluation.get("cem_regularization", 1e-6)), "block_rows": int(evaluation.get("block_rows", 32)), "signature_statistic": str(evaluation.get("signature_statistic", "mean"))}

    if args.mode in ("both", "latent"):
        rank = min(int(config["geometry"]["rank"]), core.geometry.basis.shape[1])
        projection = core.geometry.basis[:, :rank].T @ prepared.whitening
        latent = _project(cube, prepared.nuisance_statistics.center, projection, common["block_rows"])
        np.save(output / "latent_projection.npy", projection)
        np.save(output / "latent_ftccs.npy", latent)
        scores, _, _ = cem_score_map(data=latent, **common)
        np.save(output / "detection_latent_ftccs.npy", scores)
        score_maps["Latent-FTCCS"] = scores

    if args.mode in ("both", "bands"):
        bands = cube[:, :, selected]
        np.save(output / "selected_bands.npy", bands)
        scores, _, _ = cem_score_map(data=bands, **common)
        np.save(output / "detection_selected_bands.npy", scores)
        score_maps["FTCCS selected bands"] = scores

    metrics: dict[str, dict] = {}
    if support.full_positive_mask is not None:
        query = support.full_positive_mask & ~support.support_mask
        background = (support.original_map == adapter_config.background_value) & valid
        if np.any(query) and np.any(background):
            metrics = {name: evaluate_detection_scores(scores, query, background, far=float(evaluation.get("far", 1e-3))) for name, scores in score_maps.items()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_figure(score_maps, support.support_mask, output / "detection_maps.png")
    (output / "run_metadata.json").write_text(json.dumps({"scene": str(_scene_path(config)), "latent_rank": int(config["geometry"]["rank"]), "band_budget": int(config["selection"]["budget"]), "mode": args.mode}, indent=2), encoding="utf-8")
    print(f"FTCCS outputs written to {output}")


if __name__ == "__main__":
    main()
