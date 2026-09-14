# FTCCS

FTCCS is a support-conditioned method for hyperspectral target detection. It
learns task-relevant latent directions from target support pixels and
background statistics, and can select a compact subset of physical spectral
bands for subsequent detection.

## Quick start

1. Install Python 3.10 or newer and the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

2. Put one hyperspectral MAT file in `data/`, then update
   `configs/example.yaml` with its file name and MAT variable names.

3. Run FTCCS:

   ```bash
   python scripts/run_ftccs.py --config configs/example.yaml
   ```

Results are written to `outputs/<scene-name>/`. The command produces:

- `latent_ftccs.npy`: latent FTCCS representation with the configured rank;
- `selected_bands.npy`: cube restricted to the selected physical bands;
- `selected_bands.json`: selected band indices and, when available, wavelengths;
- `latent_projection.npy`: projection from the original spectrum to latent FTCCS;
- `detection_latent_ftccs.npy` and `detection_selected_bands.npy`: CEM score maps;
- `detection_maps.png`: detection-map visualization; and
- `metrics.json`: held-out detection metrics when a complete target map is supplied.

Use `--mode latent` or `--mode bands` to generate just one output path.

## Input data

The MAT file must contain:

- `data`: a numeric hyperspectral cube. Any axis order is accepted when the
  spectral axis can be inferred; otherwise set `spectral_axis` in the config.
- `map`: a two-dimensional label map, with `0` for background and positive
  values for target pixels.
- `wavelengths` (optional): one wavelength per spectral band.

Pixels assigned an `ignore_values` entry are excluded from model fitting and
evaluation. The support pixels are randomly sampled from target labels using
the configured seed; remaining target labels are reserved for evaluation.

## Reproducibility

All parameters controlling support sampling, nuisance estimation, latent rank,
band budget, and CEM detection are in the YAML configuration. The raw data are
not distributed with this repository; users must obtain and use data according
to their applicable licenses.
