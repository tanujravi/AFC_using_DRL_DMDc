# Reward Evaluation

This folder contains tools to convert reduced-order model predictions into OpenFOAM fields and evaluate aerodynamic performance using force coefficients.

> **Environment:** All commands and scripts in this folder should be run inside the `afcDrl`
> virtual environment. Activate it from the repository root with:
> ```bash
> source afcDrl/bin/activate
> ```

The workflow in `rewardEvaluation/` is intended to support the DRL integration by computing reward-related quantities such as lift and drag from reconstructed pressure and velocity fields.

## Folder structure

- `make_fields.py` — main script that reads predicted field data, uploads it to SmartRedis, generates OpenFOAM case directories from the template, and reconstructs fields using `svdToFoam`.
- `plot_coeffs_combined.py` — plotting utility for comparing reconstructed force coefficients from multiple models against original OpenFOAM references.
- `plot_coeffs_individual.py` — similar plotting utility for comparing coefficients from individual reconstructions.
- `template/` — example OpenFOAM case template used to create the reconstructed field cases. Contains case setup files, mesh data, and `Allrun`/`Allclean` scripts.

## Purpose

This module provides the following capabilities:

- convert DMDc-predicted scalar and vector fields into OpenFOAM-compatible format,
- run the OpenFOAM utility `svdToFoam` in parallel to reconstruct fields for pressure (`p`) and velocity (`U`),
- run a transient OpenFOAM solver (`pimpleFoam`) to generate force coefficients from reconstructed fields,
- plot and compare drag (`C_d`) and lift (`C_l`) histories between reconstructed models and original baseline cases.

## Usage

1. Prepare DMDc reconstruction data.

   `make_fields.py` expects a pickled gzipped dataset file containing reconstructed fields and actuation histories. The file path can be passed as the first argument; otherwise it defaults to `data/data.pkl.gz`.

   Example:
   ```bash
   python rewardEvaluation/make_fields.py data/data.pkl.gz
   ```

2. The script creates case directories under `run/cylinder/<dataset_name>` using the provided `template/` folder.
   - It writes `omega.csv` for the actuation signal associated with each dataset.
   - It sets the correct start time inside `system/controlDict`.
   - It uploads predicted fields to SmartRedis in a per-rank, per-time format that matches the OpenFOAM field reconstruction pipeline.

3. Reconstruct OpenFOAM fields and compute forces.

   `make_fields.py` launches `svdToFoam` for the fields `p` and `U`, then runs `pimpleFoam` to generate post-processed force coefficients in each reconstructed case directory.

4. Plot and compare coefficients.

   Use `plot_coeffs_combined.py` to compare reconstructed models against original reference cases across signal types. The script reads OpenFOAM-style coefficient files and produces comparison figures.

   Example:
   ```bash
   python rewardEvaluation/plot_coeffs_combined.py <model_base_path> <original_cases_path> <signal_name>
   ```

## Notes

- The `template/` directory is the OpenFOAM case template used by `make_fields.py` to create run directories.
- The script assumes reconstructed fields are available for `p` and `U` and that the template contains a working `controlDict` and mesh definition.
- `plot_coeffs_combined.py` currently expects a directory structure with `postProcessing/forces_recon/.../coefficient.dat` for model cases and `postProcessing/forces/0/coefficient.dat` for original reference cases.

This README should help you understand how to use the reward evaluation layer to turn DMDc outputs into reward-relevant OpenFOAM data and visualizations.