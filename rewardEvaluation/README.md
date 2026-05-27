# Reward Evaluation

This folder contains tools to convert reduced-order model predictions into OpenFOAM fields and evaluate aerodynamic performance using force coefficients.

> **Environment:** All commands and scripts in this folder should be run inside the `afcDrl`
> virtual environment. Activate it from the repository root with:
> ```bash
> source afcDrl/bin/activate
> ```

The workflow takes the predicted state time series produced by `dmdc/evaluation/notebooks/combined_basis.ipynb`
and converts them into OpenFOAM fields so that the standard `forceCoeffs` post-processing utility
can be used to extract lift and drag histories.

## Folder structure

- `make_fields.py` — main script that reads the predicted field data, uploads it to SmartRedis,
  generates OpenFOAM case directories from the template, reconstructs fields using `svdToFoam`,
  and computes force coefficients using `pimpleFoam` in post-process mode.
- `plot_coeffs_combined.py` — plots drag (`Cd`) and lift (`Cl`) from multiple source models
  against the original OpenFOAM reference, saving one figure per signal.
- `plot_coeffs_individual.py` — similar to `plot_coeffs_combined.py` for individual model cases.
- `template/` — OpenFOAM case template used to create the reconstructed field cases. Contains
  mesh data, initial conditions, `system/controlDict`, `system/FO_force` (force function object),
  and `Allrun`/`Allclean` scripts.

## Usage

1. **Prepare the input data.**

   The input is the compressed pickle file produced by `combined_basis.ipynb`:
   ```
   exports/data_new.pkl.gz
   ```
   The file contains a nested dictionary structured as:
   ```
   data[src_model_name][0]  →  pred_block   (predicted fields per target signal)
   data[src_model_name][1]  →  omega_block  (actuation time series per target signal)
   ```
   where each `pred_block[tgt_signal]` is a dict with keys `"p"`, `"u_x"`, `"u_y"` (and
   optionally `"u_z"`), each an array of shape `(n_points, n_times)`.

2. **Run `make_fields.py`.**

   ```bash
   python make_fields.py exports/data_new.pkl.gz
   ```

   The script:
   - Starts a local SmartSim experiment with a SmartRedis database on port 3999.
   - For each source model and target signal pair, uploads the predicted fields to SmartRedis
     per MPI rank and per time step using a fixed tensor key naming convention expected by
     `svdToFoam`.
   - Copies the `template/` directory to `run/cylinder/<src_model>/<tgt_signal>/`, writes the
     corresponding `omega.csv`, and sets `startTime` in `system/controlDict`.
   - Runs `svdToFoam` in parallel (2 MPI ranks) for fields `p` and `U` to reconstruct OpenFOAM
     fields from the SmartRedis tensors.
   - Runs `pimpleFoam -postProcess -dict system/FO_force -parallel` to compute force
     coefficients using the `forces_recon` function object defined in `system/FO_force`.
   - Stops the database when all cases are processed.

   Force coefficient results are written to:
   ```
   run/cylinder/<src_model>/<tgt_signal>/postProcessing/forces_recon/<time>/coefficient.dat
   ```

3. **Plot and compare coefficients.**

   Use `plot_coeffs_combined.py` to compare reconstructed model coefficients against the
   original OpenFOAM reference for a given target signal:

   ```bash
   python plot_coeffs_combined.py <run/cylinder> <path/to/original_cases> <tgt_signal_name>
   ```

   The script reads `coefficient.dat` from each model's `postProcessing/forces_recon/` directory
   and the original reference from `postProcessing/forces/0/coefficient.dat`. It saves separate
   `<signal>_cd.png` and `<signal>_cl.png` figures to the current directory.

## Notes

- `svdToFoam` must be built from the modified source in `auxillary/svdToFoam/` before running
  this workflow (see `auxillary/svdToFoam/README.md`).
- The number of MPI ranks (`mpi_ranks = 2`) and SVD rank (`svd_rank = 20`) are set as
  constants at the top of `make_fields.py` and must match the decomposition used when the
  data was produced.
- The `template/` directory already contains the pre-decomposed mesh for 2 processors
  (`processor0/`, `processor1/`).
