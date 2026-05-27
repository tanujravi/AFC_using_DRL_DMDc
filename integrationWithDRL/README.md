# Integration of Reduced-Order Model with Deep Reinforcement Learning

> **Environment:** All commands and scripts in this folder should be run inside the `afcDrl`
> virtual environment. Activate it from the repository root with:
> ```bash
> source afcDrl/bin/activate
> ```

This module integrates the DMDc reduced-order model into the `drlfoam` DRL framework by
replacing the full OpenFOAM environment with the DMDc surrogate, without modifying the
`drlfoam` source code. The integration is done entirely through a modified `Allrun` script and
supporting Python files inside `rotatingCylinder2D/`.

## Folder structure

- `rotatingCylinder2D/` — drop-in replacement case for `drlfoam/openfoam/test_cases/rotatingCylinder2D/`.
  - `main.py` — main execution script (see [How it works](#how-it-works)).
  - `reference.yaml` — all runtime configuration parameters.
  - `Allrun` — entry point called by `drlfoam`; sources `${DRL_BASE}/openfoam/RunFunctions`
    and runs `python main.py reference.yaml`.
  - `models_lib/` — pre-trained DMDc operator pickle files and the shared basis:
    - `common_Ur.pkl`
    - `random_walk_dmdc.pkl`, `chirp_dmdc.pkl`, `chirp_varying_amp_dmdc.pkl`, `AM_dmdc.pkl`
  - `policy.pt` — TorchScript file for the RL policy.
  - `dmdc_util.py` — DMDc one-step update and initial state construction.
  - `omega_controller.py` — wrapper around the policy for computing control actions.
  - `probes.py` — pressure probe interpolation from the reduced state.
  - `smartsim_execution.py`, `smartsim_util.py` — SmartSim/SmartRedis helpers for field
    reconstruction and force computation.
  - `system/` — OpenFOAM dictionaries (`controlDict`, `FO_force`, etc.) for post-processing.

- `plot_notebooks/` — Jupyter notebooks for analysing and comparing DRL training results
  (learning curves, reward histories, control actions) across different model configurations.

## Prerequisites

- `drlfoam` cloned and set up (see the main `README.md` for the clone command).
- DMDc operator pickle files in `models_lib/` — these are produced by
  `dmdc/evaluation/notebooks/combined_basis.ipynb` (already committed for the 2D cylinder case).
- `svdToFoam` built from `auxillary/svdToFoam/` (see that folder's `README.md`).
- OpenFOAM v2406 sourced in the shell.
- `initial_states.pkl` placed one level above `rotatingCylinder2D/` (i.e. at
  `drlfoam/openfoam/test_cases/initial_states.pkl`). This file contains the initial state
  snapshot matrix `(x0, field_names, times_used, points_xy)` used to seed the rollout.

## Setup

1. Copy the `rotatingCylinder2D/` folder into the `drlfoam` test cases directory, replacing
   the existing case:
   ```bash
   cp -r integrationWithDRL/rotatingCylinder2D \
         drlfoam/openfoam/test_cases/rotatingCylinder2D
   ```

2. Set the `DRL_BASE` environment variable to the root of the `drlfoam` repository:
   ```bash
   export DRL_BASE=/path/to/drlfoam
   ```

3. Follow the standard `drlfoam` training instructions. `drlfoam` will call `Allrun` inside
   `rotatingCylinder2D/`, which runs `python main.py reference.yaml` instead of a full
   OpenFOAM simulation.

## How it works

`main.py` replaces the OpenFOAM simulation environment with the DMDc surrogate:

1. Loads all four DMDc operator dictionaries from `models_lib/` and the shared basis
   `common_Ur.pkl`.
2. Loads the initial augmented state `x0` from `../initial_states.pkl`.
3. Starts a local SmartSim experiment with a SmartRedis database on a free port.
4. Runs the control loop for `n_steps` steps:
   - Interpolates pressure probes from the current predicted state.
   - Queries the RL policy (`OmegaController`) for the next angular velocity `omega`.
   - Converts `omega` to tangential wall speed and appends it to the control history buffer.
   - Randomly selects a DMDc model from the library every `n_shuf` steps (MORS strategy).
   - Advances the state one step using the selected model (`dmdcUtil.dmdc_step`).
5. After the loop, uploads the predicted field time series (`p`, `u_x`, `u_y`) to SmartRedis,
   runs `svdToFoam` in parallel to reconstruct OpenFOAM fields, then runs
   `pimpleFoam -postProcess -dict system/FO_force` to compute force coefficients.
6. Writes `omega.csv` and `p_samples.csv` for post-analysis, then stops the database.

## Configuration (`reference.yaml`)

| Key | Description |
|-----|-------------|
| `svd_rank` | SVD rank used for field upload and SmartRedis key naming |
| `mpi_ranks` | Number of MPI ranks for `svdToFoam` and `pimpleFoam` |
| `time_stride` | Time-index stride used in SmartRedis tensor key naming |
| `policy` | Path to the TorchScript policy file |
| `absOmegaMax` | Maximum allowable angular velocity magnitude |
| `train` | `true` to sample from Beta distribution; `false` for deterministic mean |
| `seed` | RNG seed for action sampling |
| `timeStart` | Simulation time at which control begins |
| `executeInterval` | Time step between control actions (`dt_control`) |
| `n_steps` | Number of DMDc rollout steps per episode |
| `n_probes` | Number of pressure probe locations |
| `radius` | Cylinder radius (used to convert `omega` to tangential wall speed) |

## Analysing results

After training, use the notebooks in `plot_notebooks/` to inspect reward histories, control
action trajectories, and comparisons across signal types or learning rates.
