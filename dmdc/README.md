# DMDc Model Building and Evaluation

This folder contains the Dynamic Mode Decomposition with Control (DMDc) components used to build reduced-order models for the rotating cylinder active flow control problem.

> **Environment:** All commands and notebooks in this folder should be run inside the `afcDrl`
> virtual environment. Activate it from the repository root with:
> ```bash
> source afcDrl/bin/activate
> ```

The `dmdc/` module is split into two main parts:

- `signal_library/` — generates and stores actuation signals used to excite the flow and build DMDc models.
- `evaluation/` — contains notebooks and helper code for building, evaluating, and selecting DMDc models from the simulation data.

## Purpose

This folder provides the tools needed to:

- create forcing signals for the rotating cylinder control case,
- run the corresponding OpenFOAM simulations to collect training data,
- build time-delayed DMDc models from that data,
- evaluate model performance, and
- select DMDc settings for use in the integrated DRL workflow.

## Folder structure

- `signal_library/cylinder2D/`
  - `omega.csv` — example actuation signal data.
  - `signal_generator.ipynb` — notebook for generating the cylinder angular velocity signals.
  - `template/` — a case template that shows the OpenFOAM setup for using the generated `omega.csv` signal with the rotating cylinder boundary condition.

- `evaluation/`
  - `DMDcDataset.py` — utility for assembling snapshot and control matrices for DMDc.
  - `DMDcPlotter.py` — plotting utilities for DMDc model diagnostics.
  - `GroupedBasis.py` — tools for building combined bases across multiple signals.
  - `TimeDelayedDMDc.py` — implementation of time-delayed DMDc.
  - `notebooks/` — Jupyter notebooks for model selection, parameter sweeps, validation, and comparison across signals.

## Usage

1. **Generate actuation signals**

   Open `dmdc/signal_library/cylinder2D/signal_generator.ipynb` and run the notebook to generate the desired cylinder rotation signals. This produces `omega.csv` and/or additional signal files used by the OpenFOAM test case.

2. **Run OpenFOAM simulations**

   Use the `signal_library/cylinder2D/template` case to run OpenFOAM with the generated signal, by copying the generated `omega.csv` into the OpenFOAM case. The output data from these simulations provides the snapshot and control histories required to train DMDc.

3. **Build DMDc models**

   Use the notebooks in `dmdc/evaluation/notebooks/` to construct DMDc models from the collected simulation data. Several notebooks explore rank selection, time-delay embedding, and model accuracy.

4. **Evaluate and select parameters**

   The evaluation notebooks compare model predictions against validation data and help choose the best SVD rank and time-delay settings for each signal type.

5. **Build the combined basis and MORS operators**

   Run `dmdc/evaluation/notebooks/combined_basis.ipynb`. This notebook:

   - Builds a shared reduced basis (`common_Ur`) by computing a combined SVD across all four
     actuation signals (Random walk, Chirp, Chirp with varying amplitude, Amplitude-modulated)
     using `GroupedBasis`. The basis is saved as:
     ```
     common_Ur.pkl          # numpy array, shape (n_state, svd_rank)
     ```

   - Fits a time-delayed DMDc model for each signal projected onto this common basis and saves
     each as a pickle file containing a dictionary with the following keys:
     ```
     random_walk_dmdc.pkl
     chirp_dmdc.pkl
     chirp_varying_amp_dmdc.pkl
     AM_dmdc.pkl
     ```
     Each pickle file is a dictionary with:
     | Key | Shape | Description |
     |-----|-------|-------------|
     | `"basis"` | `(n_aug, r)` | Reduced basis used for the model (local, signal-specific) |
     | `"Atilde"` | `(r, r)` | Reduced linear state transition operator |
     | `"Btilde"` | `(r, m)` | Reduced control operator projected onto the basis |

   - Tests the models against a representative DRL validation signal using both individual
     models and the MORS strategy. The predictions (predicted state time series) are stored
     in a compressed pickle file (`exports/data_new.pkl.gz`), which is subsequently used in
     `rewardEvaluation/` to compute the lift and drag histories.

6. **Export models for integration**

   The pickle files produced in step 5 are consumed directly by the DRL integration. Point the
   `integrationWithDRL/rotatingCylinder2D/reference.yaml` configuration to the folder containing
   these files before running the DRL training.

## Notes

- The `dmdc/` folder is the source of truth for model construction and evaluation. It is separate from the DRL integration layer, which consumes the resulting DMDc models.
- The `signal_library` provides both the signal generation workflow and an example OpenFOAM template for applying the actuation to the cylinder boundary.
