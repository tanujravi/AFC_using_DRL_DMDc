# Flow modeling and control employing dynamic mode decomposition and deep reinforcement learning

## Abstract

The high dimensionality of fluid flow simulations poses significant challenges in terms of
computational cost and memory usage, motivating the development of reduced-order modeling
techniques. Dynamic Mode Decomposition with Control (DMDc) is one such technique that provides a
framework for approximating the time evolution of the flow state while accounting for flow
actuation. In this work, DMDc is used as a surrogate model to accelerate the optimization of
control laws in closed-loop active flow control applications using deep reinforcement learning.
To address memory limitations in large-scale flow problems, a partitioned singular value
decomposition algorithm is employed to enable memory-efficient DMDc model construction. Overall,
this project aims to reduce the computational cost and turnaround time of flow simulations while
alleviating memory constraints in data-driven flow modeling and control.

## Overview

This repository accompanies a master's thesis that develops a framework for coupling a
reduced-order model (ROM) into a deep reinforcement learning (DRL) loop in a memory-efficient
manner. The sample problem used is the flow past a 2D cylinder controlled through the angular velocity
of the cylinder. The [drlfoam](https://github.com/OFDataCommittee/drlfoam) library is used as the
basis for the DRL framework, and Dynamic Mode Decomposition with Control (DMDc) is tested as the
reduced-order model.

The repository is organized into four components, each in its own folder, that together make up
the pipeline required for this coupling:

1. **`dmdc/`** — Builds a library of actuation signals and the corresponding DMDc models that
   capture the flow dynamics under different excitations. Time-delay embedding is implemented and
   tested to improve the robustness of the model predictions.

2. **`partitionedSVD/`** — The DMDc algorithm requires the singular value decomposition (SVD) of
   the snapshot matrix, which becomes a memory bottleneck for large-scale flows. This component
   implements a partitioned SVD that can be computed in a streaming fashion while the simulation
   runs in the background, greatly reducing storage and memory requirements.

3. **`rewardEvaluation/`** — Shows the mechanism for evaluating the lift and drag coefficients
   from the reduced-order model. The reduced fields exist in Python as NumPy arrays; these are
   converted into OpenFOAM fields using the `svdToFoam` utility so that standard OpenFOAM
   post-processing can be applied. The utility is borrowed from the
   [openfoam-smartsim](https://github.com/OFDataCommittee/openfoam-smartsim) repository, with the
   necessary changes provided in the `auxillary/` folder.

4. **`integrationWithDRL/`** — Shows how the reduced-order model is integrated into the `drlfoam`
   library by replacing the environment (a full OpenFOAM simulation) with the reduced-order model,
   without modifying the base code of `drlfoam` itself.

Each folder contains its own `README.md` with detailed usage instructions for that component.

## Requirements

The pipeline relies on the following software. Refer to the individual folder READMEs for
component-specific details.

- **OpenFOAM** (v2406 was used) — CFD solver for the flow past a cylinder.
- **Python 3** with: [PyDMD](https://github.com/PyDMD/PyDMD) (DMDc model fitting),
  [flowTorch](https://github.com/AndreWeiner/flowtorch) (loading OpenFOAM data),
  NumPy, and standard scientific Python libraries.
- [**SmartSim / SmartRedis**](https://www.craylabs.org/docs/index.html) — used by the
  partitioned SVD component for coupling OpenFOAM with Python.
- [**drlfoam**](https://github.com/OFDataCommittee/drlfoam) — the DRL–OpenFOAM coupling
  framework into which the ROM is integrated.
- [**openfoam-smartsim**](https://github.com/OFDataCommittee/openfoam-smartsim) — provides the
  `svdToFoam` utility (a modified version is included in `auxillary/`).

## Workflow used for the project

Below is the step-by-step workflow showing how the different components are built and used. In
the thesis this is done for the flow past a 2D cylinder. The partitioned SVD algorithm would
normally be used when building the DMDc model; however, because the 2D cylinder case is not
memory-intensive, it is not used here. Instead, the partitioned SVD code is validated
independently (see `partitionedSVD/`).

1. **Set up the environment.** Create and activate a Python virtual environment called `afcDrl`
   at the repository root, then install all required libraries.

   ```bash
   # repository top-level
   python3 -m venv afcDrl
   source afcDrl/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > **Note:** `flowtorch` is installed directly from its GitHub repository
   > (`git+https://github.com/FlowModelingControl/flowtorch`) as listed in `requirements.txt`.

   **SmartSim** requires an additional build step after the pip install to compile the
   backend (RedisAI and the SmartRedis C library):

   ```bash
   # CPU-only build — replace --device cpu with --device gpu for GPU support
   smart build --device cpu
   ```

   **svdToFoam** (used for reward evaluation) is a modified version of the utility from the
   [openfoam-smartsim](https://github.com/OFDataCommittee/openfoam-smartsim) repository.
   Follow the build instructions in `auxillary/svdToFoam/README.md`.

   **drlfoam** is required for the final integration step. Clone it alongside this repository
   and follow its own setup instructions before proceeding to step 7:

   ```bash
   git clone https://github.com/OFDataCommittee/drlfoam.git
   ```

2. **Generate the actuation signals.** A required signal is generated in Python as shown in
   `dmdc/signal_library/cylinder2D/signal_generator.ipynb`. The signal data is stored as time
   series data in a file called `omega.csv`. Four signals are used in this study: Random walk,
   Chirp, Chirp with varying amplitude, and Amplitude-modulated.

3. **Run the OpenFOAM simulations.** Once the signal data is generated, it is used to run an
   OpenFOAM simulation by enforcing a rotating-wall-velocity boundary condition, supplied with
   the time series angular velocity data. A template setup is provided in
   `dmdc/signal_library/cylinder2D/template`.

4. **Build the DMDc models.** Once the simulations are run for the different signals, the data
   required to build a DMDc model is available. A time-delayed version of DMDc is implemented to
   improve robustness, so the model has two hyperparameters: the number of time delays and the
   SVD rank. The individual models are tested in the notebooks under
   `dmdc/evaluation/notebooks`, and the optimum parameters are selected. These parameters are
   then used to find the individual DMDc linear operators.

5. **Combine the models (MORS).** A multi-operator random shuffling (MORS) approach is proposed,
   where each of these models is used in a random way to advance the time step. To achieve this,
   the state is projected onto a common subspace built from the data of all signals, as shown in
   `dmdc/evaluation/notebooks/combined_basis.ipynb`. The operators are stored and can then be
   used to advance the state in time for a given initial state and actuation.

6. **Evaluate rewards from the reduced fields.** The predicted states are obtained as time
   series data (NumPy arrays). To evaluate the rewards, the OpenFOAM post-processing utility `forceCoeffs` is
   used. To convert the predicted NumPy arrays into OpenFOAM fields, the `svdToFoam` utility from
   `openfoam-smartsim` is used. Multiple changes were made to the existing code to accommodate
   both scalar and vector fields; the modified code is provided in `auxillary/svdToFoam`. Once
   the lift and drag histories are computed, they are tested against the original predictions.
   This completes the components required to build the environment (see `rewardEvaluation/`).

7. **Integrate with DRL.** The environment is integrated with DRL using the `drlfoam` library.
   In the current implementation no changes are made to the contents of the `drlfoam` library
   itself; the integration is done through changes to the `Allrun` script. A MORS strategy is
   used, making use of all the models built from the Random walk, Chirp, Chirp with varying
   amplitude, and Amplitude-modulated signals. By using this `Allrun` script in the `drlfoam`
   library, the DRL training can be run by following the same steps as in `drlfoam` (see
   `integrationWithDRL/`).

## Results summary

In the current study, no learning was observed when training with the DMDc-based environment:
the averaged reward remained essentially constant across training episodes. Further analysis
showed that, despite significantly different actuation inputs, the predicted lift and drag
histories were nearly identical. This indicates that the DMDc model built was not accurate
enough to predict the rapid changes in dynamics associated with a typical RL actuation signal. See the thesis (Chapter 6 and Chapter 7) for a detailed
discussion and suggested directions for future work.


## Report

The report of this thesis can be found here: <https://zenodo.org/records/18375343>

BibTeX citation:

```bibtex
@misc{ravi_2026_18375343,
  author       = {Ravi, Tanuj and
                  Weiner, Andre and
                  Geise, Janis},
  title        = {Flow modeling and control employing dynamic mode
                  decomposition and deep reinforcement learning},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18375343},
  url          = {https://doi.org/10.5281/zenodo.18375343}
}
```

## References

- The original [drlfoam](https://github.com/OFDataCommittee/drlfoam) repository, currently
  maintained by Andre Weiner.

- Implementation of the (model-free) PPO algorithm for active flow control:
  - Thummar, Darshan. *Active flow control in simulations of fluid flows based on deep
    reinforcement learning.* <https://doi.org/10.5281/zenodo.4897961> (May 2021).
  - Gabriel, Fabian. *Aktive Regelung einer Zylinderumströmung bei variierender Reynoldszahl
    durch bestärkendes Lernen.* <https://doi.org/10.5281/zenodo.5634050> (October 2021).

- Geise, J. *Robust model-based deep reinforcement learning for flow control* (February 2023).
- Weiner, A. and Geise, J. *Model-based deep reinforcement learning for accelerated learning
  from flow simulations.* Meccanica 60(12):1771–1788, 2024.
- Proctor, J. L., Brunton, S. L., and Kutz, J. N. *Dynamic mode decomposition with control*
  (2014).
- Liang, F., Shi, R., and Mo, Q. *A split-and-merge approach for singular value decomposition
  of large-scale matrices.* Statistics and Its Interface 9:453–459, 2016.
- Maric, T., Fadeli, M. E., Rigazzi, A., Shao, A., and Weiner, A. *Combining machine learning
  with computational fluid dynamics using OpenFOAM and SmartSim.* Meccanica 60(6):1831–1850,
  2024.