### Integration of Reduced-Order Model with Deep Reinforcement Learning

> **Environment:** All commands and scripts in this folder should be run inside the `afcDrl`
> virtual environment. Activate it from the repository root with:
> ```bash
> source afcDrl/bin/activate
> ```

This module demonstrates the integration of the Dynamic Mode Decomposition with Control (DMDc) reduced-order model into the deep reinforcement learning (DRL) framework provided by the `drlfoam` library. The integration replaces the full OpenFOAM simulation environment with the DMDc surrogate model, enabling faster training of control policies for active flow control applications without modifying the core `drlfoam` codebase.

The example provided is for the flow past a 2D rotating cylinder, where the control action is the angular velocity of the cylinder. The DMDc model predicts the flow state evolution based on actuation inputs, and rewards are computed from the predicted lift and drag coefficients.

#### Key Components
- **`rotatingCylinder2D/`**: Contains the modified DRL environment setup that uses the DMDc model instead of running full OpenFOAM simulations. Includes Python scripts for execution, configuration files, and OpenFOAM case files for post-processing.
- **`plot_notebooks/`**: Jupyter notebooks for analyzing and comparing DRL training results under different actuation signals (e.g., chirp with varying amplitude, learning rates, random walk).

#### Prerequisites
- Python environment with required packages (NumPy, PyTorch, SmartSim, SmartRedis, etc.)
- Trained DMDc model (e.g., from `dmdc/` folder)
- `drlfoam` library installed and configured
- OpenFOAM for post-processing (optional, for reward evaluation)

#### Usage
1. **Prepare the `drlfoam` library**: Install and configure the original `drlfoam` repository as usual.
2. **Replace the test case folder**: Copy or symlink the current `rotatingCylinder2D/` folder into `drlfoam/openfoam/test_cases/rotatingCylinder2D`, replacing the existing case.
3. **Set `DRL_BASE`**: Ensure the `DRL_BASE` environment variable points to the root of the `drlfoam` repository before running the case.
4. **Run the case via original `drlfoam` workflow**: Use the same commands and entry points that `drlfoam` expects for running its test cases.
While Allrun is executed from drlfoam, in this integration the script launches `python main.py reference.yaml` from the current folder and uses the integrated DMDc surrogate model.
5. **Analyze results**: After the run completes, inspect output files and use the notebooks in `plot_notebooks/` to visualize performance, reward histories, and control actions.

#### Configuration Options
- `dmdc_path`: Path to the pickled DMDc model file.
- `policy`: Path to the PyTorch policy file.
- `n_steps`: Number of control time steps.
- `absOmegaMax`: Maximum allowable angular velocity.
- `train`: Boolean flag for training mode (affects action sampling).
- `DRL_BASE`: Should be set to the root of the original `drlfoam` repository so `Allrun` can source `openfoam/RunFunctions` correctly.
