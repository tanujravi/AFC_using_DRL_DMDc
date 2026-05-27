# svdToFoam Replacement Source

This folder contains a modified `svdToFoam.C` source file used to support the reconstructed field workflow in this project.


## Purpose

The file is intended to replace the original `svdToFoam.C` implementation from the `openfoam-smartsim` library. The replacement allows reconstructed DMDc fields to be converted into OpenFOAM fields for use in downstream post-processing and reward evaluation.

## Usage

1. Clone or obtain the original `openfoam-smartsim` repository from GitHub.

2. Locate the `svdToFoam` utility source directory in that repository:
   ```
   openfoam-smartsim/applications/utilities/svdToFoam
   ```

3. Copy this file into the `openfoam-smartsim` source directory, replacing the existing `svdToFoam.C`:
   ```bash
   cp /path/to/AFC_using_DRL_DMDc/auxillary/svdToFoam/svdToFoam.C \
      /path/to/openfoam-smartsim/applications/utilities/svdToFoam/svdToFoam.C
   ```

4. Build the `openfoam-smartsim` library using that repository's build instructions.
   - Follow the instructions provided in the `openfoam-smartsim` repository README.
   - Typically this will involve building the utility with `wmake` or the standard OpenFOAM build commands used by the project.

## Notes

- This file is a project-specific modification and should only be used in the `openfoam-smartsim` build when you want the modified field reconstruction behavior.
- After building, the rebuilt `svdToFoam` utility can be used by `rewardEvaluation/make_fields.py` to convert reconstructed fields into OpenFOAM format.
