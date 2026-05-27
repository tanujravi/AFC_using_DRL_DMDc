# svdToFoam Replacement Source

This folder contains a modified `svdToFoam.C` source file used to support the reconstructed field workflow in this project.


## Purpose

This folder contains two modified source files from the `openfoam-smartsim` library:

- **`svdToFoam.C`** — extended to support both scalar and vector field reconstruction, allowing
  reconstructed DMDc fields (NumPy arrays) to be converted into OpenFOAM fields for
  post-processing and reward evaluation.
- **`smartRedisClient.C`** — modified to track dataset time indices per MPI rank using a
  SmartRedis list, so that datasets can be accumulated and retrieved by rank during the
  partitioned SVD workflow.

## Usage

1. Clone the `openfoam-smartsim` repository from GitHub.

2. Copy both modified files into the repository, replacing the originals:
   ```bash
   # Replace svdToFoam.C
   cp /path/to/AFC_using_DRL_DMDc/auxillary/svdToFoam/svdToFoam.C \
      /path/to/openfoam-smartsim/applications/utilities/svdToFoam/svdToFoam.C

   # Replace smartRedisClient.C (locate the file in the openfoam-smartsim source tree)
   cp /path/to/AFC_using_DRL_DMDc/auxillary/svdToFoam/smartRedisClient.C \
      /path/to/openfoam-smartsim/src/smartRedis/smartRedisClient.C
   ```

3. Build the `openfoam-smartsim` library following that repository's build instructions.
   - Source your OpenFOAM environment first (adjust path as needed):
     ```bash
     source /usr/lib/openfoam/openfoam2406/etc/bashrc
     ```
   - Then build with `wmake` or the standard OpenFOAM build commands described in the
     `openfoam-smartsim` README.

## Notes

- These files are project-specific modifications. Only use them when building `openfoam-smartsim`
  for this project.
- After building, the rebuilt `svdToFoam` utility is called by `rewardEvaluation/make_fields.py`
  to convert reconstructed fields into OpenFOAM format.
