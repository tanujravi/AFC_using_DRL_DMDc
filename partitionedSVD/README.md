### Online Partitioned SVD for streaming OpenFOAM simulations
This module implements an online eigen-partitioned Singular Value Decomposition (SVD) pipeline designed to process high-dimensional simulation data from a live OpenFOAM run in a distributed and memory-efficient manner. The SVD computation is split across MPI ranks to enable streaming-based compression and mode extraction in parallel, making it suitable for large-scale CFD simulations.

The system integrates with the SmartSim orchestration framework to launch ensembles of SVD tasks and manage data transfers via the SmartRedis client. It supports incremental updates to the decomposition, reconstruction of modes, and conversion back to OpenFOAM fields for post-processing.
The required input parameters and configuration options are specified in the provided `example_config_local.yaml` file.

An example simulation given is for flow over 2D cylinder. 

The code can be run using
```
python3 psvd.py example_config_local.yaml
```

