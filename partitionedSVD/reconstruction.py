"""Assemble global left singular vectors and compute reconstruction.
"""

import numpy as np
from smartredis import Client, Dataset
from redis import Redis
import ast
import sys

# setting coming from driver program
mpi_rank = ;mpi_rank;
svd_rank = ;svd_rank;
time_indices = str(;time_indices;)
time_indices = ast.literal_eval(time_indices)

field_name = str(;field_name;)
ref_map = ast.literal_eval(str(;ref_map;))
ref_value = ref_map.get(field_name, 1.0)

# connect to database
client = Client(cluster=False)

def _get_tensor_or_none(key: str):
    try:
        return client.get_tensor(key)
    except Exception:
        return None

def _put_scalar_series(name_prefix: str, mat: np.ndarray):
    """mat: (n_points, K) -> write (n_points,) per time index."""
    for i, ti in enumerate(time_indices):
        key = f"{name_prefix}_time_index_{ti}"
        # store as (n_points,) vector
        client.put_tensor(key, np.copy(mat[:, i]))

s = client.get_tensor("s_incremental")          # (r,)
VT = client.get_tensor("VT_incremental")        # (r, K)
K = VT.shape[1]
if field_name != "U":
    # Left singular block for THIS field on THIS rank: (n_points, r)
    U_field = _get_tensor_or_none(f"U_incremental_field_name_{field_name}_mpi_rank_{mpi_rank}")
    if U_field is None:
        raise RuntimeError(
            f"Missing U block for field '{field_name}' on rank {mpi_rank}: "
            f"key 'U_incremental_field_name_{field_name}_mpi_rank_{mpi_rank}' not found."
        )

    # Reconstruction: (n_points, r) * (r,) -> (n_points, r), then @ (r, K) -> (n_points, K)
    rec = (U_field * s) @ VT
    rec *= ref_value
    name_prefix = f"rank_{svd_rank}_field_name_{field_name}_mpi_rank_{mpi_rank}"
    _put_scalar_series(name_prefix, rec)
    for i in range(svd_rank):
        name = f"global_U_field_name_{field_name}_mpi_rank_{mpi_rank}_mode_{i}"
        client.put_tensor(name, np.copy(U_field[:, i]))

else:
    # Try to get component U blocks; any missing -> zeros of correct shape
    Ux = _get_tensor_or_none(f"U_incremental_field_name_Ux_mpi_rank_{mpi_rank}")
    Uy = _get_tensor_or_none(f"U_incremental_field_name_Uy_mpi_rank_{mpi_rank}")
    Uz = _get_tensor_or_none(f"U_incremental_field_name_Uz_mpi_rank_{mpi_rank}")
    if Ux is None and Uy is None and Uz is None:
        raise RuntimeError(
            "No velocity component U blocks found for reconstruction. "
            "Expected any of keys: "
            f"[U_incremental_field_name_Ux_mpi_rank_{mpi_rank}, "
            f"U_incremental_field_name_Uy_mpi_rank_{mpi_rank}, "
            f"U_incremental_field_name_Uz_mpi_rank_{mpi_rank}]."
        )

    shape_ref = next(m for m in (Ux, Uy, Uz) if m is not None)
    n_points, r = shape_ref.shape

    # Fill missing with zeros of shape (n_points, r)
    if Ux is None: Ux = np.zeros((n_points, r), dtype=shape_ref.dtype)
    if Uy is None: Uy = np.zeros((n_points, r), dtype=shape_ref.dtype)
    if Uz is None: Uz = np.zeros((n_points, r), dtype=shape_ref.dtype)

    # Reconstruct each component (n_points, K)
    rec_x = (Ux * s) @ VT
    rec_x *= ref_value
    rec_y = (Uy * s) @ VT
    rec_y *= ref_value

    rec_z = (Uz * s) @ VT
    rec_z *= ref_value

    # Write combined vector snapshots (n_points, 3) per time
    for i, ti in enumerate(time_indices):
        vec = np.stack([rec_x[:, i], rec_y[:, i], rec_z[:, i]], axis=1)  # (n_points, 3)
        key = f"rank_{svd_rank}_field_name_U_mpi_rank_{mpi_rank}_time_index_{ti}"
        client.put_tensor(key, np.copy(vec))
    
    for i in range(svd_rank):
        name = f"global_U_field_name_U_mpi_rank_{mpi_rank}_mode_{i}"
        vec = np.stack([Ux[:, i], Uy[:, i], Uz[:, i]], axis=1)
        client.put_tensor(name, np.copy(vec))

print("done")
