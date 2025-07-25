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

field_name = "U"

# connect to database
client = Client(cluster=False)
"""
r = Redis(host="localhost")  # Default Redis address
keys = r.keys("*")
print("All Redis keys:", keys)
"""
# compute global left singular vectors
U = client.get_tensor(f"U_incremental_{mpi_rank}")

# optional: delete Ui from the database to save space
#client.delete_tensor(f"svd_ensemble_{mpi_rank}.partSVD_U_mpi_rank_{mpi_rank}")

# compute and save rank-r reconstruction
s = client.get_tensor(f"s_incremental")
VT = client.get_tensor(f"VT_incremental")
rec = U @ np.diag(s) @ VT
print(rec.shape)
print(len(time_indices))
n_points = rec.shape[0] // 3
for i, ti in enumerate(time_indices):
    name = f"rank_{svd_rank}_field_name_{field_name}_mpi_rank_{mpi_rank}_time_index_{ti}"
    client.put_tensor(name, np.copy(rec[:, i].reshape((n_points, 3))))

# optional: save global U into database for visualization
for i in range(svd_rank):
    name = f"global_U_mpi_rank_{mpi_rank}_mode_{i}"
    client.put_tensor(name, np.copy(U[:, i].reshape((n_points, 3))))


