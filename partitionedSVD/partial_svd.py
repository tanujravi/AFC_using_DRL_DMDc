"""Assemble partial data matrix and compute SVD.
"""

import numpy as np
from smartredis import Client
import ast
import sys
# setting coming from driver program
mpi_rank = ;mpi_rank;
svd_rank = ;svd_rank;
# settings to fetch data from database
#time_indices = list(range(10, 1001, 10))
time_indices = str(;time_indices;)
batch_no = ;batch_no;
type_matrix = ;type;
fo_name = "dataToSmartRedis"
field_name = "U"
time_indices = ast.literal_eval(time_indices)
# connect to database
client = Client(cluster=False)
def fetch_snapshot(time_index):
    dataset_name = f"{fo_name}_time_index_{time_index}_mpi_rank_{mpi_rank}"
    if client.dataset_exists(dataset_name):
        dataset = client.get_dataset(dataset_name)
        return dataset.get_tensor(f"field_name_{field_name}_patch_internal").flatten()
    else:
        return None

if type_matrix == "svd_new_matrix":
    data_matrix = np.vstack([fetch_snapshot(ti) for ti in time_indices]).T
elif type_matrix == "z_matrix":
    data_matrix = client.get_tensor(f"z_{batch_no}_{mpi_rank}")
else:
    raise ValueError(f"Unknown type_matrix")
# compute and store the partial SVD
U, s, VT = np.linalg.svd(data_matrix, full_matrices=False)
U = U[:, :svd_rank]
s = s[:svd_rank]
VT = VT[:svd_rank]
client.put_tensor(f"partSVD_U_mpi_rank_{mpi_rank}", U)
client.put_tensor(f"partSVD_VT_mpi_rank_{mpi_rank}", VT)
client.put_tensor(f"partSVD_s_mpi_rank_{mpi_rank}", s)
print("done")
