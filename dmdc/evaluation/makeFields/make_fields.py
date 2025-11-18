import sys, time, numpy as np, torch as pt
from yaml import safe_load
from os import makedirs
from os.path import join
from smartsim import Experiment
from smartsim.settings import RunSettings
from smartsim.status import SmartSimStatus
from smartredis import Client
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
import os
import sys, types
import shutil
# Map old private module name → current public one
import numpy.core.numeric as numeric
shim = types.ModuleType("numpy._core.numeric")
shim.__dict__.update(numeric.__dict__)
sys.modules["numpy._core.numeric"] = shim


def wait_for_completion(exp, entities, poll_interval=5, timeout=None):

    """Block until all entities in a SmartSim experiment complete.

    Periodically checks whether each entity has completed. Raises an
    exception if any entity fails or if a timeout occurs.

    :param exp: SmartSim experiment instance.
    :type exp: smartsim.Experiment
    :param entities: List or group of SmartSim entities (Model or Ensemble).
    :type entities: list or Ensemble or Model
    :param poll_interval: Time (in seconds) between status checks, defaults to 5.
    :type poll_interval: int, optional
    :param timeout: Maximum wait time in seconds, defaults to None (waits indefinitely).
    :type timeout: float, optional

    :raises RuntimeError: If one or more entities fail.
    :raises TimeoutError: If the timeout duration is exceeded.
    """    
    start = time.time()
    while True:
        statuses = exp.get_status(entities)
        if all(s == SmartSimStatus.STATUS_COMPLETED for s in statuses): break
        if any(s == SmartSimStatus.STATUS_FAILED for s in statuses):
            raise RuntimeError("One or more entities failed.")
        if timeout and (time.time() - start) > timeout:
            raise TimeoutError("Timeout waiting for SmartSim entities to complete.")
        time.sleep(poll_interval)



def split_1d_indices(N: int, n: int):
    """OpenFOAM-simple split of N rows into n contiguous chunks.
    Returns list of (start, stop) indices (stop exclusive)."""
    q, r = divmod(N, n)
    sizes = [(q + 1) if i < r else q for i in range(n)]
    starts = np.cumsum([0] + sizes[:-1]).tolist()
    return [(s, s + sz) for s, sz in zip(starts, sizes)]

def tensor_key(field: str, svd_rank: int, mpi_rank: int, time_index: int):
    """Match your exact naming convention."""
    return (
        f"rec_ensemble_r{svd_rank}_field_name_{field}_{mpi_rank}"
        f".rank_{svd_rank}_field_name_{field}_mpi_rank_{mpi_rank}"
        f"_time_index_{time_index}"
    )

def put_scalar_series(client, base_field: str, arr: np.ndarray, svd_rank: int,
                      nprocs: int, time_stride: int):
    """Upload scalar field per-rank, per-time (arr: [n_points, n_times])."""
    n_points, n_times = arr.shape
    splits = split_1d_indices(n_points, nprocs)
    arr = np.asarray(arr, dtype=np.float64, order="C")

    for rank, (s, e) in enumerate(splits):
        for i in range(n_times):
            ti = (i + 1) * time_stride
            key = tensor_key(base_field, svd_rank, rank, ti)
            # slice is contiguous in the first axis; copy to be safe for the client
            vec = np.copy(arr[s:e, i])
            client.put_tensor(key, vec)

def put_vector_series(client, base_field: str, ux: np.ndarray, uy: np.ndarray,
                      uz: np.ndarray | None, svd_rank: int, nprocs: int,
                      time_stride: int):
    """Upload vector field U per-rank, per-time (each input: [n_points, n_times])."""
    assert ux.shape == uy.shape, "u_x and u_y shape mismatch"
    n_points, n_times = ux.shape
    if uz is None:
        uz = np.zeros_like(ux, dtype=np.float64)
    else:
        assert uz.shape == ux.shape, "u_z shape mismatch"

    splits = split_1d_indices(n_points, nprocs)

    # Ensure dtype float64
    ux = np.asarray(ux, dtype=np.float64, order="C")
    uy = np.asarray(uy, dtype=np.float64, order="C")
    uz = np.asarray(uz, dtype=np.float64, order="C")

    for rank, (s, e) in enumerate(splits):
        for i in range(n_times):
            ti = (i + 1) * time_stride
            # stack into (n_chunk, 3)
            vec = np.stack([ux[s:e, i], uy[s:e, i], uz[s:e, i]], axis=1)
            key = tensor_key(base_field, svd_rank, rank, ti)
            client.put_tensor(key, np.copy(vec))

# ---------------------------
# Configuration
# ---------------------------

svd_rank = 20
mpi_ranks = 2                  # total MPI ranks
time_stride = 20            # matches your prior (i+1)*20 convention
fo_name = "dataToSmartRedis"
template_dir = "template"
parent_dir = join("run", "cylinder")
os.makedirs(parent_dir)

exp = Experiment(name = "cylinder", exp_path= parent_dir, launcher = "local")
db = exp.create_database(port=1995, interface="lo")
exp.start(db)
client = Client(address=db.get_address()[0], cluster=False)

import pickle, gzip
path_data = sys.argv[1] if len(sys.argv) > 1 else "data/data.pkl.gz"

with gzip.open(path_data, "rb") as f:
    data = pickle.load(f)
for src_dataset_name in data.keys():
    pred_block = data[src_dataset_name][0]
    omega_act_block = data[src_dataset_name][1]
    src_dataset_name_refined = "_".join(src_dataset_name.split())
    src_dir = join(parent_dir, src_dataset_name_refined)
    os.makedirs(src_dir, exist_ok=True)
    # ---------------------------
    # Iterate all datasets & fields in pred_block
    # ---------------------------
    #pred_block = true_block
    for tgt_dataset_name, fields_dict in pred_block.items():
        # Detect presence of vector components
        has_ux = "u_x" in fields_dict
        has_uy = "u_y" in fields_dict
        has_uz = "u_z" in fields_dict
        tgt_dataset_name_refined = "_".join(tgt_dataset_name.split())
        # If vector components exist, upload U
        if has_ux and has_uy:
            rec_x = np.asarray(fields_dict["u_x"], dtype=np.float64)
            rec_y = np.asarray(fields_dict["u_y"], dtype=np.float64)
            rec_z = (np.asarray(fields_dict["u_z"], dtype=np.float64)
                    if has_uz else None)
            put_vector_series(client, "U", rec_x, rec_y, rec_z, svd_rank, mpi_ranks, time_stride)

        # Upload all scalar fields (anything that's not a vector component)
        for field_name, arr in fields_dict.items():
            if field_name in {"u_x", "u_y", "u_z"}:
                continue  # handled above as U
            rec = np.asarray(arr, dtype=np.float64)
            if rec.ndim != 2:
                raise ValueError(f"Expected 2D array for field '{field_name}' (points x times), got shape {rec.shape}")
            put_scalar_series(client, field_name, rec, svd_rank, mpi_ranks, time_stride)
        
        dest = os.path.join(src_dir, tgt_dataset_name_refined)

        shutil.copytree(template_dir, dest)
        startTime = omega_act_block[tgt_dataset_name][0, 0]
        dt = omega_act_block[tgt_dataset_name][1, 0] - omega_act_block[tgt_dataset_name][0, 0]
        startTime = startTime - dt
        control_dict = os.path.join(dest, "system", "controlDict")
        print(startTime)
        os.system(f"foamDictionary {control_dict} -entry startTime  -set {startTime}")

        np.savetxt(
            join(dest, "omega.csv"),
            omega_act_block[tgt_dataset_name],
            delimiter=",",
            header="time,omega",
            comments="",        # keeps the header clean (no leading '#')
            fmt="%.8g"          # adjust precision if you like
        )


        fields = ["p", "U"]

        for field in fields:
            settings = exp.create_run_settings(
                exe="svdToFoam",
                exe_args=f"-fieldName {field} -svdRank {svd_rank} "
                        f"-FOName {fo_name} -parallel",
                run_command="mpirun", run_args={"np": f"{mpi_ranks}"}
            )
            model = exp.create_model(name=f"svdToFoam_r{svd_rank}_field_name_{field}_tgt_{tgt_dataset_name_refined}_src_{src_dataset_name_refined}",
                                        run_settings=settings, path=dest)
            exp.start(model, summary=False, block=False)
            wait_for_completion(exp, model)

        settings = exp.create_run_settings(
            exe="pimpleFoam",
            exe_args=f"-postProcess -dict system/FO_force -parallel",
            run_command="mpirun", run_args={"np": f"{mpi_ranks}"}
        )
        model = exp.create_model(name=f"forces_reconstructed_fields_tgt_{tgt_dataset_name_refined}_src_{src_dataset_name_refined}",
                                    run_settings=settings, path=dest)
        exp.start(model, summary=False, block=False)
        wait_for_completion(exp, model)

exp.stop(db)
