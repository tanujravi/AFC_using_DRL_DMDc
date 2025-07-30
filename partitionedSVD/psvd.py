import os
import shutil
from smartsim import Experiment
from smartredis import Client
import numpy as np
import matplotlib.pyplot as plt
import sys
from yaml import safe_load
from os import makedirs
from os.path import join
from smartsim.settings import RunSettings
from redis import Redis
from smartredis import Client
import torch as pt
import time
from smartsim.status import SmartSimStatus


def fetch_snapshot(time_index, mpi_rank):
    dataset_name = f"{fo_name}_time_index_{time_index}_mpi_rank_{mpi_rank}"
    # print(dataset_name)
    if client.dataset_exists(dataset_name):
        dataset = client.get_dataset(dataset_name)
        return dataset.get_tensor(f"field_name_{field_name}_patch_internal").flatten()
    else:
        return None


def fetch_timeseries(time_indices, mpi_rank):
    return np.vstack([fetch_snapshot(ti, mpi_rank) for ti in time_indices]).T


def wait_for_completion(exp, entities, poll_interval=5, timeout=None):
    start_time = time.time()

    while True:
        statuses = exp.get_status(entities)

        if all(status == SmartSimStatus.STATUS_COMPLETED for status in statuses):
            break

        if any(status == SmartSimStatus.STATUS_FAILED for status in statuses):
            raise RuntimeError("One or more entities failed.")

        if timeout and (time.time() - start_time) > timeout:
            raise TimeoutError("Timeout waiting for SmartSim entities to complete.")

        time.sleep(poll_interval)

def start_openfoam_sim(exp, config):
    sim_config = config["simulation"]
    base_case_path = sim_config["base_case"]
    # base_name = base_case_path.split("/")[-1]
    rs = RunSettings(exe="bash", exe_args="Allrun")
    base_sim = exp.create_model(
        "base_sim",
        run_settings=rs,
    )

    base_sim.attach_generator_files(to_copy=base_case_path)

    exp.generate(base_sim)
    exp.start(base_sim, block=False, summary=False)
    return base_sim

def run_first_svd(exp, time_indices, svd_rank, num_mpi_ranks, batch_no):
    svd_settings = exp.create_run_settings(
        exe="python3", exe_args=f"partial_svd.py"
    )
    params_svd = {
        "mpi_rank": list(range(num_mpi_ranks)),
        "svd_rank": svd_rank,
        "time_indices": str(time_indices),
        "type": '"svd_new_matrix"',
        "batch_no": batch_no,
    }
    svd_ensemble_name = f"svd_ensemble_batch_{batch_no}"
    svd_ensemble = exp.create_ensemble(
        svd_ensemble_name,
        params=params_svd,
        run_settings=svd_settings,
        perm_strategy="all_perm",
    )
    config_file = "./partial_svd.py"
    svd_ensemble.attach_generator_files(to_configure=config_file)

    exp.generate(svd_ensemble, overwrite=True)

    exp.start(svd_ensemble, summary=False, block=False)
    wait_for_completion(exp, svd_ensemble)

def run_second_svd(exp, time_indices, svd_rank, num_mpi_ranks, batch_no):

    params_svdz = {
        "mpi_rank": list(range(num_mpi_ranks)),
        "svd_rank": svd_rank,
        "time_indices": str(time_indices),
        "type": '"z_matrix"',
        "batch_no": batch_no,
    }
    svdz_settings = exp.create_run_settings(
        exe="python3", exe_args=f"partial_svd.py"
    )

    svdz_ensemble_name = f"svdz_ensemble_batch_{batch_no}"
    svdz_ensemble = exp.create_ensemble(
        svdz_ensemble_name,
        params=params_svdz,
        run_settings=svdz_settings,
        perm_strategy="all_perm",
    )
    config_file = "./partial_svd.py"
    svdz_ensemble.attach_generator_files(to_configure=config_file)

    exp.generate(svdz_ensemble, overwrite=True)
    exp.start(svdz_ensemble, summary=False, block=False)
    wait_for_completion(exp, svdz_ensemble)

def plot_singular_values(time_indices_for_data, s_incremental, svd_rank):
    data_matrix = pt.cat([pt.tensor(fetch_timeseries(time_indices_for_data, rank_i)) for rank_i in range(num_mpi_ranks)], dim = 0)
    _, s_pl, _ = np.linalg.svd(data_matrix, full_matrices=False)
    s_pl = s_pl[:svd_rank]
    fig, ax = plt.subplots(figsize=(6, 3))
    ns = list(range(svd_rank))
    ax.plot(ns, s_pl, label="SVD")
    ax.plot(range(len(s_incremental)), s_incremental, label="stream. SVD")
    ax.legend()
    #ax.set_xlim(0, 19)
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_title("SVD singular values")
    fig.savefig("compare_svd_stream_to_full.png")

def run_reconstruction(exp, svd_rank, num_mpi_ranks, time_indices_for_data):
    # compute global left singular vectors and reconstruction
    rec_settings = exp.create_run_settings(exe="python3", exe_args=f"reconstruction.py")
    params = {
        "mpi_rank" : list(range(num_mpi_ranks)),
        "svd_rank" : svd_rank,
        "time_indices": str(time_indices_for_data)
    }
    rec_ensemble = exp.create_ensemble(f"rec_ensemble_r{svd_rank}", params=params, run_settings=rec_settings, perm_strategy="all_perm")
    config_file = "./reconstruction.py"
    rec_ensemble.attach_generator_files(to_configure=config_file)
    exp.generate(rec_ensemble, overwrite=True)
    exp.start(rec_ensemble, summary=False, block=False)
    wait_for_completion(exp, rec_ensemble)

def run_svdToFoam(exp, svd_rank, field_name, fo_name, num_mpi_ranks, case_name):

    svdToFoam_settings = exp.create_run_settings(
        exe="svdToFoam",
        exe_args=f"-fieldName {field_name} -svdRank {svd_rank} -FOName {fo_name} -parallel",
        run_command="mpirun", run_args={"np": f"{num_mpi_ranks}"}
    )

    svdToFoam_model = exp.create_model(name=f"svdToFoam_r{svd_rank}", run_settings=svdToFoam_settings, path=case_name)
    exp.start(svdToFoam_model, summary=False, block=False)
    wait_for_completion(exp, svdToFoam_model)


def compute_first_svd_USV(client, svd_ensemble_name, num_mpi_ranks, svd_rank):
    Y = []
    for rank_i in range(num_mpi_ranks):
        s_svd = client.get_tensor(
            svd_ensemble_name + f"_{rank_i}.partSVD_s_mpi_rank_{rank_i}"
        )
        VT_svd = client.get_tensor(
            svd_ensemble_name + f"_{rank_i}.partSVD_VT_mpi_rank_{rank_i}"
        )
        Y.append(s_svd[:, np.newaxis] * VT_svd)
    Y = np.concatenate(Y, axis=0)
    Uy, sy, VTy = np.linalg.svd(Y, full_matrices=False)
    Uy = Uy[:, :svd_rank]
    sy = sy[:svd_rank]
    VTy = VTy[:svd_rank]
    U_li = [
        client.get_tensor(
            f"{svd_ensemble_name}_{rank_i}.partSVD_U_mpi_rank_{rank_i}"
        )
        for rank_i in range(num_mpi_ranks)
    ]
    n_times_svd = U_li[0].shape[1]

    U = np.concatenate(
        [
            (U_svd @ Uy[ind * n_times_svd : (ind + 1) * n_times_svd])
            for ind, U_svd in enumerate(U_li)
        ],
        axis=0,
    )
    s = sy
    VT = VTy
    return U, s, VT, U_li

def compute_second_svd_USV(client, svdz_ensemble_name, num_mpi_ranks, svd_rank):
    Yz = []
    for rank_i in range(num_mpi_ranks):
        s_svdz = client.get_tensor(
            svdz_ensemble_name + f"_{rank_i}.partSVD_s_mpi_rank_{rank_i}"
        )
        VT_svdz = client.get_tensor(
            svdz_ensemble_name + f"_{rank_i}.partSVD_VT_mpi_rank_{rank_i}"
        )
        Yz.append(s_svdz[:, np.newaxis] * VT_svdz)
    Yz = np.concatenate(Yz, axis=0)
    Uyz, syz, VTyz = np.linalg.svd(Yz, full_matrices=False)
    Uyz = Uyz[:, :svd_rank]
    syz = syz[:svd_rank]
    VTyz = VTyz[:svd_rank]

    Uz_li = [
        client.get_tensor(
            f"{svdz_ensemble_name}_{rank_i}.partSVD_U_mpi_rank_{rank_i}"
        )
        for rank_i in range(num_mpi_ranks)
    ]
    
    n_times_svdz = Uz_li[0].shape[1]
    Uz = np.concatenate(
        [
            (Uz_svd @ Uyz[ind * n_times_svdz : (ind + 1) * n_times_svdz])
            for ind, Uz_svd in enumerate(Uz_li)
        ],
        axis=0,
    )
    sz = syz
    VTz = VTyz
    return Uz, sz, VTz

if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        with open(config_file, "r") as cf:
            config = safe_load(cf)
    except Exception as e:
        print(e)

    # function object name
    fo_name = "dataToSmartRedis"
    # field of which to compute the SVD
    field_name = "U"

    # exp = Experiment("partitioned-svd-cylinder", launcher="local")
    makedirs(config["experiment"]["exp_path"], exist_ok=True)
    case_name = join(config["experiment"]["exp_path"], "base_sim")
    exp = Experiment(**config["experiment"])
    db = exp.create_database(port=1900, interface="lo")
    exp.start(db)
    try:

        client = Client(address=db.get_address()[0], cluster=False)
        
        config_svd = config["svd_params"]
        num_mpi_ranks = config_svd["num_mpi_ranks"]
        svd_rank = config_svd["svd_rank"]
        i, batch = config_svd["snapshot_sampling_interval"], config_svd["batch_size"]
        n_times = config_svd["end_time"]
        batch_no = 0
        sampling_interval = config_svd["snapshot_sampling_interval"]
      
        base_sim = start_openfoam_sim(exp, config)

        while i + batch <= n_times:

            time_index = i + batch
            name = f"list_time_index_{time_index}"
            fields_updated = client.poll_list_length(
                f"list_time_index_{time_index}", num_mpi_ranks, 10, 60000
            )
            if not fields_updated:
                raise ValueError("Fields dataset list not updated.")

            time_indices = list(range(i, min(i + batch + 1, n_times + 1), sampling_interval))
            svd_ensemble_name = f"svd_ensemble_batch_{batch_no}"
            run_first_svd(exp, time_indices, svd_rank, num_mpi_ranks, batch_no)
            U, s, VT, U_li = compute_first_svd_USV(client, svd_ensemble_name, num_mpi_ranks, svd_rank)
            if batch_no == 0:
                U_incremental = U
                s_incremental = s
                VT_incremental = VT
            else:

                z = np.concatenate((U_incremental * s_incremental, U * s), axis=1)
                z_chunks = np.array_split(z, num_mpi_ranks, axis=0)
                for mpi_rank, z_chunk in enumerate(z_chunks):
                    client.put_tensor(f"z_{batch_no}_{mpi_rank}", z_chunk)
            
                svdz_ensemble_name = f"svdz_ensemble_batch_{batch_no}"
                run_second_svd(exp, time_indices, svd_rank, num_mpi_ranks, batch_no)
                Uz, sz, VTz = compute_second_svd_USV(client, svdz_ensemble_name, num_mpi_ranks, svd_rank)
                U_incremental = Uz
                s_incremental = sz
                VT_incremental = np.concatenate(
                    (VT_incremental.T @ VTz.T[:svd_rank], VT.T @ VTz.T[svd_rank:]),
                    axis=0,
                ).T

            U_incremental = U_incremental[:, :svd_rank]
            s_incremental = s_incremental[:svd_rank]
            VT_incremental = VT_incremental[:svd_rank]
            i += batch
            batch_no += 1
            print("svd stream done till time step = {}".format(i))

        exp.stop(base_sim)
        
        time_indices_for_data = list(range(sampling_interval, i+1, sampling_interval))

        plot_singular_values(time_indices_for_data, s_incremental, svd_rank)

        client.put_tensor(f"s_incremental", s_incremental)
        client.put_tensor(f"VT_incremental", VT_incremental)
        sizes = [U_l.shape[0] for U_l in U_li]
        Us_incremental = pt.split(pt.tensor(U_incremental), sizes, dim=0)
        for mpi_rank, U_incre in enumerate(Us_incremental):
            client.put_tensor(f"U_incremental_{mpi_rank}", np.array(U_incre))

        run_reconstruction(exp, svd_rank, num_mpi_ranks, time_indices_for_data)

        run_svdToFoam(exp, svd_rank, field_name, fo_name, num_mpi_ranks, case_name)

        exp.stop(db)

    except Exception as e:
        print(e)
        exp.stop(db)
        exp.stop(base_sim)
