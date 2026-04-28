import sys
import os
import time
import glob
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import csv
import numpy as np
import torch as pt
from yaml import safe_load
from os.path import join
import os
from smartsim import Experiment
from smartredis import Client
import matplotlib.pyplot as plt 
from omega_controller import OmegaController

from smartsim_execution import smartSimExecution
from probes import Probes
from smartsim_util import smartSimUtil
from dmdc_util import dmdcUtil
import socket
import random

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))         # 0 = ask OS for any free port
        return s.getsockname()[1]

def remove_first_data_line(path):
    """Remove the first non-comment line from a postProcess file."""
    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    removed = False

    for line in lines:
        stripped = line.strip()

        # keep empty or comment lines
        if stripped.startswith("#") or stripped == "":
            new_lines.append(line)
            continue

        # this is the first data line → skip it
        if not removed:
            removed = True
            continue

        # keep all remaining data lines
        new_lines.append(line)

    # overwrite file
    with open(path, "w") as f:
        f.writelines(new_lines)
if __name__ == "__main__":

    # ------------------------- load configuration ------------------------- #
    with open("reference.yaml", "r") as f:
        reference_dic = safe_load(f)

    cwd = os.getcwd()
    print(f"cwd = {cwd}")
    # ROM / database / run parameters
    svd_rank = int(reference_dic.get("svd_rank", 20))
    mpi_ranks = int(reference_dic.get("mpi_ranks", 2))
    time_stride = int(reference_dic.get("time_stride", 20))
    fo_name = reference_dic.get("fo_name", "dataToSmartRedis")
    parent_dir = join("run", "cylinder")
    os.makedirs(parent_dir, exist_ok=True)

    # SmartSim experiment + DB
    exp = Experiment(name="cylinder", exp_path=parent_dir, launcher="local")
    db_port = get_free_port()
    print(f"Using DB port {db_port}")

    db = exp.create_database(port=db_port, interface="lo")
    exp.start(db)
    client = Client(address=db.get_address()[0], cluster=False)

    import pickle
    import gzip


    
    # Load DMDc model
    
    
    path_signal = "models_lib/chirp_varying_amp_dmdc.pkl"
    with open(path_signal, "rb") as f:
        dmdc_model_chirp_varying_amp = pickle.load(f)

    path_signal = "models_lib/chirp_dmdc.pkl"
    with open(path_signal, "rb") as f:
        dmdc_model_chirp = pickle.load(f)

    path_signal = "models_lib/random_walk_dmdc.pkl"
    with open(path_signal, "rb") as f:
        dmdc_model_random_walk = pickle.load(f)


    path_signal = "models_lib/AM_dmdc.pkl"
    with open(path_signal, "rb") as f:
        dmdc_model_AM = pickle.load(f)

    model_lib = [dmdc_model_random_walk, dmdc_model_chirp, dmdc_model_chirp_varying_amp, dmdc_model_AM]
    #model_lib = [dmdc_model_chirp_varying_amp]
    path_data = "models_lib/common_Ur.pkl" 
    with open(path_data, "rb") as f:
        common_Ur = pickle.load(f)

    
    # Build initial state history x0
    q = 100
    #x0, field_names, times_used, points_xy = dmdcUtil.build_x0(q=q, reference_dic=reference_dic, path = "../initial_states")
    with open("../initial_states.pkl", "rb") as f:
        x0, field_names, times_used, points_xy = pickle.load(f)

    #x0, field_names, times_used, points_xy = dmdcUtil.build_x0(q=q, reference_dic=reference_dic, path = "../initial_states")

    pad_cols = x0.shape[1]
    n_state_total = x0.shape[0]
    n_points = points_xy.shape[0]
    get_field_block = dmdcUtil.make_field_block_extractor(n_points)

    # Control buffer (single input with window length pad_cols)
    U_hist = np.zeros((1, pad_cols), dtype=float)

    # OmegaController parameters
    policy_path = reference_dic.get("policy", "policy.pt")
    abs_omega_max = float(reference_dic.get("absOmegaMax", 50.0))
    seed = int(reference_dic.get("seed", 1234))
    train = bool(reference_dic.get("train", True))
    time_start = float(reference_dic.get("timeStart", 4.0))
    dt_control = float(reference_dic.get("executeInterval", 0.01))
    n_steps = int(reference_dic.get("n_steps", 1000))
    radius = float(reference_dic.get("radius", 0.5))
    n_probes = int(reference_dic.get("n_probes", 12))

    fig, ax = plt.subplots(figsize = (6, 4))
    ax.scatter(points_xy[:,0], points_xy[:,1])
    fig.savefig("plot.png")

    omega_controller = OmegaController(
        policy_path=policy_path,
        abs_omega_max=abs_omega_max,
        dt_control=dt_control,
        start_time=time_start,
        train=train,
        seed=seed,
        trajectory_file="trajectory.csv",
    )

    def get_omega(t: float, dt: float, p_sample: np.ndarray) -> float:
        return omega_controller.compute(t, dt, p_sample)

    # Time integration
    dt = dt_control
    t = time_start

    control_dict = os.path.join("system", "controlDict")

    os.system(f"foamDictionary {control_dict} -entry deltaT -set {dt}")

    probe_locations =   [(0.3, 0.15, 0.005),
                         (0.3, 0.2, 0.005),
                         (0.3, 0.25, 0.005),
                         (0.4, 0.15, 0.005),
                         (0.4, 0.2, 0.005),
                         (0.4, 0.25, 0.005),
                         (0.5, 0.15, 0.005),
                         (0.5, 0.2, 0.005),
                         (0.5, 0.25, 0.005),
                         (0.6, 0.15, 0.005),
                         (0.6, 0.2, 0.005),
                         (0.6, 0.25, 0.005)]
    
    time_end = 8
    smartSimExecution.update_openfoam_control_dicts(time_start, time_end)

    p_history = []  # each row: [t, p_0, p_1, ..., p_{N-1}]
    p_sample = Probes.get_pressure_interpolation(probe_locations, x0[:n_points, -1], points_xy)
    p_history.append(p_sample)
    omega_list = list()
    X_pred = []
    x_roll = x0.copy()
    random.seed(1234)
    n_shuf = 1
    dmdc_model = random.choice(model_lib)          # initial model
    try:
        for step in range(n_steps):

            # --- RL: sample probes and compute control --- #
            omega_act = get_omega(t, dt, p_sample)
            omega_list.append((t, omega_act))
            U_act = radius * omega_act  # tangential wall speed

            # --- update control history buffer --- #
            U_hist = np.roll(U_hist, -1, axis=1)
            U_hist[:, -1] = U_act

            # --- DMDc one-step --- #
            if step % n_shuf == 0:
                dmdc_model = random.choice(model_lib)

            x_next = dmdcUtil.dmdc_step(dmdc_model, x_roll, U_hist, common_Ur=common_Ur)

            # update state history (sliding window)
            x_roll = np.roll(x_roll, -1, axis=1)
            x_roll[:, -1:] = x_next

            # state for this step (just x_next)
            t += dt
            p_sample = Probes.get_pressure_interpolation(probe_locations, x_next[:n_points, -1], points_xy)
            p_history.append(p_sample)

            X_pred.append(x_next)
        # --- build fields dict from state --- #
        
        p_history = np.vstack(p_history)  # shape: (n_steps, 1 + n_probes)
        np.savetxt("p_samples.csv", p_history, delimiter=",")
        X_pred = np.concatenate(X_pred, axis=1)
        print(f"X_PRED = {X_pred.shape}") 
        print(f"t = {t}")       
        fields_dict: Dict[str, np.ndarray] = {}
        for f_idx, f_name in enumerate(field_names):
            fields_dict[f_name] = get_field_block(X_pred, f_idx)

        # --- upload fields to SmartRedis --- #
        smartSimUtil.upload_fields_to_database(
            client=client,
            fields_dict=fields_dict,
            field_names=field_names,
            svd_rank=svd_rank,
            mpi_ranks=mpi_ranks,
            time_stride=time_stride,
        )

        # --- update OpenFOAM dictionaries with time and omega --- #
        with open("omega.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "omega"])
            writer.writerows(omega_list)

        # --- reconstruct fields into OpenFOAM via svdToFoam --- #
        smartSimExecution.reconstruct_fields_with_svdtofoam(
            exp=exp,
            svd_rank=svd_rank,
            fo_name=fo_name,
            mpi_ranks=mpi_ranks,
            fields=["p", "U"],
            path = cwd,
        )

        # --- run pimpleFoam postProcess for forces --- #
        smartSimExecution.run_forces_postprocess(exp, mpi_ranks, path=cwd)

        for f in glob.glob("postProcessing/forces/*/coefficient.dat"):
            remove_first_data_line(f)

        # Remove first line in probe files (e.g. p)
        for f in glob.glob("postProcessing/probes/*/p"):
            remove_first_data_line(f)

        # 1) Probes: postProcessing/probes/<time>/p  ->  postProcessing/probes/4/p
    finally:
        # Make sure DB is stopped even if something throws
        exp.stop(db)