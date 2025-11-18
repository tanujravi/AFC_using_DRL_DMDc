#!/usr/bin/env python3
"""
Plot Cd and Cl from OpenFOAM-style force coefficient files.

- Expects whitespace-delimited .dat files with '#' comments.
- Defaults assume columns: time=0, Cd=4, Cl=5 (typical OpenFOAM forceCoeffs).
- Produces a single figure with 2 columns of subplots (Cd on left, Cl on right).

Usage:
    python plot_coeffs.py \
        --orig coefficient_orig.dat \
        --random_walk coefficient_src_Random_walk.dat \
        --chirp coefficient_src_Chirp.dat \
        --am coefficient_src_AM.dat \
        --out Cd_Cl_subplots.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os

def read_coeffs(path):
    # Read whitespace-separated columns (0=time, 4=Cd, 5=Cl) and ignore comments.
    df = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",          # <-- replaces deprecated delim_whitespace=True
        engine="python",     # regex separator needs the python engine
        header=None,
        usecols=[0, 1, 4],
        dtype={0: float, 1: float, 4: float},
    )
    df.columns = ["time", "cd", "cl"]
    df = df.sort_values("time", kind="stable").reset_index(drop=True)

    t  = df["time"].to_numpy()
    cd = df["cd"].to_numpy()
    cl = df["cl"].to_numpy()
    return t, cd, cl

if __name__ == "__main__":

    if len(sys.argv) < 2:
        prog = Path(sys.argv[0]).name
        print(f"usage: {prog} <model_path>", file=sys.stderr)
        sys.exit(2)  # conventional "bad args" exit code
    


    base_path = sys.argv[1]
    orig_path = sys.argv[2]
    
    folders_model = os.listdir(base_path)
    folders_model.remove("orchestrator")
    folders_model.remove(".smartsim")
    
    data = {}
    folders_orig = os.listdir(orig_path)
    orig_data = {}
    for ind, folder_model in enumerate(folders_model):
        
        model_path = os.path.join(base_path, folder_model)
        folders = os.listdir(model_path)
        series = {}

        for folder in folders:
            lis = os.path.join(model_path, folder, "postProcessing", "forces_recon")
            time_name = os.listdir(lis)[0]
            fpath = os.path.join(model_path, folder, "postProcessing", "forces_recon", time_name, "coefficient.dat")
            if not Path(fpath).exists():
                raise FileNotFoundError(f"Missing input file: {fpath}")

            t, cd, cl = read_coeffs(fpath)
            series[folder] = {"t": t[1:], "Cd": cd[1:], "Cl": cl[1:]}

        data[folder_model] = series

        
        #t_final = np.max(series[folder]["t"])
        
        t_ini = np.min(series[folder]["t"])
        orig_fpath = os.path.join(orig_path, folder_model, "postProcessing", "forces", "0", "coefficient.dat")

        t, cd, cl = read_coeffs(orig_fpath)
        t_orig_final = np.max(t)
        idx = np.flatnonzero(t > t_ini) 
        t_new = t[idx] 
        Cd_new = cd[idx] 
        Cl_new = cl[idx] 
        orig_data[folder_model] = {"t": t_new, "Cd": Cd_new, "Cl": Cl_new}


        #ax_cd.plot(t_new, Cd_new, label=label)
        #ax_cl.plot(t_new, Cl_new, label=label) 

    from collections import defaultdict

    # `data` is already built like: data[outer_key][signal_key] = {"t","Cd","Cl"}

    # 1) Invert/pivot: grouped_by_signal["AM"]["Random_walk"] -> series dict
    grouped_by_signal = defaultdict(dict)
    for outer_key, inner_dict in data.items():
        for signal_key, series in inner_dict.items():
            grouped_by_signal[signal_key][outer_key] = series

    # 2) Plot each signal across all outer groups
    for signal_key, series_by_outer in grouped_by_signal.items():
        if not series_by_outer:  # nothing to plot
            continue

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
        ax_cd, ax_cl = axes

        # consistent ordering of legend entries
        for outer_key in sorted(series_by_outer.keys()):
            series = series_by_outer[outer_key]
            t  = series["t"]
            Cd = series["Cd"]
            Cl = series["Cl"]

            # If you ever want the "orig" time shift, uncomment this block:
            # if outer_key == "orig":
            #     idx = np.flatnonzero(t > 5)
            #     if idx.size:
            #         t = t[idx] - t[idx][0]
            #         Cd = Cd[idx]
            #         Cl = Cl[idx]

            ax_cd.plot(t, Cd, label=outer_key)
            ax_cl.plot(t, Cl, label=outer_key)

        ax_cd.plot(orig_data[signal_key]["t"], orig_data[signal_key]["Cd"], label = "orig")
        ax_cl.plot(orig_data[signal_key]["t"], orig_data[signal_key]["Cl"], label = "orig")
        print(orig_data[signal_key]["Cd"])

        ax_cd.set_xlabel("Time")
        ax_cd.set_ylabel("Cd")
        ax_cd.set_title(f"Drag Coefficient (Cd)")
        ax_cd.grid(True, alpha=0.3)
        ax_cd.legend(title="Model")

        ax_cl.set_xlabel("Time")
        ax_cl.set_ylabel("Cl")
        ax_cl.set_title(f"Lift Coefficient (Cl)")
        ax_cl.grid(True, alpha=0.3)
        ax_cl.legend(title="Model")
        #ax_cd.set_ylim((3, 3.5))
        fig.suptitle(f"Force Coefficients — {signal_key}")
        fig_name = f"{signal_key}.png"
        fig.savefig(fig_name, dpi=200)
        print(f"Saved figure to: {fig_name}")