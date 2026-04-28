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

    if len(sys.argv) < 3:
        prog = Path(sys.argv[0]).name
        print(f"usage: {prog} <model_path>", file=sys.stderr)
        sys.exit(2)  # conventional "bad args" exit code
    
    

    base_path = sys.argv[1]
    orig_path = sys.argv[2]
    tgt_signal = sys.argv[3]

    folders_model = os.listdir(base_path)
    folders_model.remove("orchestrator")
    folders_model.remove(".smartsim")
    
    data = {}
    folders_orig = os.listdir(orig_path)
    orig_data = {}
    for ind, folder_model in enumerate(folders_model):
        
        model_path = os.path.join(base_path, folder_model)
        tgt_folder = tgt_signal
        #folders = os.listdir(model_path)
        series = {}

        #for folder in folders:
        lis = os.path.join(model_path, tgt_folder, "postProcessing", "forces_recon")
        time_name = os.listdir(lis)[0]
        fpath = os.path.join(model_path, tgt_folder, "postProcessing", "forces_recon", time_name, "coefficient.dat")
        if not Path(fpath).exists():
            raise FileNotFoundError(f"Missing input file: {fpath}")

        t, cd, cl = read_coeffs(fpath)
        series[tgt_folder] = {"t": t[1:], "Cd": cd[1:], "Cl": cl[1:]}

        data[folder_model] = series

        
        #t_final = np.max(series[folder]["t"])
        
        t_ini = np.min(series[tgt_folder]["t"])
        orig_fpath = os.path.join(orig_path, tgt_folder, "postProcessing", "forces", "0", "coefficient.dat")

        t, cd, cl = read_coeffs(orig_fpath)
        t_orig_final = np.max(t)
        idx = np.flatnonzero(t > t_ini) 
        t_new = t[idx] 
        Cd_new = cd[idx] 
        Cl_new = cl[idx] 
        orig_data[tgt_folder] = {"t": t_new, "Cd": Cd_new, "Cl": Cl_new}


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

    colors = {
        "Original": "#9467bd",
        "AM": "#1f77b4",
        "Chirp": "#ff7f0e",
        "Chirp with varying amplitude": "#2ca02c",
        "Random walk": "#d62728",
    }
    for signal_key, series_by_outer in grouped_by_signal.items():
        if not series_by_outer:  # nothing to plot
            continue


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

            if outer_key == "Chirp_with_varying_ampl":
                label = "Chirp with varying amplitude"
            elif outer_key == "Random_walk":
                label = "Random walk"
            else:
                label = outer_key


            fig_cd, ax_cd = plt.subplots(figsize=(6, 6), dpi= 160)
            fig_cl, ax_cl = plt.subplots(figsize=(6, 6), dpi = 160)

            ax_cd.plot(t*10, Cd, label=label, linestyle = "--", color = colors[label])
            ax_cl.plot(t*10, Cl, label=label, linestyle = "--", color = colors[label])
            ax_cd.plot(orig_data[signal_key]["t"]*10, orig_data[signal_key]["Cd"], label = "Original", lw = 2.0, color = colors["Original"])
            ax_cl.plot(orig_data[signal_key]["t"]*10, orig_data[signal_key]["Cl"], label = "Original", lw = 2.0, color = colors["Original"])
            ax_cd.set_xlabel(r"$\tilde{t}$", fontsize = 20)
            ax_cd.set_ylabel(r"$C_d$", fontsize = 20)
            #ax_cd.set_title(f"Drag Coefficient (Cd)")
            ax_cd.grid(True, alpha=0.3)
            ax_cd.legend(title="Model", fontsize = 15)

            ax_cl.set_xlabel(r"$\tilde{t}$", fontsize = 20)
            ax_cl.set_ylabel(r"$C_l$", fontsize = 20)
            #ax_cl.set_title(f"Lift Coefficient (Cl)")
            ax_cl.grid(True, alpha=0.3)
            ax_cl.set_ylim((-2.3, 2.3))
            ax_cl.legend(title="Model", fontsize = 14, title_fontsize=15)

            ax_cl.tick_params(axis="both", labelsize=18)
            fig_cd_name = f"{signal_key}__{outer_key}_cd.png"
            fig_cl_name = f"{signal_key}_{outer_key}_cl.png"

            fig_cd.savefig(fig_cd_name, bbox_inches="tight")
            fig_cl.savefig(fig_cl_name, bbox_inches="tight")
            print(f"Saved figure to: {fig_cd_name}")
            plt.close(fig_cl)
            plt.close(fig_cd)