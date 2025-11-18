import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.animation import FuncAnimation


class DMDcPlotter:
    """Utilities to visualize DMDc errors, spectra, and modes across datasets.

    Expects each dataset object to expose attributes produced by your
    :class:`DMDcDataset`, including:

    - ``reference_dic`` (dict): contains ``"signal"`` label used for legends.
    - ``err_reconstruction`` / ``err_prediction`` (list[float]): errors vs rank.
    - ``field_names`` (list[str]): stacked field order (e.g., ``["p","u_x","u_y"]``).
    - ``x``, ``y`` (:class:`numpy.ndarray`): vertex coordinates (size = n_points).
    - ``n_points`` (int): number of spatial points.
    - After calling :meth:`DMDcDataset.eval_data_for_plot`,
      the following must be available per dataset:
      ``integral_contribution``, ``frequency_DMDc``, ``eigs_real_part``,
      ``eigs_imag_part``, and ``selected_modes``.

    :param datasets: List of dataset objects to compare/plot.
    :type datasets: list
    :raises ValueError: If ``datasets`` is empty.
    """

    def __init__(self, datasets):
        """Constructor."""
        if not datasets:
            raise ValueError("`datasets` list is empty.")
        self.datasets = datasets

    def plot_errors(self, zoom, show=True, save=False):
        """Public wrapper to plot reconstruction/prediction errors.

        Delegates to :meth:`_plot_errors`.

        :param zoom: Tuple ``(start_rank, end_rank)`` for zoomed inset.
        :type zoom: tuple[int, int]
        :param show: If ``True``, call :func:`matplotlib.pyplot.show`.
        :type show: bool, optional
        :param save: If ``True``, save figures
            ``reconstruction_error.png`` and ``prediction_error.png``.
        :type save: bool, optional
        """
        self._plot_errors(zoom, show, save)

    def plot_modal_summary(self, n, show=True, save=False):
        """Plot modal summary using precomputed quantities from each dataset.

        Steps (once per dataset):
          1. Calls ``ds.eval_data_for_plot(n)`` to precompute modal info.
          2. Plots integral contribution vs frequency across datasets.
          3. Plots eigenvalues (``Re(λ)`` vs ``|Im(λ)|``) with unit circle.
          4. Plots real parts of mode shapes for each field and dataset.

        :param n: Number of top modes to evaluate/plot per dataset.
        :type n: int
        :param show: If ``True``, display the figures.
        :type show: bool, optional
        :param save: If ``True``, save figures with standard filenames.
        :type save: bool, optional

        :raises AttributeError: If a dataset is missing ``dmdc`` (i.e., not fitted).
        """
        # Prepare data for each dataset using eval_data_for plot() member
        self.n_datapoints = n
        for ds in self.datasets:
            if not hasattr(ds, "dmdc"):
                raise AttributeError(
                    f"Dataset {ds} is missing `dmdc`. Fit it before plotting."
                )
            ds.eval_data_for_plot(n)  # <-- ONE CALL ONLY

        # Plot modal summary using evaluated quantities
        self._plot_integral_contribution_vs_frequency(show, save)
        self._plot_eigenvalues(show, save)
        self._plot_modes_by_field(show, save)

    def _plot_errors(self, zoom, show, save):
        """Internal: plot reconstruction and prediction errors vs rank.

        Produces two 1 x 2 panels:
          - Reconstruction error (full range and zoomed)
          - Prediction error (full range and zoomed)

        :param zoom: Tuple ``(start_rank, end_rank)`` for zoomed plots (1-based).
        :type zoom: tuple[int, int]
        :param show: If ``True``, display figures.
        :type show: bool
        :param save: If ``True``, save to
            ``reconstruction_error.png`` and ``prediction_error.png``.
        :type save: bool
        """
        labels = [
            str(ds.reference_dic.get("signal", "unknown")) for ds in self.datasets
        ]
        start_zoom, end_zoom = zoom
        zoom_dic = {}
        for index, ds in enumerate(self.datasets):        
        
            zoomed_rank = []
            zoomed_recon = []
            zoomed_pred = []

            for ind,rank_zoom in enumerate(ds.ranks):
                if rank_zoom >= start_zoom and rank_zoom <= end_zoom:
                    zoomed_rank.append(rank_zoom)
                    zoomed_recon.append(ds.err_reconstruction[ind])
                    zoomed_pred.append(ds.err_prediction[ind])
                zoom_dic[index] = [zoomed_rank, zoomed_recon, zoomed_pred] 

        # Reconstruction
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_xlabel("rank")
        ax[0].set_ylabel(r"$L_{F}$")
        ax[0].set_title("Reconstruction error with rank")
        for lbl, ds in zip(labels, self.datasets):
            ax[0].plot(
                ds.ranks,
                ds.err_reconstruction,
                marker="x",
                label=lbl,
            )
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel("rank")
        ax[1].set_ylabel(r"$L_{F}$")
        ax[1].set_title("Reconstruction error with rank - Zoomed")
        for ind, (lbl, ds) in enumerate(zip(labels, self.datasets)):
            ax[1].plot(
                zoom_dic[ind][0],
                zoom_dic[ind][1],
                marker="x",
                label=lbl,
            )
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        fig.suptitle("Reconstruction errors for signals")
        if save:
            fig.savefig("reconstruction_error.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        plt.close(fig)

        # Prediction
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_xlabel("rank")
        ax[0].set_ylabel(r"$L_{L}$")
        ax[0].set_title("Prediction error with rank")
        for lbl, ds in zip(labels, self.datasets):
            ax[0].plot(
                ds.ranks,
                ds.err_prediction,
                marker="x",
                label=lbl,
            )
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel("rank")
        ax[1].set_ylabel(r"$L_{L}$")
        ax[1].set_title("Prediction error with rank - Zoomed")
        for ind, (lbl, ds) in enumerate(zip(labels, self.datasets)):
            ax[1].plot(
                zoom_dic[ind][0], zoom_dic[ind][2], marker="x", label=lbl
            )
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        fig.suptitle("Prediction errors for signals")
        if save:
            fig.savefig("prediction_error.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_integral_contribution_vs_frequency(self, show, save):
        """Internal: scatter plot of integral contribution vs modal frequency.

        Assumes :meth:`DMDcDataset.eval_data_for_plot` has filled
        ``frequency_DMDc`` and ``integral_contribution`` for each dataset.

        :param show: If ``True``, display the figure.
        :type show: bool
        :param save: If ``True``, save to
            ``integral_contribution_vs_frequency.png``.
        :type save: bool
        """
        fig = plt.figure(figsize=(10, 6))
        for ds in self.datasets:
            # Uuse precomputed values from eval_data_for_plot()
            freq = np.asarray(ds.frequency_DMDc, dtype=float)
            contrib = np.asarray(ds.integral_contribution, dtype=float)
            label = str(ds.reference_dic.get("signal", "dataset"))
            plt.scatter(freq, contrib, label=label)

        plt.xlabel("frequency")
        plt.ylabel("Integral contribution")
        plt.title("Comparison of Integral contribution v/s frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save:
            plt.savefig(
                "integral_contribution_vs_frequency.png", dpi=300, bbox_inches="tight"
            )
        if show:
            plt.show()
        plt.close(fig)

    def _plot_eigenvalues(self, show, save):
        """Internal: plot eigenvalues (real vs |imag|) with a unit circle.

        Assumes each dataset provides ``eigs_real_part`` and ``eigs_imag_part``
        (imaginary part magnitudes).

        :param show: If ``True``, display the figure.
        :type show: bool
        :param save: If ``True``, save to ``eigenvalues.png``.
        :type save: bool
        """
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="green",
            fill=False,
            linestyle="--",
            label="Unit circle",
        )
        ax.add_artist(unit_circle)

        for ds in self.datasets:
            # Use precomputed eig parts
            eigs_real = np.asarray(ds.eigs_real_part, dtype=float)
            eigs_imag = np.asarray(ds.eigs_imag_part, dtype=float)
            label = str(ds.reference_dic.get("signal", "dataset"))
            ax.plot(eigs_real, eigs_imag, "o", label=label)

        ax.set_xlabel("Real part")
        ax.set_ylabel("Imaginary part")
        ax.set_aspect("equal")
        ax.set_title("Eigenvalues")
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        ax.grid(True, alpha=0.3)
        if save:
            plt.savefig("eigenvalues.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_modes_by_field(self, show, save, cmap="seismic"):
        """Internal: plot real parts of mode shapes per field × dataset.

        For each field in ``datasets[0].field_names``, creates a grid with
        rows = number of modes (``self.n_datapoints``) and columns = number
        of datasets. Each panel shows a tricontour of the real part of the
        mode vector over ``(x, y)`` vertices; adds a cylinder overlay if
        ``cyl_centre``/``cyl_radius`` are provided in ``reference_dic``.

        :param show: If ``True``, display the figure.
        :type show: bool
        :param save: If ``True``, save to ``modes_<field>.png`` per field.
        :type save: bool
        :param cmap: Matplotlib colormap for contours.
        :type cmap: str, optional
        """
        field_names = self.datasets[0].field_names
        ncols = len(self.datasets)
        no_modes = self.n_datapoints

        def get_field_block(X, ds, field_idx):
            n = ds.n_points
            r0 = field_idx * n
            r1 = (field_idx + 1) * n
            return X[r0:r1, :]  # (n_points, K)

        for f_idx, f_name in enumerate(field_names):
            field_modes_real = []
            field_freqs = []
            for ds in self.datasets:
                modes_field = get_field_block(ds.selected_modes, ds, f_idx)
                field_modes_real.append(modes_field.real)
                field_freqs.append(np.asarray(ds.frequency_DMDc, dtype=float))

            vmin = np.nanmin([m.min() for m in field_modes_real])
            vmax = np.nanmax([m.max() for m in field_modes_real])
            levels = np.linspace(vmin, vmax, 100)

            fig, ax = plt.subplots(
                nrows=no_modes, ncols=ncols, figsize=(10 * ncols, 2.5 * no_modes)
            )
            if no_modes == 1:
                ax = np.array([ax])
            if ncols == 1:
                ax = ax.reshape(no_modes, 1)

            fig.suptitle(f"Mode shapes — {f_name}", fontsize=16, y=0.93)

            for i in range(no_modes):
                for j, ds in enumerate(self.datasets):
                    modes_field = field_modes_real[j]
                    freq_j = field_freqs[j]

                    if i >= modes_field.shape[1]:
                        ax[no_modes - 1 - i, j].axis("off")
                        continue

                    mode_vec = modes_field[:, i]
                    pc = ax[i, j].tricontourf(
                        ds.x,
                        ds.y,
                        mode_vec,
                        levels=levels,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                    )
                    ax[i, j].set_aspect("equal")

                    # Cylinder overlay from dataset
                    cyl_center = ds.reference_dic.get("cyl_centre", (0.0, 0.0))
                    cyl_radius = ds.reference_dic.get("cyl_radius", 0.0)
                    if cyl_radius and cyl_radius > 0:
                        ax[i, j].add_patch(
                            plt.Circle(
                                cyl_center, cyl_radius, color="black", lw=1, fc="w"
                            )
                        )

                    label = str(ds.reference_dic.get("signal", "dataset"))
                    freq_txt = f"{freq_j[i]:.2f}" if i < len(freq_j) else "NA"
                    ax[i, j].set_title(f"{label} - f = {freq_txt}")

            if save:
                fig.savefig(f"modes_{f_name}.png", dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    def cross_prediction_table(self, save_csv=False):
        """Compute an N×N cross-prediction error matrix across datasets.

        Row ``i`` uses dataset ``i``’s DMDc model to predict flows of dataset
        ``j`` (column), using dataset ``j``’s control signal and initial
        state. The error is computed as
        :math:`\\|X_{true} - X_{pred}\\|_F / \\|X_{true}\\|_F` over the full
        time window.

        Assumptions
        -----------
        - All datasets share the same feature layout and geometry ordering.
        - Each dataset has ``state_matrix`` (``features × T``) and
          ``signal_matrix`` (``m × (T-1)``).
        - Each dataset has ``ds.dmdc`` prepared (e.g., via ``makeDMDcModel``).
        - If available, ``ds.times_used`` may be used to set dt elsewhere,
          but dt is not required here.

        :param save_csv: If ``True``, write a CSV named
            ``cross_prediction_errors.csv`` with labels from
            ``ds.reference_dic["signal"]``.
        :type save_csv: bool, optional
        :returns: The error matrix ``errors`` shaped ``(N, N)``.
        :rtype: numpy.ndarray
        :raises ValueError: If the plotter has no datasets.
        :raises AttributeError: If any dataset is missing ``dmdc``.
        """
        if not hasattr(self, "datasets") or not self.datasets:
            raise ValueError("No datasets provided to DMDcPlotter.")

        D = self.datasets
        N = len(D)
        labels = [
            str(ds.reference_dic.get("signal", f"dataset_{k}"))
            for k, ds in enumerate(D)
        ]
        errors = np.full((N, N), np.nan, dtype=float)

        # Ensure each ds has a FITTED model and dt set
        for ds in D:
            if not hasattr(ds, "dmdc"):
                raise AttributeError(
                    f"{ds} is missing `dmdc`. Create with ds.makeDMDcModel(...) first."
                )

        # Core cross-prediction table
        for i, src in enumerate(D):
            for j, tgt in enumerate(D):
                x0  = src.x_ini              # (F x q), aligned with target grid

                X_true = tgt.state_matrix  # (features x Tj)
                U_tgt = tgt.signal_matrix  # (m x (Tj-1))
                pad_cols = x0.shape[1]         # same number of columns as x0

                m, _ = U_tgt.shape
                U_tgt_ext = np.hstack([
                    np.zeros((m, pad_cols)),   # zero padding (m x pad_cols)
                    U_tgt                      # original (m x (T-1))
                ])

                q = getattr(src, "q", 1)
                #x0 = X_true[:, :q]  # original-space initial column

                X_pred = src.time_march_dmdc(src.dmdc, U_tgt_ext, x0)  # (features x Tj)

                num = np.linalg.norm(X_true[:,:] - X_pred[:,1:])
                den = np.linalg.norm(X_true[:,:])
                errors[i, j] = num / den if den > 0 else np.nan

        if save_csv:
            # Save CSV if requested
            csv_name = "cross_prediction_errors.csv"
            if save_csv:
                with open(csv_name, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Model \\ Target"] + labels)
                    for i in range(N):
                        row = [labels[i]] + [f"{errors[i, j]:.6e}" for j in range(N)]
                        writer.writerow(row)
        labels = [str(ds.reference_dic.get("signal", f"dataset_{i}")) for i, ds in enumerate(self.datasets)]

        fig, ax = plt.subplots(figsize=(1.0*len(labels) + 6, 1.0*len(labels) + 4))

        # Mask NaNs so they show as a distinct "bad" color in the default colormap
        masked = np.ma.masked_invalid(errors)
        im = ax.imshow(errors)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$L_{F}$")

        # Ticks & labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Target dataset")
        ax.set_ylabel("Model (source)")
        ax.set_title("Cross-prediction error heatmap")
        plt.show()

    def plot_error_time_series(self, src_dataset):

        if not hasattr(self, "datasets") or not self.datasets:
            raise ValueError("No datasets provided to DMDcPlotter.")
        lbl_src = str(src_dataset.reference_dic.get("signal", f"dataset_src"))
        D = self.datasets
        N = len(D)
        labels = [
            str(ds.reference_dic.get("signal", f"dataset_{k}"))
            for k, ds in enumerate(D)
        ]
        # Ensure each ds has a FITTED model and dt set
        for ds in D:
            if not hasattr(ds, "dmdc"):
                raise AttributeError(
                    f"{ds} is missing `dmdc`. Create with ds.makeDMDcModel(...) first."
                )

        series_list = []   # list of np.arrays of per-step errors
        index_list = []    # matching time indices per target

        # Core cross-prediction table
        for j, tgt in enumerate(D):
            X_true = tgt.state_matrix  # (features x Tj)
            U_tgt = tgt.signal_matrix  # (m x (Tj-1))
            q = getattr(src_dataset, "q", 1)
            x0 = X_true[:, :q]  # original-space initial column
            X_pred = src_dataset.time_march_dmdc(src_dataset.dmdc, U_tgt, x0)  # (features x Tj)
            T = X_true.shape[1]
            errs = []
            for t in range(q - 1, T):
                num = np.linalg.norm(X_true[:, t] - X_pred[:, t - (q - 1)])
                den = np.linalg.norm(X_true[:, t])
                errs.append(num / den if den > 0 else np.nan)
            series_list.append(np.asarray(errs))
            index_list.append(np.arange(q - 1, q - 1 + len(errs)))


        fig, ax = plt.subplots(figsize=(8, 4.5))
        for lbl, idx, ser in zip(labels, index_list, series_list):
            ax.plot(idx, ser, marker='.', linewidth=1.0, label=lbl)
        ax.set_xlabel("time index t")
        ax.set_ylabel("normalized per-step error")
        ax.set_title(f"Time series cross-prediction error for {lbl_src}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.show()



    def write_all_fields(self):
        data = {}
            
        for src_dataset in self.datasets:
            if not hasattr(self, "datasets") or not self.datasets:
                raise ValueError("No datasets provided to DMDcPlotter.")
            if not hasattr(src_dataset, "dmdc"):
                raise AttributeError("src_dataset is missing `dmdc`. Fit it first.")

            D = self.datasets
            labels = [str(ds.reference_dic.get("signal", f"dataset_{k}")) for k, ds in enumerate(D)]
            q = getattr(src_dataset, "q", 1)

            # helper: split stacked features to (n_points x T) block for field f_idx
            def get_field_block(X, field_idx):
                n = src_dataset.n_points
                r0 = field_idx * n
                r1 = (field_idx + 1) * n
                return X[r0:r1, :]


            # per-target animations
            pred_block = {}
            omega_act = {}
            for tgt, label in zip(D, labels):

                sig_name = tgt.reference_dic.get("signal", "")
                
                
                means_by_name = {}

                x0     = src_dataset.x_ini              # (F x q), aligned with target grid
                X_true = tgt.state_matrix          # (F x T)
                U_tgt  = tgt.signal_matrix         # (m x (T-1))

                pad_cols = x0.shape[1]         # same number of columns as x0

                m, _ = U_tgt.shape
                U_tgt_ext = np.hstack([
                    np.zeros((m, pad_cols)),   # zero padding (m x pad_cols)
                    U_tgt                      # original (m x (T-1))
                ])

                F, T   = X_true.shape


                """
                for f_name in src_dataset.field_names:
                    attr_name = f_name.replace("_", "") + "_mean"

                    if hasattr(src_dataset, attr_name):
                        means_by_name[f_name] = getattr(tgt, attr_name)-getattr(src_dataset, attr_name)
                for f_idx, f_name in enumerate(src_dataset.field_names):
                    if f_name not in means_by_name:
                        continue  # no mean stored for this field

                    mu = np.asarray(means_by_name[f_name]).reshape(-1, 1)  # (n_points, 1)
                    
                    block = get_field_block(x0, f_idx)                     # (n_points, q)
                    block += mu  
                """
                X_pred = src_dataset.time_march_dmdc(src_dataset.dmdc, U_tgt_ext, x0)  # (F x T_pred)
                # align indices: truth starts at t0 = q-1; pred index maps to t-(q-1)

                # Prepare per-field data blocks
                field_blocks = []
                pred_block[sig_name] = {}
                
                times_actual = np.asarray(tgt.times_used, dtype="float64")
                dt = times_actual[1] - times_actual[0]
                start_actual = times_actual[0]-dt
                times_actual = np.concatenate(([start_actual], times_actual))
                omega = tgt.f_omega(times_actual)
                #omega_act[sig_name] = np.column_stack((times_arr, omega)) 
                omega_act[sig_name] = np.column_stack((times_actual, omega)) 
                
                for f_idx, f_name in enumerate(src_dataset.field_names):
                    
                    """
                    if f_name == "p":
                        normalized_coeff = src_dataset.p_mean
                    elif f_name == "u_x":
                        normalized_coeff = src_dataset.ux_mean
                    elif f_name == "u_y":
                        normalized_coeff = src_dataset.uy_mean
                    elif f_name == "u_z":
                        normalized_coeff = src_dataset.uz_mean
                    
                    else:
                        normalized_coeff = 0
                    """
                    
                    #pred_block[sig_name][f_name]  = get_field_block(X_pred, f_idx)[:, :usable_T] + normalized_coeff                          # (n_pts x usable_T)
                    pred_block[sig_name][f_name]  = get_field_block(X_pred, f_idx)
            src_signal = str(src_dataset.reference_dic.get("signal", f"dataset"))
            data[src_signal] = [pred_block, omega_act]
        
        return data

    def place_sensors(self, bottom_left, n, h, dataset_idx=0, show=True, save=None):
        """
        Place an N×N grid of sensors starting at `bottom_left` with spacing `h`,
        snap each to the closest mesh node (shared by all datasets), and plot.

        Parameters
        ----------
        bottom_left : tuple[float, float]
            (x0, y0) of the lower-left corner of the sensor grid (in physical coords).
        n : int
            Number of sensors along each axis → total sensors n*n.
        h : float
            Grid spacing between sensors.
        dataset_idx : int, optional
            Which dataset to use for the (x, y) node coordinates. Default 0.
            (All your datasets share the same geometry, so any is fine.)
        show : bool, optional
            If True, calls plt.show() for the scatter figure.
        save : str | None, optional
            If a string (e.g. "sensors.png"), save the plot to this path.

        Returns
        -------
        sensor_idx : np.ndarray, shape (k,)
            Indices of the snapped sensor nodes in the mesh (unique, in row-major
            sensor-grid order; duplicates removed).
        snapped_xy : np.ndarray, shape (k, 2)
            Coordinates of the snapped sensor nodes.
        target_xy : np.ndarray, shape (n*n, 2)
            The *intended* (unsnapped) sensor coordinates in row-major order.
        """
        if not hasattr(self, "datasets") or not self.datasets:
            raise ValueError("No datasets provided to DMDcPlotter.")

        if dataset_idx < 0 or dataset_idx >= len(self.datasets):
            raise IndexError(f"dataset_idx={dataset_idx} out of range for {len(self.datasets)} datasets")

        ds0 = self.datasets[dataset_idx]
        x_nodes = np.asarray(ds0.x).ravel()
        y_nodes = np.asarray(ds0.y).ravel()
        if x_nodes.shape != y_nodes.shape:
            raise ValueError("x and y node arrays must have the same shape")
        if x_nodes.ndim != 1:
            raise ValueError("x and y must be 1D arrays of node coordinates")

        mesh_xy = np.column_stack([x_nodes, y_nodes])  # (N_nodes, 2)

        # Build the regular sensor grid (row-major: y fastest or x fastest? → conventional x fastest)
        x0, y0 = bottom_left
        xs = x0 + h * np.arange(n, dtype=float)
        ys = y0 + h * np.arange(n, dtype=float)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")  # shape (n, n)
        target_xy = np.column_stack([XX.ravel(), YY.ravel()])  # (n*n, 2)

        # Snap each sensor to nearest mesh node (vectorized)
        # diffs: (n*n, N_nodes, 2)
        diffs = mesh_xy[None, :, :] - target_xy[:, None, :]
        dists = np.hypot(diffs[..., 0], diffs[..., 1])          # (n*n, N_nodes)
        nearest = np.argmin(dists, axis=1)                      # (n*n,)

        # De-duplicate while preserving first occurrence order (so row-major grid order is respected)
        seen = set()
        sensor_idx = []
        for idx in nearest.tolist():
            if idx not in seen:
                sensor_idx.append(idx)
                seen.add(idx)
        sensor_idx = np.asarray(sensor_idx, dtype=int)

        snapped_xy = mesh_xy[sensor_idx, :]

        # Plot: all nodes tiny, sensors as big 'x'
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.scatter(x_nodes, y_nodes, s=1, alpha=0.6)  # very small markers for the full grid
        ax.scatter(snapped_xy[:, 0], snapped_xy[:, 1], marker='x', s=60, linewidths=1.5)  # sensors
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Sensor placement (n={n}, h={h})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.2)

        if save:
            fig.savefig(save, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        # Optionally store on the instance for later use
        self.sensor_indices = sensor_idx


    def cross_prediction_table_at_sensors(self, save_csv=False):
        """Compute an N×N cross-prediction error matrix across datasets.

        Row ``i`` uses dataset ``i``’s DMDc model to predict flows of dataset
        ``j`` (column), using dataset ``j``’s control signal and initial
        state. The error is computed as
        :math:`\\|X_{true} - X_{pred}\\|_F / \\|X_{true}\\|_F` over the full
        time window.

        Assumptions
        -----------
        - All datasets share the same feature layout and geometry ordering.
        - Each dataset has ``state_matrix`` (``features × T``) and
          ``signal_matrix`` (``m × (T-1)``).
        - Each dataset has ``ds.dmdc`` prepared (e.g., via ``makeDMDcModel``).
        - If available, ``ds.times_used`` may be used to set dt elsewhere,
          but dt is not required here.

        :param save_csv: If ``True``, write a CSV named
            ``cross_prediction_errors.csv`` with labels from
            ``ds.reference_dic["signal"]``.
        :type save_csv: bool, optional
        :returns: The error matrix ``errors`` shaped ``(N, N)``.
        :rtype: numpy.ndarray
        :raises ValueError: If the plotter has no datasets.
        :raises AttributeError: If any dataset is missing ``dmdc``.
        """
        if not hasattr(self, "datasets") or not self.datasets:
            raise ValueError("No datasets provided to DMDcPlotter.")
        if not hasattr(self, "sensor_indices"):
            raise AttributeError(
                "Sensor indices not set. Run place_sensors(bottom_left, n, h, ...) first."
            )
        D = self.datasets
        N = len(D)
        labels = [
            str(ds.reference_dic.get("signal", f"dataset_{k}"))
            for k, ds in enumerate(D)
        ]
        errors = np.full((N, N), np.nan, dtype=float)

        # Ensure each ds has a FITTED model and dt set
        for ds in D:
            if not hasattr(ds, "dmdc"):
                raise AttributeError(
                    f"{ds} is missing `dmdc`. Create with ds.makeDMDcModel(...) first."
                )

        # Core cross-prediction table
        for i, src in enumerate(D):
            for j, tgt in enumerate(D):
                X_true = tgt.state_matrix  # (features x Tj)
                U_tgt = tgt.signal_matrix  # (m x (Tj-1))
                q = getattr(src, "q", 1)
                x0 = X_true[:, :q]  # original-space initial column
                X_pred = src.time_march_dmdc(src.dmdc, U_tgt, x0)  # (features x Tj)

                X_true_sensor = X_true[self.sensor_indices, :]
                X_pred_sensor = X_pred[self.sensor_indices, :]
                num = np.linalg.norm(X_true_sensor[:, q-1:] - X_pred_sensor)
                den = np.linalg.norm(X_true_sensor[:, q-1:])
                errors[i, j] = num / den if den > 0 else np.nan

        if save_csv:
            # Save CSV if requested
            csv_name = "cross_prediction_errors.csv"
            if save_csv:
                with open(csv_name, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Model \\ Target"] + labels)
                    for i in range(N):
                        row = [labels[i]] + [f"{errors[i, j]:.6e}" for j in range(N)]
                        writer.writerow(row)
        labels = [str(ds.reference_dic.get("signal", f"dataset_{i}")) for i, ds in enumerate(self.datasets)]

        fig, ax = plt.subplots(figsize=(1.0*len(labels) + 6, 1.0*len(labels) + 4))

        # Mask NaNs so they show as a distinct "bad" color in the default colormap
        masked = np.ma.masked_invalid(errors)
        im = ax.imshow(errors)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$L_{F}$")

        # Ticks & labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Target dataset")
        ax.set_ylabel("Model (source)")
        ax.set_title("Cross-prediction error heatmap")
        plt.show()
