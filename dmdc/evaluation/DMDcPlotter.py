import numpy as np
import matplotlib.pyplot as plt
import csv
class DMDcPlotter:
    def __init__(self, datasets):
        if not datasets:
            raise ValueError("`datasets` list is empty.")
        self.datasets = datasets

    # ---------- Public API ----------
    def plot_errors(self, zoom, show = True, save = False):
        self._plot_errors(zoom, show, save)

    def plot_modal_summary(self, n, show = True, save = False):
        """
        One-shot modal suite:
          1) Precompute modal data once via ds.eval_data_for_plot(n) for each dataset.
          2) Plot integral contribution vs frequency
          3) Plot eigenvalues (real vs |imag|)
          4) Plot mode shapes per field (no_modes = n)
        """
        # 1) PREP ONCE
        self.n_datapoints = n
        for ds in self.datasets:
            if not hasattr(ds, "dmdc"):
                raise AttributeError(
                    f"Dataset {ds} is missing `dmdc`. Fit it before plotting."
                )
            ds.eval_data_for_plot(n)  # <-- ONE CALL ONLY

        # 2–4) PLOTS (reuse cached attributes)
        self._plot_integral_contribution_vs_frequency(show, save)
        self._plot_eigenvalues(show, save)
        self._plot_modes_by_field(show, save)

    # ---------- Internals ----------
    def _plot_errors(self, zoom, show, save):
        labels = [str(ds.reference_dic.get("signal", "unknown")) for ds in self.datasets]
        max_len = max(len(ds.err_reconstruction) for ds in self.datasets)
        ranks_full = list(range(1, max_len + 1))
        start_zoom, end_zoom = zoom
        #ranks_zoom = list(range(start_zoom, max_len + 1))
        z0 = start_zoom - 1
        zend = end_zoom -1
        # Reconstruction
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_xlabel("rank"); ax[0].set_ylabel(r"$L_{F}$")
        ax[0].set_title("Reconstruction error with rank")
        for lbl, ds in zip(labels, self.datasets):
            ax[0].plot(range(1, len(ds.err_reconstruction)+1), ds.err_reconstruction, marker='x', label=lbl)
        ax[0].legend(); ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel("rank"); ax[1].set_ylabel(r"$L_{F}$")
        ax[1].set_title("Reconstruction error with rank - Zoomed")
        for lbl, ds in zip(labels, self.datasets):
            if len(ds.err_reconstruction) >= start_zoom:
                ranks_zoom = list(range(start_zoom, end_zoom+1))
                ax[1].plot(ranks_zoom, ds.err_reconstruction[z0:zend+1], marker='x', label=lbl)
        ax[1].legend(); ax[1].grid(True, alpha=0.3)

        fig.suptitle("Reconstruction errors for signals")
        if save:
            fig.savefig("reconstruction_error.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        plt.close(fig)

        # Prediction
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_xlabel("rank"); ax[0].set_ylabel(r"$L_{L}$")
        ax[0].set_title("Prediction error with rank")
        for lbl, ds in zip(labels, self.datasets):
            ax[0].plot(range(1, len(ds.err_prediction)+1), ds.err_prediction, marker='x', label=lbl)
        ax[0].legend(); ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel("rank"); ax[1].set_ylabel(r"$L_{L}$")
        ax[1].set_title("Prediction error with rank - Zoomed")
        for lbl, ds in zip(labels, self.datasets):
            if len(ds.err_prediction) >= start_zoom:
                ranks_zoom = list(range(start_zoom, end_zoom+1))
                ax[1].plot(ranks_zoom, ds.err_prediction[z0:zend+1], marker='x', label=lbl)
        ax[1].legend(); ax[1].grid(True, alpha=0.3)

        fig.suptitle("Prediction errors for signals")
        if save:
            fig.savefig("prediction_error.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def _plot_integral_contribution_vs_frequency(self, show, save):
        fig = plt.figure(figsize=(10, 6))
        for ds in self.datasets:
            # Reuse precomputed values from eval_data_for_plot(...)
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
            plt.savefig("integral_contribution_vs_frequency.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_eigenvalues(self, show, save):
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        unit_circle = plt.Circle((0.0, 0.0), 1.0, color="green", fill=False, linestyle="--", label="Unit circle")
        ax.add_artist(unit_circle)

        for ds in self.datasets:
            # Reuse precomputed eig parts
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
        field_names = self.datasets[0].field_names
        ncols = len(self.datasets)
        no_modes = self.n_datapoints
        def get_field_block(X, ds, field_idx):
            n = ds.n_points
            r0 = field_idx * n
            r1 = (field_idx + 1) * n
            return X[r0:r1, :]  # (n_points, K)

        for f_idx, f_name in enumerate(field_names):
            # gather across datasets for consistent color scale
            field_modes_real = []
            field_freqs = []
            for ds in self.datasets:
                modes_field = get_field_block(ds.selected_modes, ds, f_idx)
                field_modes_real.append(modes_field.real)
                field_freqs.append(np.asarray(ds.frequency_DMDc, dtype=float))

            vmin = np.nanmin([m.min() for m in field_modes_real])
            vmax = np.nanmax([m.max() for m in field_modes_real])
            levels = np.linspace(vmin, vmax, 100)

            fig, ax = plt.subplots(nrows=no_modes, ncols=ncols, figsize=(10*ncols, 2.5*no_modes))
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
                        ax[no_modes-1-i, j].axis('off')
                        continue

                    mode_vec = modes_field[:, i]
                    pc = ax[i, j].tricontourf(
                        ds.x, ds.y, mode_vec, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap
                    )
                    ax[i, j].set_aspect('equal')

                    # Cylinder overlay from dataset
                    cyl_center = ds.reference_dic.get("cyl_centre", (0.0, 0.0))
                    cyl_radius = ds.reference_dic.get("cyl_radius", 0.0)
                    if cyl_radius and cyl_radius > 0:
                        ax[i, j].add_patch(
                            plt.Circle(cyl_center, cyl_radius, color='black', lw=1, fc='w')
                        )

                    label = str(ds.reference_dic.get("signal", "dataset"))
                    freq_txt = f"{freq_j[i]:.2f}" if i < len(freq_j) else "NA"
                    ax[i, j].set_title(f"{label} - f = {freq_txt}")

            #cbar = fig.colorbar(pc, ax=ax.ravel().tolist(), shrink=0.9, fraction = 0.03)
            #cbar.set_label(f_name)
            #fmt = ScalarFormatter(useMathText=True)
            #fmt.set_powerlimits((-2, 2))   # control when scientific notation kicks in
            #cbar.ax.yaxis.set_major_formatter(fmt)
            if save:
                fig.savefig(f"modes_{f_name}.png", dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig)


    def cross_prediction_table(self, save_csv=True):
        """
        Build an N x N table of cross-prediction errors across datasets.
        Row i uses dataset i's DMDc model to predict flows of dataset j (column),
        using dataset j's control signal and initial state. Error is
        ||X_true - X_pred||_F / ||X_true||_F computed over the full time window.

        Assumptions:
        - All datasets share the same feature layout and geometry ordering
        (so projection between bases is meaningful).
        - Each dataset has: state_matrix (features x T), signal_matrix (m x (T-1))
        - Each dataset has ds.dmdc (if not fit, this method fits it on train split if available, else full).
        - If available, uses ds.times_used to set dt; otherwise dt=1.0.

        Returns
        -------
        errors : np.ndarray, shape (N, N)
            errors[i, j] = relative error when model from dataset i predicts dataset j.
        row_labels : list[str]
            Model/source labels (signals).
        col_labels : list[str]
            Target/forecast-on labels (signals).
        """
        if not hasattr(self, "datasets") or not self.datasets:
            raise ValueError("No datasets provided to DMDcPlotter.")

        D = self.datasets
        N = len(D)
        labels = [str(ds.reference_dic.get("signal", f"dataset_{k}")) for k, ds in enumerate(D)]
        errors = np.full((N, N), np.nan, dtype=float)

        # Ensure each ds has a FITTED model and dt set
        for ds in D:
            if not hasattr(ds, "dmdc"):
                raise AttributeError(f"{ds} is missing `dmdc`. Create with ds.makeDMDcModel(...) first.")


                # Core cross-prediction loop
        for i, src in enumerate(D):
            for j, tgt in enumerate(D):
                X_true = tgt.state_matrix              # (features x Tj)
                U_tgt  = tgt.signal_matrix             # (m x (Tj-1))
                x0     = X_true[:, 0]                  # original-space initial column

                X_pred = src.time_march_dmdc(src.dmdc, U_tgt, x0)   # (features x Tj)

                num = np.linalg.norm(X_true - X_pred, ord="fro")
                den = np.linalg.norm(X_true, ord="fro")
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

        return errors