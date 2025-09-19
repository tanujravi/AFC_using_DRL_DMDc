import numpy as np
import matplotlib.pyplot as plt
import csv


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
        max_len = max(len(ds.err_reconstruction) for ds in self.datasets)
        ranks_full = list(range(1, max_len + 1))
        start_zoom, end_zoom = zoom
        z0 = start_zoom - 1
        zend = end_zoom - 1
        # Reconstruction
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_xlabel("rank")
        ax[0].set_ylabel(r"$L_{F}$")
        ax[0].set_title("Reconstruction error with rank")
        for lbl, ds in zip(labels, self.datasets):
            ax[0].plot(
                range(1, len(ds.err_reconstruction) + 1),
                ds.err_reconstruction,
                marker="x",
                label=lbl,
            )
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel("rank")
        ax[1].set_ylabel(r"$L_{F}$")
        ax[1].set_title("Reconstruction error with rank - Zoomed")
        for lbl, ds in zip(labels, self.datasets):
            if len(ds.err_reconstruction) >= start_zoom:
                ranks_zoom = list(range(start_zoom, end_zoom + 1))
                ax[1].plot(
                    ranks_zoom,
                    ds.err_reconstruction[z0 : zend + 1],
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
                range(1, len(ds.err_prediction) + 1),
                ds.err_prediction,
                marker="x",
                label=lbl,
            )
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].set_xlabel("rank")
        ax[1].set_ylabel(r"$L_{L}$")
        ax[1].set_title("Prediction error with rank - Zoomed")
        for lbl, ds in zip(labels, self.datasets):
            if len(ds.err_prediction) >= start_zoom:
                ranks_zoom = list(range(start_zoom, end_zoom + 1))
                ax[1].plot(
                    ranks_zoom, ds.err_prediction[z0 : zend + 1], marker="x", label=lbl
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

    def cross_prediction_table(self, save_csv=True):
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
                X_true = tgt.state_matrix  # (features x Tj)
                U_tgt = tgt.signal_matrix  # (m x (Tj-1))
                x0 = X_true[:, 0]  # original-space initial column

                X_pred = src.time_march_dmdc(src.dmdc, U_tgt, x0)  # (features x Tj)

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
