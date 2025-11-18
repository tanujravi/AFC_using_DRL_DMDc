import numpy as np
from pydmd import DMDc
from flowtorch.data import FOAMDataloader
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pydmd.dmd_modes_tuner import ModesSelectors


class DMDcDataset:
    """Build and analyze a DMDc model from OpenFOAM snapshots.

    Loads pressure/velocity fields and a control signal (e.g., cylinder
    rotation) from an OpenFOAM case, constructs normalized state/inputs,
    trains DMDc variants, and provides utilities for rollout, rank
    selection, and qualitative plots.

    :param folder_path: Path to the OpenFOAM case directory.
    :type folder_path: str
    :param reference_dic: Dictionary of reference quantities and geometry, e.g.
        ``{"u_inlet": float, "cyl_radius": float, "cyl_centre": (x, y), "signal": "name"}``.
        ``"u_inlet"`` is required. ``"cyl_radius"``/``"cyl_centre"`` are used for plots and
        for scaling the control from angular speed to tangential speed.
    :type reference_dic: dict
    :param start_time: First write time (inclusive) to consider when building matrices.
    :type start_time: float, optional
    :param dim: Dimensionality of velocity; if ``"3d"``, include ``u_z``.
    :type dim: str, optional

    **Main attributes created**
    - ``x``, ``y`` (:class:`np.ndarray`): Vertex coordinates (1D, size = n_points).
    - ``field_names`` (:class:`list[str]`): Order of stacked fields, e.g., ``["p","u_x","u_y"]``.
    - ``times_used`` (:class:`np.ndarray`): Sorted list of times used (from ``start_time``).
    - ``state_matrix`` (:class:`np.ndarray`): Shape ``(n_fields*n_points, T)``.
    - ``signal_matrix`` (:class:`np.ndarray`): Shape ``(m, T-1)``, here ``m=1``.
    """

    def __init__(self, folder_path, reference_dic, start_time=4.0, dim="2d", verbose = True):
        """Constructor method."""
        self.folder_path = folder_path
        self.reference_dic = reference_dic
        self.start_time = start_time
        self.dim = dim
        self.verbose = verbose
        self.build_state_signal_matrix()

    def build_state_signal_matrix(self):
        """Construct normalized state and control matrices from the case.

        Reads mesh and field snapshots using :class:`flowtorch.data.FOAMDataloader`.
        Builds a stacked state matrix by concatenating (by rows) mean-subtracted
        pressure and velocity components, normalized by ``u_inlet`` (pressure by
        ``u_inlet**2``). The control signal is read from ``omega.csv`` and
        interpolated to align with state transitions.

        :returns: None. Populates attributes ``x``, ``y``, ``n_points``,
                  ``field_names``, ``times_used``, ``state_matrix``, ``signal_matrix``.
        :rtype: None

        :raises KeyError: If ``"u_inlet"`` is missing in ``reference_dic``.
        :raises FileNotFoundError: If ``omega.csv`` is not present in ``folder_path``.
        :raises ValueError: If control interpolation fails due to time coverage.
        """
        # Load mesh/time info
        loader = FOAMDataloader(self.folder_path, distributed=True)
        times = loader.write_times  # ensure ndarray for search

        times_arr = np.array(loader.write_times)
        vertices = np.asarray(loader.vertices, dtype=np.float32)
        self.x = np.asarray(vertices[:, 0])
        self.y = np.asarray(vertices[:, 1])
        self.n_points = self.x.size

        # Find the first index whose time >= start_time

        time_start_idx = int(np.searchsorted(times_arr, self.start_time, side="left"))
        
        times_used = times[time_start_idx:]

        times_initial = times[1:time_start_idx]

        u_inlet = float(self.reference_dic["u_inlet"])

        # ---- Pressure (mean-subtracted over space; normalized by u_inlet^2) ----
        p_snap = np.asarray(loader.load_snapshot("p", times_used), dtype=np.float32)
        # subtract spatial mean per time step
        #self.p_mean = np.atleast_2d(p_snap.mean(axis=1)).T  # (n_times, 1)
        #p_shifted = (p_snap - self.p_mean) / (u_inlet**2)
        
        p_shifted = (p_snap) / (u_inlet**2)

        # ---- Velocity components (mean-subtracted over space; normalized by u_inlet) ----
        U_snap = np.asarray(loader.load_snapshot("U", times_used), dtype=np.float32)

        # ux
        ux = U_snap[:, 0, :] / u_inlet
        #self.ux_mean = np.atleast_2d(ux.mean(axis=1)).T
        #ux = ux - self.ux_mean
        
        # uy
        uy = U_snap[:, 1, :] / u_inlet
        #self.uy_mean = np.atleast_2d(uy.mean(axis=1)).T
        #uy = uy - self.uy_mean

        components = [p_shifted, ux, uy]

        field_names = ["p", "u_x", "u_y"]

        if self.dim.lower() == "3d":
            # uz (assumes third component exists)
            uz = U_snap[:, 2, :] / u_inlet
            #uz_mean = np.atleast_2d(uz.mean(axis=1)).T
            #uz = uz - uz_mean
            components.append(uz)
            field_names.append("u_z")



        p_snap_ini = np.asarray(loader.load_snapshot("p", times_initial), dtype=np.float32)
        # subtract spatial mean per time step
        #self.p_mean = np.atleast_2d(p_snap.mean(axis=1)).T  # (n_times, 1)
        #p_shifted = (p_snap - self.p_mean) / (u_inlet**2)
        
        p_shifted_ini = (p_snap_ini) / (u_inlet**2)

        # ---- Velocity components (mean-subtracted over space; normalized by u_inlet) ----
        U_snap_ini = np.asarray(loader.load_snapshot("U", times_initial), dtype=np.float32)

        # ux
        ux_ini = U_snap_ini[:, 0, :] / u_inlet
        #self.ux_mean = np.atleast_2d(ux.mean(axis=1)).T
        #ux = ux - self.ux_mean
        
        # uy
        uy_ini = U_snap_ini[:, 1, :] / u_inlet
        #self.uy_mean = np.atleast_2d(uy.mean(axis=1)).T
        #uy = uy - self.uy_mean

        components_ini = [p_shifted_ini, ux_ini, uy_ini]


        if self.dim.lower() == "3d":
            # uz (assumes third component exists)
            uz_ini = U_snap[:, 2, :] / u_inlet
            #uz_mean = np.atleast_2d(uz.mean(axis=1)).T
            #uz = uz - uz_mean
            components_ini.append(uz_ini)

        self.field_names = field_names
        self.times_used = times_used
        self.state_matrix = np.vstack(components).astype(np.float32)
        self.state_matrix_ini = np.vstack(components_ini).astype(np.float32)

        if self.verbose:
            print("Vertex arrays:", self.x.shape, self.y.shape)

            print(
                "Shapes:",
                "p:",
                p_shifted.shape,
                "ux:",
                ux.shape,
                "uy:",
                uy.shape,
                *(("uz:", uz.shape) if self.dim.lower() == "3d" else ()),
                "state_matrix:",
                self.state_matrix.shape,
            )

        omega_csv_path = os.path.join(self.folder_path, "omega.csv")
        data = np.genfromtxt(omega_csv_path, delimiter=",", names=True, dtype=float)
        t_src = np.asarray(data["time"], dtype=float)
        omega_src = np.asarray(data["omega"], dtype=float)

        # Interpolation function
        self.f_omega = interp1d(t_src, omega_src)

        # Signal matrix construction
        omega_interp = self.f_omega(np.array(times_used[:-1]))  # shape (T,)
        u_act = omega_interp * self.reference_dic["cyl_radius"]
        self.signal_matrix = np.atleast_2d(u_act.astype(np.float32))  # shape (1, T)


    def train_test_state_signal_split(self, train_test_ratio=0.75):
        """Split state and control into train/test windows (time-wise).

        Splits columns of ``state_matrix`` at ``train_test_ratio``, and aligns
        control matrices so that ``U[:, k]`` drives the transition
        ``X[:, k] -> X[:, k+1]``.

        :param train_test_ratio: Fraction of time columns to use for training.
        :type train_test_ratio: float, optional

        :returns: None. Sets ``train_data``, ``test_data``, ``train_u``, ``test_u``,
                  and keeps copies ``*_original`` for error evaluation.
        :rtype: None
        """
        index = round(train_test_ratio * self.state_matrix.shape[1])
        self.train_data = self.state_matrix[:, :index]
        self.test_data = self.state_matrix[:, index:]

        self.train_u = self.signal_matrix[:, : index - 1]
        self.test_u = self.signal_matrix[:, index - 1 :]
        self.train_data_original = self.train_data
        self.test_data_original = self.test_data

        if self.verbose:
            print(f"Train state shape: {self.train_data.shape}")
            print(f"Test state shape:  {self.test_data.shape}")

            print(f"Train signal shape: {self.train_u.shape}")
            print(f"Test signal shape:  {self.test_u.shape}")

    def time_march_dmdc(self, dmdc_model, U, x0):
        """Roll out a reduced DMDc model and lift to full space.

        Evolves the reduced state ``z`` by ``z_{k+1} = Atilde z_k + Btilde u_k``
        with the provided input sequence, starting from ``x0`` in full space.
        Uses ``basis`` to project/lift between full and reduced coordinates.

        :param dmdc_model: Dictionary with keys ``"basis"``, ``"Atilde"``, ``"Btilde"``.
        :type dmdc_model: dict[str, np.ndarray]
        :param U: Control sequence shaped ``(m, T-1)`` (or broadcastable to it).
        :type U: np.ndarray
        :param x0: Initial full state column vector (``(n,)`` or ``(n,1)``).
        :type x0: np.ndarray
        :returns: Predicted full-state trajectory of shape ``(n, T)``.
        :rtype: np.ndarray
        """
        x0 = np.asarray(x0)
        if x0.ndim == 1:
            x0 = x0[:, None]  # (n,1)

        U = np.asarray(U)
        if U.ndim == 1:
            U = U[None, :]  # (1, T-1)
        elif U.ndim == 2 and U.shape[0] > U.shape[1]:
            # If user passed (T-1, m), transpose to (m, T-1)
            U = U.T

        # DMD operators
        Phi = dmdc_model["basis"]  # (n, r)
        Atilde = dmdc_model["Atilde"]  # (r, r)
        Btilde = dmdc_model["Btilde"]  # (r, m)

        # Reduced initial state
        x_red = Phi.T @ x0  # (r,1)

        # Rollout in reduced space
        Tm1 = U.shape[1]  # number of transitions
        z_cols = [x_red]
        for k in range(Tm1):
            u_k = U[:, k : k + 1]  # (m,1)
            x_red = Atilde @ x_red + Btilde @ u_k
            z_cols.append(x_red)
        Z = np.hstack(z_cols)  # (r, T)

        # Lift back to full space
        X_pred = Phi @ Z  # (n, T)
        return X_pred

    def evaluate_svd_rank(self, ranks):
        """Grid search over SVD ranks for reconstruction and prediction error.

        For ranks ``1..max_rank``, fits :class:`pydmd.DMDc` on the train window,
        reconstructs train data, and predicts the test window by time marching.
        Populates ``err_reconstruction`` and ``err_prediction`` and prints the
        best ranks.

        :param max_rank: Maximum truncation rank to evaluate.
        :type max_rank: int
        :returns: None. Populates ``err_reconstruction``, ``err_prediction``.
        :rtype: None
        """
        #ranks = list(range(1, max_rank + 1))
        self.ranks = ranks
        self.err_reconstruction = []
        self.err_prediction = []

        train_data, test_data = self.train_data, self.test_data
        train_u, test_u = self.train_u, self.test_u

        for r in ranks:
            dmdc_forced_model = DMDc(svd_rank=r)
            dmdc_forced_model.fit(train_data, train_u)
            dmdc_forced = {
                "basis": dmdc_forced_model.basis,  # (n, r)
                "Atilde": dmdc_forced_model._Atilde._Atilde,  # (r, r)
                "Btilde": dmdc_forced_model.basis.T @ dmdc_forced_model.B,  # (r, m)
            }

            reconstructed_data = self.time_march_dmdc(
                dmdc_forced, train_u, train_data[:, 0]
            )
            error = np.linalg.norm(
                self.train_data_original - reconstructed_data
            ) / np.linalg.norm(self.train_data_original)
            self.err_reconstruction.append(error)

            predicted_flow = self.time_march_dmdc(
                dmdc_forced, test_u, train_data[:, -1]
            )
            error = np.linalg.norm(
                self.test_data_original - predicted_flow[:, 1:]
            ) / np.linalg.norm(self.test_data_original)
            self.err_prediction.append(error)

        rank_best_reconstruction = np.argmin(np.array(self.err_reconstruction))
        rank_best_prediction = np.argmin(np.array(self.err_prediction))
        if self.verbose:
            print(f"Details for {self.reference_dic['signal']}")
            print(
                f"Best reconstruction rank: {ranks[rank_best_reconstruction]} "
                f"(error = {self.err_reconstruction[rank_best_reconstruction]:.4e})"
            )
            print(
                f"Best prediction rank:     {ranks[rank_best_prediction]} "
                f"(error = {self.err_prediction[rank_best_prediction]:.4e})"
            )

    def plot_reconstruction_prediction_all_fields(
        self, svd_rank, no_points=4, cmap="seismic", show=True, save=False
    ):
        """Qualitative train/test comparisons per field with tricontours.

        Fits DMDc at the given rank, reconstructs the train window, and
        predicts the test window. For each field in ``field_names``, plots
        a small set of evenly spaced time slices: Original vs Reconstructed
        (train) and Original vs Predicted (test).

        :param svd_rank: SVD rank used in DMDc fitting.
        :type svd_rank: int
        :param no_points: Number of time indices to display (evenly spaced).
        :type no_points: int, optional
        :param cmap: Matplotlib colormap passed to ``tricontourf``.
        :type cmap: str, optional
        :param show: If ``True``, show figures with ``plt.show()``.
        :type show: bool, optional
        :param save: If ``True``, save figures as PNGs named
            ``reconstruction_<field>_r<rank>_<signal>.png`` and
            ``prediction_<field>_r<rank>_<signal>.png``.
        :type save: bool, optional
        :returns: None.
        :rtype: None
        """
        train_data, test_data = self.train_data, self.test_data
        train_u, test_u = self.train_u, self.test_u

        # --- DMDc model fit ---
        dmdc_model = DMDc(svd_rank=svd_rank)
        dmdc_model.fit(train_data, train_u)
        dmdc = {
            "basis": dmdc_model.basis,  # (n, r)
            "Atilde": dmdc_model._Atilde._Atilde,  # (r, r)
            "Btilde": dmdc_model.basis.T @ dmdc_model.B,  # (r, m)
        }

        reconstructed_data = self.time_march_dmdc(dmdc, train_u, train_data[:, 0])
        predicted_flow = self.time_march_dmdc(dmdc, test_u, train_data[:, -1])[:, 1:]

        def get_field_block(X, field_idx):
            """Return (n_points, T) block for field_idx from stacked features matrix X."""
            n = self.n_points
            r0 = field_idx * n
            r1 = (field_idx + 1) * n
            return X[r0:r1, :]  # (n_points, T)

        T_train = self.train_data_original.shape[1]
        T_test = self.test_data_original.shape[1]

        idx_train = np.unique(np.linspace(0, T_train - 1, no_points, dtype=int))
        idx_test = np.unique(np.linspace(0, T_test - 1, no_points, dtype=int))

        for f_idx, f_name in enumerate(self.field_names):
            # Extract original & model snapshots for this field
            train_orig = get_field_block(
                self.train_data_original, f_idx
            )  # (n_points, T_train)
            train_reco = get_field_block(reconstructed_data, f_idx)

            test_orig = get_field_block(
                self.test_data_original, f_idx
            )  # (n_points, T_test)
            test_pred = get_field_block(predicted_flow, f_idx)

            string_signal = self.reference_dic["signal"].replace(" ", "_")
            vmin = np.nanmin(train_orig)
            vmax = np.nanmax(train_orig)
            levels = np.linspace(vmin, vmax, 100)
            fig, ax = plt.subplots(
                nrows=len(idx_train), ncols=2, figsize=(3 * len(idx_train), 6)
            )

            if len(idx_train) == 1:
                ax = np.array([ax])

            for i, t in enumerate(idx_train):
                # left
                im0 = ax[i, 0].tricontourf(
                    self.x,
                    self.y,
                    train_orig[:, t],
                    levels=levels,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                ax[i, 0].set_aspect("equal")
                cyl_center = self.reference_dic.get("cyl_centre", (0.0, 0.0))
                cyl_radius = self.reference_dic.get("cyl_radius", 0.0)

                ax[i, 0].add_patch(plt.Circle(cyl_center, cyl_radius, color="w"))
                tlabel = f"{self.times_used[t]} s"
                ax[i, 0].set_title(f"Original flow {f_name}, t={tlabel}")

                # right
                im1 = ax[i, 1].tricontourf(
                    self.x,
                    self.y,
                    train_reco[:, t],
                    levels=levels,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                ax[i, 1].set_aspect("equal")
                ax[i, 1].add_patch(plt.Circle(cyl_center, cyl_radius, color="w"))
                ax[i, 1].set_title(f"Reconstructed flow {f_name}, t={tlabel}")

            fig.suptitle(
                f"DMDc model reconstruction - {f_name} - {self.reference_dic.get('signal','signal')}"
            )
            cbar = fig.colorbar(im1, ax=ax.ravel().tolist(), shrink=0.95)
            cbar.set_label(f_name)
            if save:
                fig.savefig(
                    f"reconstruction_{f_name}_r{svd_rank}_{string_signal}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            if show:
                plt.show()

            # Plot Test: Original vs Predicted
            vmin = np.nanmin(test_orig)
            vmax = np.nanmax(test_orig)
            levels = np.linspace(vmin, vmax, 100)
            fig, ax = plt.subplots(
                nrows=len(idx_test), ncols=2, figsize=(3 * len(idx_test), 6)
            )

            if len(idx_test) == 1:
                ax = np.array([ax])

            for i, t in enumerate(idx_test):
                # left
                im0 = ax[i, 0].tricontourf(
                    self.x,
                    self.y,
                    test_orig[:, t],
                    levels=levels,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                ax[i, 0].set_aspect("equal")
                ax[i, 0].add_patch(plt.Circle((0.2, 0.2), 0.05, color="w"))
                tlabel = f"{self.times_used[t+T_train]} s"
                ax[i, 0].set_title(f"Original flow {f_name}, t={tlabel}")

                # right
                im1 = ax[i, 1].tricontourf(
                    self.x,
                    self.y,
                    test_pred[:, t],
                    levels=levels,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                ax[i, 1].set_aspect("equal")
                ax[i, 1].add_patch(plt.Circle((0.2, 0.2), 0.05, color="w"))
                ax[i, 1].set_title(f"Predicted flow {f_name}, t={tlabel}")

            fig.suptitle(
                f"DMDc model prediction - {f_name} - {self.reference_dic.get('signal','signal')}"
            )
            cbar = fig.colorbar(im1, ax=ax.ravel().tolist(), shrink=0.95)
            cbar.set_label(f_name)
            if save:
                fig.savefig(
                    f"prediction_{f_name}_r{svd_rank}_{string_signal}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

            if show:
                plt.show()

    def get_indices_based_on_integral_contributions(self, n):
        """Select top-|n| DMD modes by integral contribution (non-negative freq).

        Uses :func:`pydmd.dmd_modes_tuner.ModesSelectors._compute_integral_contribution`
        already stored in ``self.contrib`` and filters via ``self.valid`` (modes
        with non-negative frequency).

        :param n: Number of modes to select.
        :type n: int
        :returns: A tuple ``(indices, values)`` where ``indices`` are the selected
                  global mode indices and ``values`` their integral contributions.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        k = min(n, self.valid.size)
        local_topk = np.argpartition(self.contrib[self.valid], -k)[-k:]
        local_order = np.argsort(self.contrib[self.valid][local_topk])[::-1]
        indices_to_plot = self.valid[local_topk[local_order]]

        return indices_to_plot, self.contrib[indices_to_plot]

    def makeDMDcModel(self, svd_rank=15):
        """Fit a full DMDc model on all data and cache analysis artifacts.

        Fits :class:`pydmd.DMDc` using ``state_matrix`` and ``signal_matrix``,
        computes integral contributions for every mode, caches eigen-information,

        and stores reduced operators for rollout.
        :param svd_rank: Truncation rank passed to :class:`pydmd.DMDc`.
        :type svd_rank: int, optional
        :returns: None. Sets attributes:
                  ``dynamics``, ``modes``, ``contrib``, ``freq``, ``valid``,
                  ``frequency_all``, ``eigs_real_part_all``, ``eigs_imag_part_all``,
                  and ``dmdc`` (dict with ``basis``, ``Atilde``, ``Btilde``).
        :rtype: None
        """
        dmdc = DMDc(svd_rank=svd_rank)
        dmdc.fit(self.state_matrix, self.signal_matrix)
        self.dynamics = dmdc.dynamics
        self.modes = dmdc.modes
        self.contrib = np.array(
            [
                ModesSelectors._compute_integral_contribution(m, d)
                for m, d in zip(self.modes.T, self.dynamics)
            ],
            dtype=float,
        )

        self.freq = np.asarray(dmdc.frequency, dtype=float)

        # 1) keep only non-negative frequencies for integral contribution sorting
        self.valid = np.flatnonzero(self.freq >= 0)
        try:
            if hasattr(self, "times_used") and len(self.times_used) > 1:
                dt = float(np.mean(np.diff(np.asarray(self.times_used, dtype=float))))
                dmdc.original_time["dt"] = dt
        except Exception:
            pass

        self.frequency_all = dmdc.frequency
        self.eigs_real_part_all = dmdc.eigs.real
        self.eigs_imag_part_all = dmdc.eigs.imag
        self.dmdc = {
            "basis": dmdc.basis,  # (n, r)
            "Atilde": dmdc._Atilde._Atilde,  # (r, r)
            "Btilde": dmdc.basis.T @ dmdc.B,  # (r, m)
        }

    def clear_memory(self):
        """Free large train/test arrays after model fitting/plotting.

        Sets large arrays to ``None`` to release memory, if present:
        ``train_data``, ``test_data``, ``train_u``, ``test_u``,
        ``train_data_original``, ``test_data_original``.

        :returns: None.
        :rtype: None
        """
        # Drop very large arrays after fitting
        for name in (
            "train_data",
            "test_data",
            "train_u",
            "test_u",
            "train_data_original",
            "test_data_original",
            "state_matrix_ini"
        ):
            if hasattr(self, name):
                setattr(self, name, None)

    def eval_data_for_plot(self, n):
        """Prepare top-|n| mode data for downstream plotting.

        Selects top-|n| modes using integral contribution, then slices and
        caches frequency/eigenvalue components and the subset of modes.

        :param n: Number of modes to keep for plotting.
        :type n: int
        :returns: None. Populates ``integral_contribution``, ``frequency_DMDc``,
                  ``eigs_real_part``, ``eigs_imag_part``, and ``selected_modes``.
        :rtype: None
        """
        indices_integral_cont, integral_contribution = (
            self.get_indices_based_on_integral_contributions(n)
        )
        self.integral_contribution = integral_contribution

        self.frequency_DMDc = self.frequency_all[indices_integral_cont]
        self.eigs_real_part = self.eigs_real_part_all[indices_integral_cont]
        self.eigs_imag_part = np.abs(self.eigs_imag_part_all[indices_integral_cont])
        self.selected_modes = self.modes[:, indices_integral_cont]

