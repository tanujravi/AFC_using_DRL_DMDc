import numpy as np
from pydmd import DMDc
from flowtorch.data import FOAMDataloader
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pydmd.dmd_modes_tuner import ModesSelectors

class DMDcDataset:
    def __init__(self, folder_path, reference_dic, start_time=4.0, dim='2d'):
        self.folder_path = folder_path
        self.reference_dic = reference_dic
        self.start_time = start_time
        self.dim = dim
        self.build_state_signal_matrix()

    def build_state_signal_matrix(self):
        """
        Build a state matrix from OpenFOAM snapshots.

        Parameters
        ----------
        folder_path : str
            Path to the OpenFOAM case folder.
        reference_dic : dict
            Must contain "u_inlet" (float).
        start_time : float
            Start building the dataset from the first write time >= this value.
        dim : {'2d', '3d'}
            If '3d', include the z-velocity component in the state.

        Returns
        -------
        state_matrix : ndarray, shape = (n_fields * n_times, n_points)
            Stacked [p, ux, uy] (and + uz if 3d), each mean-subtracted and normalized.
        time_start_idx : int
            Index into `times` used as the starting point.
        times_used : list/array
            The times actually used (times[time_start_idx:]).
        """
        # Load mesh/time info
        loader = FOAMDataloader(self.folder_path, distributed=True)
        times = loader.write_times  # ensure ndarray for search

        times_arr = np.array(loader.write_times)
        vertices = np.asarray(loader.vertices, dtype = np.float32)
        self.x = np.asarray(vertices[:, 0])

        self.y = np.asarray(vertices[:, 1])
        self.n_points = self.x.size
        #x, y = np.array(vertices[:, 0]), np.array(vertices[:, 1])
        print("Vertex arrays:", self.x.shape, self.y.shape)

        # Find the first index whose time >= start_time
        # (leftmost insertion point keeps time >= start_time)
        
        #time_start_idx = int(np.searchsorted(times_arr, self.start_time, side='left'))
        time_start_idx = int(np.searchsorted(times_arr, self.start_time, side='left'))

        times_used = times[time_start_idx:]
        
        u_inlet = float(self.reference_dic["u_inlet"])

        # ---- Pressure (mean-subtracted over space; normalized by u_inlet^2) ----
        # Expect shape ~ (n_times, n_points)
        p_snap = np.asarray(loader.load_snapshot("p", times_used), dtype=np.float32)
        # subtract spatial mean per time step
        p_mean = np.atleast_2d(p_snap.mean(axis=1)).T  # (n_times, 1)
        p_shifted = (p_snap - p_mean) / (u_inlet ** 2)

        # ---- Velocity components (mean-subtracted over space; normalized by u_inlet) ----
        # Expect U with shape ~ (n_times, 3, n_points) for 3D / (n_times, 2, n_points) for 2D fields
        U_snap = np.asarray(loader.load_snapshot("U", times_used), dtype=np.float32)

        # ux
        ux = U_snap[:, 0, :] / u_inlet
        ux_mean = np.atleast_2d(ux.mean(axis=1)).T
        ux = ux - ux_mean

        # uy
        uy = U_snap[:, 1, :] / u_inlet
        uy_mean = np.atleast_2d(uy.mean(axis=1)).T
        uy = uy - uy_mean

        components = [p_shifted, ux, uy]
        
        field_names = ["p", "u_x", "u_y"]

        if self.dim.lower() == '3d':
            # uz (assumes third component exists)
            uz = U_snap[:, 2, :] / u_inlet
            uz_mean = np.atleast_2d(uz.mean(axis=1)).T
            uz = uz - uz_mean
            components.append(uz)
            field_names.append("u_z")
        
        self.field_names = field_names
        self.times_used = times_used
        # Stack fields along time-first axis: we want shape (n_fields * n_times, n_points)
        # Each component currently (n_times, n_points)
        self.state_matrix = np.vstack(components).astype(np.float32)

        print("Shapes:",
            "p:", p_shifted.shape,
            "ux:", ux.shape,
            "uy:", uy.shape,
            *(("uz:", uz.shape) if self.dim.lower() == '3d' else ()),
            "state_matrix:", self.state_matrix.shape)


        omega_csv_path = os.path.join(self.folder_path, "omega.csv")
        data = np.genfromtxt(omega_csv_path, delimiter=',', names=True, dtype=float)
        t_src = np.asarray(data['time'], dtype=float)
        omega_src = np.asarray(data['omega'], dtype=float)

        # Interpolation function
        f_omega = interp1d(t_src, omega_src)

        # Evaluate on times_used
        omega_interp = f_omega(np.array(times_used[:-1]))       # shape (T,)
        u_act = omega_interp*self.reference_dic["cyl_radius"]
        self.signal_matrix = np.atleast_2d(u_act.astype(np.float32))  # shape (1, T)

    def train_test_state_signal_split(self, train_test_ratio=0.75):
        index = round(train_test_ratio*self.state_matrix.shape[1])
        self.train_data = self.state_matrix[:,:index]
        self.test_data = self.state_matrix[:,index:]
        print(f"Train state shape: {self.train_data.shape}")
        print(f"Test state shape:  {self.test_data.shape}")

        self.train_u = self.signal_matrix[:,:index-1]
        self.test_u = self.signal_matrix[:,index-1:]    
        self.train_data_original = self.train_data
        self.test_data_original = self.test_data

        print(f"Train signal shape: {self.train_u.shape}")
        print(f"Test signal shape:  {self.test_u.shape}")
    
    def time_march_dmdc(self, dmdc_model, U, x0):
        """
        Roll out a fitted PyDMD DMDc model forward in time.

        Parameters
        ----------
        dmdc_model : pydmd.DMDc
            A fitted DMDc model (must have .basis, ._Atilde._Atilde, .B).
        U : array_like, shape (m, T-1) or (T-1,) or (T-1, )
            Control inputs for each step from k -> k+1.
            Will be reshaped to (m, T-1).
        x0_full : array_like, shape (n,) or (n,1)
            Initial full-state column at time k=0.

        Returns
        -------
        X_pred : ndarray, shape (n, T)
            Predicted full-state trajectory for all times [0..T-1],
            where T = U.shape[1] + 1.
        """
        # Ensure column shapes
        x0 = np.asarray(x0)
        if x0.ndim == 1:
            x0 = x0[:, None]          # (n,1)

        U = np.asarray(U)
        if U.ndim == 1:
            U = U[None, :]                      # (1, T-1)
        elif U.ndim == 2 and U.shape[0] > U.shape[1]:
            # If user passed (T-1, m), transpose to (m, T-1)
            # (comment out this heuristic if you want strict shape checking)
            U = U.T

        # Extract operators
        Phi    = dmdc_model["basis"]               # (n, r)
        Atilde = dmdc_model["Atilde"]     # (r, r)
        Btilde = dmdc_model["Btilde"]
                   # (r, m)

        # Reduced initial state
        x_red = Phi.T @ x0                 # (r,1)

        # Rollout in reduced space
        Tm1 = U.shape[1]                        # number of transitions
        z_cols = [x_red]
        for k in range(Tm1):
            u_k = U[:, k:k+1]                   # (m,1)
            x_red = Atilde @ x_red + Btilde @ u_k
            z_cols.append(x_red)
        Z = np.hstack(z_cols)                   # (r, T)

        # Lift back to full space
        X_pred = Phi @ Z                        # (n, T)
        return X_pred

    def evaluate_svd_rank(self, max_rank):
        ranks = list(range(1, max_rank+1))
        self.err_reconstruction = []
        self.err_prediction = []

        train_data, test_data = self.train_data, self.test_data
        train_u, test_u = self.train_u, self.test_u

        for r in ranks:
            dmdc_forced_model = DMDc(svd_rank = r)
            dmdc_forced_model.fit(train_data, train_u)
            dmdc_forced = {
            "basis": dmdc_forced_model.basis,               # (n, r)
            "Atilde": dmdc_forced_model._Atilde._Atilde,     # (r, r)
            "Btilde": dmdc_forced_model.basis.T @ dmdc_forced_model.B          # (r, m)
            }

            reconstructed_data = self.time_march_dmdc(dmdc_forced, train_u, train_data[:,0])
            error = np.linalg.norm(self.train_data_original-reconstructed_data)/np.linalg.norm(train_data)
            self.err_reconstruction.append(error)

            predicted_flow = self.time_march_dmdc(dmdc_forced, test_u, train_data[:,-1])
            error = np.linalg.norm(self.test_data_original-predicted_flow[:, 1:])/np.linalg.norm(test_data)
            self.err_prediction.append(error)

        rank_best_reconstruction = np.argmin(np.array(self.err_reconstruction))
        rank_best_prediction = np.argmin(np.array(self.err_prediction))
        
        print(f"Details for {self.reference_dic['signal']}")
        print(f"Best reconstruction rank: {ranks[rank_best_reconstruction]} "
            f"(error = {self.err_reconstruction[rank_best_reconstruction]:.4e})")
        print(f"Best prediction rank:     {ranks[rank_best_prediction]} "
            f"(error = {self.err_prediction[rank_best_prediction]:.4e})")

    def plot_reconstruction_prediction_all_fields(self, svd_rank, no_points=4,
                                                  cmap="seismic", show = True, save = False):
        """
        For a given SVD rank, fit DMDc (using self.train_data/self.train_u),
        reconstruct on train window, predict on test window, and
        plot Original vs Reconstructed (train) and Original vs Predicted (test)
        for EACH field in self.field_names.

        Parameters
        ----------
        svd_rank : int
        n_rows : int
            Number of evenly spaced time slices to plot.
        train_times : array-like or None
            Physical times for train columns (len == self.train_data.shape[1]).
        test_times : array-like or None
            Physical times for test columns (len == self.test_data.shape[1]).
        cmap : str
        """
        train_data, test_data = self.train_data, self.test_data
        train_u, test_u = self.train_u, self.test_u

        # --- Fit model once ---
        dmdc_model = DMDc(svd_rank=svd_rank)
        dmdc_model.fit(train_data, train_u)
        dmdc = {
        "basis": dmdc_model.basis,               # (n, r)
        "Atilde": dmdc_model._Atilde._Atilde,     # (r, r)
        "Btilde": dmdc_model.basis.T @ dmdc_model.B           # (r, m)
        }

        reconstructed_data = self.time_march_dmdc(dmdc, train_u, train_data[:, 0])
        predicted_flow = self.time_march_dmdc(dmdc, test_u, train_data[:, -1])[:, 1:]
        
        # --- Field extraction helper ---
        def get_field_block(X, field_idx):
            """Return (n_points, T) block for field_idx from stacked features matrix X."""
            n = self.n_points
            r0 = field_idx * n
            r1 = (field_idx + 1) * n
            return X[r0:r1, :]                         # (n_points, T)

        # --- Evenly spaced indices ---
        T_train = self.train_data_original.shape[1]
        T_test  = self.test_data_original.shape[1]

        idx_train = np.unique(np.linspace(0, T_train-1, no_points, dtype=int))
        idx_test  = np.unique(np.linspace(0, T_test-1, no_points, dtype=int))
        #idx_test = idx_test + T_train
        # --- Loop over fields and plot ---
        for f_idx, f_name in enumerate(self.field_names):
            # Extract original & model snapshots for this field
            train_orig = get_field_block(self.train_data_original, f_idx)   # (n_points, T_train)
            train_reco = get_field_block(reconstructed_data,   f_idx)

            test_orig  = get_field_block(self.test_data_original, f_idx)   # (n_points, T_test)
            test_pred  = get_field_block(predicted_flow, f_idx)
            
            string_signal = self.reference_dic["signal"].replace(" ", "_")
            # Plot Train: Original vs Reconstructed
            vmin = np.nanmin(train_orig)
            vmax = np.nanmax(train_orig)
            levels = np.linspace(vmin, vmax, 100)
            fig, ax = plt.subplots(nrows=len(idx_train), ncols=2,
                                   figsize=(3*len(idx_train), 6))

            if len(idx_train) == 1:
                ax = np.array([ax])

            for i, t in enumerate(idx_train):
                # left
                im0 = ax[i, 0].tricontourf(self.x, self.y, train_orig[:, t],
                                           levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[i, 0].set_aspect("equal")
                cyl_center = self.reference_dic.get("cyl_centre", (0.0, 0.0))
                cyl_radius = self.reference_dic.get("cyl_radius", 0.0)

                ax[i,0].add_patch(plt.Circle(cyl_center, cyl_radius, color='w'))
                tlabel = f"{self.times_used[t]} s"
                ax[i, 0].set_title(f"Original flow {f_name}, t={tlabel}")

                # right
                im1 = ax[i, 1].tricontourf(self.x, self.y, train_reco[:, t],
                                           levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[i, 1].set_aspect("equal")
                ax[i,1].add_patch(plt.Circle(cyl_center, cyl_radius, color='w'))
                ax[i, 1].set_title(f"Reconstructed flow {f_name}, t={tlabel}")

            fig.suptitle(f"DMDc model reconstruction - {f_name} - {self.reference_dic.get('signal','signal')}")
            cbar = fig.colorbar(im1, ax=ax.ravel().tolist(), shrink=0.95)
            cbar.set_label(f_name)
            if save:
                fig.savefig(f"reconstruction_{f_name}_r{svd_rank}_{string_signal}.png",
                        dpi=300, bbox_inches='tight')
            if show:
                plt.show()

            # Plot Test: Original vs Predicted
            vmin = np.nanmin(test_orig)
            vmax = np.nanmax(test_orig)
            levels = np.linspace(vmin, vmax, 100)
            fig, ax = plt.subplots(nrows=len(idx_test), ncols=2,
                                   figsize=(3*len(idx_test), 6))

            if len(idx_test) == 1:
                ax = np.array([ax])

            for i, t in enumerate(idx_test):
                # left
                im0 = ax[i, 0].tricontourf(self.x, self.y, test_orig[:, t],
                                           levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[i, 0].set_aspect("equal")
                ax[i,0].add_patch(plt.Circle((0.2, 0.2), 0.05, color='w'))
                tlabel = f"{self.times_used[t+T_train]} s"
                ax[i, 0].set_title(f"Original flow {f_name}, t={tlabel}")

                # right
                im1 = ax[i, 1].tricontourf(self.x, self.y, test_pred[:, t],
                                           levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[i, 1].set_aspect("equal")
                ax[i, 1].add_patch(plt.Circle((0.2, 0.2), 0.05, color='w'))
                ax[i, 1].set_title(f"Predicted flow {f_name}, t={tlabel}")

            fig.suptitle(f"DMDc model prediction - {f_name} - {self.reference_dic.get('signal','signal')}")
            cbar = fig.colorbar(im1, ax=ax.ravel().tolist(), shrink=0.95)
            cbar.set_label(f_name)
            if save:
                fig.savefig(f"prediction_{f_name}_r{svd_rank}_{string_signal}.png",
                        dpi=300, bbox_inches='tight')

            if show:
                plt.show()
    
    def get_indices_based_on_integral_contributions(self, n):
        # Compute per-mode integral contributions
        if self.valid.size == 0:
            # no non-negative frequencies available
            return np.array([], dtype=int), np.array([])

        # 2) top-k within the valid set
        k = min(n, self.valid.size)
        # argpartition for O(N) top-k, then sort those k in descending contrib
        local_topk = np.argpartition(self.contrib[self.valid], -k)[-k:]
        local_order = np.argsort(self.contrib[self.valid][local_topk])[::-1]
        indices_to_plot = self.valid[local_topk[local_order]]

        return indices_to_plot, self.contrib[indices_to_plot]
    
    def makeDMDcModel(self, svd_rank = 15):
        """
        Makes a dmdc model using full state matrix and control signal for given svd rank
        """
        dmdc = DMDc(svd_rank = svd_rank)
        dmdc.fit(self.state_matrix, self.signal_matrix)
        self.dynamics = dmdc.dynamics
        self.modes = dmdc.modes
        self.contrib = np.array([ModesSelectors._compute_integral_contribution(m, d)
                            for m, d in zip(self.modes.T, self.dynamics)], dtype=float)

        self.freq = np.asarray(dmdc.frequency, dtype=float)

        # 1) keep only non-negative frequencies
        self.valid = np.flatnonzero(self.freq >= 0)
        try:
            if hasattr(self, "times_used") and len(self.times_used) > 1:
                dt = float(np.mean(np.diff(np.asarray(self.times_used, dtype=float))))
                dmdc.original_time['dt'] = dt
        except Exception:
            pass

        self.frequency_all = dmdc.frequency
        self.eigs_real_part_all = dmdc.eigs.real
        self.eigs_imag_part_all = dmdc.eigs.imag
        self.dmdc = {
        "basis": dmdc.basis,               # (n, r)
        "Atilde": dmdc._Atilde._Atilde,     # (r, r)
        "Btilde": dmdc.basis.T @ dmdc.B           # (r, m)
        }

    def clear_memory(self):
        # Drop very large arrays you don't need after fitting
        for name in (
            "train_data", "test_data", "train_u", "test_u", "train_data_original",
            "test_data_original"
            # keep only *original* small windows for plotting if needed:
            # "train_data_original", "test_data_original",
        ):
            if hasattr(self, name):
                setattr(self, name, None)

    def eval_data_for_plot(self, n):
        indices_integral_cont, integral_contribution = self.get_indices_based_on_integral_contributions(n)
        self.integral_contribution = integral_contribution
        # set dt best-effort for frequency scaling if needed

        self.frequency_DMDc = self.frequency_all[indices_integral_cont]
        self.eigs_real_part = self.eigs_real_part_all[indices_integral_cont]
        self.eigs_imag_part = np.abs(self.eigs_imag_part_all[indices_integral_cont])
        self.selected_modes = self.modes[:, indices_integral_cont]
