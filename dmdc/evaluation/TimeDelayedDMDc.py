import numpy as np
from pydmd import DMDc
from pydmd.dmd_modes_tuner import ModesSelectors
from DMDcDataset import DMDcDataset
import matplotlib.pyplot as plt

class TimeDelayedDMDc(DMDcDataset):
    """
    Time-delay (Hankel) DMDc on top of DMDcDataset.

    Key additions:
    - Delay parameters: q (number of delays), step (lag in samples between delays)
    - Builds delayed state/control matrices: state_matrix_delayed (f*q x C), signal_matrix_delayed (m x (C-1))
    - Train/test split in delayed space with original-space alignment
    - Fitting on delayed matrices (makeDMDcModel)
    - Reconstruction/prediction mapped back to original space (top block)
    - eval_data_for_plot returns modes *for the top block only* (physically interpretable)
    """
    def __init__(self, folder_path, reference_dic, start_time=4.0, dim='2d', q=5, step=1):
        self.q = int(q)
        self.step = int(step)
        super().__init__(folder_path, reference_dic, start_time=start_time, dim=dim)

        # original features & control

        # Infer original feature count (top block size)
        # If your base class stacks per-field blocks vertically, f = n_points * len(field_names)
        self.f_orig = self.state_matrix.shape[0]
        self.init_policy = "repeat"
        # Build delayed matrices
        self._build_delay_matrices()

    # --------------------------- Delay builders ---------------------------

    def set_init_policy(self, policy = "repeat"):
        self.init_policy = policy

    @staticmethod
    def _delay_embed_state(X, q, step=1):
        """
        Forward (Hankel) delay embedding.

        Build columns:
            Xd[:, k] = [x_k; x_{k+step}; x_{k+2*step}; ...; x_{k+(q-1)step}]
        for k = 0 .. C-1, where C = T - (q-1)*step.

        Parameters
        ----------
        X : (f, T)
            Original state snapshots (features x time).
        q : int
            Number of forward-shifted blocks (window length).
        step : int
            Lag between successive blocks.

        Returns
        -------
        Xd : (f*q, C)
            Forward-embedded (Hankel) state matrix.
        """
        f, T = X.shape
        C = T - (q - 1) * step
        if C <= 0:
            raise ValueError(f"Not enough samples for requested delays: q={q}, step={step}, T={T}")
        # Each block is a forward-shifted slice of length C
        blocks = [X[:, d*step : d*step + C] for d in range(q)]
        return np.vstack(blocks)  # (f*q, C)

    def _build_delay_matrices(self):
        """Create forward-embedded (Hankel) state/control matrices aligned for DMDc."""
        q, step = self.q, self.step
        f, T = self.state_matrix.shape
        m, Tu = self.signal_matrix.shape
        if Tu != T - 1:
            raise ValueError(f"signal_matrix must have T-1 columns; got {Tu} vs T={T}")

        # Forward-embedded states
        self.state_matrix_delayed = self._delay_embed_state(self.state_matrix, q=q, step=step)  # (f*q, C)
        C = self.state_matrix_delayed.shape[1]

        # Controls aligned with transitions Xd[:, k] -> Xd[:, k+1] use u_k
        # Since k starts at 0 in forward embedding, k0 = 0
        self.signal_matrix_delayed = self.signal_matrix[:, : (C - 1)]                             # (m, C-1)

        self.delay_start = 0                             # first column corresponds to x_0 (top block)
        self.delay_C = C
        # For compatibility with the rest of your pipeline, we’ll *not* overwrite self.state_matrix
        # (so you still have access to original-space arrays). Delayed arrays have explicit names.
        print(
            "Delayed matrices shapes:",
            "q:", q, "step:", step,
            "state_matrix_delayed:", self.state_matrix_delayed.shape,
            "signal_matrix_delayed:", self.signal_matrix_delayed.shape,
        )
    # --------------------------- Train/Test (delayed) ---------------------------
    def _build_augmented_x0(self, x0, policy=None):
        q = self.q
        f = self.f_orig
        x0 = np.asarray(x0)
        if x0.ndim == 1:
            x0 = x0[:, None]         # (f,1) expected
        policy = policy

        blocks = [x0]  # x_0 on top
        if policy == "repeat":
            filler = x0
        elif policy == "zero":
            filler = np.zeros_like(x0)
        else:
            raise ValueError(f"Unknown init_policy: {policy}")
        for _ in range(q - 1):
            blocks.append(filler)

        return np.vstack(blocks)  # (f*q, 1)
    
    def train_test_state_signal_split(self, train_test_ratio=0.75):
        """
        Split train/test in delayed space (default) and keep original-space alignment.
        Sets:
          - (delayed)  self.train_data, self.test_data, self.train_u, self.test_u
          - (original) self.train_data_original, self.test_data_original
        """
        Xd = self.state_matrix_delayed               # (f*q, C)
        Ud = self.signal_matrix_delayed              # (m, C-1)
        C = Xd.shape[1]
        idx = int(round(train_test_ratio * C))
        idx = np.clip(idx, 1, C-1)

        # Delayed splits used for fitting
        self.train_data = Xd[:, :idx]                # (f*q, idx)
        self.test_data  = Xd[:, idx:]                # (f*q, C-idx)
        self.train_u    = Ud[:, :idx-1]              # (m, idx-1) aligns transitions
        self.test_u     = Ud[:, idx-1:]              # (m, (C-1)-(idx-1))

        # Original-space windows aligned with delayed columns
        k0 = self.delay_start
        self.train_data_original = self.state_matrix[:, : idx]   # (f, idx)
        self.test_data_original  = self.state_matrix[:, idx :C]   # (f, C-idx)

        print("Delayed train data:", self.train_data.shape)
        print("Delayed test  data:", self.test_data.shape)
        print("Delayed train_u  :", self.train_u.shape)
        print("Delayed test_u   :", self.test_u.shape)
        print("Orig train data:", self.train_data_original.shape)
        print("Orig test data:", self.test_data_original.shape)

    # --------------------------- Fitting (delayed) ---------------------------

    def makeDMDcModel(self, svd_rank=15):
        """
        Build & fit a DMDc model.
        fit_on: 'delayed' (default) -> fit on (state_matrix_delayed, signal_matrix_delayed)
                'original'          -> fall back to base class behavior on (state_matrix, signal_matrix)
        """
        dmdc = DMDc(svd_rank=svd_rank)
        dmdc.fit(self.state_matrix_delayed, self.signal_matrix_delayed)

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
            "state_matrix_delayed", "signal_matrix_delayed",
            "train_data", "test_data", "train_u", "test_u", "train_data_original",
            "test_data_original"
            # keep only *original* small windows for plotting if needed:
            # "train_data_original", "test_data_original",
        ):
            if hasattr(self, name):
                setattr(self, name, None)
                        
    def time_march_dmdc(self, dmdc_model, u_seq, x0):
        """
        Roll out a time-delayed (Hankel) DMDc model given only the current state x0 and a control sequence.

        Parameters
        ----------
        dmdc_model : pydmd.DMDc
            Fitted on delayed states of size (f*q) with basis Phi, Atilde, B.
        u_seq : array_like, shape (m, T-1) or (T-1,) or (T-1, m)
            Control inputs for transitions k -> k+1.
        x0 : array_like, shape (f,) or (f,1)
            Current ORIGINAL state at time k = 0 (not the delayed stack).
        q : int
            Number of delays used during training (size of augmented stack).
        step : int, default=1
            Lag in samples between delayed blocks (must match training).
        past_blocks : array_like or None
            Optional past states to seed the stack. If provided, should be a list/array
            of length (q-1) with each element shape (f,) for times
            x_{-step}, x_{-2*step}, ..., x_{-(q-1)*step} in that order.
        init_policy : {'repeat','zero','mean'}
            Policy to fill missing past blocks when past_blocks is None:
            - 'repeat': use x0 for all missing blocks
            - 'zero'  : use zeros
            - 'mean'  : use the spatial mean of x0 (broadcasted)

        Returns
        -------
        X_pred : ndarray, shape (f, T)
            Predicted ORIGINAL states for all times k = 0..T-1, i.e., the top block of the
            augmented state at each step. Here T = u_seq.shape[1] + 1.

        Notes
        -----
        - This assumes the model was trained on delayed states of size (f*q).
        - The augmented initial state is constructed as:
            tilt_x0 = [ x0, x_{-step}, x_{-2*step}, ..., x_{-(q-1)*step} ]^T
        (stacked vertically).
        - If you *do* have real past samples, pass them via `past_blocks` for best accuracy.
        """
        # normalize x0 to column (f,1)
        x0 = np.asarray(x0)
        if x0.ndim == 1:
            x0 = x0[:, None]

        q = self.q
        fq = dmdc_model["basis"].shape[0]
        f = self.f_orig
        if x0.shape[0] == fq:
            tilt_x0 = x0                     # already augmented
        elif x0.shape[0] == f:
            tilt_x0 = self._build_augmented_x0(x0, policy=self.init_policy)  # build augmented
        else:
            raise ValueError(
                f"x0 has {x0.shape[0]} rows; expected {f} (original) or {fq} (augmented)."
            )        

        # normalize controls to (m, T-1)
        U = np.asarray(u_seq)
        if U.ndim == 1:
            U = U[None, :]                  # (1, T-1)
        elif U.shape[0] < U.shape[1] and U.shape[0] != dmdc_model["Btilde"].shape[1]:
            # if user passed (T-1, m), transpose
            U = U.T


        X_aug = super().time_march_dmdc(dmdc_model, U, tilt_x0)
        X_pred = X_aug[:f, :]                # (f, T)
        return X_pred



    def eval_data_for_plot(self, n):
        """
        Prepares modal data for plotting (compatible with your Plotter):
        - selected_modes = TOP BLOCK ONLY (f_orig x n)
        - frequency_DMDc, eigs parts, integral_contribution
        """
        idx, integral_contribution = self.get_indices_based_on_integral_contributions(n)
        self.integral_contribution = integral_contribution
        # set dt best-effort for frequency scaling if needed

        self.frequency_DMDc = self.frequency_all[idx]
        self.eigs_real_part = self.eigs_real_part_all[idx]
        self.eigs_imag_part = np.abs(self.eigs_imag_part_all[idx])

        # Only the TOP block of each mode for physical plotting
        f = self.f_orig
        self.selected_modes = self.modes[:f, :][:, idx]   # (f, n)
