import numpy as np
from pydmd import DMDc
from pydmd.dmd_modes_tuner import ModesSelectors
from DMDcDataset import DMDcDataset
import matplotlib.pyplot as plt


class TimeDelayedDMDc(DMDcDataset):
    """Time-delay (Hankel) DMDc built on top of :class:`DMDcDataset`.

    Adds forward-delay embedding of the state (Hankel stacking) before fitting
    DMDc. Control inputs remain in original time (aligned to transitions).

    :param folder_path: Path to the OpenFOAM case directory.
    :type folder_path: str
    :param reference_dic: Dictionary of reference quantities/geometry (must include
        ``"u_inlet"``; optional: ``"cyl_radius"``, ``"cyl_centre"``, ``"signal"``).
    :type reference_dic: dict
    :param start_time: First write time (inclusive) to consider when building matrices.
    :type start_time: float, optional
    :param dim: If ``"3d"``, include ``u_z`` in the state.
    :type dim: str, optional
    :param q: Number of forward delay blocks in Hankel embedding.
    :type q: int, optional
    :param step: Lag (in samples) between successive delay blocks.
    :type step: int, optional

    **Attributes added**
    - ``q`` (int): Number of delays.
    - ``step`` (int): Delay stride.
    - ``f_orig`` (int): Original state feature dimension (before Hankel).
    - ``init_policy`` (str): Policy for building the augmented initial state 
        (``"repeat"`` or ``"zero"``).
    - ``state_matrix_delayed`` (:class:`numpy.ndarray`): Shape ``(f*q, C)``.
    - ``signal_matrix_delayed`` (:class:`numpy.ndarray`): Shape ``(m, C-1)``.
    - ``delay_start`` (int): Starting alignment index (0).
    - ``delay_C`` (int): Number of Hankel columns.
    """

    def __init__(
        self, folder_path, reference_dic, start_time=4.0, dim="2d", q=5, step=1
    ):
        """Constructor: builds base matrices and delayed (Hankel) versions."""
        self.q = int(q)
        self.step = int(step)
        super().__init__(folder_path, reference_dic, start_time=start_time, dim=dim)

        # f = n_points * len(field_names)
        self.f_orig = self.state_matrix.shape[0]
        self.init_policy = "repeat"
        # Build delayed matrices
        self._build_delay_matrices()

    def set_init_policy(self, policy="repeat"):
        """Set policy for initializing the augmented initial state.

        :param policy: One of ``"repeat"`` (default) or ``"zero"``.
        :type policy: str
        """
        self.init_policy = policy

    @staticmethod
    def _delay_embed_state(X, q, step=1):
        """Forward (Hankel) delay embedding of a state snapshot matrix.

        Constructs columns:
        ``Xd[:, k] = [x_k; x_{k+step}; ...; x_{k+(q-1)step}]`` for
        ``k = 0..C-1``, where ``C = T - (q-1)step``.

        :param X: Original state snapshots (features × time).
        :type X: numpy.ndarray
        :param q: Number of forward-shifted blocks (window length).
        :type q: int
        :param step: Lag between successive blocks.
        :type step: int, optional
        :returns: Forward-embedded (Hankel) state matrix of shape ``(f*q, C)``.
        :rtype: numpy.ndarray
        :raises ValueError: If not enough samples for requested delays.
        """
        f, T = X.shape
        C = T - (q - 1) * step
        if C <= 0:
            raise ValueError(
                f"Not enough samples for requested delays: q={q}, step={step}, T={T}"
            )
        blocks = [X[:, d * step : d * step + C] for d in range(q)]
        return np.vstack(blocks)  # (f*q, C)

    def _build_delay_matrices(self):
        """Create forward-embedded (Hankel) state/control matrices aligned for DMDc.

        Uses ``self.state_matrix`` and ``self.signal_matrix`` to build:
        - ``state_matrix_delayed`` with shape ``(f*q, C)``,
        - ``signal_matrix_delayed`` with shape ``(m, C-1)``.

        :returns: None. Populates delayed matrices and alignment metadata.
        :rtype: None
        :raises ValueError: If ``signal_matrix`` does not satisfy ``Tu = T - 1``.
        """
        q, step = self.q, self.step
        f, T = self.state_matrix.shape
        m, Tu = self.signal_matrix.shape
        if Tu != T - 1:
            raise ValueError(f"signal_matrix must have T-1 columns; got {Tu} vs T={T}")

        self.state_matrix_delayed = self._delay_embed_state(
            self.state_matrix, q=q, step=step
        )  # (f*q, C)
        C = self.state_matrix_delayed.shape[1]

        self.signal_matrix_delayed = self.signal_matrix[:, : (C - 1)]  # (m, C-1)

        self.delay_start = 0  # first column corresponds to x_0 (top block)
        self.delay_C = C
        print(
            "Delayed matrices shapes:",
            "q:",
            q,
            "step:",
            step,
            "state_matrix_delayed:",
            self.state_matrix_delayed.shape,
            "signal_matrix_delayed:",
            self.signal_matrix_delayed.shape,
        )

    def _build_augmented_x0(self, x0, policy=None):
        """Create an augmented initial state for Hankel rollout.

        Stacks ``q`` blocks vertically using either repetition of ``x0`` or zeros
        for the lower blocks.

        :param x0: Original-space initial state (``(f,)`` or ``(f,1)``).
        :type x0: numpy.ndarray
        :param policy: ``"repeat"`` (use ``x0`` for all blocks) or ``"zero"``.
            If ``None``, uses :attr:`init_policy`.
        :type policy: str, optional
        :returns: Augmented initial vector with shape ``(f*q, 1)``.
        :rtype: numpy.ndarray
        :raises ValueError: If policy is unknown.
        """
        q = self.q
        f = self.f_orig
        x0 = np.asarray(x0)
        if x0.ndim == 1:
            x0 = x0[:, None]  # (f,1) expected
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
        """Split train/test in delayed space and preserve original-space copies.

        Sets (delayed):
          - ``train_data`` (``f*q × idx``), ``test_data`` (``f*q × (C-idx)``)
          - ``train_u`` (``m × (idx-1)``), ``test_u`` (``m × ((C-1)-(idx-1))``)

        Sets (original, for comparison/visualization):
          - ``train_data_original`` (``f × idx``), ``test_data_original`` (``f × (C-idx)``)

        :param train_test_ratio: Fraction of delayed columns used for training.
        :type train_test_ratio: float, optional
        :returns: None.
        :rtype: None
        """
        Xd = self.state_matrix_delayed  # (f*q, C)
        Ud = self.signal_matrix_delayed  # (m, C-1)
        C = Xd.shape[1]
        idx = int(round(train_test_ratio * C))
        idx = np.clip(idx, 1, C - 1)

        # Delayed splits used for fitting
        self.train_data = Xd[:, :idx]  # (f*q, idx)
        self.test_data = Xd[:, idx:]  # (f*q, C-idx)
        self.train_u = Ud[:, : idx - 1]  # (m, idx-1) aligns transitions
        self.test_u = Ud[:, idx - 1 :]  # (m, (C-1)-(idx-1))

        k0 = self.delay_start
        self.train_data_original = self.state_matrix[:, :idx]  # (f, idx)
        self.test_data_original = self.state_matrix[:, idx:C]  # (f, C-idx)

        print("Delayed train data:", self.train_data.shape)
        print("Delayed test  data:", self.test_data.shape)
        print("Delayed train_u  :", self.train_u.shape)
        print("Delayed test_u   :", self.test_u.shape)
        print("Orig train data:", self.train_data_original.shape)
        print("Orig test data:", self.test_data_original.shape)

    def makeDMDcModel(self, svd_rank=15):
        """Fit a DMDc model in delayed space and cache analysis artifacts.

        Fits :class:`pydmd.DMDc` on ``(state_matrix_delayed, signal_matrix_delayed)``,
        computes per-mode integral contributions, and stores reduced operators
        for rollout (``basis``, ``Atilde``, ``Btilde``).

        :param svd_rank: Truncation rank passed to :class:`pydmd.DMDc`.
        :type svd_rank: int, optional
        :returns: None. Sets: ``dynamics``, ``modes``, ``contrib``, ``freq``,
                  ``valid``, ``frequency_all``, eigen parts, and ``dmdc`` dict.
        :rtype: None
        """
        dmdc = DMDc(svd_rank=svd_rank)
        dmdc.fit(self.state_matrix_delayed, self.signal_matrix_delayed)

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

        # 1) keep only non-negative frequencies
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
        """Free large delayed/original arrays after fitting/plotting.

        Sets large attributes to ``None`` if present:
        ``state_matrix_delayed``, ``signal_matrix_delayed``,
        ``train_data``, ``test_data``, ``train_u``, ``test_u``,
        ``train_data_original``, ``test_data_original``.
        """
        # Drop very large arrays you don't need after fitting
        for name in (
            "state_matrix_delayed",
            "signal_matrix_delayed",
            "train_data",
            "test_data",
            "train_u",
            "test_u",
            "train_data_original",
            "test_data_original",
        ):
            if hasattr(self, name):
                setattr(self, name, None)

    def time_march_dmdc(self, dmdc_model, u_seq, x0):
        """Roll out a time-delayed (Hankel) DMDc model from an original state.

        Builds an augmented initial state consistent with the delay embedding
        and uses the reduced operators to march forward, returning only the
        top (original) block at each time.

        :param dmdc_model: Dictionary with keys ``"basis"``, ``"Atilde"``, ``"Btilde"`` trained in delayed space.
        :type dmdc_model: dict[str, numpy.ndarray]
        :param u_seq: Control inputs per step, shaped ``(m, T-1)`` or broadcastable.
        :type u_seq: numpy.ndarray
        :param x0: Original-space initial state (``(f,)`` or ``(f,1)``) or already-augmented (``(f*q, 1)``).
        :type x0: numpy.ndarray
        :returns: Predicted ORIGINAL states for all times ``k = 0..T-1`` (shape ``(f, T)``).
        :rtype: numpy.ndarray
        :raises ValueError: If ``x0`` size does not match original or augmented dimension.
        """
        # normalize x0 to column (f,1)
        x0 = np.asarray(x0)
        if x0.ndim == 1:
            x0 = x0[:, None]

        q = self.q
        fq = dmdc_model["basis"].shape[0]
        f = self.f_orig
        if x0.shape[0] == fq:
            tilt_x0 = x0  # already augmented
        elif x0.shape[0] == f:
            tilt_x0 = self._build_augmented_x0(
                x0, policy=self.init_policy
            )  # build augmented
        else:
            raise ValueError(
                f"x0 has {x0.shape[0]} rows; expected {f} (original) or {fq} (augmented)."
            )

        # normalize controls to (m, T-1)
        U = np.asarray(u_seq)
        if U.ndim == 1:
            U = U[None, :]  # (1, T-1)
        elif U.shape[0] < U.shape[1] and U.shape[0] != dmdc_model["Btilde"].shape[1]:
            # if user passed (T-1, m), transpose
            U = U.T

        X_aug = super().time_march_dmdc(dmdc_model, U, tilt_x0)
        X_pred = X_aug[:f, :]  # (f, T)
        return X_pred

    def eval_data_for_plot(self, n):
        """Prepare top-``n`` modal data (original top block) for plotting.

        After fitting, selects top modes by integral contribution and slices
        the *top block only* of each mode for physical plotting in original
        space.

        :param n: Number of modes to keep.
        :type n: int
        :returns: None. Populates ``integral_contribution``, ``frequency_DMDc``,
                  eigen parts, and ``selected_modes`` (shape ``(f_orig, n)``).
        :rtype: None
        """
        idx, integral_contribution = self.get_indices_based_on_integral_contributions(n)
        self.integral_contribution = integral_contribution
        self.frequency_DMDc = self.frequency_all[idx]
        self.eigs_real_part = self.eigs_real_part_all[idx]
        self.eigs_imag_part = np.abs(self.eigs_imag_part_all[idx])

        # Only the TOP block of each mode for physical plotting
        f = self.f_orig
        self.selected_modes = self.modes[:f, :][:, idx]  # (f, n)
