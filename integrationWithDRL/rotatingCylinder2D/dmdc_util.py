from flowtorch.data import FOAMDataloader
import numpy as np
from typing import Optional, Dict, List, Tuple


class dmdcUtil:
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

    @staticmethod
    def dmdc_step(dmdc_model: Dict, x: np.ndarray, u: np.ndarray, common_Ur: np.ndarray) -> np.ndarray:
        """One-step DMDc update in reduced space."""
        Phi = np.asarray(dmdc_model["basis"])
        Atilde = np.asarray(dmdc_model["Atilde"])
        Btilde = np.asarray(dmdc_model["Btilde"])
        rank_dr = common_Ur.shape[1]
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        #print(f"U_hist = {u.shape}")
        #print(f"common ur = {common_Ur.shape}, x = {x.shape}")
        tilt_x0 = common_Ur.T @ x   # build augmented


        tilt_x0 = tilt_x0.reshape(-1,1, order= 'F')

        u = np.asarray(u)
        if u.ndim == 1:
            u = u[:, None]

        Ud = dmdcUtil._delay_embed_state(u, 100, step = 1)
        #print(f"Ud = {Ud.shape}")

        z = Phi.T @ tilt_x0                      # (r,1)
        z_next = Atilde @ z + Btilde @ Ud   # (r,1)
        x_next = Phi @ z_next              # (n_state,1)

        return common_Ur @ x_next[-rank_dr:, :]

    @staticmethod
    def build_x0(q: int, reference_dic: Dict, path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Build initial (normalized) state snapshot matrix from OpenFOAM case."""
        loader = FOAMDataloader(path, distributed=True)
        times = loader.write_times

        if len(times) < q:
            raise ValueError(f"Not enough write times ({len(times)}) for q={q}")
        times_used = times[-q:]
        print(times_used)
        u_inlet = float(reference_dic["u_inlet"])
        dim = reference_dic.get("dim", "2D").lower()
        vertices = np.asarray(loader.vertices, dtype=np.float32)

        # Pressure normalized by u_inlet^2
        p_snap = np.asarray(loader.load_snapshot("p", times_used), dtype=np.float32)
        p_norm = p_snap / (u_inlet ** 2)

        # Velocity normalized by u_inlet
        U_snap = np.asarray(loader.load_snapshot("U", times_used), dtype=np.float32)
        ux = U_snap[:, 0, :] / u_inlet
        uy = U_snap[:, 1, :] / u_inlet

        components = [p_norm, ux, uy]
        field_names = ["p", "u_x", "u_y"]

        if dim == "3d":
            uz = U_snap[:, 2, :] / u_inlet
            components.append(uz)
            field_names.append("u_z")

        X0 = np.vstack(components).astype(np.float32)  # (n_state, q)
        return X0, field_names, times_used, vertices[:,0:2]

    @staticmethod
    def infer_n_points(n_state_total: int, n_fields: int) -> int:
        """Infer number of spatial points from total state size and number of fields."""
        if n_state_total % n_fields != 0:
            raise ValueError(
                f"Cannot infer n_points: n_state_total={n_state_total} not divisible by n_fields={n_fields}"
            )
        return n_state_total // n_fields

    @staticmethod
    def make_field_block_extractor(n_points: int):
        """Return a function get_field_block(X, field_idx) based on n_points."""
        def get_field_block(X: np.ndarray, field_idx: int) -> np.ndarray:
            X = np.asarray(X)
            r0 = field_idx * n_points
            r1 = (field_idx + 1) * n_points
            return X[r0:r1, :]
        return get_field_block