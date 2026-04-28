import numpy as np 
from flowtorch.data import FOAMDataloader
import torch as pt

class GroupedBasis:

    def __init__(self, ref_dic_list):
        grouped_snapshot = list()
        for ref_dic in ref_dic_list:
            state_matrix = self.build_state_matrix(ref_dic)
            grouped_snapshot.append(state_matrix)

        self.combined_state_matrix = np.concatenate(grouped_snapshot, axis=1)

    def build_state_matrix(self, ref_dic):
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
        folder_path = ref_dic["path"]
        u_inlet = ref_dic["u_inlet"]
        start_time = ref_dic["start_time"]
        dim = ref_dic["dim"]

        # Load mesh/time info
        loader = FOAMDataloader(folder_path, distributed=True)
        times = loader.write_times  # ensure ndarray for search

        times_arr = np.array(loader.write_times)

        time_start_idx = int(np.searchsorted(times_arr, start_time, side="left"))
        
        times_used = times[time_start_idx:]


        p_snap = np.asarray(loader.load_snapshot("p", times_used), dtype=np.float32)
        
        p_shifted = (p_snap) / (u_inlet**2)

        U_snap = np.asarray(loader.load_snapshot("U", times_used), dtype=np.float32)

        ux = U_snap[:, 0, :] / u_inlet
        uy = U_snap[:, 1, :] / u_inlet

        components = [p_shifted, ux, uy]

        field_names = ["p", "u_x", "u_y"]

        if dim.lower() == "3d":
            uz = U_snap[:, 2, :] / u_inlet
            components.append(uz)
            field_names.append("u_z")

        state_matrix = np.vstack(components).astype(np.float32)

        print(
            "Shapes:",
            "p:",
            p_shifted.shape,
            "ux:",
            ux.shape,
            "uy:",
            uy.shape,
            *(("uz:", uz.shape) if dim.lower() == "3d" else ()),
            "state_matrix:",
            state_matrix.shape,
        )

        return state_matrix
    
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
            "combined_state_matrix",
        ):
            if hasattr(self, name):
                setattr(self, name, None)        

    def compute_basis(self, svd_rank):
        
        Ur, _, _ = np.linalg.svd(self.combined_state_matrix, full_matrices = False)
        
        return Ur[:, :svd_rank]