import sys, time, numpy as np, torch as pt
from yaml import safe_load
from os import makedirs
from os.path import join
from smartsim import Experiment
from smartsim.settings import RunSettings
from smartsim.status import SmartSimStatus
from smartredis import Client
import matplotlib.pyplot as plt
from pandas import read_csv


class OnlineRowPartitionedSVD:
    """Row-partitioned streaming SVD (Liang split & merge).

    Coordinates SmartSim/SmartRedis jobs that compute an online SVD over
    row-partitioned OpenFOAM fields, merges per-partition partial SVDs,
    and (optionally) reconstructs/export fields for post-processing.

    :param config: Experiment + SVD configuration dictionary. Expected keys include
    ``experiment.exp_path`` (output path), ``svd_params.num_mpi_ranks``,
    ``svd_params.svd_rank``, ``svd_params.snapshot_sampling_interval``,
    ``svd_params.batch_size``, ``svd_params.end_time``, and optional IO fields
    under ``io.fo_name`` and ``svd_params.fields``.
    :type config: dict
    :param exp: Active SmartSim :class:`smartsim.Experiment` used to create and
        manage Models/Ensembles (OpenFOAM run, partial SVD jobs, merges, post-proc).
    :type exp: :class:`smartsim.Experiment`
    :param client: SmartRedis client handle used to read/write partitioned tensors
        and datasets (e.g., partial SVD U/S/VT blocks, Z-matrix chunks, factors).
    :type client: :class:`smartredis.Client`

    Attributes initialized from ``config`` include:
    - ``fo_name`` (str): FunctionObject/IO prefix, default ``"dataToSmartRedis"``.
    - ``fields`` (list[str]): Field names to process, default ``["p","Ux","Uy"]``.
    - ``exp_path`` (str): Experiment directory; ``case_name`` is ``exp_path/base_sim``.
    - ``num_mpi_ranks`` (int): Number of row partitions (MPI ranks).
    - ``svd_rank`` (int): Truncation rank for incremental SVD.
    - ``sampling_interval`` (int): Snapshot stride used when sampling times.
    - ``batch_size`` (int): Number of snapshots per streaming batch.
    - ``n_times`` (int): Final time index to process.

    The class itself does not start work on construction; methods provide:
    - Launch/monitor OpenFOAM and SVD Ensembles
    - Compute/merge SVD factors across batches
    - Save factors back to the DB, reconstruct snapshots, and export fields
    """

    def __init__(self, config, exp, client):
        """Constructor method"""

        self.config = config
        self.fo_name = self.config.get("io", {}).get("fo_name", "dataToSmartRedis")
        self.fields = self.config.get("svd_params", {}).get("fields", ["p", "Ux", "Uy"])
        exp_cfg = self.config["experiment"]
        self.exp_path = exp_cfg["exp_path"]
        makedirs(self.exp_path, exist_ok=True)
        self.case_name = join(self.exp_path, "base_sim")

        svd_cfg = self.config["svd_params"]
        self.num_mpi_ranks = svd_cfg["num_mpi_ranks"]
        self.svd_rank = svd_cfg["svd_rank"]
        self.sampling_interval = svd_cfg["snapshot_sampling_interval"]
        self.batch_size = svd_cfg["batch_size"]
        self.n_times = svd_cfg["end_time"]
        self.exp = exp
        self.client = client

    def wait_for_completion(self, entities, poll_interval=5, timeout=None):
        """Block until all entities in a SmartSim experiment complete.

        Periodically checks whether each entity has completed. Raises an
        exception if any entity fails or if a timeout occurs.

        :param exp: SmartSim experiment instance.
        :type exp: smartsim.Experiment
        :param entities: List or group of SmartSim entities (Model or Ensemble).
        :type entities: list or Ensemble or Model
        :param poll_interval: Time (in seconds) between status checks, defaults to 5.
        :type poll_interval: int, optional
        :param timeout: Maximum wait time in seconds, defaults to None (waits indefinitely).
        :type timeout: float, optional

        :raises RuntimeError: If one or more entities fail.
        :raises TimeoutError: If the timeout duration is exceeded.
        """
        start = time.time()
        while True:
            statuses = self.exp.get_status(entities)
            if all(s == SmartSimStatus.STATUS_COMPLETED for s in statuses):
                break
            if any(s == SmartSimStatus.STATUS_FAILED for s in statuses):
                raise RuntimeError("One or more entities failed.")
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError("Timeout waiting for SmartSim entities to complete.")
            time.sleep(poll_interval)

    def start_openfoam_sim(self):
        """Launch the base OpenFOAM simulation via SmartSim.

        Creates a Model that runs `Allrun` in the provided base case,
        generates the run directory, and starts it non-blocking.

        :return: The SmartSim Model running the OpenFOAM case.
        :rtype: Model
        """
        sim_config = self.config["simulation"]
        base_case_path = sim_config["base_case"]
        rs = RunSettings(exe="bash", exe_args="Allrun")
        base_sim = self.exp.create_model("base_sim", run_settings=rs)
        base_sim.attach_generator_files(to_copy=base_case_path)
        self.exp.generate(base_sim)
        self.exp.start(base_sim, block=False, summary=False)
        self.base_sim_path = base_sim.path
        return base_sim

    def run_first_svd(self, time_indices, batch_no):
        """Launch first-pass partial SVD ensembles (per field x rank).

        Spawns an Ensemble that computes per-partition SVD factors for the
        current batch window (`type="svd_new_matrix"`).

        :param time_indices: Sample indices used to build the batch data matrix.
        :type time_indices: list[int] | range | str
        :param batch_no: Batch counter used to namespace artifacts.
        :type batch_no: int
        """
        svd_settings = self.exp.create_run_settings(
            exe="python3", exe_args="partial_svd.py"
        )
        quoted_fields = [f"'{s}'" for s in self.fields]
        quoted_fo_name = f"'{self.fo_name}'"
        params_svd = {
            "field_name": quoted_fields,
            "mpi_rank": list(range(self.num_mpi_ranks)),
            "svd_rank": self.svd_rank,
            "fo_name": quoted_fo_name,
            "time_indices": str(time_indices),
            "type": '"svd_new_matrix"',
            "batch_no": batch_no,
        }
        name = f"svd_ensemble_batch_{batch_no}"
        ens = self.exp.create_ensemble(
            name, params=params_svd, run_settings=svd_settings, perm_strategy="all_perm"
        )
        ens.attach_generator_files(to_configure="./partial_svd.py")
        self.exp.generate(ens, overwrite=True)
        self.exp.start(ens, summary=False, block=False)
        self.wait_for_completion(ens)

    def run_second_svd(self, time_indices, batch_no):
        """Launch second-pass SVD on W-matrix (online eigen merge).

        Spawns an Ensemble that performs SVD on the stacked matrix W formed
        from previous and current batch contributions.

        :param time_indices: Sample indices corresponding to this batch.
        :type time_indices: list[int] | range | str
        :param batch_no: Batch counter used to namespace artifacts.
        :type batch_no: int
        """
        svdW_settings = self.exp.create_run_settings(
            exe="python3", exe_args="partial_svd.py"
        )
        quoted_fields = [f"'{s}'" for s in self.fields]
        quoted_fo_name = f"'{self.fo_name}'"
        params_svdW = {
            "field_name": quoted_fields,
            "mpi_rank": list(range(self.num_mpi_ranks)),
            "svd_rank": self.svd_rank,
            "fo_name": quoted_fo_name,
            "time_indices": str(time_indices),
            "type": '"W_matrix"',
            "batch_no": batch_no,
        }
        name = f"svdW_ensemble_batch_{batch_no}"
        ens = self.exp.create_ensemble(
            name,
            params=params_svdW,
            run_settings=svdW_settings,
            perm_strategy="all_perm",
        )
        ens.attach_generator_files(to_configure="./partial_svd.py")
        self.exp.generate(ens, overwrite=True)
        self.exp.start(ens, summary=False, block=False)
        self.wait_for_completion(ens)

    def compute_first_svd_USV(self, svd_ensemble_name):
        """Assemble global SVD (U, S, VT) from partial SVD outputs.

        Pulls per-partition {s, VT, U} from the database for all fields/ranks,
        forms Y = blockstack(s * VT), computes SVD(Y), and maps the local U's
        into a global left singular basis.

        :param svd_ensemble_name: Base name of the partial SVD ensemble.
        :type svd_ensemble_name: str
        :return: Truncated U, S, VT of the batch and stores partition sizes.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        Y, U_li = [], []
        r = 0
        for field in self.fields:
            for rank in range(self.num_mpi_ranks):

                s_svd = self.client.get_tensor(
                    f"{svd_ensemble_name}_{r}.partSVD_s_field_name_{field}_mpi_rank_{rank}"
                )
                VT_svd = self.client.get_tensor(
                    f"{svd_ensemble_name}_{r}.partSVD_VT_field_name_{field}_mpi_rank_{rank}"
                )
                Y.append(s_svd[:, np.newaxis] * VT_svd)
                r += 1
        Y = np.concatenate(Y, axis=0)
        Uy, sy, VTy = np.linalg.svd(Y, full_matrices=False)
        Uy, sy, VTy = Uy[:, : self.svd_rank], sy[: self.svd_rank], VTy[: self.svd_rank]

        r = 0
        for field in self.fields:
            # clean_field = field.strip("'\"")
            for rank in range(self.num_mpi_ranks):
                U_li.append(
                    self.client.get_tensor(
                        f"{svd_ensemble_name}_{r}.partSVD_U_field_name_{field}_mpi_rank_{rank}"
                    )
                )
                r += 1
        self.parititon_sizes = [U_l.shape[0] for U_l in U_li]
        n_times_svd = U_li[0].shape[1]
        U = np.concatenate(
            [
                (U_svd @ Uy[i * n_times_svd : (i + 1) * n_times_svd])
                for i, U_svd in enumerate(U_li)
            ],
            axis=0,
        )
        return U, sy, VTy

    def compute_second_svd_USV(self, svdW_ensemble_name):
        """Merge batches via Z-matrix SVD to update (U, S, VT).

        Pulls per-partition Z-matrix SVD factors, computes SVD over the
        concatenated Yz, and projects local Uz blocks to form the new U.

        :param svdz_ensemble_name: Base name of the Z-matrix SVD ensemble.
        :type svdz_ensemble_name: str
        :return: Updated U, S, VT after online merge.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        Yw, Uw_li = [], []
        r = 0
        for field in self.fields:
            for rank in range(self.num_mpi_ranks):
                s_svdw = self.client.get_tensor(
                    f"{svdW_ensemble_name}_{r}.partSVD_s_field_name_{field}_mpi_rank_{rank}"
                )
                VT_svdw = self.client.get_tensor(
                    f"{svdW_ensemble_name}_{r}.partSVD_VT_field_name_{field}_mpi_rank_{rank}"
                )
                Yw.append(s_svdw[:, np.newaxis] * VT_svdw)
                r += 1
        Yw = np.concatenate(Yw, axis=0)
        Uyw, syw, VTyw = np.linalg.svd(Yw, full_matrices=False)
        Uyw, syw, VTyw = (
            Uyw[:, : self.svd_rank],
            syw[: self.svd_rank],
            VTyw[: self.svd_rank],
        )

        r = 0
        for field in self.fields:
            for rank in range(self.num_mpi_ranks):
                Uw_li.append(
                    self.client.get_tensor(
                        f"{svdW_ensemble_name}_{r}.partSVD_U_field_name_{field}_mpi_rank_{rank}"
                    )
                )
                r += 1
        n_times_svdz = Uw_li[0].shape[1]
        Uw = np.concatenate(
            [
                (Uz_svd @ Uyw[i * n_times_svdz : (i + 1) * n_times_svdz])
                for i, Uz_svd in enumerate(Uw_li)
            ],
            axis=0,
        )
        return Uw, syw, VTyw

    def delete_tensors_from_ensemble(self, ensemble_name):
        """Remove partitioned SVD tensors for a given ensemble.

        Deletes {partSVD_s, partSVD_VT, partSVD_U} tensors across all
        fields and MPI ranks to free database memory.

        :param ensemble_name: Base ensemble name used for tensor keys.
        :type ensemble_name: str
        """
        r = 0
        for field in self.fields:
            for rank in range(self.num_mpi_ranks):
                self.client.delete_tensor(
                    f"{ensemble_name}_{r}.partSVD_s_field_name_{field}_mpi_rank_{rank}"
                )
                self.client.delete_tensor(
                    f"{ensemble_name}_{r}.partSVD_VT_field_name_{field}_mpi_rank_{rank}"
                )
                self.client.delete_tensor(
                    f"{ensemble_name}_{r}.partSVD_U_field_name_{field}_mpi_rank_{rank}"
                )
                r += 1

    def run_reconstruction(self, time_indices_for_data):
        """Reconstruct full-field snapshots from incremental SVD factors.

        For each physical field to output (collapsing Ux/ Uy/ Uz - U),
        launches an Ensemble that runs `reconstruction.py` across ranks
        for the requested `time_indices_for_data`.

        :param time_indices_for_data: Indices at which to reconstruct fields.
        :type time_indices_for_data: list[int] | range
        """
        fields_local = []
        u_added = False

        for f in self.fields:
            if f in ("Ux", "Uy", "Uz"):
                if not u_added:
                    fields_local.append("U")
                    u_added = True
            else:
                fields_local.append(f)

        quoted_fields = [f"'{s}'" for s in fields_local]

        for field in quoted_fields:
            rec_settings = self.exp.create_run_settings(
                exe="python3", exe_args="reconstruction.py"
            )
            params = {
                "field_name": field,
                "mpi_rank": list(range(self.num_mpi_ranks)),
                "svd_rank": self.svd_rank,
                "time_indices": str(time_indices_for_data),
            }
            clean_field = field.strip("'\"")

            ens = self.exp.create_ensemble(
                f"rec_ensemble_r{self.svd_rank}_field_name_{clean_field}",
                params=params,
                run_settings=rec_settings,
                perm_strategy="all_perm",
            )
            ens.attach_generator_files(to_configure="./reconstruction.py")
            self.exp.generate(ens, overwrite=True)
            self.exp.start(ens, summary=False, block=False)
            self.wait_for_completion(ens)

    def run_svdToFoam(self, fields=["p", "U"]):
        """Export reconstructed factors to OpenFOAM field format.

        Invokes `svdToFoam` in parallel for each field in `fields`, using
        the current truncation rank and FO functionObject name.

        :param fields: List of field names to export (e.g., ["p","U"]).
        :type fields: list[str]
        """
        for field in fields:
            settings = self.exp.create_run_settings(
                exe="svdToFoam",
                exe_args=f"-fieldName {field} -svdRank {self.svd_rank} "
                f"-FOName {self.fo_name} -parallel",
                run_command="mpirun",
                run_args={"np": f"{self.num_mpi_ranks}"},
            )
            model = self.exp.create_model(
                name=f"svdToFoam_r{self.svd_rank}_field_name_{field}",
                run_settings=settings,
                path=self.case_name,
            )
            self.exp.start(model, summary=False, block=False)
            self.wait_for_completion(model)

    def evaluate_increSVD(self, time_indices, batch_no):
        """Run one streaming-SVD update step for a batch.

        1) Compute partial SVDs for the current batch.
        2) If batch 0: take (U,S,VT) directly. Else:
        - Build Z = [U_prev S_prev, U_curr S_curr]
        - Partition and push Z blocks
        - Run Z-matrix SVD merge to get updated (U,S,VT)
        3) Cleanup per-batch tensors and truncate to `self.svd_rank`.

        :param time_indices: Sample indices forming the current batch window.
        :type time_indices: list[int] | range
        :param batch_no: Zero-based batch index.
        :type batch_no: int
        """
        # 1) partial SVD
        svd_ensemble_name = f"svd_ensemble_batch_{batch_no}"
        OnlineSVD.run_first_svd(time_indices, batch_no)

        # 2) merge
        U, s, VT = OnlineSVD.compute_first_svd_USV(svd_ensemble_name)

        if batch_no == 0:
            self.U_inc, self.s_inc, self.VT_inc = U, s, VT
        else:
            W = np.concatenate((self.U_inc * self.s_inc, U * s), axis=1)
            W_chunks = pt.split(pt.tensor(W), self.parititon_sizes, dim=0)
            r = 0
            for field in self.fields:
                for rank in range(self.num_mpi_ranks):
                    self.client.put_tensor(
                        f"W_{batch_no}_field_name_{field}_mpi_rank_{rank}",
                        np.array(W_chunks[r]),
                    )
                    r += 1

            svdW_ensemble_name = f"svdW_ensemble_batch_{batch_no}"
            OnlineSVD.run_second_svd(time_indices, batch_no)
            Uw, sw, VTw = OnlineSVD.compute_second_svd_USV(svdW_ensemble_name)
            self.U_inc, self.s_inc = Uw, sw
            self.VT_inc = np.concatenate(
                (self.VT_inc.T @ VTw.T[: self.svd_rank], VT.T @ VTw.T[self.svd_rank :]),
                axis=0,
            ).T
            self.delete_tensors_from_ensemble(svdW_ensemble_name)

        self.delete_tensors_from_ensemble(svd_ensemble_name)
        self.U_inc = self.U_inc[:, : self.svd_rank]
        self.s_inc = self.s_inc[: self.svd_rank]
        self.VT_inc = self.VT_inc[: self.svd_rank]

    def save_tensors(self):
        """Persist incremental SVD factors to the database.

        Stores s_incremental, VT_incremental, and splits U_incremental
        along original partition sizes to write one tensor per field x rank.
        """
        self.client.put_tensor("s_incremental", self.s_inc)
        self.client.put_tensor("VT_incremental", self.VT_inc)
        Us_inc = pt.split(pt.tensor(self.U_inc), self.parititon_sizes, dim=0)

        r = 0
        for field in self.fields:
            # clean_field = field.strip("'\"")
            for rank in range(self.num_mpi_ranks):
                self.client.put_tensor(
                    f"U_incremental_field_name_{field}_mpi_rank_{rank}",
                    np.array(Us_inc[r]),
                )
                r += 1

    def fetch_snapshot(self, time_index, mpi_rank, field_name):
        """Fetch a single snapshot vector for a field and rank.

        Loads dataset `{fo_name}_time_index_{t}_mpi_rank_{r}` and extracts
        the internal patch tensor for the requested field. For velocity
        components, slices the corresponding column from `U`.

        :param time_index: Simulation time index.
        :type time_index: int
        :param mpi_rank: Partition/MPI rank id.
        :type mpi_rank: int
        :param field_name: Field key (e.g., "p", "Ux", "Uy", "Uz", "U").
        :type field_name: str
        :return: Flattened snapshot array or None if dataset missing.
        :rtype: np.ndarray | None
        """
        dataset_name = f"{self.fo_name}_time_index_{time_index}_mpi_rank_{mpi_rank}"
        if client.dataset_exists(dataset_name):
            dataset = client.get_dataset(dataset_name)
            if field_name == "Ux":
                matrix = dataset.get_tensor(f"field_name_U_patch_internal")[
                    :, 0
                ].flatten()

            elif field_name == "Uy":
                matrix = dataset.get_tensor(f"field_name_U_patch_internal")[
                    :, 1
                ].flatten()
            elif field_name == "Uz":
                matrix = dataset.get_tensor(f"field_name_U_patch_internal")[
                    :, 2
                ].flatten()
            else:
                matrix = dataset.get_tensor(
                    f"field_name_{field_name}_patch_internal"
                ).flatten()

            return matrix
        else:
            return None

    def fetch_timeseries(self, time_indices, mpi_rank, field_name):
        """Stack snapshots across time for one rank and field.

        :param time_indices: Iterable of time indices to fetch.
        :type time_indices: list[int] | range
        :param mpi_rank: Partition/MPI rank id.
        :type mpi_rank: int
        :param field_name: Field key to fetch.
        :type field_name: str
        :return: 2D matrix shaped (n_dofs_per_rank, len(time_indices)).
        :rtype: np.ndarray
        """
        return np.vstack(
            [self.fetch_snapshot(ti, mpi_rank, field_name) for ti in time_indices]
        ).T

    def plot_singular_values(self, time_indices_for_data):
        """Compare full SVD vs streaming SVD singular values.

        Builds the full data matrix (concatenating all fields × ranks) for
        `time_indices_for_data`, computes its singular values, and overlays
        them with the current incremental spectrum. Saves:
        `compare_svd_stream_to_full.png`.

        :param time_indices_for_data: Indices used to assemble the full matrix.
        :type time_indices_for_data: list[int] | range
        """
        matrix = []
        for field in self.fields:
            # clean_field = field.strip("'\"")

            data_matrix = pt.cat(
                [
                    pt.tensor(
                        self.fetch_timeseries(time_indices_for_data, rank_i, field)
                    )
                    for rank_i in range(self.num_mpi_ranks)
                ],
                dim=0,
            )
            # print(data_matrix.shape)
            matrix.append(data_matrix)
        matrix = pt.cat(matrix, dim=0)
        _, s_pl, _ = np.linalg.svd(matrix, full_matrices=False)
        s_pl = s_pl[: self.svd_rank]
        fig, ax = plt.subplots(figsize=(6, 3))
        ns = list(range(self.svd_rank))
        ax.plot(ns, s_pl, label="SVD")
        ax.plot(range(len(self.s_inc)), self.s_inc, label="stream. SVD")
        ax.legend()
        # ax.set_xlim(0, 19)
        ax.set_xlabel(r"$i$")
        ax.set_ylabel(r"$\sigma_i$")
        ax.set_title("SVD singular values")
        fig.savefig("compare_svd_stream_to_full.png")

    def runForcesFO(self):
        """Post-process reconstructed fields to compute force coefficients.

        Runs `pimpleFoam -postProcess -dict system/FO_force -parallel`
        in the case directory to generate `postProcessing/forces_recon` outputs.
        """
        settings = self.exp.create_run_settings(
            exe="pimpleFoam",
            exe_args=f"-postProcess -dict system/FO_force -parallel",
            run_command="mpirun",
            run_args={"np": f"{self.num_mpi_ranks}"},
        )
        model = self.exp.create_model(
            name="forces_reconstructed_fields",
            run_settings=settings,
            path=self.case_name,
        )
        self.exp.start(model, summary=False, block=False)
        self.wait_for_completion(model)

    def getLiftDragforces(self, force_folder):
        """Read lift/drag coefficient time series from OpenFOAM output.

        Expects `postProcessing/{force_folder}/0/coefficient.dat` with the
        standard header and columns. Returns time, Cd, Cl arrays.

        :param force_folder: Subfolder under `postProcessing/` to read from
                            (e.g., "forces" or "forces_recon").
        :type force_folder: str
        :return: Time, drag coefficient (Cd), lift coefficient (Cl).
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        path = join(
            self.base_sim_path, "postProcessing", force_folder, "0", "coefficient.dat"
        )
        df = read_csv(
            path,
            header=None,
            sep=r"\s+",
            skiprows=13,
            usecols=[0, 1, 4],
            names=["t", "Cd", "Cl"],
        )
        return df["t"].to_numpy(), df["Cd"].to_numpy(), df["Cl"].to_numpy()

    def plotLiftDragforcedComparison(self, save_path="forces_vs_recon.png"):
        """Plot lift/drag comparison: original vs reconstructed.

        Reads coefficients from `forces` and `forces_recon`, plots Cd (left)
        and Cl (right) vs time, and saves the figure instead of showing it.

        :param save_path: Output path for the PNG figure.
        :type save_path: str
        """

        t1, Cd1, Cl1 = self.getLiftDragforces("forces")
        t2, Cd2, Cl2 = self.getLiftDragforces("forces_recon")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        # Drag plot
        axes[0].plot(t1[1:], Cd1[1:], label="forces", lw=1.5)
        axes[0].plot(t2[1:-1], Cd2[1:-1], label="forced_recon", lw=1.5, linestyle="--")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Drag Coefficient (Cd)")
        axes[0].legend()
        axes[0].grid(True)

        # Lift plot
        axes[1].plot(t1[1:], Cl1[1:], label="forces", lw=1.5)
        axes[1].plot(t2[1:-1], Cl2[1:-1], label="forced_recon", lw=1.5, linestyle="--")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Lift Coefficient (Cl)")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)  # close to free memory


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        with open(config_file, "r") as cf:
            cfg = safe_load(cf)
    except Exception as e:
        print(e)

    makedirs(cfg["experiment"]["exp_path"], exist_ok=True)
    case_name = join(cfg["experiment"]["exp_path"], "base_sim")
    exp = Experiment(**cfg["experiment"])
    db = exp.create_database(port=1901, interface="lo")
    exp.start(db)

    config_svd = cfg["svd_params"]

    # number of mpi ranks used in openfoam simulation
    num_mpi_ranks = config_svd["num_mpi_ranks"]
    # batch size for the online eigen SVD
    i, batch_size = config_svd["snapshot_sampling_interval"], config_svd["batch_size"]
    # end time
    n_times = config_svd["end_time"]
    batch_no = 0
    # indices to sample from the database in intervals
    sampling_interval = config_svd["snapshot_sampling_interval"]

    client = Client(address=db.get_address()[0], cluster=False)
    OnlineSVD = OnlineRowPartitionedSVD(cfg, exp, client)

    try:
        base_sim = OnlineSVD.start_openfoam_sim()
        i = OnlineSVD.sampling_interval
        batch_no = 0

        while i + batch_size <= n_times:
            time_index = i + batch_size

            # Wait for data for this batch
            ok = client.poll_list_length(
                f"list_time_index_{time_index}", num_mpi_ranks, 10, 60000
            )
            if not ok:
                raise ValueError("Fields dataset list not updated.")

            time_indices = list(
                range(i, min(i + batch_size + 1, n_times + 1), sampling_interval)
            )

            OnlineSVD.evaluate_increSVD(time_indices, batch_no)
            i += batch_size
            batch_no += 1
            print(f"svd stream done till time step = {i}")

        exp.stop(base_sim)

        # Validation indices
        time_indices_for_data = list(range(sampling_interval, i + 1, sampling_interval))

        # Save factors
        OnlineSVD.save_tensors()
        # Recon + export
        OnlineSVD.plot_singular_values(time_indices_for_data)
        OnlineSVD.run_reconstruction(time_indices_for_data)
        OnlineSVD.run_svdToFoam(fields=["p", "U"])
        OnlineSVD.runForcesFO()
        OnlineSVD.plotLiftDragforcedComparison()
        exp.stop(db)

    except Exception as e:
        print(e)
        exp.stop(db)
        exp.stop(base_sim)
