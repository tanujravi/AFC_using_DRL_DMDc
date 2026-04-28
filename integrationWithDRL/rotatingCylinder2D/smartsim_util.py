import numpy as np
import torch as pt
from yaml import safe_load
from os.path import join
from smartsim import Experiment
from smartsim.settings import RunSettings
from smartsim.status import SmartSimStatus
from smartredis import Client
from typing import Optional, Dict, List, Tuple
import time


class smartSimUtil:

    @staticmethod
    def wait_for_completion(exp: Experiment, entities, poll_interval=5, timeout=None):
        """Block until all entities in a SmartSim experiment complete."""
        start = time.time()
        while True:
            statuses = exp.get_status(entities)
            if all(s == SmartSimStatus.STATUS_COMPLETED for s in statuses):
                break
            if any(s == SmartSimStatus.STATUS_FAILED for s in statuses):
                raise RuntimeError("One or more entities failed.")
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError("Timeout waiting for SmartSim entities to complete.")
            time.sleep(poll_interval)

    @staticmethod
    def split_1d_indices(N: int, n: int):
        """Split N rows into n contiguous chunks. Returns list of (start, stop)."""
        q, r = divmod(N, n)
        sizes = [(q + 1) if i < r else q for i in range(n)]
        starts = np.cumsum([0] + sizes[:-1]).tolist()
        return [(s, s + sz) for s, sz in zip(starts, sizes)]

    @staticmethod
    def tensor_key(field: str, svd_rank: int, mpi_rank: int, time_index: int) -> str:
        """Match dataToSmartRedis naming convention."""
        return (
            f"rec_ensemble_r{svd_rank}_field_name_{field}_{mpi_rank}"
            f".rank_{svd_rank}_field_name_{field}_mpi_rank_{mpi_rank}"
            f"_time_index_{time_index}"
        )

    @staticmethod
    def put_scalar_series(
        client: Client,
        base_field: str,
        arr: np.ndarray,
        svd_rank: int,
        nprocs: int,
        time_stride: int,
    ):
        """Upload scalar field per-rank, per-time (arr: [n_points, n_times])."""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        n_points, n_times = arr.shape
        splits = smartSimUtil.split_1d_indices(n_points, nprocs)
        arr = np.asarray(arr, dtype=np.float64, order="C")

        for rank, (s, e) in enumerate(splits):
            for i in range(n_times):
                ti = (i + 1) * time_stride + 8000
                key = smartSimUtil.tensor_key(base_field, svd_rank, rank, ti)
                vec = np.copy(arr[s:e, i])
                client.put_tensor(key, vec)

    @staticmethod
    def put_vector_series(
        client: Client,
        base_field: str,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: Optional[np.ndarray],
        svd_rank: int,
        nprocs: int,
        time_stride: int,
    ):
        """Upload vector field U per-rank, per-time (each input: [n_points, n_times])."""
        ux = np.asarray(ux)
        uy = np.asarray(uy)
        assert ux.shape == uy.shape, "u_x and u_y shape mismatch"
        n_points, n_times = ux.shape

        if uz is None:
            uz = np.zeros_like(ux, dtype=np.float64)
        else:
            uz = np.asarray(uz)
            assert uz.shape == ux.shape, "u_z shape mismatch"

        splits = smartSimUtil.split_1d_indices(n_points, nprocs)

        ux = np.asarray(ux, dtype=np.float64, order="C")
        uy = np.asarray(uy, dtype=np.float64, order="C")
        uz = np.asarray(uz, dtype=np.float64, order="C")

        for rank, (s, e) in enumerate(splits):
            for i in range(n_times):
                ti = (i + 1) * time_stride + 8000
                vec = np.stack([ux[s:e, i], uy[s:e, i], uz[s:e, i]], axis=1)
                key = smartSimUtil.tensor_key(base_field, svd_rank, rank, ti)
                client.put_tensor(key, np.copy(vec))

    @staticmethod
    def upload_fields_to_database(
        client: Client,
        fields_dict: Dict[str, np.ndarray],
        field_names: List[str],
        svd_rank: int,
        mpi_ranks: int,
        time_stride: int,
    ):
        """Upload scalar and vector fields for a single time step to SmartRedis."""
        has_ux = "u_x" in fields_dict
        has_uy = "u_y" in fields_dict
        has_uz = "u_z" in fields_dict

        #time_index = 
        # Vector field U
        if has_ux and has_uy:
            ux = np.asarray(fields_dict["u_x"], dtype=np.float64)
            uy = np.asarray(fields_dict["u_y"], dtype=np.float64)
            uz = (
                np.asarray(fields_dict["u_z"], dtype=np.float64)
                if has_uz
                else None
            )
            smartSimUtil.put_vector_series(
                client, "U", ux, uy, uz, svd_rank, mpi_ranks, time_stride
            )

        # Scalar fields (everything except velocity components)
        for fname, arr in fields_dict.items():
            if fname in {"u_x", "u_y", "u_z"}:
                continue
            rec = np.asarray(arr, dtype=np.float64)
            if rec.ndim != 2:
                raise ValueError(
                    f"Expected 2D array for field '{fname}' (points x times), got shape {rec.shape}"
                )
            smartSimUtil.put_scalar_series(client, fname, rec, svd_rank, mpi_ranks, time_stride)

