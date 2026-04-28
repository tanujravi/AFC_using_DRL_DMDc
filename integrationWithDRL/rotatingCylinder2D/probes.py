import os
import glob
from typing import Tuple

import numpy as np

# Assumed to exist somewhere in your code:
# DEFAULT_PROBES_PATH_PATTERN = "postProcessing/probes/*/p"
# PROBES_FILE = None
# PROBES_CACHE = None


class Probes:
    @staticmethod
    def get_pressure_interpolation(
        probe_locations,
        p_field,
        points_xy,
        k=4,
        eps=1e-12,
    ):
        """
        Interpolate pressure values at the given probe locations using
        inverse-distance weighting on a set of scattered 2D points.

        Parameters
        ----------
        probe_locations : list of (x, y, z)
            Probe positions; z is ignored.
        p_field : array-like, shape (n_points,)
            Field values at mesh points (e.g. x0[:n_points, -1]).
        points_xy : array-like, shape (n_points, 2)
            (x, y) coordinates of the mesh points, in the same ordering as p_field.
        k : int, optional
            Number of nearest neighbors for interpolation.
        eps : float, optional
            Small number to avoid division by zero in weights.

        Returns
        -------
        p_sample : np.ndarray, shape (n_probes,)
            Interpolated pressure values at the probe locations.
        """
        points_xy = np.asarray(points_xy)          # (n_points, 2)
        p_field = np.asarray(p_field).ravel()      # (n_points,)
        n_points = points_xy.shape[0]

        if p_field.shape[0] != n_points:
            raise ValueError("p_field and points_xy must have same length")

        # Extract only (x, y) from probe_locations, ignore z
        probe_xy = np.array([(x, y) for (x, y, _z) in probe_locations])
        n_probes = probe_xy.shape[0]
        p_sample = np.empty(n_probes, dtype=float)

        for i in range(n_probes):
            px, py = probe_xy[i]
            dx = points_xy[:, 0] - px
            dy = points_xy[:, 1] - py
            dist2 = dx*dx + dy*dy

            # If we hit a point almost exactly, just take its value
            j0 = np.argmin(dist2)
            if dist2[j0] < 1e-20:
                p_sample[i] = p_field[j0]
                continue

            # Use k nearest neighbors
            k_eff = min(k, n_points)
            idx = np.argpartition(dist2, k_eff-1)[:k_eff]
            w = 1.0 / (dist2[idx] + eps)
            w /= w.sum()
            p_sample[i] = np.dot(w, p_field[idx])

        return p_sample
    @staticmethod
    def find_probes_file(root: str = "./") -> str:
        """
        Find OpenFOAM pressure probes file 'p', using the *latest* time directory
        under postProcessing/probes/<time>/p.
        """
        base = os.path.join(root, "postProcessing", "probes")

        # Try to find the latest numeric time directory
        if os.path.isdir(base):
            time_dirs = [
                d for d in glob.glob(os.path.join(base, "*"))
                if os.path.isdir(d)
            ]

            numeric_dirs = []
            for d in time_dirs:
                name = os.path.basename(d)
                try:
                    t = float(name)
                except ValueError:
                    continue  # skip non-numeric directories
                numeric_dirs.append((t, d))

            if numeric_dirs:
                # pick the directory with the largest time value
                latest_dir = max(numeric_dirs, key=lambda x: x[0])[1]
                latest_path = os.path.join(latest_dir, "p")
                if os.path.isfile(latest_path):
                    print(latest_path)
                    return latest_path



    @staticmethod
    def load_probes_data(probes_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse OpenFOAM probes file into (times, data).

        Expected format:
        #          Time
                   4.01  v1 v2 ... vN
                   4.02  v1 v2 ... vN
        """
        times = []
        rows = []

        with open(probes_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                t_val = float(parts[0])
                vals = [float(v) for v in parts[1:]]
                times.append(t_val)
                rows.append(vals)

        if not times:
            raise RuntimeError(f"No data lines found in probes file: {probes_file}")

        times_arr = np.asarray(times, dtype=float)       # (Nt,)
        data_arr = np.asarray(rows, dtype=float)         # (Nt, Nprobes)
        return times_arr, data_arr

    @staticmethod
    def sample_pressure_for_policy(
        t: float,
        dt: float,
        root: str = "./",
    ) -> np.ndarray:
        """
        Return *last available* probe pressures, ignoring the requested time t.
        t and dt are kept only for API compatibility.
        """
        global PROBES_FILE, PROBES_CACHE

        # Locate probes file (from latest time directory)
        PROBES_FILE = Probes.find_probes_file(root)

        # Load and cache full probes data (times, data)
        PROBES_CACHE = Probes.load_probes_data(PROBES_FILE)

        times, data = PROBES_CACHE  # times: (Nt,), data: (Nt, Nprobes)

        # We only want the last time-step
        last_values = data[-1, :]  # (Nprobes,)

        return last_values.copy()