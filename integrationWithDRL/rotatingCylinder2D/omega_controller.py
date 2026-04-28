import numpy as np
import torch as pt
from dataclasses import dataclass
from pathlib import Path
import csv

# ----------------- state container ----------------- #

@dataclass
class OmegaState:
    omega: float = 0.0       # current target omega_
    omega_old: float = 0.0   # previous omega_old_
    control_time: float = 0.0
    update_omega: bool = True   # corresponds to update_omega_ in C++


# ----------------- controller ----------------- #

class OmegaController:
    """
    Python version of agentRotatingWallVelocityFvPatchVectorField control logic.

    You call `compute(t, dt, p_sample)` each time step and get the
    *smoothly ramped* omega(t) back.
    """

    def __init__(
        self,
        policy_path: str,
        abs_omega_max: float,
        dt_control: float,
        start_time: float = 0.0,
        train: bool = True,
        seed: int = 0,
        trajectory_file: str = "trajectory.csv",
    ):
        # load TorchScript policy (same as C++ torch::jit::load)
        self.policy = pt.jit.load(policy_path)
        self.policy.eval()

        self.abs_omega_max = float(abs_omega_max)
        self.dt_control = float(dt_control)
        self.start_time = float(start_time)
        self.train = bool(train)
        self.rng = np.random.default_rng()
        self.state = OmegaState()
        self.trajectory_file = trajectory_file

    # --------- internals ---------- #

    def _ensure_trajectory_header(self):
        path = Path(self.trajectory_file)
        if not path.exists():
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "omega", "alpha", "beta"])

    def _save_trajectory(self, t, omega, alpha, beta):
        self._ensure_trajectory_header()
        with open(self.trajectory_file, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"{t:.15g}", f"{omega:.15g}", f"{alpha:.15g}", f"{beta:.15g}"])

    # --------- main step function ---------- #

    @pt.no_grad()
    def compute(self, t: float, dt: float, p_sample: np.ndarray) -> float:
        """
        Parameters
        ----------
        t : float
            Current simulation time (OpenFOAM timeOutputValue()).
        dt : float
            Current time step (deltaTValue()).
        p_sample : np.ndarray
            1D or 2D array with pressure samples (what probes_.sample(p) returns).

        Returns
        -------
        omega_now : float
            Angular velocity at time t (with linear ramp applied).
        """

        t = float(t)
        dt = float(dt)

        # Flatten pressure samples to match C++ features = from_blob(p_sample.data(), {1, p_sample.size()})
        p_flat = np.asarray(p_sample, dtype=np.float64).ravel()
        features = pt.from_numpy(p_flat[None, :])  # shape (1, N)

        s = self.state

        # --- corresponds to `timeToControl` in C++ --- #
        # control_.execute() is "time to act according to probes dict"
        # Here we simply control every dt_control, starting at ~start_time
        time_since_last = t - s.control_time
        time_to_control = (
            t >= self.start_time - 0.5 * dt and
            (time_since_last >= self.dt_control - 0.5 * dt)
        )

        time_to_control = True
        # --- if it's time to change omega and we're allowed to --- #
        if time_to_control: #and s.update_omega:
            s.omega_old = s.omega
            s.control_time = t

            # policy_.forward(policyFeatures)
            dist_parameters = self.policy(features).to(pt.float64)
            #print(dist_parameters)
            alpha = float(dist_parameters[0, 0].item())
            beta = float(dist_parameters[0, 1].item())

            # --- same Beta sampling logic as C++ --- #
            if self.train:
                n1 = self.rng.gamma(alpha, 1.0)
                n2 = self.rng.gamma(beta, 1.0)
                omega_pre_scale = n1 / (n1 + n2)
            else:
                omega_pre_scale = alpha / (alpha + beta)

            # rescale [0,1] -> [-abs_omega_max, +abs_omega_max]
            s.omega = (omega_pre_scale - 0.5) * 2.0 * self.abs_omega_max

            # trajectory.csv logging
            self._save_trajectory(t, s.omega, alpha, beta)

            # avoid updating within p-U coupling
            s.update_omega = False
        
        """
        # --- "activate update of angular velocity after p-U coupling" --- #
        if (not time_to_control) and (not s.update_omega):
            s.update_omega = True
        """
        # --- linear transition from old to new over dt_control_ --- #
        # d_omega = (omega_ - omega_old_) / dt_control_ * (t - control_time_)
        #d_omega = (s.omega - s.omega_old) / self.dt_control * (t - s.control_time)
        #omega_now = s.omega_old + d_omega

        omega_now = s.omega
        return float(omega_now)