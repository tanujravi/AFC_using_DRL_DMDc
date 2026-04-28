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
import numpy as np
import os
import sys, types
import shutil
from flowtorch.data import FOAMDataloader
from omega_controller import OmegaController

# Map old private module name → current public one
import numpy.core.numeric as numeric
shim = types.ModuleType("numpy._core.numeric")
shim.__dict__.update(numeric.__dict__)
sys.modules["numpy._core.numeric"] = shim

@staticmethod
def _delay_embed_state(X, q):
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
    C = T - (q - 1)
    if C <= 0:
        raise ValueError(
            f"Not enough samples for requested delays: q={q}, T={T}"
        )
    blocks = [X[:,d: d + C] for d in range(q)]
    return np.vstack(blocks)  # (f*q, C)

def time_march_dmdc(dmdc_model, u_seq, x0):
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
    
    q = dmdc_model["q"]

    fq = dmdc_model["basis"].shape[0]
    f = dmdc_model["U_dr"].shape[1]
    if x0.shape[0] == fq:
        tilt_x0 = x0  # already augmented

    elif x0.shape[0] == f and x0.shape[1] > 1:
        tilt_x0 = dmdc_model["U_dr"].T @ x0   # build augmented
        tilt_x0 = tilt_x0.reshape(-1,1, order= 'F')

    else:
        raise ValueError(
            f"x0 has {x0.shape[0]} rows; expected {f} (original) or {fq} (augmented)."
        )

    mu_expected = dmdc_model["Btilde"].shape[1]     # rows expected for control at rollout
    # normalize controls to (m, T-1)
    U = np.asarray(u_seq)
    if U.shape[0] == mu_expected:
        Ud = U
    else:
        Ud = _delay_embed_state(U, q)

    if Ud.shape[0] != mu_expected:
        raise ValueError(
            f"After delay embedding, control rows={Ud.shape[0]} "
            f"but Btilde expects {mu_expected}"
        )

    Phi = dmdc_model["basis"]  # (n, r)
    Atilde = dmdc_model["Atilde"]  # (r, r)
    Btilde = dmdc_model["Btilde"]  # (r, m)

    # Reduced initial state
    x_red = Phi.T @ x0  # (r,1)

    # Rollout in reduced space
    Tm1 = Ud.shape[1]  # number of transitions
    z_cols = [x_red]
    for k in range(Tm1):
        u_k = Ud[:, k : k + 1]  # (m,1)
        x_red = Atilde @ x_red + Btilde @ u_k
        z_cols.append(x_red)
    Z = np.hstack(z_cols)  # (r, T)

    # Lift back to full space
    X_aug = Phi @ Z  # (n, T)

    rank_dr = dmdc_model["U_dr"].shape[1] 
    X_pred = dmdc_model["U_dr"] @ X_aug[rank_dr:]
    return X_pred


def wait_for_completion(exp, entities, poll_interval=5, timeout=None):

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
        statuses = exp.get_status(entities)
        if all(s == SmartSimStatus.STATUS_COMPLETED for s in statuses): break
        if any(s == SmartSimStatus.STATUS_FAILED for s in statuses):
            raise RuntimeError("One or more entities failed.")
        if timeout and (time.time() - start) > timeout:
            raise TimeoutError("Timeout waiting for SmartSim entities to complete.")
        time.sleep(poll_interval)



def split_1d_indices(N: int, n: int):
    """OpenFOAM-simple split of N rows into n contiguous chunks.
    Returns list of (start, stop) indices (stop exclusive)."""
    q, r = divmod(N, n)
    sizes = [(q + 1) if i < r else q for i in range(n)]
    starts = np.cumsum([0] + sizes[:-1]).tolist()
    return [(s, s + sz) for s, sz in zip(starts, sizes)]

def tensor_key(field: str, svd_rank: int, mpi_rank: int, time_index: int):
    """Match your exact naming convention."""
    return (
        f"rec_ensemble_r{svd_rank}_field_name_{field}_{mpi_rank}"
        f".rank_{svd_rank}_field_name_{field}_mpi_rank_{mpi_rank}"
        f"_time_index_{time_index}"
    )

def put_scalar_series(client, base_field: str, arr: np.ndarray, svd_rank: int,
                      nprocs: int, time_stride: int):
    """Upload scalar field per-rank, per-time (arr: [n_points, n_times])."""
    n_points, n_times = arr.shape
    splits = split_1d_indices(n_points, nprocs)
    arr = np.asarray(arr, dtype=np.float64, order="C")

    for rank, (s, e) in enumerate(splits):
        for i in range(n_times):
            ti = (i + 1) * time_stride
            key = tensor_key(base_field, svd_rank, rank, ti)
            # slice is contiguous in the first axis; copy to be safe for the client
            vec = np.copy(arr[s:e, i])
            client.put_tensor(key, vec)

def put_vector_series(client, base_field: str, ux: np.ndarray, uy: np.ndarray,
                      uz: np.ndarray | None, svd_rank: int, nprocs: int,
                      time_stride: int):
    """Upload vector field U per-rank, per-time (each input: [n_points, n_times])."""
    assert ux.shape == uy.shape, "u_x and u_y shape mismatch"
    n_points, n_times = ux.shape
    if uz is None:
        uz = np.zeros_like(ux, dtype=np.float64)
    else:
        assert uz.shape == ux.shape, "u_z shape mismatch"

    splits = split_1d_indices(n_points, nprocs)

    # Ensure dtype float64
    ux = np.asarray(ux, dtype=np.float64, order="C")
    uy = np.asarray(uy, dtype=np.float64, order="C")
    uz = np.asarray(uz, dtype=np.float64, order="C")

    for rank, (s, e) in enumerate(splits):
        for i in range(n_times):
            ti = (i + 1) * time_stride
            # stack into (n_chunk, 3)
            vec = np.stack([ux[s:e, i], uy[s:e, i], uz[s:e, i]], axis=1)
            key = tensor_key(base_field, svd_rank, rank, ti)
            client.put_tensor(key, np.copy(vec))

def build_x0(q, reference_dic):
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
    loader = FOAMDataloader("./", distributed=True)
    times = loader.write_times  # ensure ndarray for search

    times_arr = np.array(loader.write_times)
    
    # Find the first index whose time >= start_time


    times_used = times[-q:]

    u_inlet = float(reference_dic["u_inlet"])

    # ---- Pressure (mean-subtracted over space; normalized by u_inlet^2) ----
    p_snap = np.asarray(loader.load_snapshot("p", times_used), dtype=np.float32)
    # subtract spatial mean per time step
    p_shifted = (p_snap) / (u_inlet**2)

    # ---- Velocity components (mean-subtracted over space; normalized by u_inlet) ----
    U_snap = np.asarray(loader.load_snapshot("U", times_used), dtype=np.float32)

    # ux
    ux = U_snap[:, 0, :] / u_inlet

    # uy
    uy = U_snap[:, 1, :] / u_inlet

    components = [p_shifted, ux, uy]

    field_names = ["p", "u_x", "u_y"]

    if reference_dic["dim"].lower() == "3d":
        # uz (assumes third component exists)
        uz = U_snap[:, 2, :] / u_inlet
        components.append(uz)
        field_names.append("u_z")

    return np.vstack(components).astype(np.float32)


def get_omega():
    pass


svd_rank = 20
mpi_ranks = 2                  # total MPI ranks
time_stride = 20            # matches your prior (i+1)*20 convention
fo_name = "dataToSmartRedis"
template_dir = "template"
parent_dir = join("run", "cylinder")
os.makedirs(parent_dir)

exp = Experiment(name = "cylinder", exp_path= parent_dir, launcher = "local")
db = exp.create_database(port=1993, interface="lo")
exp.start(db)
client = Client(address=db.get_address()[0], cluster=False)

import pickle, gzip
path_data = "dmdc.pkl.gz"

with gzip.open(path_data, "rb") as f:
    dmdc_model = pickle.load(f)

referece_dic = {}
#Load the initial state vector (with time delay)
def get_field_block(X, field_idx):
    n = dmdc_model["U_dr"].shape[0]
    r0 = field_idx * n
    r1 = (field_idx + 1) * n
    return X[r0:r1, :]

x0 = build_x0(q=dmdc_model["q"], reference_dic= referece_dic)
pad_cols = x0.shape[1]         # same number of columns as x0
U = np.zeros((1, pad_cols))

policy_path   = "policy.pt"                  # dict.get<word>("policy")
abs_omega_max = 50.0                         # dict.get<scalar>("absOmegaMax")
seed          = 1234                         # dict.get<int>("seed")
train         = True                         # dict.get<bool>("train")
time_start    = 0.0                          # probesDict.getOrDefault("timeStart", 0.0)
dt_control    = 0.1                          # probesDict.getOrDefault("executeInterval", 0.1)

omega_controller = OmegaController(
    policy_path   = policy_path,
    abs_omega_max = abs_omega_max,
    dt_control    = dt_control,
    start_time    = time_start,
    train         = train,
    seed          = seed,
    trajectory_file="trajectory.csv",
)


def get_omega(t, dt, p_sample):
    """Thin wrapper to match your old C++ semantics."""
    return omega_controller.compute(t, dt, p_sample)

while t < t_end:

    tilt_x0 = dmdc_model["U_dr"].T @ x0   # build augmented
    tilt_x0 = tilt_x0.reshape(-1,1, order= 'F')

    fields_dict = {}
    omega_act = get_omega()
    U_act = referece_dic["radius"] * omega_act
    U = np.roll(U, -1, axis=1)
    U[-1] = U_act
    X_full = dmdc_model["Atilde"] @ x0 + dmdc_model["Btilde"] @ U
    X_pred = dmdc_model["U_dr"] @ X_full[-dmdc_model["svd_rank"]:]
    x0 = np.roll(x0, -1, axis = 1)
    x0[:-1] = X_pred
    for f_idx, f_name in enumerate(dmdc_model["field_names"]):
        fields_dict[f_name]  = get_field_block(X_pred, f_idx)                          # (n_pts x usable_T)



    # Detect presence of vector components
    has_ux = "u_x" in fields_dict
    has_uy = "u_y" in fields_dict
    has_uz = "u_z" in fields_dict

    # If vector components exist, upload U
    if has_ux and has_uy:
        rec_x = np.asarray(fields_dict["u_x"], dtype=np.float64)
        rec_y = np.asarray(fields_dict["u_y"], dtype=np.float64)
        rec_z = (np.asarray(fields_dict["u_z"], dtype=np.float64)
                if has_uz else None)
        put_vector_series(client, "U", rec_x, rec_y, rec_z, svd_rank, mpi_ranks, time_stride)

    # Upload all scalar fields (anything that's not a vector component)
    for field_name, arr in fields_dict.items():
        if field_name in {"u_x", "u_y", "u_z"}:
            continue  # handled above as U
        rec = np.asarray(arr, dtype=np.float64)
        if rec.ndim != 2:
            raise ValueError(f"Expected 2D array for field '{field_name}' (points x times), got shape {rec.shape}")
        put_scalar_series(client, field_name, rec, svd_rank, mpi_ranks, time_stride)

    control_dict = os.path.join("system", "controlDict")
    os.system(f"foamDictionary {control_dict} -entry startTime  -set {t}")
    os.system(f"foamDictionary 0.org/U -entry \"boundaryField.cylinder.omega\" -set {omega_act}")

    fields = ["p", "U"]

    for field in fields:
        settings = exp.create_run_settings(
            exe="svdToFoam",
            exe_args=f"-fieldName {field} -svdRank {svd_rank} "
                    f"-FOName {fo_name} -parallel",
            run_command="mpirun", run_args={"np": f"{mpi_ranks}"}
        )
        model = exp.create_model(name=f"svdToFoam_r{svd_rank}_field_name_{field}",
                                    run_settings=settings, path="./")
        exp.start(model, summary=False, block=False)
        wait_for_completion(exp, model)

    settings = exp.create_run_settings(
        exe="pimpleFoam",
        exe_args=f"-postProcess -dict system/FO_force -parallel",
        run_command="mpirun", run_args={"np": f"{mpi_ranks}"}
    )
    model = exp.create_model(name=f"forces_reconstructed_fields",
                                run_settings=settings, path="./")
    exp.start(model, summary=False, block=False)
    wait_for_completion(exp, model)

exp.stop(db)
