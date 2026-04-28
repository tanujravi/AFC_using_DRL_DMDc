from flowtorch.data import FOAMDataloader
import numpy as np
import pickle
from dmdc_util import dmdcUtil
import random



#path = "../initial_states"
loader = FOAMDataloader("./", distributed=True)
times = loader.write_times
q = 100
if len(times) < q:
    raise ValueError(f"Not enough write times ({len(times)}) for q={q}")
times_used = times[-q:]
print(times_used)
u_inlet = 1
dim = "2d"
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
"""
n_steps = 300
X_pred = []


path_signal = "models_lib/chirp_varying_amp_dmdc.pkl"
with open(path_signal, "rb") as f:
    dmdc_model_chirp_varying_amp = pickle.load(f)

path_signal = "models_lib/chirp_dmdc.pkl"
with open(path_signal, "rb") as f:
    dmdc_model_chirp = pickle.load(f)

path_signal = "models_lib/random_walk_dmdc.pkl"
with open(path_signal, "rb") as f:
    dmdc_model_random_walk = pickle.load(f)


path_signal = "models_lib/AM_dmdc.pkl"
with open(path_signal, "rb") as f:
    dmdc_model_AM = pickle.load(f)

model_lib = [dmdc_model_random_walk, dmdc_model_chirp, dmdc_model_chirp_varying_amp, dmdc_model_AM]
#model_lib = [dmdc_model_chirp_varying_amp]
path_data = "models_lib/common_Ur.pkl" 
with open(path_data, "rb") as f:
    common_Ur = pickle.load(f)

n_shuf = 1
pad_cols = X0.shape[1]
U_hist = np.zeros((1, pad_cols), dtype=float)
x_roll = X0.copy()
dmdc_model = random.choice(model_lib)          # initial model

for step in range(n_steps):

    # --- RL: sample probes and compute control --- #
    U_act = 0  # tangential wall speed

    # --- update control history buffer --- #
    U_hist = np.roll(U_hist, -1, axis=1)
    U_hist[:, -1] = U_act

    # --- DMDc one-step --- #
    if step % n_shuf == 0:
        dmdc_model = random.choice(model_lib)

    x_next = dmdcUtil.dmdc_step(dmdc_model, x_roll, U_hist, common_Ur=common_Ur)

    # update state history (sliding window)
    x_roll = np.roll(x_roll, -1, axis=1)
    x_roll[:, -1:] = x_next

    # state for this step (just x_next)

    X_pred.append(x_next)


X_pred = np.concatenate(X_pred, axis=1)
"""
data = (X0, field_names, times_used, vertices[:, 0:2])

#data = (X_pred[:, -q:], field_names, times_used, vertices[:, 0:2])


# write to pickle
with open("../initial_states.pkl", "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)