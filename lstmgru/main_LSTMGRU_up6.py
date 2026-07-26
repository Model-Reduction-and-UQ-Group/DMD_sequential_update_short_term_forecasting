"""
GRACE basin-mean monthly forecasting (GRU / LSTM).

"""

import os
import time
import random
from itertools import product
import numpy as np
import torch
import matplotlib.pyplot as plt
from NN import IntegratedModel   
import pandas as pd
from scipy.io import savemat


# ====================================
# parameters
# ====================================
TRAINVAL = 241   # months 0..240 = train+val, test starts at 241
X_STEP = 6       # retraining step
basin = "Po"


# ====================================
# environment / reproducibility
# ====================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================
# hyper-parameters
# ====================================
batch_size    = 32
learning_rate = 1e-3
step_size     = 100
gamma         = 0.5
epochs        = 200

bw = 96          # lookback (months)
fw = X_STEP      # horizon (months)
dim_input_all         = [bw]
sampling_interval_all = [1]
dim_hidden_all        = [16]

# only GRU and LSTM (attention and SSM blocks forced to 0)
gl = [c for c in product(range(3), repeat=2) if c != (0, 0)]
num_blocks_all = [[g, l, 0, 0] for (g, l) in gl]      # 8 combinations

DIM_IN = 4   # [anom_norm, mask, sin_month, cos_month]
DIM_SEQ = len(range(0, bw, sampling_interval_all[0]))


# ======================================================================
# helpers
# ======================================================================
def build_monthly_grid(values, instants):
    """Two raw vectors -> regular monthly grid (np.nan in the voids).
    instants: datetime64 / strings / decimal years / YYYYMM ints."""
    values = np.asarray(values, dtype=np.float64)
    inst = np.asarray(instants)
    assert len(values) == len(inst), \
        f"values ({len(values)}) and instants ({len(inst)}) differ in length"
    k = inst.dtype.kind
    if k in ("M", "U", "S", "O"):                    # dates / strings
        ms = inst.astype("datetime64[M]").astype(int)
        year, mon0 = 1970 + ms // 12, ms % 12
    elif k == "f":                                   # decimal years
        year = np.floor(inst).astype(int)
        mon0 = np.floor((inst - year) * 12 + 1e-9).astype(int)
    else:                                            # YYYYMM integers
        year, mon0 = inst // 100, inst % 100 - 1

    abs_month = year * 12 + mon0
    start_abs = int(abs_month.min())
    idx = abs_month - start_abs
    N = int(idx.max()) + 1

    sums = np.zeros(N); counts = np.zeros(N)
    np.add.at(sums, idx, values)                     # duplicates -> averaged
    np.add.at(counts, idx, 1)
    series = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)

    start_year, start_month = start_abs // 12, start_abs % 12 + 1
    month = ((np.arange(N) + (start_month - 1)) % 12) + 1
    return series, month, int(start_year), int(start_month)


def train_climatology(series, month, train_end):
    """Monthly climatology from TRAINING observations only."""
    obs = ~np.isnan(series)
    gmean = np.nanmean(series[:train_end])
    clim = np.zeros(12)
    for m in range(1, 13):
        sel = (month == m) & obs
        sel[train_end:] = False
        clim[m - 1] = series[sel].mean() if sel.any() else gmean
    return clim


def impute_anomaly(series, seas):
    """Fill voids by linear interpolation of the ANOMALY (series - seas).
    Returns filled anomaly + observation mask."""
    anom = series - seas
    idx = np.arange(len(series))
    ok = ~np.isnan(anom)
    anom_filled = np.interp(idx, idx[ok], anom[ok])
    return anom_filled.astype(np.float32), ok.astype(np.float32)


def masked_mse(pred, target, mask):
    se = (pred - target) ** 2 * mask
    return se.sum() / mask.sum().clamp(min=1)


def build_windows(anom_norm, obs_mask, sin_m, cos_m, tot,
                   bw_, fw_, step, lo, hi):
    """Builds the (X, Y, M, target indices) windows for a given split."""
    X, Y, M, TI = [], [], [], []
    for t0 in range(0, tot - bw_ - fw_ + 1):
        tgt = t0 + bw_
        if not (lo <= tgt < hi):
            continue
        si = slice(t0, t0 + bw_, step)
        so = slice(tgt, tgt + fw_)
        X.append(np.stack([anom_norm[si], obs_mask[si],
                            sin_m[si], cos_m[si]], axis=-1))
        Y.append(anom_norm[so])
        M.append(obs_mask[so])
        TI.append(np.arange(tgt, tgt + fw_))
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    return t(X), t(Y), t(M), np.array(TI, dtype=int).reshape(-1, fw_)


def make_model(num_blocks):
    torch.manual_seed(SEED)
    return IntegratedModel(
        dim_in=DIM_IN,
        GRU_blocks=num_blocks[0], LSTM_blocks=num_blocks[1],
        Atten_blocks=num_blocks[2], SSM_blocks=num_blocks[3],
        d_intermediate=32, d_hidden=dim_hidden_all[0], d_outfeature=fw,
        dim_atten=64, n_heads=1,
        dim_seq=DIM_SEQ,
    ).to(device)


def train_model(model, Xtr, Ytr, Mtr, Xva_d, Yva_d, Mva_d):
    """Standard training loop (200 epochs, best-on-validation checkpoint)."""
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr, Mtr),
        batch_size=batch_size, shuffle=True)

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            loss = masked_mse(model(x), y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        scheduler.step()
        if len(Xva_d):
            model.eval()
            with torch.no_grad():
                vl = masked_mse(model(Xva_d), Yva_d, Mva_d).item()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_state


def to_physical(pred_norm, tgt_idx, sd_a, mu_a, seas):
    """normalized anomaly -> physical units: de-norm + add climatology."""
    return pred_norm * sd_a + mu_a + seas[tgt_idx]


def reconstruct_with_fixed_model(model, anom_norm, obs_mask, sin_m, cos_m, tot,
                                  bw_, fw_, step, lo, hi, sd_a, mu_a, seas):
    """Reconstructs a time series (of total length, with NaN values where 
    reconstruction is not possible) by applying a fixed-weight model to sliding 
    windows with target indices in [lo, hi). When multiple windows cover 
    the same time step, the predictions are averaged."""
    Xr, Yr, Mr, TIr = build_windows(anom_norm, obs_mask, sin_m, cos_m, tot,
                                     bw_, fw_, step, lo, hi)
    recon = np.full(tot, np.nan, dtype=float)
    if len(Xr) == 0:
        return recon, Xr, TIr
    with torch.no_grad():
        pred = model(Xr.to(device)).cpu().numpy()
    pred_phys = to_physical(pred, TIr, sd_a, mu_a, seas)  # (n_win, fw_)
    rsum = np.zeros(tot); rcnt = np.zeros(tot)
    for row, tgt_idx in enumerate(TIr):
        rsum[tgt_idx] += pred_phys[row]
        rcnt[tgt_idx] += 1
    valid = rcnt > 0
    recon[valid] = rsum[valid] / rcnt[valid]
    return recon, Xr, TIr


def matlab_datenum(dt64_array):
    """np.datetime64 array -> MATLAB datenum (compatible with datestr/datetime)."""
    unix_epoch = np.datetime64("1970-01-01T00:00:00")
    days = (np.asarray(dt64_array).astype("datetime64[s]") - unix_epoch) / np.timedelta64(1, "D")
    return days.astype(float) + 719529.0   # datenum('1970-01-01') in MATLAB


def save_npy_and_mat(base_path, var_name, array):
    """save same array as .npy (Python) and as .mat (MATLAB)."""
    np.save(base_path + ".npy", array)
    savemat(base_path + ".mat", {var_name: array})


# ======================================================================
# data: pixel map + instants -> UNICA serie media del bacino
# ======================================================================
pixel_map = np.load("grace/1_dati/GRACE_basins/" + basin + "_pixels_time.npy")   # (P, T)
instants_full = np.load("grace/1_dati/GRACE_basins/instants.npy")                                                  # (T,)

current_directory = os.getcwd()
case_dir = current_directory + "/grace/2_codici/LSTM_GRU/MeanBasin_rec"+str(X_STEP)+"/" + basin + "/"
os.makedirs(case_dir, exist_ok=True)
save_models_to = case_dir + "model/"
os.makedirs(save_models_to, exist_ok=True)

basin_row_mean = np.nanmean(pixel_map, axis=0)          # (T,) pixels mean
valid_arch = ~np.isnan(basin_row_mean)
values = basin_row_mean[valid_arch]
instants = instants_full                              

series, month, start_year, start_month = build_monthly_grid(values, instants)
tot = len(series)
d0 = np.datetime64(f"{start_year}-{start_month:02d}")
grid_dates = d0 + np.arange(tot)
print(f"grid (basin mean): {tot} months from {start_year}-{start_month:02d}, "
      f"{int(np.isnan(series).sum())} voids")


# ======================================================================
# 1) ARCHITECTURE SEARCH + STEP 0, performed only once on the [0, TRAINVAL) split.
#    The selected model is directly used as step 0 of the recursive procedure,
#    avoiding redundant retraining on the same split.
# ======================================================================
print("\n################ architecture search + step0 (basin mean) ################")

trainval_k   = TRAINVAL
train_end_k  = int(trainval_k * 0.85)
val_start_k  = train_end_k
val_end_k    = trainval_k
test_start_k = trainval_k

clim = train_climatology(series, month, train_end_k)
seas = clim[month - 1].astype(np.float32)
anom_filled, obs_mask = impute_anomaly(series, seas)
tr_obs = obs_mask[:train_end_k].astype(bool)
mu_a = float(anom_filled[:train_end_k][tr_obs].mean())
sd_a = float(anom_filled[:train_end_k][tr_obs].std() + 1e-8)
anom_norm = (anom_filled - mu_a) / sd_a
sin_m = np.sin(2 * np.pi * month / 12).astype(np.float32)
cos_m = np.cos(2 * np.pi * month / 12).astype(np.float32)

Xtr, Ytr, Mtr, _    = build_windows(anom_norm, obs_mask, sin_m, cos_m, tot,
                                     bw, fw, sampling_interval_all[0],
                                     0, train_end_k)
Xva, Yva, Mva, _    = build_windows(anom_norm, obs_mask, sin_m, cos_m, tot,
                                     bw, fw, sampling_interval_all[0],
                                     val_start_k, val_end_k)
Xte, Yte, Mte, TIte = build_windows(anom_norm, obs_mask, sin_m, cos_m, tot,
                                     bw, fw, sampling_interval_all[0],
                                     test_start_k, test_start_k + 1)
print(f"windows -> train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}")

Xva_d, Yva_d, Mva_d = Xva.to(device), Yva.to(device), Mva.to(device)

best_arch = None   # {'l':..., 'val':..., 'num_blocks':..., 'state':..., 'pred':...}
for l, num_blocks in enumerate(num_blocks_all):
    model = make_model(num_blocks)
    model, val_loss, state = train_model(model, Xtr, Ytr, Mtr, Xva_d, Yva_d, Mva_d)
    print(f"  arch {l} {num_blocks} -> val_loss={val_loss:.5f}")
    if best_arch is None or val_loss < best_arch["val"]:
        with torch.no_grad():
            pred = model(Xte.to(device)).cpu().numpy() if len(Xte) else None
        best_arch = dict(l=l, val=val_loss, num_blocks=num_blocks, state=state, pred=pred)

best_architecture = best_arch["num_blocks"]
best_architecture_id = best_arch["l"]
np.save(case_dir + "best_architecture_id.npy", best_architecture_id)
print(f"Architettura scelta: id={best_architecture_id}, blocks={best_architecture}")

torch.save(best_arch["state"], save_models_to + "model_step0.pt")

# ------------------------------------------------------------------
# Reconstruct the training and validation time series using the
# step-0 architecture (fixed weights, before any recursive retraining).
# Sliding windows are used with target indices in [bw, train_end_k) for
# the training set and in [val_start_k, val_end_k) for the validation
# set. When multiple windows cover the same time step (because fw > 1),
# the corresponding predictions are averaged.
# ------------------------------------------------------------------
model_step0 = make_model(best_architecture)
model_step0.load_state_dict(best_arch["state"])
model_step0.eval()

train_reconstruction, Xrec, TIrec = reconstruct_with_fixed_model(
    model_step0, anom_norm, obs_mask, sin_m, cos_m, tot,
    bw, fw, sampling_interval_all[0], bw, train_end_k, sd_a, mu_a, seas)
print(f"windows ricostruzione train (arch. step0, pesi fissi) -> {len(Xrec)}")

val_reconstruction, Xvarec, TIvarec = reconstruct_with_fixed_model(
    model_step0, anom_norm, obs_mask, sin_m, cos_m, tot,
    bw, fw, sampling_interval_all[0], val_start_k, val_end_k, sd_a, mu_a, seas)
print(f"windows ricostruzione val (arch. step0, pesi fissi) -> {len(Xvarec)}")

save_npy_and_mat(case_dir + "basin_train_reconstruction_step0",
                  "train_reconstruction_step0", train_reconstruction)
save_npy_and_mat(case_dir + "basin_val_reconstruction_step0",
                  "val_reconstruction_step0", val_reconstruction)

save_npy_and_mat(case_dir + "basin_series_observed", "series_observed", series)

pred_phys = to_physical(best_arch["pred"], TIte, sd_a, mu_a, seas).ravel()
true_phys = to_physical(Yte.numpy(), TIte, sd_a, mu_a, seas).ravel()
mk = Mte.numpy().ravel()
test_dates_k = grid_dates[TIte[0]]
err = (pred_phys - true_phys) * mk
rmse_k = np.sqrt((err ** 2).sum() / max(mk.sum(), 1))

recursive_steps = [dict(
    step=0,
    best_l=best_architecture_id,
    best_val=best_arch["val"],
    rmse=rmse_k,
    pred_phys=pred_phys,
    true_phys=true_phys,
    mask=mk,
    test_dates=test_dates_k,
    train_end=train_end_k,
    val_end=val_end_k,
)]


# ======================================================================
# 2) RECURSIVE FRAMEWORK: retrain the model every X_STEP months over a
#    forecasting horizon of fw. The process starts at TRAINVAL + X_STEP,
#    since step 0 has already been completed above.
# ======================================================================
trainval_k = TRAINVAL + X_STEP
step_idx = 1

while trainval_k + fw <= tot:
    train_start_k = 0
    train_end_k   = int(trainval_k * 0.85)
    val_start_k   = train_end_k
    val_end_k     = trainval_k
    test_start_k  = trainval_k

    print(f"\n---- step {step_idx}: train=[0,{train_end_k}) "
          f"val=[{val_start_k},{val_end_k}) test=[{test_start_k},{test_start_k+fw}) ----")

    # ---- Recompute the climatology and anomaly normalization for the current training set ----
    clim = train_climatology(series, month, train_end_k)
    seas = clim[month - 1].astype(np.float32)
    anom_filled, obs_mask = impute_anomaly(series, seas)
    tr_obs = obs_mask[:train_end_k].astype(bool)
    mu_a = float(anom_filled[:train_end_k][tr_obs].mean())
    sd_a = float(anom_filled[:train_end_k][tr_obs].std() + 1e-8)
    anom_norm = (anom_filled - mu_a) / sd_a
    sin_m = np.sin(2 * np.pi * month / 12).astype(np.float32)
    cos_m = np.cos(2 * np.pi * month / 12).astype(np.float32)

    Xtr, Ytr, Mtr, _    = build_windows(anom_norm, obs_mask, sin_m, cos_m, tot,
                                         bw, fw, sampling_interval_all[0],
                                         train_start_k, train_end_k)
    Xva, Yva, Mva, _    = build_windows(anom_norm, obs_mask, sin_m, cos_m, tot,
                                         bw, fw, sampling_interval_all[0],
                                         val_start_k, val_end_k)
    Xte, Yte, Mte, TIte = build_windows(anom_norm, obs_mask, sin_m, cos_m, tot,
                                         bw, fw, sampling_interval_all[0],
                                         test_start_k, test_start_k + 1)
    print(f"windows -> train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}")

    Xva_d, Yva_d, Mva_d = Xva.to(device), Yva.to(device), Mva.to(device)

    model = make_model(best_architecture)
    model, best_val, best_state = train_model(model, Xtr, Ytr, Mtr, Xva_d, Yva_d, Mva_d)

    with torch.no_grad():
        pred = model(Xte.to(device)).cpu().numpy()

    pred_phys = to_physical(pred, TIte, sd_a, mu_a, seas).ravel()
    true_phys = to_physical(Yte.numpy(), TIte, sd_a, mu_a, seas).ravel()
    mk = Mte.numpy().ravel()
    test_dates_k = grid_dates[TIte[0]]
    err = (pred_phys - true_phys) * mk
    rmse_k = np.sqrt((err ** 2).sum() / max(mk.sum(), 1))

    torch.save(best_state, save_models_to + f"model_step{step_idx}.pt")

    recursive_steps.append(dict(
        step=step_idx,
        best_l=best_architecture_id,
        best_val=best_val,
        rmse=rmse_k,
        pred_phys=pred_phys,
        true_phys=true_phys,
        mask=mk,
        test_dates=test_dates_k,
        train_end=train_end_k,
        val_end=val_end_k,
    ))

    step_idx += 1
    trainval_k += X_STEP

np.save(save_models_to + "recursive_steps.npy", recursive_steps, allow_pickle=True)


# ======================================================================
# final evaluation: aggregate metrics, reconstruct the time series, save outputs
# ======================================================================
all_pred = np.concatenate([s["pred_phys"] for s in recursive_steps])
all_true = np.concatenate([s["true_phys"] for s in recursive_steps])
all_mask = np.concatenate([s["mask"]      for s in recursive_steps])
err_all = (all_pred - all_true) * all_mask
rmse_overall = np.sqrt((err_all ** 2).sum() / max(all_mask.sum(), 1))
print(f"\nRMSE complessivo forecast ricorsivo (basin mean): {rmse_overall:.3f}")

basin_pred_mean = np.full(tot, np.nan, dtype=float)
basin_true_mean = np.full(tot, np.nan, dtype=float)

date_to_idx = {d: i for i, d in enumerate(grid_dates)}
for s in recursive_steps:
    idx = np.array([date_to_idx[d] for d in s["test_dates"]])
    basin_pred_mean[idx] = s["pred_phys"]
    basin_true_mean[idx] = s["true_phys"]

save_npy_and_mat(case_dir + "basin_test_prediction_mean", "prediction_mean", basin_pred_mean)
save_npy_and_mat(case_dir + "basin_test_true_mean", "truth_mean", basin_true_mean)

np.save(case_dir + "grid_dates.npy", grid_dates)
savemat(case_dir + "grid_dates.mat", {
    "grid_dates_datenum": matlab_datenum(grid_dates),
    "grid_dates_str": grid_dates.astype("datetime64[D]").astype(str),
})

df = pd.DataFrame({
    "date": grid_dates.astype("datetime64[D]"),
    "prediction_mean": basin_pred_mean,
    "truth_mean": basin_true_mean,
    "train_reconstruction_step0": train_reconstruction,
    "val_reconstruction_step0": val_reconstruction,
})
df.to_csv(case_dir + "basin_test_mean_timeseries.csv", index=False)


# ==========================================================
# Plot: time series
# ==========================================================
plt.figure(figsize=(14, 5))
plt.plot(grid_dates, series, color="black", linewidth=1, label="Observed basin-averaged time series.")
plt.plot(grid_dates, train_reconstruction, color="tab:green", linewidth=1.2, linestyle="--",
          label="# Train reconstruction (step-0 architecture, fixed weights).")
plt.plot(grid_dates, val_reconstruction, color="tab:orange", linewidth=1.2, linestyle="--",
          label="# Val reconstruction (step-0 architecture, fixed weights).")

cmap = plt.cm.plasma(np.linspace(0, 1, len(recursive_steps)))
for s, c in zip(recursive_steps, cmap):
    plt.plot(
        s["test_dates"], s["pred_phys"], "o-", color=c, linewidth=1.5,
        label=f"Step {s['step']} (retrain @ {grid_dates[s['train_end']]})"
    )
    plt.axvline(s["test_dates"][0], color="gray", linestyle=":", linewidth=0.5)

plt.title(f"{basin}: Recursive forecasting of the basin-mean time series.")
plt.xlabel("Data")
plt.ylabel("GRACE value")
plt.legend(loc="best", fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(case_dir + "basin_recursive_forecast.png", dpi=150)
plt.show()

mask = ~np.isnan(basin_true_mean) & ~np.isnan(basin_pred_mean)
rmse_basin = np.sqrt(np.mean((basin_pred_mean[mask] - basin_true_mean[mask]) ** 2))
print(f"RMSE medio del bacino = {rmse_basin:.3f}")


# ==========================================================
# Plot: predicted vs. observed values (train/validation/test)
#       with regression lines.
# ==========================================================
def valid_pairs(true_arr, pred_arr):
    """Valid (observed, predicted) pairs (excluding NaNs in either series)."""
    m = ~np.isnan(true_arr) & ~np.isnan(pred_arr)
    return true_arr[m], pred_arr[m]

# "series" is the raw observed time series (NaN where no observations are
# available). Using it as the ground truth is consistent for both the
# training/validation sets (step-0 reconstructions) and the test set
# (basin_pred_mean), while avoiding comparisons against interpolated values.
train_true_pts, train_pred_pts = valid_pairs(series, train_reconstruction)
val_true_pts,   val_pred_pts   = valid_pairs(series, val_reconstruction)
test_true_pts,  test_pred_pts  = valid_pairs(series, basin_pred_mean)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(train_true_pts, train_pred_pts, s=18, color="tab:green", alpha=0.7,
           label=f"Train (n={len(train_true_pts)})")
ax.scatter(val_true_pts, val_pred_pts, s=18, color="tab:orange", alpha=0.7,
           label=f"Val (n={len(val_true_pts)})")
ax.scatter(test_true_pts, test_pred_pts, s=18, color="tab:blue", alpha=0.7,
           label=f"Test (n={len(test_true_pts)})")

def plot_group_regression(ax, x, y, color, label_prefix):
    """Plot the regression line for each group using the same color as the data points."""
    if len(x) < 2:
        return
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ax.plot(x_line, y_line, color=color, linewidth=1.8, linestyle="-",
            label=f"{label_prefix} fit: y={slope:.2f}x+{intercept:.2f} (R²={r2:.3f})")

plot_group_regression(ax, train_true_pts, train_pred_pts, "tab:green", "Train")
plot_group_regression(ax, val_true_pts, val_pred_pts, "tab:orange", "Val")
plot_group_regression(ax, test_true_pts, test_pred_pts, "tab:blue", "Test")

all_true = np.concatenate([train_true_pts, val_true_pts, test_true_pts])
all_pred = np.concatenate([train_pred_pts, val_pred_pts, test_pred_pts])

if len(all_true) >= 2:
    lo_lim = float(min(all_true.min(), all_pred.min()))
    hi_lim = float(max(all_true.max(), all_pred.max()))
    lims = [lo_lim, hi_lim]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("Observed values")
ax.set_ylabel("Model estimates")
ax.set_title(f"{basin}: estimated vs observed (train / val / test)")
ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(case_dir + "basin_pred_vs_true_scatter.png", dpi=150)
plt.show()

print("\nDone.")