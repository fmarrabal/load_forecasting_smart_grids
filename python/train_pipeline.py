"""
Training and evaluation pipeline — STLF V4.

Protocol ([FIX-6]):
  * Training samples slide with stride 1 (data richness).
  * Val/test are evaluated at stride = pred_horizon: one forecast per day,
    exactly the operational day-ahead setting. This removes the overlapping-
    window autocorrelation that inflated V3 metrics and makes the
    Diebold-Mariano test well-posed.
  * Every experiment runs across SEEDS; results report mean +/- std.
  * Error correction is CAUSAL ([FIX-3]): trained on validation errors,
    applied at test time using only errors of already-realized forecasts.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (DEVICE, TRAIN_PARAMS, MODEL_PARAMS, DECOMP_PARAMS,
                    ERROR_CORRECTION, MODELS_DIR)
from data_utils import STLFDataset, make_loaders
from metrics_stats import compute_metrics
from model_proposed import CPTB, CausalErrorCorrection, create_proposed
from models_baselines import (SeasonalNaive, LSTMModel, BiLSTMModel, GRUModel,
                              TCNModel, TransformerModel, CNNLSTMModel,
                              GRUTCNAttention, DLinear, PatchTST, TiDE)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════ TRAINING LOOP ═══════════════════

def train_model(model, train_loader, val_loader, tag="model"):
    model = model.to(DEVICE)
    P = TRAIN_PARAMS
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"],
                           weight_decay=P["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=P["epochs"], eta_min=P["min_lr"])
    crit = nn.HuberLoss(delta=P["huber_delta"])
    best_vl, patience, best_state = float("inf"), 0, None
    t0 = time.time()

    for ep in range(P["epochs"]):
        model.train()
        for xl, xc, xf, y, _ in train_loader:
            xl, xc = xl.to(DEVICE), xc.to(DEVICE)
            xf, y = xf.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            yp, _ = model(xl, xc, xf)
            loss = crit(yp, y)
            if not torch.isfinite(loss):
                continue        # skip pathological batch, never poison params
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), P["grad_clip"])
            if not torch.isfinite(gn):
                opt.zero_grad()
                continue
            opt.step()
        sched.step()

        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for xl, xc, xf, y, _ in val_loader:
                xl, xc = xl.to(DEVICE), xc.to(DEVICE)
                xf, y = xf.to(DEVICE), y.to(DEVICE)
                yp, _ = model(xl, xc, xf)
                vl += crit(yp, y).item()
                nb += 1
        vl /= max(nb, 1)

        if vl < best_vl - 1e-6:
            best_vl, patience = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= P["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, time.time() - t0


@torch.no_grad()
def predict(model, loader):
    """Returns predictions/targets in z-space plus issue indices and the mean
    per-forecast inference latency in ms."""
    model.eval().to(DEVICE)
    preds, trues, idxs, attns = [], [], [], []
    lat = []
    for xl, xc, xf, y, t in loader:
        xl, xc, xf = xl.to(DEVICE), xc.to(DEVICE), xf.to(DEVICE)
        t0 = time.time()
        yp, att = model(xl, xc, xf)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        lat.append((time.time() - t0) / max(len(xl), 1))
        preds.append(yp.cpu().numpy())
        trues.append(y.numpy())
        idxs.append(np.asarray(t))
        if att is not None:
            attns.append(att.cpu().numpy())
    return (np.concatenate(preds), np.concatenate(trues),
            np.concatenate(idxs),
            np.concatenate(attns) if attns else None,
            float(np.mean(lat) * 1000))


# ═══════════════════ CAUSAL ERROR CORRECTION ═══════════════════

def build_error_series(y_true, y_pred, issue_idx, horizon):
    """Flatten non-overlapping day-ahead forecasts into a time-indexed error
    series err[t]. Assumes stride == horizon (operational protocol)."""
    t0 = issue_idx[0]
    T = issue_idx[-1] + horizon - t0
    err = np.full(T, np.nan)
    for i, t in enumerate(issue_idx):
        err[t - t0: t - t0 + horizon] = y_true[i] - y_pred[i]
    return err, t0


def fit_error_correction(val_true, val_pred, val_idx, horizon, seed):
    EC = ERROR_CORRECTION
    # one week of OBSERVED errors, in steps (horizon == steps per day),
    # so the window scales correctly for half-hourly data (AEMO)
    W = 7 * horizon
    err, _ = build_error_series(val_true, val_pred, val_idx, horizon)
    X, Y = [], []
    for s in range(W, len(err) - horizon + 1, horizon):
        xw, yw = err[s - W:s], err[s:s + horizon]
        if np.isnan(xw).any() or np.isnan(yw).any():
            continue
        X.append(xw)
        Y.append(yw)
    if len(X) < 32:
        return None
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    ec = CausalErrorCorrection(W, horizon, EC["hidden_size"]).to(DEVICE)
    opt = torch.optim.Adam(ec.parameters(), lr=EC["lr"])
    crit = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(X, Y)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True,
                                     generator=torch.Generator().manual_seed(seed))
    for _ in range(EC["epochs"]):
        ec.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(ec(xb), yb).backward()
            opt.step()
    return ec


@torch.no_grad()
def apply_error_correction(ec, y_pred, y_true, issue_idx, horizon):
    """
    Strictly causal application: the correction for the forecast issued at t
    uses err[t-W:t], and err[s] is only defined from forecasts issued at
    s' <= s - horizon (fully realized at t). With stride == horizon, the
    previous forecasts' errors are realized exactly at issue time.
    """
    if ec is None:
        return y_pred
    W = ec.W
    err, t0 = build_error_series(y_true, y_pred, issue_idx, horizon)
    corrected = y_pred.copy()
    for i, t in enumerate(issue_idx):
        rel = t - t0
        if rel >= W:
            xw = err[rel - W:rel]
            if np.isnan(xw).any():
                continue
            corr = ec(torch.tensor(xw, dtype=torch.float32)
                      .unsqueeze(0).to(DEVICE))
            corrected[i] = y_pred[i] + corr.cpu().numpy().flatten()
    return corrected


# ═══════════════════ EXPERIMENT RUNNERS ═══════════════════

def _make_dl_model(name, data, seed):
    cfg = data["cfg"]
    ncp, ncf, H, L = (data["n_cov_past"], data["n_cov_fut"],
                      cfg["pred_horizon"], cfg["input_window"])
    set_seed(seed)
    table = {
        "LSTM": lambda: LSTMModel(ncp, H, ncf),
        "BiLSTM": lambda: BiLSTMModel(ncp, H, ncf),
        "GRU": lambda: GRUModel(ncp, H, ncf),
        "TCN": lambda: TCNModel(ncp, H, ncf),
        "Transformer": lambda: TransformerModel(ncp, H, ncf),
        "CNN_LSTM": lambda: CNNLSTMModel(ncp, H, ncf),
        "GRU_TCN_Attention": lambda: GRUTCNAttention(ncp, H, ncf),
        "DLinear": lambda: DLinear(L, H, ncf),
        "PatchTST": lambda: PatchTST(L, H, ncf),
        "TiDE": lambda: TiDE(L, H, ncp, ncf),
        "Proposed": lambda: create_proposed(cfg, MODEL_PARAMS, DECOMP_PARAMS,
                                            ncp, ncf),
    }
    if name.startswith("Proposed_"):
        flag = name.split("Proposed_")[1]           # e.g. "noStage2"
        mapping = {
            "noStage1": {"use_stage1": False},
            "noStage2": {"use_stage2": False},
            "noGate": {"use_gate": False},
            "noPatch": {"use_patch": False},
            "noCross": {"use_cross_att": False},
            "noBiGRU": {"use_bigru": False},
            "noRevIN": {"use_revin": False},
            "noFutureCov": {"use_future_cov": False},
            "noLinearSkip": {"use_linear_skip": False},
            "noCovSkip": {"use_cov_skip": False},
        }
        flags = mapping[flag]
        return create_proposed(cfg, MODEL_PARAMS, DECOMP_PARAMS, ncp, ncf,
                               **flags)
    return table[name]()


def run_dl_experiment(name, data, seed, use_ec=False, save_tag=None):
    """Train + evaluate one deep model with one seed. Returns results dict
    with metrics on the ORIGINAL scale and per-forecast predictions."""
    cfg = data["cfg"]
    H = cfg["pred_horizon"]
    set_seed(seed)
    tr, va, te = make_loaders(data, TRAIN_PARAMS["batch_size"],
                              eval_stride=H)
    model = _make_dl_model(name, data, seed)
    n_par = count_parameters(model)
    model, t_train = train_model(model, tr, va, tag=f"{name}_s{seed}")

    yp, yt, idx, attn, lat_ms = predict(model, te)
    sc = data["scaler_load"]
    yp_mw = sc.inverse_transform(yp)
    yt_mw = sc.inverse_transform(yt)

    result = {
        "metrics": compute_metrics(yt_mw, yp_mw),
        "y_pred": yp_mw, "y_true": yt_mw, "issue_idx": idx,
        "attn": attn, "n_params": n_par,
        "train_time_s": t_train, "inference_ms": lat_ms, "seed": seed,
    }

    if use_ec:
        vp, vt, vidx, _, _ = predict(model, va)
        vp_mw, vt_mw = sc.inverse_transform(vp), sc.inverse_transform(vt)
        ec = fit_error_correction(vt_mw, vp_mw, vidx, H, seed)
        yp_ec = apply_error_correction(ec, yp_mw, yt_mw, idx, H)
        result["metrics_ec"] = compute_metrics(yt_mw, yp_ec)
        result["y_pred_ec"] = yp_ec

    if save_tag:
        torch.save(model.state_dict(),
                   os.path.join(MODELS_DIR, f"{save_tag}_s{seed}.pt"))
    return result


def run_seasonal_naive(data):
    cfg = data["cfg"]
    H, spd = cfg["pred_horizon"], cfg["steps_per_day"]
    season = 7 * spd
    te = STLFDataset(data, data["val_end"], len(data["load_z"]), stride=H)
    sc = data["scaler_load"]
    yps, yts, idxs = [], [], []
    naive = SeasonalNaive(season, H)
    for i in range(len(te)):
        xl, _, _, y, t = te[i]
        yps.append(naive.predict(xl.unsqueeze(0)).numpy()[0])
        yts.append(y.numpy())
        idxs.append(t)
    yp = sc.inverse_transform(np.array(yps))
    yt = sc.inverse_transform(np.array(yts))
    return {"metrics": compute_metrics(yt, yp), "y_pred": yp, "y_true": yt,
            "issue_idx": np.array(idxs), "n_params": 0,
            "train_time_s": 0.0, "inference_ms": 0.0, "seed": 0}


def run_gbm(name, data, seed):
    """XGBoost / LightGBM with one regressor per horizon step. Features:
    down-sampled past window + future calendar covariates ([FIX-5])."""
    cfg = data["cfg"]
    H = cfg["pred_horizon"]
    tr = STLFDataset(data, 0, data["train_end"], stride=2)
    te = STLFDataset(data, data["val_end"], len(data["load_z"]), stride=H)

    def to_xy(ds):
        X, Y = [], []
        for i in range(len(ds)):
            xl, xc, xf, y, _ = ds[i]
            feats = np.concatenate([
                xl.numpy()[::2],                       # past load (every 2nd)
                xc.numpy()[-H:].flatten(),             # last-day covariates
                xf.numpy().flatten()])                 # future covariates
            X.append(feats)
            Y.append(y.numpy())
        return np.array(X), np.array(Y)

    Xtr, Ytr = to_xy(tr)
    Xte, Yte = to_xy(te)
    yp = np.zeros_like(Yte)
    if name == "XGBoost":
        import xgboost as xgb
        for h in range(H):
            m = xgb.XGBRegressor(n_estimators=300, max_depth=6,
                                 learning_rate=0.05, random_state=seed,
                                 n_jobs=-1, verbosity=0)
            m.fit(Xtr, Ytr[:, h])
            yp[:, h] = m.predict(Xte)
    else:
        import lightgbm as lgb
        for h in range(H):
            m = lgb.LGBMRegressor(n_estimators=300, max_depth=6,
                                  learning_rate=0.05, random_state=seed,
                                  n_jobs=-1, verbose=-1)
            m.fit(Xtr, Ytr[:, h])
            yp[:, h] = m.predict(Xte)
    sc = data["scaler_load"]
    yp_mw, yt_mw = sc.inverse_transform(yp), sc.inverse_transform(Yte)
    idxs = te.issue_idx
    return {"metrics": compute_metrics(yt_mw, yp_mw), "y_pred": yp_mw,
            "y_true": yt_mw, "issue_idx": idxs, "n_params": 0,
            "train_time_s": 0.0, "inference_ms": 0.0, "seed": seed}


def run_arima(data, seed=0):
    """Seasonal-differenced ARIMA refit per test day on recent history.
    Honest configuration: works on the ORIGINAL scale, seasonal differencing
    at the daily period, 4 weeks of history."""
    from statsmodels.tsa.arima.model import ARIMA
    cfg = data["cfg"]
    H, spd = cfg["pred_horizon"], cfg["steps_per_day"]
    hist_len = 28 * spd
    load = data["load_raw"]
    te = STLFDataset(data, data["val_end"], len(data["load_z"]), stride=H)
    yps, yts, idxs = [], [], []
    for i in range(len(te)):
        t = int(te.issue_idx[i])
        hist = load[max(0, t - hist_len):t]
        try:
            m = ARIMA(hist, order=(2, 0, 1),
                      seasonal_order=(1, 1, 1, spd)).fit(method_kwargs={
                          "maxiter": 50})
            f = np.asarray(m.forecast(H))
        except Exception:
            f = load[t - 7 * spd: t - 7 * spd + H]     # naive fallback
        yps.append(f)
        yts.append(load[t:t + H])
        idxs.append(t)
    yp, yt = np.array(yps), np.array(yts)
    return {"metrics": compute_metrics(yt, yp), "y_pred": yp, "y_true": yt,
            "issue_idx": np.array(idxs), "n_params": 0,
            "train_time_s": 0.0, "inference_ms": 0.0, "seed": seed}
