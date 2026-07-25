"""
Evaluation metrics and statistical significance tests — STLF V4.

[FIX-6] Adds the Diebold-Mariano test (with Harvey small-sample correction and
HAC variance) so that improvement claims in the manuscript are backed by a
formal test instead of the V3 assertion that they "would survive any
reasonable statistical test".

Metrics are computed on the ORIGINAL scale (MW) against UNMODIFIED targets.
"""
import numpy as np
from scipy import stats


# ═══════════════════════════════════════════
# POINT METRICS
# ═══════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    """Standard STLF metrics. Inputs: arrays broadcastable to the same shape."""
    yt = np.asarray(y_true, dtype=np.float64).flatten()
    yp = np.asarray(y_pred, dtype=np.float64).flatten()
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # Standard MAPE. Guard only against literal zeros (documented, not silent).
    denom = np.where(np.abs(yt) < 1e-8, np.nan, np.abs(yt))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    smape = float(np.mean(2 * np.abs(err) / (np.abs(yt) + np.abs(yp) + 1e-8)) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "sMAPE": smape}


def metrics_mean_std(metric_dicts):
    """Aggregate a list of per-seed metric dicts into mean/std dicts. Uses the
    SAMPLE standard deviation (ddof=1) — the correct estimator for the
    variability of a small seed sample reported as mean ± std (round-3 audit).
    Falls back to 0 for a single seed."""
    keys = metric_dicts[0].keys()
    n = len(metric_dicts)
    ddof = 1 if n > 1 else 0
    mean = {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}
    std = {k: float(np.std([m[k] for m in metric_dicts], ddof=ddof)) for k in keys}
    return mean, std


# ═══════════════════════════════════════════
# DIEBOLD-MARIANO TEST
# ═══════════════════════════════════════════

def _dm_from_diff(d, hac_lags=None, hln_horizon=1):
    """Core DM test on a precomputed loss-differential series d (one value per
    non-overlapping issue day). Shared by diebold_mariano and
    diebold_mariano_multiseed so the variance/HLN math cannot drift
    (round-4 audit: a duplicated copy had diverged)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar

    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    hac_lags = max(0, min(hac_lags, n - 1))

    # HAC variance of the mean (Bartlett weights, autocovariances scaled by 1/n)
    gamma0 = np.sum(dc ** 2) / n
    var = gamma0
    for k in range(1, hac_lags + 1):
        w = 1.0 - k / (hac_lags + 1)
        var += 2.0 * w * np.sum(dc[k:] * dc[:-k]) / n
    fallback = False
    if var <= 0:
        var = gamma0                      # degenerate HAC -> naive variance
        fallback = True
    var = var / n
    dm = dbar / np.sqrt(var) if var > 0 else 0.0

    # Harvey-Leybourne-Newbold correction: (n + 1 - 2h + h(h-1)/n)/n,
    # which for the non-overlapping horizon h=1 is (n-1)/n.
    hh = hln_horizon
    radicand = (n + 1 - 2 * hh + hh * (hh - 1) / n) / n
    correction = np.sqrt(radicand) if radicand > 0 else 1.0
    dm_hln = dm * correction
    p = 2 * stats.t.sf(np.abs(dm_hln), df=n - 1) if n > 1 else 1.0
    return {"dm_stat": float(dm_hln), "p_value": float(p),
            "mean_loss_diff": float(dbar), "n": int(n),
            "hac_lags": int(hac_lags), "hac_fallback": bool(fallback)}


def diebold_mariano(y_true, pred_a, pred_b, loss="mse", hac_lags=None,
                    hln_horizon=1):
    """
    DM test on day-ahead forecasts (round-3 statistics audit).

    The forecasts are issued once per horizon (stride = H), so the loss
    differential series d_i (one value per issue day) is NON-OVERLAPPING.
    Consequently:
      * the HLN small-sample horizon is 1, not H (the MA(h-1) structure of
        overlapping multi-step forecasts does not apply here); this makes the
        Harvey-Leybourne-Newbold factor sqrt((n-1)/n), a mild correction;
      * the HAC (Newey-West / Bartlett) bandwidth defaults to the automatic
        rule floor(4·(n/100)^(2/9)) to absorb any residual (e.g. weekly)
        autocorrelation in d_i, rather than the horizon H;
      * autocovariances are scaled by 1/n (preserving the PSD guarantee that
        motivates Bartlett weighting).

    Args:
        y_true, pred_a, pred_b : (n_forecasts, horizon) arrays, original scale
        loss : "mse" or "mae"
        hac_lags : HAC truncation lag (default: Newey-West automatic)
        hln_horizon : horizon for the HLN factor (1 for non-overlapping)
    Returns: dict(dm_stat, p_value, mean_loss_diff, n, hac_lags, hac_fallback)
    """
    yt = np.asarray(y_true, dtype=np.float64)
    pa = np.asarray(pred_a, dtype=np.float64)
    pb = np.asarray(pred_b, dtype=np.float64)
    if yt.ndim == 1:
        yt, pa, pb = yt[:, None], pa[:, None], pb[:, None]

    if loss == "mse":
        la = ((yt - pa) ** 2).mean(axis=1)
        lb = ((yt - pb) ** 2).mean(axis=1)
    else:
        la = np.abs(yt - pa).mean(axis=1)
        lb = np.abs(yt - pb).mean(axis=1)

    return _dm_from_diff(la - lb, hac_lags=hac_lags, hln_horizon=hln_horizon)


def _issue_losses(y_true, pred, loss):
    yt = np.asarray(y_true, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    if yt.ndim == 1:
        yt, pp = yt[:, None], pp[:, None]
    return (((yt - pp) ** 2) if loss == "mse" else np.abs(yt - pp)).mean(axis=1)


def diebold_mariano_multiseed(y_true, ref_preds, base_preds, loss="mse"):
    """DM test whose per-issue loss differential is AVERAGED across seeds
    before the test (round-3 referee audit). This tests the 5-seed mean loss
    profile — the quantity the tables actually report — with n = #issue days
    (no pseudo-replication from stacking 5·n differentials as if independent).

    ref_preds / base_preds : lists of (n, H) arrays, one per seed, all on the
    same deterministic issue grid.
    """
    n = min(min(len(p) for p in ref_preds), min(len(p) for p in base_preds),
            len(y_true))
    yt = y_true[:n]
    d_ref = np.mean([_issue_losses(yt, p[:n], loss) for p in ref_preds], axis=0)
    d_base = np.mean([_issue_losses(yt, p[:n], loss) for p in base_preds], axis=0)
    # Same shared core as the single-seed test -> identical variance/HLN math.
    return _dm_from_diff(d_ref - d_base)


def holm_bonferroni(pvalues):
    """Holm-Bonferroni step-down adjusted p-values for a family of tests.
    Input/return: dict name -> p. Adjusted p's are monotone and control the
    family-wise error rate (round-3 referee audit: ~42 simultaneous DM tests
    need multiplicity control)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        a = (m - i) * p
        running = max(running, a)         # enforce monotonicity
        adj[name] = float(min(1.0, running))
    return adj


def dm_table(y_true, preds_by_model, reference="Proposed", loss="mse"):
    """DM test of `reference` against every other model, with Holm-adjusted
    p-values across the family of comparisons. Runs both MSE- and MAE-loss
    variants so significance can be reported for the loss the claims use
    (round-3 referee audit)."""
    out = {}
    ref = preds_by_model[reference]
    for name, p in preds_by_model.items():
        if name == reference:
            continue
        n = min(len(ref), len(p), len(y_true))
        res = diebold_mariano(y_true[:n], ref[:n], p[:n], loss=loss)
        mae = diebold_mariano(y_true[:n], ref[:n], p[:n], loss="mae")
        res["dm_mae"] = mae["dm_stat"]
        res["p_value_mae"] = mae["p_value"]
        out[name] = res
    # Holm-Bonferroni across the comparison family (both losses)
    holm = holm_bonferroni({k: v["p_value"] for k, v in out.items()})
    holm_mae = holm_bonferroni({k: v["p_value_mae"] for k, v in out.items()})
    for k in out:
        out[k]["p_holm"] = holm[k]
        out[k]["p_holm_mae"] = holm_mae[k]
    return out
