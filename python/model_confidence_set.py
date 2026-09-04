#!/usr/bin/env python3
"""Model confidence set (Hansen, Lunde & Nason, Econometrica 2011) over the
saved day-ahead forecasts of every model, per dataset.

Why
---
Section 5.3 of the paper reports pairwise Diebold-Mariano tests of the proposed
model against each baseline, Holm-corrected. A pairwise test answers "does this
baseline beat the proposed model"; it does not answer "which models are
statistically indistinguishable from the best one". The MCS does. It starts from
the full set, tests the null that all remaining models have equal expected loss
(T_max statistic), eliminates the worst model when the null is rejected, and
stops when it is not; each model receives an MCS p-value, and the 90% MCS is
the set of models with p >= 0.10.

Data
----
results/predictions_v4.pkl holds, per dataset and model, y_pred (the five-seed
mean prediction, or the single deterministic prediction) and y_true, one row per
non-overlapping day-ahead issue time. The loss profile used is the per-issue
mean squared error across the horizon (the headline DM loss) and, separately,
the per-issue mean absolute error. Cross-model dependence is preserved by
resampling issue times jointly for all models with a moving-block bootstrap,
block length = HAC bandwidth + 1 (4(n/100)^(2/9) as in the DM test, +1).

    python scripts/model_confidence_set.py            # writes results/mcs_v4.json
    python scripts/model_confidence_set.py --selftest # synthetic sanity checks

Degenerate forecasters
----------------------
The T_max statistic standardises each model's mean loss difference by a
bootstrap standard deviation, and d_i. is measured against the average loss of
the surviving set. A forecaster with heavy-tailed losses inflates that average
and its variance for every other model, and the procedure loses power against
everything (a first run with seasonal ARIMA on AEMO, whose R^2 is -72 because a
handful of forecasts are off by an order of magnitude, kept all fifteen models
in the 90 % set). Such a model is not a candidate for "best" under any reading,
so --exclude removes it from the candidate set; the default excludes exactly
that one case and the exclusion is recorded in the output file.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
RESULTS = os.path.join(CODE, "results")
PRED = os.path.join(RESULTS, "predictions_v4.pkl")
OUT = os.path.join(RESULTS, "mcs_v4.json")

MODELS = ["Proposed", "SeasonalNaive", "ARIMA", "XGBoost", "LightGBM", "LSTM",
          "BiLSTM", "GRU", "TCN", "Transformer", "CNN_LSTM", "GRU_TCN_Attention",
          "DLinear", "PatchTST", "TiDE"]


def hac_lags(n):
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def per_issue_loss(y_pred, y_true, loss):
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    if loss == "mse":
        return np.mean(e ** 2, axis=1)
    if loss == "mae":
        return np.mean(np.abs(e), axis=1)
    raise ValueError(loss)


def bootstrap_means(L, block, B, rng, chunk=500):
    """Moving-block bootstrap means of the columns of L, shape (B, m)."""
    n, m = L.shape
    nb = int(np.ceil(n / block))
    out = np.empty((B, m))
    offs = np.arange(block)
    done = 0
    while done < B:
        b = min(chunk, B - done)
        starts = rng.integers(0, n - block + 1, size=(b, nb))
        idx = (starts[:, :, None] + offs[None, None, :]).reshape(b, -1)[:, :n]
        out[done:done + b] = L[idx].mean(axis=1)
        done += b
    return out


def mcs(L, B=10000, block=None, seed=0):
    """L: (n, m) loss matrix. Returns (p-values per column, elimination order).

    T_max version of Hansen et al. (2011): at each step, with the surviving set
    S, d_i. = mean_j (L_i - L_j) over j in S, t_i = d_i. / sd(d_i.) with the
    bootstrap standard deviation, T_max = max_i t_i; the bootstrap distribution
    of T_max is that of max_i (d*_i. - d_i.) / sd(d_i.). The model with the
    largest t_i is eliminated and receives p = max(p_step, p_previous)."""
    n, m = L.shape
    if block is None:
        block = hac_lags(n) + 1
    rng = np.random.default_rng(seed)
    Lbar = L.mean(0)
    Lb = bootstrap_means(L, block, B, rng)
    included = list(range(m))
    pvals = np.full(m, np.nan)
    order = []
    p_prev = 0.0
    while len(included) > 1:
        S = np.array(included)
        dbar = Lbar[S] - Lbar[S].mean()
        dbar_b = Lb[:, S] - Lb[:, S].mean(axis=1, keepdims=True)
        var = np.mean((dbar_b - dbar) ** 2, axis=0)
        sd = np.sqrt(np.maximum(var, 1e-300))
        t = dbar / sd
        Tmax = t.max()
        Tmax_b = ((dbar_b - dbar) / sd).max(axis=1)
        p = float(np.mean(Tmax_b >= Tmax))
        p_mcs = max(p, p_prev)
        p_prev = p_mcs
        worst = int(S[np.argmax(t)])
        pvals[worst] = p_mcs
        order.append(worst)
        included.remove(worst)
    pvals[included[0]] = 1.0
    order.append(included[0])
    return pvals, order, block


def selftest():
    rng = np.random.default_rng(1)
    n, m = 300, 6
    base = rng.standard_normal((n, 1)) * 0.5
    # (a) model 0 clearly best, the rest identical in distribution
    L = 1.0 + 0.3 * rng.standard_normal((n, m)) + base
    L[:, 0] -= 0.5
    p, order, _ = mcs(L, B=2000)
    assert p[0] == 1.0 and (p[1:] < 0.10).all(), (p, order)
    # (b) all models identical in distribution: no model should be dropped at 10 %
    #     in most runs; check the best model keeps p = 1 and that at least three
    #     survive (a loose check, this is a randomised procedure)
    L = 1.0 + 0.3 * rng.standard_normal((n, m)) + base
    p, order, _ = mcs(L, B=2000)
    assert (p >= 0.10).sum() >= 3, (p, order)
    # (c) two clearly best, tied: they must be the last two standing, and at the
    #     10 % level both should survive in most replicates (nominal size 10 %)
    both = 0
    for rep in range(20):
        L = 1.0 + 0.3 * rng.standard_normal((n, m)) + base
        L[:, 0] -= 0.5
        L[:, 1] -= 0.5
        p, order, _ = mcs(L, B=1000, seed=rep)
        assert set(order[-2:]) == {0, 1} and (p[2:] < 0.10).all(), (p, order)
        both += int(p[0] >= 0.10 and p[1] >= 0.10)
    assert both >= 14, both
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--exclude", default="AEMO:ARIMA",
                    help="comma-separated dataset:model pairs excluded from the candidate set")
    a = ap.parse_args()
    excl = {}
    for item in filter(None, a.exclude.split(",")):
        ds_, m_ = item.split(":")
        excl.setdefault(ds_, set()).add(m_)
    if a.selftest:
        selftest()
        return
    with open(PRED, "rb") as f:
        P = pickle.load(f)
    out = {"method": "Hansen-Lunde-Nason (2011) MCS, T_max statistic, moving-block "
                     "bootstrap (block = HAC bandwidth + 1), losses per day-ahead "
                     "issue time on the five-seed mean prediction",
           "B": a.B, "seed": a.seed, "alpha_reported": 0.10, "datasets": {}}
    for ds in ("GEFCom2014", "PJM", "AEMO"):
        d = P[ds]
        names = [m for m in MODELS if m in d and m not in excl.get(ds, set())]
        y_true = np.asarray(d["Proposed"]["y_true"], float)
        for m in names:
            assert np.allclose(np.asarray(d[m]["y_true"], float), y_true), (ds, m)
        rec = {"n": int(y_true.shape[0]), "models": names,
               "excluded": sorted(excl.get(ds, set()))}
        for loss in ("mse", "mae"):
            L = np.stack([per_issue_loss(d[m]["y_pred"], y_true, loss) for m in names], 1)
            p, order, block = mcs(L, B=a.B, seed=a.seed)
            rec["block_length"] = block
            rec[loss] = {"p_mcs": {m: float(p[i]) for i, m in enumerate(names)},
                         "elimination_order": [names[i] for i in order],
                         "mcs_90": [m for i, m in enumerate(names) if p[i] >= 0.10],
                         "mean_loss": {m: float(L[:, i].mean()) for i, m in enumerate(names)}}
        out["datasets"][ds] = rec
        print(f"{ds}: n={rec['n']} block={rec['block_length']}")
        for loss in ("mse", "mae"):
            print(f"  [{loss}] 90% MCS = {rec[loss]['mcs_90']}")
            print("        p:", {m: round(v, 3) for m, v in rec[loss]["p_mcs"].items()})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
