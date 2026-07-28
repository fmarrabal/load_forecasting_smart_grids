#!/usr/bin/env python3
"""
Leakage quantification experiment — STLF V4 (Section 5.4 of the manuscript).

Reproduces the V3 'decompose-then-split' protocol in a CONTROLLED way and
compares it with the causal V4 protocol using the SAME downstream learner,
so the difference isolates the information leak:

  Protocol A (invalid, V3): CEEMDAN-SE-VMD over the FULL series -> split each
      sub-signal chronologically -> train one small learner per sub-signal ->
      sum test predictions.
  Protocol B (valid): identical learner and splits, but each input window is
      decomposed CAUSALLY (only samples inside the window), mirroring the
      sample-wise decomposition literature (VMDNet 2025; Yang et al. 2024).

To keep the experiment affordable, both protocols use ridge regression on
lagged features of each component — the point is the PROTOCOL gap, not the
learner. Requires: pip install EMD-signal vmdpy scikit-learn.

Usage:
    python leakage_demo.py --dataset GEFCom2014 [--max-n 20000]
"""
import argparse
import os

import numpy as np

from config import DATASETS, RESULTS_DIR
from data_utils import prepare_dataset
from metrics_stats import compute_metrics


def _ridge_per_component(train_X, train_y, test_X):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0)
    m.fit(train_X, train_y)
    return m.predict(test_X)


def _windows(sig, L, H, idxs):
    X = np.stack([sig[t - L:t] for t in idxs])
    Y = np.stack([sig[t:t + H] for t in idxs])
    return X, Y


def _issue_indices(load_len, L, H, start, stop, stride, observed=None):
    """Issue times in [start, stop); when an observed mask is given, drop any
    whose TARGET window contains a fabricated step (mirrors STLFDataset), so
    the leakage comparison never scores against gap-filled synthetic load."""
    idx = np.arange(max(start, L), stop - H + 1, stride)
    if observed is not None and len(idx):
        miss = np.concatenate([[0], np.cumsum(~observed)])
        idx = idx[(miss[idx + H] - miss[idx]) == 0]
    return idx


def protocol_A_decompose_then_split(load, L, H, train_end, val_end,
                                    observed=None, verbose=True):
    """V3 protocol: global CEEMDAN(-SE-VMD lite) on the FULL series."""
    from PyEMD import CEEMDAN
    ce = CEEMDAN(trials=20, epsilon=0.2)
    ce.noise_seed(42)
    if verbose:
        print(f"  [A] CEEMDAN over the FULL series (n={len(load)}) — "
              "this is the leaky step...")
    imfs = ce(load)
    comps = list(imfs) + [load - imfs.sum(axis=0)]

    test_idx = _issue_indices(len(load), L, H, val_end, len(load), H, observed)
    train_idx = _issue_indices(len(load), L, H, L, train_end, 4, observed)

    y_pred = np.zeros((len(test_idx), H))
    for c in comps:
        Xtr, Ytr = _windows(c, L, H, train_idx)
        Xte, _ = _windows(c, L, H, test_idx)
        y_pred += _ridge_per_component(Xtr, Ytr, Xte)
    _, y_true = _windows(load, L, H, test_idx)
    return compute_metrics(y_true, y_pred)


def protocol_B_causal(load, L, H, train_end, val_end, kernels=(12, 24, 168),
                      observed=None, verbose=True):
    """Causal protocol: same learner, but per-window causal decomposition
    (multi-scale moving-average split computed inside each window)."""
    def causal_split(w):
        # mirrors Stage 1 of the V4 model, numpy version
        out = []
        prev = w
        for k in kernels:
            k = min(k, len(w))
            ma = np.convolve(np.concatenate([np.full(k - 1, w[0]), w]),
                             np.ones(k) / k, mode="valid")
            out.append(prev - ma)
            prev = ma
        out.append(prev)
        return np.stack(out)          # (len(kernels)+1, L)

    test_idx = _issue_indices(len(load), L, H, val_end, len(load), H, observed)
    train_idx = _issue_indices(len(load), L, H, L, train_end, 4, observed)
    if verbose:
        print(f"  [B] causal per-window decomposition "
              f"({len(train_idx)} train / {len(test_idx)} test windows)...")

    def build(idxs):
        X = []
        for t in idxs:
            comps = causal_split(load[t - L:t])
            X.append(comps.flatten())
        Y = np.stack([load[t:t + H] for t in idxs])
        return np.array(X), Y

    Xtr, Ytr = build(train_idx)
    Xte, y_true = build(test_idx)
    y_pred = _ridge_per_component(Xtr, Ytr, Xte)
    return compute_metrics(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GEFCom2014")
    ap.add_argument("--max-n", type=int, default=20000,
                    help="cap series length to keep CEEMDAN affordable")
    args = ap.parse_args()

    # Preflight: this standalone entrypoint does not go through main.py's
    # dependency check, so fail fast with a clear message (round-4 audit).
    try:
        import PyEMD  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as e:
        raise SystemExit("leakage_demo needs EMD-signal and scikit-learn: "
                         "pip install EMD-signal scikit-learn  (" + str(e) + ")")

    cfg = DATASETS[args.dataset]
    data = prepare_dataset(args.dataset)
    load = data["load_raw"]
    observed = data["observed"]
    if len(load) > args.max_n:
        load = load[-args.max_n:]
        observed = observed[-args.max_n:]
    n = len(load)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    L, H = cfg["input_window"], cfg["pred_horizon"]
    if n < L + 10 * H:
        raise SystemExit(f"--max-n={args.max_n} too small for {args.dataset} "
                         f"(need >= L+10H = {L + 10 * H}).")
    # Scale the causal Stage-1 kernels to the dataset resolution, exactly as
    # create_proposed does (round-4: was hardcoded to hourly).
    scale = cfg["steps_per_day"] / 24.0
    kernels = tuple(max(2, int(round(k * scale))) for k in (12, 24, 168))

    print(f"\nLeakage quantification on {args.dataset} (n={n}, "
          f"observed={100*observed.mean():.1f}%)")
    mA = protocol_A_decompose_then_split(load, L, H, train_end, val_end,
                                         observed=observed)
    mB = protocol_B_causal(load, L, H, train_end, val_end, kernels=kernels,
                           observed=observed)

    print("\n  Same learner, same splits — only the decomposition protocol differs:")
    print(f"    A decompose-then-split : MAPE={mA['MAPE']:.2f}%  "
          f"MAE={mA['MAE']:.2f}  RMSE={mA['RMSE']:.2f}")
    print(f"    B causal (valid)       : MAPE={mB['MAPE']:.2f}%  "
          f"MAE={mB['MAE']:.2f}  RMSE={mB['RMSE']:.2f}")
    infl = (mB["MAPE"] - mA["MAPE"]) / mB["MAPE"] * 100
    print(f"    Apparent (illusory) improvement from leakage: {infl:.1f}%")

    # [round-5 audit] PERSIST the measurement. This experiment produces the
    # paper's headline leakage figure, and until now it only printed it: the
    # number reached the manuscript and Fig. 11 as a literal typed by hand.
    # For a paper whose contribution is a leakage argument, the one controlled
    # measurement of leakage has to be the one a referee can re-derive from a
    # released file, so it is written out and the figure reads it back.
    import json
    out = {
        "dataset": args.dataset, "n": int(n),
        "observed_fraction": float(observed.mean()),
        "input_window": int(L), "pred_horizon": int(H),
        "stage1_kernels": list(kernels),
        "protocol_A_decompose_then_split": {k: float(v) for k, v in mA.items()},
        "protocol_B_causal": {k: float(v) for k, v in mB.items()},
        "illusory_improvement_pct": float(infl),
    }
    path = os.path.join(RESULTS_DIR, f"leakage_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  written: {path}")

    try:
        import figures_results as FR
        FR.fig11_leakage(mA, mB)
    except Exception as e:
        print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()
