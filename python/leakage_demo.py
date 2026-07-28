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


def _flat_ridge(Xtr, Ytr, Xte):
    """One ridge over the concatenated components — the SAME learner topology
    in both arms of the controlled contrast, so that a protocol comparison is
    not silently also a topology comparison."""
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0)
    m.fit(Xtr, Ytr)
    return m.predict(Xte)


_CEEMDAN_FALLBACKS = [0]     # windows whose sifting failed; reported, not hidden


def _ceemdan_fixed_k(sig, K, trials, seed=42):
    """CEEMDAN truncated/padded to exactly K+1 rows (K IMFs + residual) so that
    every window yields the same feature width. The residual absorbs whatever
    is left, so the decomposition still reconstructs the signal exactly.

    On a short window the sifting occasionally fails to converge and PyEMD
    returns non-finite values. Rather than let a NaN reach the regression (or
    silently drop the window, which would change the sample between arms), the
    signal is placed unmodified in the residual row — the identity
    decomposition — and the occurrence is counted so the run can report how
    often it happened.
    """
    from PyEMD import CEEMDAN
    # parallel=False: PyEMD spawns a process pool per call, which on a
    # 168-point window costs far more than the decomposition itself (and needs
    # a __main__ guard on Windows).
    ce = CEEMDAN(trials=trials, epsilon=0.2, parallel=False)
    ce.noise_seed(seed)
    sig = np.asarray(sig, dtype=float)
    out = np.zeros((K + 1, len(sig)))
    try:
        imfs = np.atleast_2d(ce(sig, max_imf=K))[:K]
    except Exception:
        imfs = np.zeros((0, len(sig)))
    if imfs.size and not np.isfinite(imfs).all():
        imfs = np.zeros((0, len(sig)))
    if imfs.shape[0] == 0:
        _CEEMDAN_FALLBACKS[0] += 1
    out[:len(imfs)] = imfs
    out[K] = sig - imfs.sum(axis=0)
    return out


def protocol_A2_global_ceemdan(load, L, H, train_end, val_end, K=8,
                               observed=None, trials=20, verbose=True):
    """LEAKY arm of the controlled contrast: CEEMDAN over the FULL series, each
    window's slice of the components flattened into one feature vector.
    Identical decomposition family, feature layout and learner to protocol C —
    the only difference is that the transform saw beyond the issue time."""
    if verbose:
        print(f"  [A2] CEEMDAN over the FULL series (n={len(load)}, K={K}) — "
              "leaky, matched to C in every other respect...")
    comps = _ceemdan_fixed_k(load, K, trials)
    test_idx = _issue_indices(len(load), L, H, val_end, len(load), H, observed)
    train_idx = _issue_indices(len(load), L, H, L, train_end, 4, observed)
    build = lambda idxs: (
        np.stack([comps[:, t - L:t].ravel() for t in idxs]),
        np.stack([load[t:t + H] for t in idxs]))
    Xtr, Ytr = build(train_idx)
    Xte, Yte = build(test_idx)
    return compute_metrics(Yte, _flat_ridge(Xtr, Ytr, Xte))


def protocol_C_causal_ceemdan(load, L, H, train_end, val_end, K=8,
                              observed=None, trials=20, verbose=True):
    """CAUSAL arm of the controlled contrast: the SAME CEEMDAN, feature layout
    and learner as A2, but recomputed inside each input window, so it never
    sees beyond t0.

    [round-5 audit] A2 vs C is the clean protocol effect. The original A vs B
    contrast is NOT: it also swaps the decomposition family (CEEMDAN vs a
    3-kernel causal moving average) and the learner topology (per-component
    ridges summed vs one ridge on concatenated features), so its gap cannot be
    attributed to leakage alone.
    """
    test_idx = _issue_indices(len(load), L, H, val_end, len(load), H, observed)
    train_idx = _issue_indices(len(load), L, H, L, train_end, 4, observed)
    if verbose:
        print(f"  [C] per-window CEEMDAN (K={K}, {len(train_idx)} train / "
              f"{len(test_idx)} test windows) — this is the slow arm...")
    _CEEMDAN_FALLBACKS[0] = 0
    def build(idxs):
        X = np.stack([_ceemdan_fixed_k(load[t - L:t], K, trials).ravel()
                      for t in idxs])
        return X, np.stack([load[t:t + H] for t in idxs])
    Xtr, Ytr = build(train_idx)
    Xte, Yte = build(test_idx)
    nfb = _CEEMDAN_FALLBACKS[0]
    if nfb and verbose:
        print(f"       ({nfb} of {len(train_idx) + len(test_idx)} windows fell "
              f"back to the identity decomposition: sifting did not converge)")
    m = compute_metrics(Yte, _flat_ridge(Xtr, Ytr, Xte))
    m["_fallback_windows"] = float(nfb)
    return m


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
    ap.add_argument("--no-matched", action="store_true",
                    help="skip the matched A2-vs-C contrast (it recomputes "
                         "CEEMDAN inside every window and takes ~1 h)")
    ap.add_argument("--trials", type=int, default=20,
                    help="CEEMDAN ensemble trials, identical in both arms")
    ap.add_argument("--K", type=int, default=8,
                    help="IMFs kept per window (+1 residual) so that every "
                         "window yields the same feature width")
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

    print("\n  Same splits, same series — the decomposition protocol differs:")
    print(f"    A decompose-then-split : MAPE={mA['MAPE']:.2f}%  "
          f"MAE={mA['MAE']:.2f}  RMSE={mA['RMSE']:.2f}")
    print(f"    B causal (valid)       : MAPE={mB['MAPE']:.2f}%  "
          f"MAE={mB['MAE']:.2f}  RMSE={mB['RMSE']:.2f}")
    infl = (mB["MAPE"] - mA["MAPE"]) / mB["MAPE"] * 100
    print(f"    Apparent (illusory) improvement: {infl:.1f}%")
    print("    NOTE: A and B differ in decomposition family and learner "
          "topology as well as\n          in protocol, so this gap is an "
          "upper bound on the leakage effect. The\n          matched contrast "
          "below (A2 vs C) isolates the protocol.")

    # ── the matched contrast: identical decomposition, features and learner ──
    mA2 = mC = infl2 = None
    if not args.no_matched:
        mA2 = protocol_A2_global_ceemdan(load, L, H, train_end, val_end,
                                         K=args.K, observed=observed,
                                         trials=args.trials)
        mC = protocol_C_causal_ceemdan(load, L, H, train_end, val_end,
                                       K=args.K, observed=observed,
                                       trials=args.trials)
        infl2 = (mC["MAPE"] - mA2["MAPE"]) / mC["MAPE"] * 100
        print("\n  MATCHED contrast — same CEEMDAN, same features, same "
              "learner; only the protocol differs:")
        print(f"    A2 global CEEMDAN (leaky) : MAPE={mA2['MAPE']:.2f}%  "
              f"MAE={mA2['MAE']:.2f}  RMSE={mA2['RMSE']:.2f}")
        print(f"    C  per-window CEEMDAN     : MAPE={mC['MAPE']:.2f}%  "
              f"MAE={mC['MAE']:.2f}  RMSE={mC['RMSE']:.2f}")
        print(f"    Leakage effect, isolated  : {infl2:.1f}%")

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
    if mA2 is not None:
        out["ceemdan_trials"] = int(args.trials)
        out["ceemdan_K"] = int(args.K)
        out["protocol_A2_global_ceemdan"] = {k: float(v) for k, v in mA2.items()}
        out["protocol_C_causal_ceemdan"] = {k: float(v) for k, v in mC.items()}
        out["leakage_effect_matched_pct"] = float(infl2)
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
