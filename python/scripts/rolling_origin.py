#!/usr/bin/env python3
"""Rolling-origin evaluation — does the ranking hold across test periods?

Why this exists
---------------
Every headline number in the paper rests on ONE chronological 70/15/15 split
per dataset, which Section 5.7(ii) lists as a threat to validity: a ranking
established on a single test period may be an artefact of that period. PJM is
the only benchmark long enough to answer the question — seventeen years, so
several disjoint test years can be carved out while keeping a training segment
of realistic length.

Protocol
--------
For each of K origins the series is cut at a moving point: everything before it
is split 82/18 into train/validation, and the following `test_days` days form
the test period. Origins are spaced so that the test periods do not overlap, so
the K estimates are independent samples of "a year in which this model might be
deployed". Windows are non-overlapping day-ahead forecasts, exactly as in the
main protocol, and the observed-data mask applies identically.

Everything else — preprocessing, scaling fitted on that origin's training
segment only, seeds, early stopping — matches main.py. No statistic from a
later origin can reach an earlier one.

Usage
-----
    python scripts/rolling_origin.py --dataset PJM --origins 4 --seeds 42 43
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import DATASETS, MODEL_PARAMS, DECOMP_PARAMS, TRAIN_PARAMS, RESULTS_DIR
from data_utils import prepare_dataset, STLFDataset
from metrics_stats import compute_metrics, diebold_mariano_multiseed
from model_proposed import create_proposed
from train_pipeline import train_model, predict, set_seed, _make_dl_model
from torch.utils.data import DataLoader

DEFAULT_MODELS = ["Proposed", "PatchTST", "DLinear", "TCN", "TiDE"]


def loaders_for(data, tr_end, va_end, te_end, batch, H):
    """Train/val/test loaders for one origin. Mirrors make_loaders but with
    explicit boundaries instead of the global 70/15/15 fractions."""
    tr = STLFDataset(data, 0, tr_end)
    va = STLFDataset(data, tr_end, va_end, stride=H)
    te = STLFDataset(data, va_end, te_end, stride=H)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch, shuffle=sh,
                                   drop_last=sh, num_workers=0)
    return mk(tr, True), mk(va, False), mk(te, False), len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PJM")
    ap.add_argument("--origins", type=int, default=4)
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    data = prepare_dataset(args.dataset)
    cfg = data["cfg"]
    H, spd = cfg["pred_horizon"], cfg["steps_per_day"]
    n = len(data["load_z"])
    test_len = args.test_days * spd

    # Place the last origin's test period at the very end of the series and
    # walk backwards in non-overlapping blocks.
    ends = [n - i * test_len for i in range(args.origins)][::-1]
    print(f"{args.dataset}: n={n:,}  horizon={H}  test block={test_len:,} steps "
          f"({args.test_days} d)\norigins (test end index): {ends}", flush=True)

    out = {"dataset": args.dataset, "test_days": args.test_days,
           "seeds": args.seeds, "origins": []}
    for oi, te_end in enumerate(ends):
        va_end = te_end - test_len
        tr_end = int(va_end * 0.82)
        idx = data["index"]
        print(f"\n=== origin {oi + 1}/{len(ends)} === train -> {idx[tr_end].date()}"
              f" | val -> {idx[va_end].date()} | test -> {idx[te_end - 1].date()}",
              flush=True)
        rec = {"train_end": str(idx[tr_end].date()),
               "val_end": str(idx[va_end].date()),
               "test_end": str(idx[te_end - 1].date()), "models": {}}
        preds_by_model = {}
        for name in models:
            per_seed, mets = [], []
            for s in args.seeds:
                set_seed(s)
                tr, va, te, n_te = loaders_for(data, tr_end, va_end, te_end,
                                               TRAIN_PARAMS["batch_size"], H)
                model = _make_dl_model(name, data, s)
                model, _ = train_model(model, tr, va)
                yp, yt, _, _, _ = predict(model, te)
                sc = data["scaler_load"]
                yp, yt = sc.inverse_transform(yp), sc.inverse_transform(yt)
                per_seed.append(yp)
                mets.append(compute_metrics(yt, yp))
            y_true = yt
            mean = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
            std = {k: float(np.std([m[k] for m in mets], ddof=1))
                   if len(mets) > 1 else 0.0 for k in mets[0]}
            rec["models"][name] = {"mean": mean, "std": std, "n_test": n_te}
            preds_by_model[name] = per_seed
            print(f"   {name:10s} MAPE={mean['MAPE']:.3f} "
                  f"± {std['MAPE']:.3f}  (n={n_te})", flush=True)

        # DM of the proposed model against each baseline, within this origin
        if "Proposed" in preds_by_model:
            rec["dm"] = {}
            for name, p in preds_by_model.items():
                if name == "Proposed":
                    continue
                d = diebold_mariano_multiseed(y_true, preds_by_model["Proposed"], p)
                rec["dm"][name] = {"dm_stat": float(d["dm_stat"]),
                                   "p_value": float(d["p_value"])}
        out["origins"].append(rec)

    # Summary: per-model mean MAPE across origins and how often it ranks first
    names = models
    tbl = {m: [o["models"][m]["mean"]["MAPE"] for o in out["origins"]]
           for m in names if all(m in o["models"] for o in out["origins"])}
    out["per_origin_MAPE"] = tbl
    out["mean_across_origins"] = {m: float(np.mean(v)) for m, v in tbl.items()}
    out["wins"] = {m: int(sum(1 for i in range(len(out["origins"]))
                              if min(tbl, key=lambda k: tbl[k][i]) == m))
                   for m in tbl}
    print("\n" + "=" * 62 + "\nMAPE (%) by origin\n" + "=" * 62)
    hdr = "model      " + "".join(f"  origin {i+1}" for i in range(len(ends))) \
          + "     mean   wins"
    print(hdr)
    for m, v in sorted(tbl.items(), key=lambda kv: np.mean(kv[1])):
        print(f"{m:10s}" + "".join(f"  {x:8.3f}" for x in v)
              + f"  {np.mean(v):7.3f}  {out['wins'][m]:4d}")

    path = os.path.join(RESULTS_DIR, f"rolling_origin_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
