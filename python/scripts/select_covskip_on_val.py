#!/usr/bin/env python3
"""Re-derive the covariate-skip configuration using VALIDATION ONLY.

Why this script exists
----------------------
An earlier version of this project enabled the full-resolution covariate path
(Section 3.7) with the rule `use_cov_skip = has_temperature`. That rule was not
adopted a priori: it was chosen after inspecting TEST scores, which is exactly
the selection-leakage channel the paper criticises elsewhere. Resting on a
legitimate property of the dataset does not launder it — what contaminates the
result is how the rule was picked.

This script replaces that with an explicit, reproducible selection rule that
never touches the test split:

    enable the full-resolution covariate path on a dataset iff it lowers the
    VALIDATION MAPE of the proposed model on that dataset, averaged over seeds.

Whatever it prints is copied verbatim into `config.COV_SKIP_BY_DATASET`,
including where it contradicts the earlier test-driven choice.

Usage
-----
    python scripts/select_covskip_on_val.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import (DATASETS, MODEL_PARAMS, DECOMP_PARAMS, TRAIN_PARAMS,
                    RESULTS_DIR)
from data_utils import prepare_dataset, make_loaders
from model_proposed import create_proposed
from train_pipeline import train_model, predict, set_seed
from metrics_stats import compute_metrics

SEEDS = [42, 43, 44]


def main():
    print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    print("Selection rule: enable cov_skip iff it lowers VALIDATION MAPE.\n",
          flush=True)

    table, record = {}, {}
    for ds in ["GEFCom2014", "PJM", "AEMO"]:
        data = prepare_dataset(ds, verbose=False)
        cfg = DATASETS[ds]
        H = cfg["pred_horizon"]
        row = {}
        for use_skip in (True, False):
            vals = []
            for s in SEEDS:
                print(f"    [{ds} cov_skip={use_skip} seed={s}] training...",
                      flush=True)
                set_seed(s)
                tr, va, _ = make_loaders(data, TRAIN_PARAMS["batch_size"],
                                         eval_stride=H)
                model = create_proposed(cfg, MODEL_PARAMS, DECOMP_PARAMS,
                                        data["n_cov_past"], data["n_cov_fut"],
                                        use_cov_skip=use_skip)
                model, _ = train_model(model, tr, va)
                # VALIDATION ONLY — the test loader is never built or touched
                vp, vt, _, _, _ = predict(model, va)
                sc = data["scaler_load"]
                m = compute_metrics(sc.inverse_transform(vt),
                                    sc.inverse_transform(vp))
                vals.append(m["MAPE"])
            row[use_skip] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
            print(f"  {ds:11s} cov_skip={str(use_skip):5s} "
                  f"val MAPE = {row[use_skip][0]:.3f} ± {row[use_skip][1]:.3f}",
                  flush=True)
        decision = row[True][0] < row[False][0]
        table[ds] = decision
        record[ds] = {
            "val_MAPE_with_cov_skip": {"mean": row[True][0], "std": row[True][1]},
            "val_MAPE_without_cov_skip": {"mean": row[False][0],
                                          "std": row[False][1]},
            "delta_pp": row[True][0] - row[False][0],
            "selected": bool(decision),
        }
        print(f"  -> {ds}: cov_skip {'ENABLED' if decision else 'DISABLED'} "
              f"(val delta {row[True][0] - row[False][0]:+.3f} pp)\n",
              flush=True)

    print("VALIDATION-SELECTED CONFIGURATION:", table, flush=True)

    # Persist the evidence. A selection rule that only prints to a terminal is
    # not falsifiable from the release; this file is what lets a reader check
    # that config.COV_SKIP_BY_DATASET is what the rule actually produced.
    out = {"rule": "enable the full-resolution covariate path on a dataset iff "
                   "it lowers mean VALIDATION MAPE on that dataset",
           "seeds": SEEDS, "split": "validation only; the test loader is never "
                                    "built in this script",
           "per_dataset": record, "selected_configuration": table}
    path = os.path.join(RESULTS_DIR, "covskip_selection.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written: {path}")
    print("\nCopy the mapping above into config.COV_SKIP_BY_DATASET and re-run "
          "main.py on any dataset whose flag changed.")


if __name__ == "__main__":
    main()
