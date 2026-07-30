#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  STLF V4 Experiment Pipeline — Applied Energy (leak-free)

  Usage:
    python main.py                          # full: 3 datasets, all models,
                                            #       5 seeds, ablation, DM tests
    python main.py --dataset GEFCom2014     # one dataset
    python main.py --quick                  # 1 seed, reduced baselines
    python main.py --smoke                  # tiny run to validate the code
    python main.py --ablation-only
    python main.py --figures-only           # regenerate tables/figures
═══════════════════════════════════════════════════════════════
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np

# Make console output robust on Windows cp1252 terminals (the banners and
# progress markers use Unicode box-drawing characters).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import (DATASETS, SEEDS, PRIMARY_SEED, RESULTS_DIR, FIGURES_DIR,
                    BASELINES, TRAIN_PARAMS, DEVICE)
from data_utils import prepare_dataset
from metrics_stats import (compute_metrics, metrics_mean_std, dm_table,
                           diebold_mariano_multiseed, holm_bonferroni)
from train_pipeline import (run_dl_experiment, run_seasonal_naive, run_gbm,
                            run_arima, set_seed)


def preflight_dependencies(model_list, strict=True):
    """Import every optional dependency the requested models need, up front,
    so the multi-day run does not silently drop baselines (round-3 audit)."""
    need = {}
    if "XGBoost" in model_list:
        need["xgboost"] = "xgboost"
    if "LightGBM" in model_list:
        need["lightgbm"] = "lightgbm"
    if "ARIMA" in model_list:
        need["statsmodels"] = "statsmodels.tsa.arima.model"
    missing = []
    for pkg, mod in need.items():
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        msg = ("Missing optional dependencies for requested baselines: "
               + ", ".join(missing) + ". Install them (pip install -r "
               "requirements.txt) or drop those models.")
        if strict:
            raise SystemExit("PREFLIGHT ABORT: " + msg)
        print("WARNING: " + msg)
    return missing

DL_BASELINES = ["LSTM", "BiLSTM", "GRU", "TCN", "Transformer", "CNN_LSTM",
                "GRU_TCN_Attention", "DLinear", "PatchTST", "TiDE"]
ABLATION_VARIANTS = [
    ("Full (Proposed)", "Proposed", {}),
    ("w/o Stage 1 (multi-scale split)", "Proposed_noStage1", {}),
    ("w/o Stage 2 (filter bank)", "Proposed_noStage2", {}),
    ("w/o adaptive gating", "Proposed_noGate", {}),
    ("w/o patch embedding", "Proposed_noPatch", {}),
    ("w/o cross-attention", "Proposed_noCross", {}),
    ("w/o BiGRU decoder", "Proposed_noBiGRU", {}),
    ("w/o RevIN", "Proposed_noRevIN", {}),
    ("w/o future covariates", "Proposed_noFutureCov", {}),
    ("w/o linear skip", "Proposed_noLinearSkip", {}),
    ("w/o covariate skip", "Proposed_noCovSkip", {}),
    ("w/ causal error correction", "Proposed", {"use_ec": True}),
]


def run_model_multiseed(name, data, seeds, use_ec=False):
    """Run a model across seeds; returns per-seed results + aggregate."""
    per_seed = []
    for s in seeds:
        print(f"    seed {s}...", flush=True)
        if name == "SeasonalNaive":
            r = run_seasonal_naive(data)
        elif name == "ARIMA":
            r = run_arima(data)
        elif name in ("XGBoost", "LightGBM"):
            r = run_gbm(name, data, s)
        else:
            r = run_dl_experiment(name, data, s, use_ec=use_ec,
                                  save_tag=name if s == PRIMARY_SEED else None)
        per_seed.append(r)
        m = r["metrics"]
        print(f"      MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"MAPE={m['MAPE']:.2f}%  R2={m['R2']:.4f}")
        if name in ("SeasonalNaive", "ARIMA"):
            break                       # deterministic — one run suffices
    mean, std = metrics_mean_std([r["metrics"] for r in per_seed])
    out = {"per_seed": per_seed, "metrics_mean": mean, "metrics_std": std}

    # [V4.1] Seed ensemble: mean of the per-seed test predictions (all seeds
    # forecast the SAME issue times, so alignment is exact). Standard, honest
    # variance-reduction; reported as an additional table row.
    if len(per_seed) > 1 and all("y_pred" in r for r in per_seed):
        n_min = min(r["y_pred"].shape[0] for r in per_seed)
        y_ens = np.mean([r["y_pred"][:n_min] for r in per_seed], axis=0)
        y_true = per_seed[0]["y_true"][:n_min]
        out["ensemble"] = {"metrics": compute_metrics(y_true, y_ens),
                           "y_pred": y_ens, "y_true": y_true}
    return out


def run_dataset(ds_name, seeds, model_list, use_ec_on_proposed=True):
    print(f"\n{'#' * 64}\n# DATASET: {ds_name}\n{'#' * 64}")
    data = prepare_dataset(ds_name)
    results = {}
    for name in model_list:
        print(f"\n  ── {name} ──")
        try:
            results[name] = run_model_multiseed(
                name, data, seeds,
                use_ec=(use_ec_on_proposed and name == "Proposed"))
        except ImportError as e:
            print(f"    SKIPPED (missing dependency: {e})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")

    # ── Diebold-Mariano significance testing ──
    # Headline test uses the 5-seed mean loss profile (dm_multiseed); the
    # single-seed (seed 42) test is kept for reference. Holm-Bonferroni
    # controls the family-wise error across baselines. Ensembles are compared
    # like-with-like (Proposed-Ens vs each baseline-Ens). [round-3 audit]
    dm = {}
    dm_ms = {}
    dm_ens = {}
    n_test = 0
    if "Proposed" in results and results["Proposed"]["per_seed"]:
        ref = results["Proposed"]["per_seed"][0]
        y_true = ref["y_true"]
        n_test = len(y_true)
        ref_all = [s["y_pred"] for s in results["Proposed"]["per_seed"]]

        preds = {"Proposed": ref["y_pred"]}
        for name, r in results.items():
            if name != "Proposed" and r["per_seed"]:
                preds[name] = r["per_seed"][0]["y_pred"]
        dm = dm_table(y_true, preds, reference="Proposed")

        # multi-seed averaged-loss DM (the honest headline)
        raw_ms = {}
        for name, r in results.items():
            if name == "Proposed" or not r["per_seed"]:
                continue
            base_all = [s["y_pred"] for s in r["per_seed"]]
            raw_ms[name] = diebold_mariano_multiseed(y_true, ref_all, base_all)
            mae = diebold_mariano_multiseed(y_true, ref_all, base_all,
                                            loss="mae")
            raw_ms[name]["dm_mae"] = mae["dm_stat"]
            raw_ms[name]["p_value_mae"] = mae["p_value"]
        holm_ms = holm_bonferroni({k: v["p_value"] for k, v in raw_ms.items()})
        for k in raw_ms:
            raw_ms[k]["p_holm"] = holm_ms[k]
        dm_ms = raw_ms

        # symmetric ensemble comparison
        if "ensemble" in results["Proposed"]:
            ens_preds = {"Proposed-Ens": results["Proposed"]["ensemble"]["y_pred"]}
            for name, r in results.items():
                if name != "Proposed" and "ensemble" in r:
                    ens_preds[name + "-Ens"] = r["ensemble"]["y_pred"]
            if len(ens_preds) > 1:
                dm_ens = dm_table(results["Proposed"]["ensemble"]["y_true"],
                                  ens_preds, reference="Proposed-Ens")

        print(f"\n  Diebold-Mariano (Proposed vs baselines), {ds_name} "
              f"[n={n_test} day-ahead forecasts]:")
        print(f"    {'baseline':22s} {'DM(5-seed)':>11s} {'p_Holm':>8s}  "
              f"{'DM(seed42)':>11s}")
        for k in dm_ms:
            v = dm_ms[k]
            sig = "**" if v["p_holm"] < 0.01 else ("*" if v["p_holm"] < 0.05 else "")
            s42 = dm.get(k, {}).get("dm_stat", float("nan"))
            print(f"    {k:22s} {v['dm_stat']:+11.3f} {v['p_holm']:8.4f}  "
                  f"{s42:+11.3f} {sig}")

    # [V5] Wall-clock stamp of the forecasts. With stride == horizon == one
    # day every forecast is issued at the same clock time, so lead-time
    # position and time of day are perfectly confounded; Fig. 10 needs this to
    # show which of the two is actually driving the error.
    idx = data["index"]
    step_min = int((idx[1] - idx[0]).total_seconds() // 60)
    t0 = results["Proposed"]["per_seed"][0].get("issue_idx") \
        if "Proposed" in results else None
    clock = None
    if t0 is not None and len(t0):
        hrs = {idx[int(t)].hour + idx[int(t)].minute / 60 for t in t0}
        if len(hrs) == 1:
            clock = {"issue_hour": hrs.pop(), "step_min": step_min}

    return {"results": results, "dm": dm, "dm_multiseed": dm_ms,
            "dm_ensemble": dm_ens, "clock": clock,
            "data_meta": {
                "n": len(data['load_z']), "train_end": data['train_end'],
                "val_end": data['val_end'], "n_cov_past": data['n_cov_past'],
                "n_cov_fut": data['n_cov_fut'], "n_test_forecasts": n_test}}


def run_ablation(ds_name, seeds):
    """Every variant differs from Full in EXACTLY ONE component ([FIX] for
    the V3 confounded ablation)."""
    print(f"\n{'#' * 64}\n# ABLATION on {ds_name}\n{'#' * 64}")
    data = prepare_dataset(ds_name)
    out = {"_dataset": ds_name}
    # Skip any variant that would be a no-op on this dataset: disabling a
    # component the selection rule already leaves off would train the SAME
    # model twice and report the gap as an ablation result (round-5 audit).
    from config import COV_SKIP_BY_DATASET
    variants = list(ABLATION_VARIANTS)
    if not COV_SKIP_BY_DATASET.get(ds_name, True):
        variants = [v for v in variants if v[1] != "Proposed_noCovSkip"]
        print("  (skipping 'w/o covariate skip': the path is already off on "
              f"{ds_name} under the validation-selected configuration)")
    for label, model_name, kw in variants:
        print(f"\n  ── {label} ──")
        try:
            r = run_model_multiseed(model_name, data, seeds, **kw)
            key = ("metrics_ec" if kw.get("use_ec") else "metrics")
            per_seed_metrics = [s.get(key, s["metrics"]) for s in r["per_seed"]]
            mean, std = metrics_mean_std(per_seed_metrics)
            out[label] = {"mean": mean, "std": std}
            print(f"    -> MAPE={mean['MAPE']:.2f}±{std['MAPE']:.2f}%")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED: {e}")
    # Persist unconditionally so an --ablation-only run leaves artifacts even
    # when all_results is empty (round-3 audit).
    with open(os.path.join(RESULTS_DIR, "ablation_v4.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  Saved: {RESULTS_DIR}/ablation_v4.json")
    return out


def save_all(all_results, ablation, tag="v4"):
    """Persist a compact summary (JSON) + full pickle for figures.

    [round-5 audit] MERGES into whatever is already on disk instead of
    replacing it. A `--ablation-only` run has an empty all_results, so the
    previous version wrote {"results": {}} over the file and destroyed the
    entire multi-seed protocol; a single-dataset run silently dropped the other
    two. Runs are additive: a dataset absent from this call keeps its stored
    results, and ablations accumulate per dataset under "ablations".
    """
    sum_path = os.path.join(RESULTS_DIR, f"summary_{tag}.json")
    prev = {}
    if os.path.exists(sum_path):
        try:
            with open(sum_path) as f:
                prev = json.load(f)
        except Exception as e:
            print(f"  (existing summary unreadable, starting fresh: {e})")
    summary = dict(prev.get("results") or {})
    ablations = dict(prev.get("ablations") or {})
    if prev.get("ablation") and not ablations:      # migrate the old scalar
        old = prev["ablation"]
        ablations[old.get("_dataset", "GEFCom2014")] = old

    for ds, block in all_results.items():
        # Merge at the MODEL level, not the dataset level: running a subset
        # (e.g. --models Proposed,TiDE to add one baseline) must not delete the
        # twelve models already measured for that dataset.
        summary.setdefault(ds, {})
        for name, r in block["results"].items():
            summary[ds][name] = {
                "mean": r["metrics_mean"], "std": r["metrics_std"],
                "n_params": r["per_seed"][0].get("n_params", 0),
                "train_time_s": float(np.mean(
                    [s.get("train_time_s", 0) for s in r["per_seed"]])),
                "inference_ms": float(np.mean(
                    [s.get("inference_ms", 0) for s in r["per_seed"]])),
            }
            if any("metrics_ec" in s for s in r["per_seed"]):
                ec_ms = [s["metrics_ec"] for s in r["per_seed"]
                         if "metrics_ec" in s]
                summary[ds][name]["mean_ec"], summary[ds][name]["std_ec"] = \
                    metrics_mean_std(ec_ms)
            if "ensemble" in r:
                summary[ds][name]["ensemble"] = r["ensemble"]["metrics"]
        # DM entries merge too, and the multiplicity correction is then
        # RECOMPUTED over the enlarged family: adding a fourteenth baseline
        # makes every stored p_holm stale, and leaving them would understate
        # the correction the paper claims to apply.
        for key in ("_dm", "_dm_multiseed", "_dm_ensemble"):
            src = block.get(key.lstrip("_") if key != "_dm" else "dm") or {}
            if key == "_dm_multiseed":
                src = block.get("dm_multiseed", {})
            elif key == "_dm_ensemble":
                src = block.get("dm_ensemble", {})
            merged = dict(summary[ds].get(key) or {})
            merged.update(src)
            summary[ds][key] = merged
        for key in ("_dm_multiseed", "_dm_ensemble"):
            fam = summary[ds].get(key) or {}
            raw = {k: v["p_value"] for k, v in fam.items()
                   if isinstance(v, dict) and "p_value" in v}
            if raw:
                for k, ph in holm_bonferroni(raw).items():
                    fam[k]["p_holm"] = ph
                print(f"  [{ds}] {key}: Holm recomputed over "
                      f"{len(raw)} comparisons")
        summary[ds]["_meta"] = block.get("data_meta", {})
    if ablation:
        ablations[ablation.get("_dataset", "GEFCom2014")] = ablation
    with open(sum_path, "w") as f:
        json.dump({"results": summary, "ablations": ablations,
                   # kept so older readers still find the primary ablation
                   "ablation": ablations.get("GEFCom2014")
                               or (next(iter(ablations.values()), None))},
                  f, indent=2, default=float)

    pkl_path = os.path.join(RESULTS_DIR, f"predictions_{tag}.pkl")
    slim = {}
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                slim = pickle.load(f)
        except Exception as e:
            print(f"  (existing predictions unreadable, starting fresh: {e})")
    for ds, block in all_results.items():
        # Model-level merge here too. The summary was fixed for this and the
        # pickle was not, so a `--models TiDE` run kept every model's summary
        # row and deleted every model's PREDICTIONS — which are what Figs. 5-7,
        # 9 and 10 are drawn from. One incomplete fix is worse than none,
        # because the surviving summary made the loss look impossible.
        slim.setdefault(ds, {})
        for name, r in block["results"].items():
            p0 = r["per_seed"][0]
            slim[ds][name] = {k: p0.get(k) for k in
                              ("y_pred", "y_true", "issue_idx", "attn")}
            # [V6] ALL seeds' forecasts, not just the primary one. The
            # five-seed Diebold-Mariano test needs every seed's loss profile,
            # and storing only seed 42 meant that adding one baseline later
            # forced a full retrain of the proposed model just to obtain the
            # comparison - which perturbed its own metrics each time.
            slim[ds][name]["y_pred_all"] = np.stack(
                [s["y_pred"] for s in r["per_seed"] if "y_pred" in s])                 if all("y_pred" in s for s in r["per_seed"]) else None
        if block.get("clock"):
            slim[ds]["_clock"] = block["clock"]     # needed by Fig. 10
    with open(pkl_path, "wb") as f:
        pickle.dump(slim, f)
    kept = sorted(set(summary) - set(all_results))
    print(f"\nSaved: {sum_path} + predictions_{tag}.pkl"
          + (f"  (merged; kept stored results for {', '.join(kept)})"
             if kept else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset to run, e.g. 'Proposed,TiDE'. "
                         "Results merge into the stored summary; models "
                         "not listed keep their stored values.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny validation run (few epochs, 1 seed, 1 dataset)")
    ap.add_argument("--ablation-only", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    t0 = time.time()

    if args.figures_only:
        from figures_tables import regenerate_from_saved
        regenerate_from_saved()
        return

    seeds = SEEDS
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    model_list = ["Proposed", "SeasonalNaive", "ARIMA", "XGBoost", "LightGBM"] \
        + DL_BASELINES

    if args.quick:
        seeds = [PRIMARY_SEED]
        model_list = ["Proposed", "SeasonalNaive", "DLinear", "PatchTST",
                      "GRU_TCN_Attention"]
    if args.smoke:
        TRAIN_PARAMS["epochs"] = 2
        TRAIN_PARAMS["patience"] = 2
        seeds = [PRIMARY_SEED]
        datasets = datasets[:1]
        model_list = ["Proposed", "SeasonalNaive", "DLinear"]

    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in want if m not in model_list + ["Proposed"]]
        if unknown:
            raise SystemExit(f"unknown model(s): {unknown}")
        model_list = want
        print(f"running subset: {model_list}")

    # Preflight: abort early if a requested baseline's dependency is missing,
    # instead of silently dropping it after a multi-day run (round-3 audit).
    preflight_dependencies(model_list, strict=not (args.smoke or args.quick))

    all_results = {}
    if not args.ablation_only:
        for ds in datasets:
            all_results[ds] = run_dataset(ds, seeds, model_list)
            # Checkpoint after each dataset so a late failure cannot discard
            # the datasets already completed (round-3 audit).
            try:
                save_all(all_results, None, tag="v4_partial")
            except Exception as e:
                print(f"(partial checkpoint skipped: {e})")

    ablation = None
    if not args.no_ablation and not args.smoke:
        ablation = run_ablation(datasets[0], seeds if not args.quick
                                else [PRIMARY_SEED])
    elif args.smoke:
        # smoke-test one ablation variant to exercise the flag machinery
        data = prepare_dataset(datasets[0])
        r = run_dl_experiment("Proposed_noStage2", data, PRIMARY_SEED)
        print(f"  smoke ablation OK: MAPE={r['metrics']['MAPE']:.2f}%")

    if all_results or ablation:
        save_all(all_results, ablation)
        try:
            from figures_tables import generate_all
            generate_all(all_results, ablation)
        except Exception as e:
            print(f"(figures/tables skipped: {e})")

    print(f"\nTotal: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
