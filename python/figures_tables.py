"""
Figures and tables — STLF V4.

Every figure/table is generated FROM ACTUAL RESULT FILES. There are no
hardcoded placeholder values (the V3 Fig. 7 ablation chart and Fig. 10
error-correction histograms were rendered from invented / np.random data;
that class of bug is structurally impossible here: generators take the
results dict as their only data source and raise if it is missing).
"""
import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, FIGURES_DIR, FIG_PARAMS as FP, DATASETS

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FP["font_size"],
    "axes.titlesize": FP["title_size"],
    "axes.labelsize": FP["label_size"],
    "xtick.labelsize": FP["tick_size"],
    "ytick.labelsize": FP["tick_size"],
    "legend.fontsize": FP["legend_size"],
    "savefig.dpi": FP["dpi"],
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


# ═══════════════════ TABLES ═══════════════════

def _fmt(mean, std, dec=2):
    return f"{mean:.{dec}f} ± {std:.{dec}f}"


def table_results(summary, ds_name, table_num, dm=None):
    """Main results table: mean ± std over seeds + DM significance marks.
    Stars use the Holm-adjusted p-value of the 5-seed mean-loss-profile DM
    test (round-3 headline); falls back to the single-seed p-value if the
    multiseed entry is absent."""
    import pandas as pd
    rows = []
    for name, r in summary.items():
        if name.startswith("_"):
            continue
        m, s = r["mean"], r["std"]
        star = ""
        if dm and name in dm:
            e = dm[name]
            p = e.get("p_holm", e.get("p_value", 1.0))
            star = " **" if p < 0.01 else (" *" if p < 0.05 else "")
        rows.append({
            "Model": name.replace("_", "-") + star,
            "MAE": _fmt(m["MAE"], s["MAE"]),
            "RMSE": _fmt(m["RMSE"], s["RMSE"]),
            "MAPE (%)": _fmt(m["MAPE"], s["MAPE"]),
            "R2": _fmt(m["R2"], s["R2"], 4),
            "sMAPE (%)": _fmt(m["sMAPE"], s["sMAPE"]),
            "_sort": m["MAPE"],
        })
        # [round-5 audit] NO ensemble row here. Earlier versions inserted the
        # proposed model's five-seed ensemble into this single-seed ranking —
        # and only its own, no baseline's — which is exactly the comparison
        # the paper says elsewhere would not be like-for-like, and on PJM and
        # AEMO it placed the proposal at the top of the table. Every model's
        # ensemble is ranked against every other in table_ensembles().
    df = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")
    df.to_csv(os.path.join(RESULTS_DIR, f"Table{table_num}_{ds_name}.csv"),
              index=False)
    with open(os.path.join(RESULTS_DIR, f"Table{table_num}_{ds_name}.tex"),
              "w") as f:
        f.write(df.to_latex(index=False,
                caption=(f"Day-ahead results on {ds_name} "
                         "(mean $\\pm$ std over 5 seeds; "
                         "*/** : DM test vs Proposed, p<0.05/0.01)."),
                label=f"tab:results_{ds_name.lower()}"))
    print(df.to_string(index=False))
    return df


def table_ablation(ablation, ds_name=None, table_num=6):
    """The dataset is taken from the ablation record itself so the file can
    never be labelled with the wrong benchmark (round-5 audit)."""
    import pandas as pd
    ds_name = ablation.get("_dataset", ds_name or "GEFCom2014")
    rows = []
    for label, r in ablation.items():
        if label.startswith("_"):
            continue
        m, s = r["mean"], r["std"]
        rows.append({"Variant": label,
                     "MAE": _fmt(m["MAE"], s["MAE"]),
                     "RMSE": _fmt(m["RMSE"], s["RMSE"]),
                     "MAPE (%)": _fmt(m["MAPE"], s["MAPE"]),
                     "R2": _fmt(m["R2"], s["R2"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR,
                           f"Table{table_num}_ablation_{ds_name}.csv"),
              index=False)
    with open(os.path.join(RESULTS_DIR,
                           f"Table{table_num}_ablation_{ds_name}.tex"),
              "w") as f:
        f.write(df.to_latex(index=False,
                caption=("Ablation on " + ds_name + ": every variant differs "
                         "from the full model in exactly one component "
                         "(mean $\\pm$ std over 5 seeds)."),
                label="tab:ablation"))
    print(df.to_string(index=False))
    return df


def table_ensembles(summary, ds_name, table_num="8b"):
    """Like-for-like seed-ensemble comparison: EVERY model's 5-seed ensemble,
    ranked together. Table 3-5 report single models; mixing an ensemble into
    that ranking would be an unfair comparison (round-5 audit)."""
    import pandas as pd
    rows = [{"Model": n.replace("_", "-") + "-Ens",
             "MAE": r["ensemble"]["MAE"], "RMSE": r["ensemble"]["RMSE"],
             "MAPE (%)": r["ensemble"]["MAPE"], "R2": r["ensemble"]["R2"]}
            for n, r in summary.items()
            if not n.startswith("_") and "ensemble" in r]
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("MAPE (%)").reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    for c in ("MAE", "RMSE", "MAPE (%)"):
        df[c] = df[c].map(lambda v: f"{v:.2f}")
    df["R2"] = df["R2"].map(lambda v: f"{v:.4f}")
    df.to_csv(os.path.join(RESULTS_DIR,
                           f"Table{table_num}_ensembles_{ds_name}.csv"),
              index=False)
    return df


def table_efficiency(summary, ds_name, table_num=7):
    import pandas as pd
    rows = []
    for name, r in summary.items():
        if name.startswith("_"):
            continue
        rows.append({"Model": name.replace("_", "-"),
                     "Parameters": f"{r['n_params']:,}",
                     "Train time (s)": f"{r['train_time_s']:.0f}",
                     # amortised over the evaluation batch (64), not a
                     # single-window latency measurement (round-5 audit)
                     "Inference (ms/forecast, amortised)":
                         f"{r['inference_ms']:.2f}"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR,
                           f"Table{table_num}_efficiency_{ds_name}.csv"),
              index=False)
    with open(os.path.join(RESULTS_DIR,
                           f"Table{table_num}_efficiency_{ds_name}.tex"),
              "w") as f:
        f.write(df.to_latex(index=False,
                caption=("Computational cost (single model per dataset). "
                         "Inference is the wall-clock time of one day-ahead "
                         "forecast amortised over the evaluation batch of "
                         "64, not a batch-1 latency measurement."),
                label="tab:efficiency"))
    return df


def table_dm(dm_block, ds_name, table_num=8):
    """DM table: statistic + raw and Holm-adjusted p (MSE loss), plus the
    MAE-loss statistic when present, and the sample size n."""
    import pandas as pd
    rows = []
    for k, v in dm_block.items():
        row = {"Baseline": k.replace("_", "-"),
               "DM (MSE)": f"{v['dm_stat']:+.3f}",
               "p (raw)": f"{v.get('p_value', float('nan')):.2e}"}
        if "p_holm" in v:
            row["p (Holm)"] = f"{v['p_holm']:.2e}"
        if "dm_mae" in v:
            row["DM (MAE)"] = f"{v['dm_mae']:+.3f}"
        if "n" in v:
            row["n"] = v["n"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"Table{table_num}_DM_{ds_name}.csv"),
              index=False)
    return df


# ═══════════════════ FIGURES ═══════════════════
# [V5] Every figure now lives in figures_diagrams.py (schematics) and
# figures_results.py (result figures), which share one validated publication
# style (figstyle.py). The earlier per-dataset figure helpers that lived here
# were removed rather than kept alongside: two figure pipelines writing
# similarly-named files into the same directory is how a stale figure ends up
# in a manuscript.


# ═══════════════════ DRIVERS ═══════════════════

def _tables(summary, ablation):
    """Every table, in manuscript order. Shared by both drivers so that a
    regeneration from saved results can never emit a different set than the
    run that produced them."""
    tno = 3
    for ds in summary:
        # Prefer the 5-seed mean-loss-profile DM (with Holm-adjusted p) for
        # stars and Table 8; fall back to single-seed if unavailable.
        dm = summary[ds].get("_dm_multiseed") or summary[ds].get("_dm", {})
        print(f"\nTable {tno}: {ds}")
        table_results(summary[ds], ds, tno, dm)
        table_dm(dm, ds)
        table_ensembles(summary[ds], ds)      # like-for-like ensemble ranking
        tno += 1
    if ablation:
        table_ablation(ablation)
    if summary:                    # guard --ablation-only (empty results)
        first = next(iter(summary))
        table_efficiency(summary[first], first)


def _figures(tag, summary, ablation):
    """Figures 1-11, all from figstyle-styled modules and saved result files.

    Imported lazily: the schematics need only matplotlib, but Fig. 3 loads a
    trained checkpoint, so a tables-only run should not pay for either.
    """
    import figures_diagrams as FD
    import figures_results as FR

    FD.fig1_protocol()
    FD.fig2_architecture()
    try:
        FR.fig3_decomposition()
    except Exception as e:                     # no checkpoint yet
        print(f"  [skip] Fig 3 (causal decomposition): {e}")
    if summary:
        FR.fig4_overview({"results": summary})
    if ablation:
        FR.fig8_ablation(ablation)

    pkl = os.path.join(RESULTS_DIR, f"predictions_{tag}.pkl")
    if not os.path.exists(pkl):
        print(f"  [skip] Figs 5-7, 9, 10: no {os.path.basename(pkl)}")
        return
    with open(pkl, "rb") as f:
        preds = pickle.load(f)
    for i, ds in enumerate(["GEFCom2014", "PJM", "AEMO"]):
        if ds in preds:
            FR.fig_forecast(preds, ds, fig_num=5 + i)
    FR.fig9_attention(preds)
    FR.fig10_leadtime(preds)
    FR.fig11_leakage()          # measured values; see leakage_demo.py


def generate_all(all_results, ablation=None, tag="v4"):
    print("\nGenerating tables/figures from ACTUAL results...")
    with open(os.path.join(RESULTS_DIR, f"summary_{tag}.json")) as f:
        saved = json.load(f)
    summary = saved["results"]
    _tables(summary, ablation)
    _figures(tag, summary, ablation)


def regenerate_from_saved(tag="v4"):
    path = os.path.join(RESULTS_DIR, f"summary_{tag}.json")
    if not os.path.exists(path):
        print(f"No saved results at {path}. Run the pipeline first "
              f"(or pass the correct tag).")
        return
    with open(path) as f:
        saved = json.load(f)
    summary, ablation = saved["results"], saved.get("ablation")
    _tables(summary, ablation)
    _figures(tag, summary, ablation)
