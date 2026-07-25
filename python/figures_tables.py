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
        # [V4.1] seed-ensemble row, shown only for the Proposed model to avoid
        # doubling the table; every model's ensemble is still computed and the
        # symmetric ensemble-vs-ensemble DM comparison lives in _dm_ensemble
        # (round-4 audit: was emitting an -Ens row for every baseline).
        if "ensemble" in r and name == "Proposed":
            e = r["ensemble"]
            rows.append({
                "Model": name.replace("_", "-") + "-Ens",
                "MAE": f"{e['MAE']:.2f}", "RMSE": f"{e['RMSE']:.2f}",
                "MAPE (%)": f"{e['MAPE']:.2f}", "R2": f"{e['R2']:.4f}",
                "sMAPE (%)": f"{e['sMAPE']:.2f}", "_sort": e["MAPE"],
            })
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


def table_ablation(ablation, ds_name="GEFCom2014", table_num=6):
    import pandas as pd
    rows = []
    for label, r in ablation.items():
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


def table_efficiency(summary, ds_name, table_num=7):
    import pandas as pd
    rows = []
    for name, r in summary.items():
        if name.startswith("_"):
            continue
        rows.append({"Model": name.replace("_", "-"),
                     "Parameters": f"{r['n_params']:,}",
                     "Train time (s)": f"{r['train_time_s']:.0f}",
                     "Inference (ms/forecast)": f"{r['inference_ms']:.2f}"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR,
                           f"Table{table_num}_efficiency_{ds_name}.csv"),
              index=False)
    with open(os.path.join(RESULTS_DIR,
                           f"Table{table_num}_efficiency_{ds_name}.tex"),
              "w") as f:
        f.write(df.to_latex(index=False,
                caption=("Computational cost (single model per dataset; "
                         "inference = one full day-ahead forecast, batch "
                         "size 1 equivalent)."),
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

def fig_decomposition(data_load_window, comps, labels, name="Fig3_causal_decomposition"):
    """Causal decomposition of ONE input window (what the model actually sees)."""
    M = comps.shape[0]
    fig, axes = plt.subplots(M + 1, 1, figsize=(10, 1.3 * (M + 1)),
                             sharex=True)
    axes[0].plot(data_load_window, color="black", lw=0.9)
    axes[0].set_ylabel("Input\nwindow", fontsize=7, rotation=0,
                       labelpad=28, va="center")
    for i in range(M):
        axes[i + 1].plot(comps[i], lw=0.8,
                         color=FP["colors"]["hf_group"] if i < M // 2
                         else FP["colors"]["lf_group"])
        axes[i + 1].set_ylabel(labels[i], fontsize=7, rotation=0,
                               labelpad=28, va="center")
    axes[-1].set_xlabel("Time step within input window")
    fig.suptitle("Causal adaptive dual-stage decomposition (single window)",
                 fontweight="bold")
    _save(fig, name)


def fig_pred_vs_actual(y_true, y_pred, ds_name, fig_num, n_days=7,
                       steps_per_day=24):
    T = min(n_days * steps_per_day, y_true.size)
    yt, yp = y_true.flatten()[:T], y_pred.flatten()[:T]
    t = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    ax1.plot(t, yt, color=FP["colors"]["actual"], label="Actual", lw=1.2)
    ax1.plot(t, yp, color=FP["colors"]["proposed"], label="Proposed",
             lw=1.2, ls="--")
    ax1.set_ylabel("Load (MW)")
    ax1.set_title(f"Day-ahead forecasts — {ds_name} test set (operational "
                  "protocol, non-overlapping)", fontweight="bold")
    ax1.legend(loc="upper right")
    err = yt - yp
    ax2.fill_between(t, err, 0, alpha=0.3, color=FP["colors"]["proposed"])
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Error (MW)")
    ax2.set_xlabel("Time step")
    _save(fig, f"Fig{fig_num}_predicted_vs_actual_{ds_name}")


def fig_ablation(ablation, name="Fig7_ablation"):
    labels = list(ablation.keys())
    means = [ablation[k]["mean"]["MAPE"] for k in labels]
    stds = [ablation[k]["std"]["MAPE"] for k in labels]
    colors = [FP["colors"]["proposed"] if "Full" in v else "#5B9BD5"
              for v in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=3,
           color=colors, edgecolor="gray", width=0.6)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f"{m:.2f}", ha="center", fontsize=8,
                fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Ablation (mean ± std over 5 seeds)", fontweight="bold")
    _save(fig, name)


def fig_attention(attn, n_past, n_fut, name="Fig8_cross_attention"):
    """attn: (n_samples, N_load_tokens, N_cov_tokens). Mean over samples.
    Axis labels state EXACTLY what is plotted (past patches + future patches)."""
    A = attn.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(A, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(A.shape[1]))
    ax.set_xticklabels([f"P{i+1}" for i in range(n_past)] +
                       [f"F{i+1}" for i in range(A.shape[1] - n_past)],
                       fontsize=7)
    ax.set_yticks(range(A.shape[0]))
    ax.set_yticklabels([f"Day −{A.shape[0]-i}" for i in range(A.shape[0])],
                       fontsize=7)
    ax.axvline(n_past - 0.5, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Covariate tokens (P=past patch, F=future horizon patch)")
    ax.set_ylabel("Load patch tokens (query)")
    ax.set_title("Cross-attention weights, averaged over the full test set",
                 fontweight="bold")
    fig.colorbar(im, label="Attention weight")
    _save(fig, name)


def fig_error_by_leadtime(y_true, y_pred, name="Fig9_error_by_leadtime"):
    """Honest version of V3's Fig 9: APE vs LEAD TIME (labelled as such —
    the V3 figure labelled lead time as wall-clock hour)."""
    ape = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8) * 100
    H = y_true.shape[1]
    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot([ape[:, h] for h in range(H)], positions=range(H),
                    widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="red", lw=1.5))
    for p in bp["boxes"]:
        p.set_facecolor("#AED6F1")
    ax.set_xlabel("Forecast lead time (steps ahead)")
    ax.set_ylabel("APE (%)")
    ax.set_title("Error vs lead time — test set", fontweight="bold")
    ax.set_xticks(range(0, H, max(1, H // 12)))
    _save(fig, name)


def fig_leakage(leaky_metrics, honest_metrics, name="Fig10_leakage_effect"):
    """The V4 headline figure: same architecture, leaky vs causal protocol."""
    labels = ["MAE", "RMSE", "MAPE"]
    leaky = [leaky_metrics[k] for k in labels]
    honest = [honest_metrics[k] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, leaky, 0.38, label="Decompose-then-split (invalid)",
           color="#B22222")
    ax.bar(x + 0.2, honest, 0.38, label="Causal protocol (V4)",
           color="#2E75B6")
    for i, (l, h) in enumerate(zip(leaky, honest)):
        ax.text(i - 0.2, l, f"{l:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Look-ahead leakage inflates apparent accuracy",
                 fontweight="bold")
    ax.legend()
    _save(fig, name)


# ═══════════════════ DRIVERS ═══════════════════

def generate_all(all_results, ablation=None, tag="v4"):
    print("\nGenerating tables/figures from ACTUAL results...")
    with open(os.path.join(RESULTS_DIR, f"summary_{tag}.json")) as f:
        saved = json.load(f)
    summary = saved["results"]

    tno = 3
    for ds in summary:
        # Prefer the 5-seed mean-loss-profile DM (with Holm-adjusted p) for
        # stars and Table 8; fall back to single-seed if unavailable.
        dm = summary[ds].get("_dm_multiseed") or summary[ds].get("_dm", {})
        print(f"\nTable {tno}: {ds}")
        table_results(summary[ds], ds, tno, dm)
        table_dm(dm, ds)
        tno += 1
    if ablation:
        table_ablation(ablation)
        fig_ablation(ablation)
    if summary:                    # guard --ablation-only (empty results)
        first = next(iter(summary))
        table_efficiency(summary[first], first)

    fig_num = 4
    for ds, block in all_results.items():
        r = block["results"].get("Proposed")
        if r and r["per_seed"]:
            p0 = r["per_seed"][0]
            fig_pred_vs_actual(p0["y_true"], p0["y_pred"], ds, fig_num,
                               steps_per_day=DATASETS[ds]["steps_per_day"])
            if p0.get("attn") is not None:
                n_past = DATASETS[ds]["input_window"] // DATASETS[ds]["patch_length"]
                n_fut = p0["attn"].shape[-1] - n_past
                fig_attention(p0["attn"], n_past, n_fut,
                              name=f"Fig8_cross_attention_{ds}")
            fig_error_by_leadtime(p0["y_true"], p0["y_pred"],
                                  name=f"Fig9_error_by_leadtime_{ds}")
        fig_num += 1


def regenerate_from_saved(tag="v4"):
    path = os.path.join(RESULTS_DIR, f"summary_{tag}.json")
    if not os.path.exists(path):
        print(f"No saved results at {path}. Run the pipeline first "
              f"(or pass the correct tag).")
        return
    with open(path) as f:
        saved = json.load(f)
    summary, ablation = saved["results"], saved.get("ablation")
    tno = 3
    for ds in summary:
        dm = summary[ds].get("_dm_multiseed") or summary[ds].get("_dm", {})
        table_results(summary[ds], ds, tno, dm)
        table_dm(dm, ds)
        tno += 1
    if ablation:
        table_ablation(ablation)
        fig_ablation(ablation)
    pkl = os.path.join(RESULTS_DIR, f"predictions_{tag}.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            slim = pickle.load(f)
        fig_num = 4
        for ds, models in slim.items():
            if "Proposed" in models and models["Proposed"]["y_pred"] is not None:
                p0 = models["Proposed"]
                fig_pred_vs_actual(np.asarray(p0["y_true"]),
                                   np.asarray(p0["y_pred"]), ds, fig_num,
                                   steps_per_day=DATASETS[ds]["steps_per_day"])
            fig_num += 1
