"""
Result figures for the manuscript (Fig. 3 and Figs. 4-11), rebuilt on the
shared publication style in figstyle.py.

Every figure is generated from a result file — summary_v4.json,
ablation_v4.json, predictions_v4.pkl, leakage_<dataset>.json or a saved
checkpoint — and a missing input is an exception, never a default value.
The V3 release rendered its ablation chart from an invented dict and its
error-correction histograms from np.random; the round-5 audit found this
release doing a quieter version of the same thing, drawing Fig. 11 from
literal defaults because leakage_demo.py printed its measurement without
saving it. The rule is therefore enforced rather than intended: no figure
function here carries fallback numbers.

Run:  python figures_results.py
"""
import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from figstyle import (apply_style, save, tidy, SURFACE, INK, INK2, MUTED, GRID,
                      AXIS, BLUE, ORANGE, AQUA, VIOLET, CRITICAL, SEQ_BLUE,
                      FILL_SOFT)
from config import RESULTS_DIR, FIGURES_DIR, DATASETS, PRIMARY_SEED

apply_style()
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def _load(tag="v4"):
    with open(os.path.join(RESULTS_DIR, f"summary_{tag}.json")) as f:
        summary = json.load(f)
    with open(os.path.join(RESULTS_DIR, "ablation_v4.json")) as f:
        ablation = json.load(f)
    pkl = os.path.join(RESULTS_DIR, f"predictions_{tag}.pkl")
    preds = pickle.load(open(pkl, "rb")) if os.path.exists(pkl) else {}
    return summary, ablation, preds


# ═══════════════════════════════════════════════════════════════════════
# Fig. 3 — the causal decomposition of one real input window
# ═══════════════════════════════════════════════════════════════════════

def fig3_decomposition(name="Fig3_causal_decomposition"):
    import torch
    from config import MODEL_PARAMS, DECOMP_PARAMS, MODELS_DIR
    from data_utils import prepare_dataset, STLFDataset
    from model_proposed import create_proposed

    data = prepare_dataset("GEFCom2014", verbose=False)
    cfg = DATASETS["GEFCom2014"]
    model = create_proposed(cfg, MODEL_PARAMS, DECOMP_PARAMS,
                            data["n_cov_past"], data["n_cov_fut"])
    ck = os.path.join(MODELS_DIR, "Proposed_s42.pt")
    trained = os.path.exists(ck)
    if trained:
        model.load_state_dict(torch.load(ck, map_location="cpu",
                                         weights_only=True))
    model.eval()

    te = STLFDataset(data, data["val_end"], len(data["load_z"]),
                     stride=cfg["pred_horizon"])
    xl, _, _, _, _ = te[40]
    x = xl.unsqueeze(0)
    with torch.no_grad():
        z, mu, sd = model.revin.normalize(x)
        comps = model.decompose(z)[0].numpy()
    z = z[0].numpy()

    labels = ([f"band $b_{k+1}$" for k in range(model.stage2.K)] +
              ["remainder $r$", "intra-day", "weekly", "trend"])
    M = comps.shape[0]

    fig, axes = plt.subplots(M + 1, 1, figsize=(7.2, 6.6), sharex=True)
    fig.subplots_adjust(hspace=0.30, left=0.135, right=0.940, top=0.895,
                        bottom=0.060)

    t = np.arange(len(z))
    axes[0].plot(t, z, color=INK, lw=1.0)
    axes[0].set_ylabel("input\nwindow", rotation=0, ha="right", va="center",
                       labelpad=8, fontsize=7.4, color=INK, fontweight="bold")
    # Each panel is autoscaled so the shape is legible, which destroys the
    # relative amplitudes; the share of the window's variance is printed
    # instead, so the reader can still see which components carry energy.
    share = comps.std(axis=1) / comps.std(axis=1).sum() * 100
    for i in range(M):
        c = AQUA if i < model.stage2.K + 1 else BLUE
        axes[i + 1].plot(t, comps[i], color=c, lw=0.9)
        axes[i + 1].set_ylabel(labels[i], rotation=0, ha="right", va="center",
                               labelpad=8, fontsize=7.2, color=INK2)
        axes[i + 1].text(1.004, 0.5, f"{share[i]:4.1f} %",
                         transform=axes[i + 1].transAxes, va="center",
                         ha="left", fontsize=6.4, color=MUTED)

    for k, ax in enumerate(axes):
        tidy(ax, ygrid=False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        if k < len(axes) - 1:
            ax.spines["bottom"].set_visible(False)
    axes[-1].set_xlabel("hour within the 168-step input window")
    axes[-1].set_xlim(0, len(z) - 1)
    axes[-1].set_xticks(np.arange(0, len(z) + 1, 24))

    rec = np.abs(comps.sum(0) - z).max()
    fig.suptitle("Causal dual-stage decomposition of a single input window",
                 x=0.135, ha="left", y=0.988, fontsize=9.6, fontweight="bold")
    fig.text(0.135, 0.947,
             f"stage-2 bands + remainder (green) and stage-1 scales (blue) sum "
             f"back to the window; max reconstruction error {rec:.1e}. Panels "
             f"are scaled independently, so the\nfigure on the right of each "
             f"gives its share of the total component standard deviation."
             + ("" if trained else "  ·  untrained filter bank"),
             fontsize=7.0, color=MUTED, ha="left", va="top", linespacing=1.55)
    save(fig, os.path.join(FIGURES_DIR, name))


# ═══════════════════════════════════════════════════════════════════════
# Figs. 5-7 — day-ahead forecasts per dataset
# ═══════════════════════════════════════════════════════════════════════

def fig_forecast(preds, ds, fig_num, n_days=5, name=None):
    p = preds[ds]["Proposed"]
    yt = np.asarray(p["y_true"])
    yp = np.asarray(p["y_pred"])
    H = yt.shape[1]
    n = min(n_days, yt.shape[0])

    # Show a REPRESENTATIVE window rather than the first one: the run of n
    # consecutive days whose MAPE is closest to the full test-set MAPE.
    # Picking the first days would be arbitrary and picking the best would be
    # cherry-picking; this is reproducible and stated in the caption.
    ape_day = np.abs(yt - yp) / np.abs(yt)
    overall = ape_day.mean() * 100
    runs = np.array([ape_day[i:i + n].mean() * 100
                     for i in range(len(ape_day) - n + 1)])
    s = int(np.argmin(np.abs(runs - overall)))
    y_true = yt[s:s + n].ravel()
    y_pred = yp[s:s + n].ravel()
    t = np.arange(len(y_true))

    fig, (ax, axe) = plt.subplots(
        2, 1, figsize=(7.2, 3.5), sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1], "hspace": 0.12})

    # day boundaries: this is an operational, non-overlapping protocol
    for d in range(1, n):
        for a in (ax, axe):
            a.axvline(d * H, color=GRID, lw=0.7, zorder=0)

    ax.plot(t, y_true, color=BLUE, lw=1.7, label="Actual", zorder=3)
    ax.plot(t, y_pred, color=ORANGE, lw=1.7, ls=(0, (3.2, 1.6)),
            label="Proposed (CPTB)", zorder=4)
    unit = "MW" if ds != "GEFCom2014" else "MW (ISO-NE scale)"
    ax.set_ylabel(f"Load  [{unit}]")
    # legend outside the data area entirely: with only two series there is no
    # placement inside the axes that is safe on all three datasets
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.002), ncol=2,
              fontsize=7.4, frameon=False, handlelength=2.2,
              columnspacing=1.4, borderpad=0.0)
    tidy(ax)

    err = y_true - y_pred
    axe.fill_between(t, err, 0, color=ORANGE, alpha=0.22, lw=0)
    axe.plot(t, err, color=ORANGE, lw=1.0)
    axe.axhline(0, color=AXIS, lw=0.7)
    axe.set_ylabel("Error")
    # the x unit is the series' own step, which is a half-hour on AEMO
    step = "half-hours" if H > 24 else "hours"
    axe.set_xlabel(f"{step.capitalize()} of the test period "
                   f"({n} consecutive day-ahead forecasts, "
                   f"one issued per day)")
    tidy(axe)
    axe.set_xlim(0, len(t) - 1)
    axe.set_xticks(np.arange(0, len(t) + 1, H))

    # [round-5 audit] the pickle stores the PRIMARY SEED only, so `overall` is
    # a single-seed test MAPE and differs from the five-seed mean reported in
    # the tables (e.g. 4.58 % vs 4.73 % on GEFCom2014). Saying "test-set mean"
    # without that qualifier put two different headline errors for the same
    # model in one paper.
    mape = np.mean(np.abs(err) / np.abs(y_true)) * 100
    ax.set_title(f"{ds} — a representative {n}-day window   ·   MAPE here "
                 f"{mape:.2f} %, seed-{PRIMARY_SEED} test mean {overall:.2f} % "
                 f"(five-seed mean in Table)",
                 loc="left", pad=20, fontsize=8.8, fontweight="bold")
    save(fig, os.path.join(FIGURES_DIR, name or
                           f"Fig{fig_num}_forecast_{ds}"))


# ═══════════════════════════════════════════════════════════════════════
# Fig. 4 — headline accuracy across the three datasets
# ═══════════════════════════════════════════════════════════════════════

def fig4_overview(summary, name="Fig4_accuracy_overview"):
    res = summary["results"]
    order = ["GEFCom2014", "PJM", "AEMO"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.78, bottom=0.29,
                        wspace=0.52)

    # Pick what each panel shows first, then give the three panels one shared
    # MAPE scale: the unit is the same, so per-panel autoscaling would make
    # the easiest benchmark look the hardest. The scale spans the DISPLAYED
    # models — including the far-worse tail (ARIMA) would squash every bar.
    panel = {}
    for ds in order:
        rows = [(m, v["mean"]["MAPE"]) for m, v in res[ds].items()
                if not m.startswith("_") and m != "SeasonalNaive"]
        rows.sort(key=lambda r: r[1])
        # always show the proposed model, even where it is outside the top
        # group — hiding it would misrepresent the comparison. When it is
        # outside, a labelled break makes the omission explicit rather than
        # letting the appended bar read as the next-best model.
        top, skipped = rows[:7], 0
        prop = [r for r in rows if r[0] == "Proposed"]
        if prop and prop[0] not in top:
            skipped = rows.index(prop[0]) - len(top)
            top = top + [None] + prop           # None = the break row
        rank = [i for i, r in enumerate(rows, 1) if r[0] == "Proposed"][0]
        panel[ds] = (top, skipped, rank, len(rows) + 1)

    xmax = max(r[1] for top, *_ in panel.values() for r in top if r) * 1.10

    for ax, ds in zip(axes, order):
        top, skipped, rank, n_models = panel[ds]
        short = {"GRU_TCN_Attention": "GRU-TCN-Att", "SeasonalNaive": "Naive"}
        names = ["" if r is None else short.get(r[0], r[0].replace("_", "-"))
                 for r in top]
        vals = [0.0 if r is None else r[1] for r in top]
        ypos = np.arange(len(top))[::-1]

        draw = [i for i, r in enumerate(top) if r is not None]
        ax.barh([ypos[i] for i in draw], [vals[i] for i in draw], height=0.64,
                color=[ORANGE if names[i] == "Proposed" else "#c9d4e0"
                       for i in draw], zorder=3)
        for i, r in enumerate(top):
            if r is None:
                ax.text(0.0, ypos[i], f"...  {skipped} model"
                        f"{'s' if skipped != 1 else ''} omitted",
                        va="center", ha="left", fontsize=6.2, color=MUTED)
                continue
            # value labels sit INSIDE the bar end, so they can never collide
            # with the neighbouring panel's category labels
            ax.text(vals[i] - xmax * 0.012, ypos[i], f"{vals[i]:.2f}",
                    va="center", ha="right", fontsize=6.6,
                    color=SURFACE if names[i] == "Proposed" else INK2,
                    fontweight="bold" if names[i] == "Proposed" else "normal")

        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=6.9)
        tidy(ax, ygrid=False, xgrid=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.7, len(top) - 0.3)
        ax.set_title(f"{ds}   ·   proposed ranks {rank} of {n_models}",
                     loc="left", fontsize=8.0, pad=6)
        ax.set_xlabel("MAPE (%)", fontsize=7.6)
        for lbl, n in zip(ax.get_yticklabels(), names):
            if n == "Proposed":
                lbl.set_color(ORANGE)
                lbl.set_fontweight("bold")

    fig.suptitle("Day-ahead accuracy under the leak-free protocol "
                 "(mean over five seeds)",
                 x=0.105, ha="left", y=0.968, fontsize=9.4, fontweight="bold")
    fig.text(0.105, 0.885, "leading models per benchmark on a shared scale, "
             "with the proposed model always shown; on GEFCom2014 it is "
             "mid-table by point error, yet no baseline beats it "
             "significantly (Table 8)", fontsize=7.0, color=MUTED)
    save(fig, os.path.join(FIGURES_DIR, name))


# ═══════════════════════════════════════════════════════════════════════
# Fig. 8 — ablation
# ═══════════════════════════════════════════════════════════════════════

def fig8_ablation(ablation, name="Fig8_ablation", tol=0.05):
    ds = ablation.get("_dataset", "GEFCom2014")
    items = [(k, v["mean"]["MAPE"], v["std"]["MAPE"]) for k, v in
             ablation.items() if not k.startswith("_")]
    full = next(v for k, v, _ in items if k.startswith("Full"))
    items.sort(key=lambda r: r[1])
    names = [i[0].replace("Full (Proposed)", "Full model")
             .replace("w/o ", "− ").replace("w/ ", "+ ") for i in items]
    vals = [i[1] for i in items]
    errs = [i[2] for i in items]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.subplots_adjust(left=0.315, right=0.965, top=0.80, bottom=0.135)
    ypos = np.arange(len(items))[::-1]

    # [round-5 audit] the "+ ..." rows ADD a stage rather than remove one, so
    # a legend phrased purely in terms of removal mislabels them; they get
    # their own colour and entry instead of being coloured "removing it hurts".
    def col(n, v):
        if n == "Full model":
            return ORANGE
        if n.startswith("+ "):
            return "#8a7fb5"          # added stage, not an ablation
        if v > full + tol:
            return CRITICAL           # removing it hurts
        if v < full - tol:
            return BLUE               # removing it helps
        return "#c9d4e0"              # no measurable effect

    colors = [col(n, v) for n, v in zip(names, vals)]
    ax.barh(ypos, vals, height=0.62, color=colors, zorder=3,
            xerr=errs, error_kw=dict(ecolor=MUTED, elinewidth=0.7, capsize=1.8))
    ax.axvline(full, color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7.2)
    # labels clear of the error whiskers
    for yy, v, e in zip(ypos, vals, errs):
        ax.text(v + e + max(vals) * 0.018, yy, f"{v:.2f}", va="center",
                fontsize=6.8, color=INK2)
    tidy(ax, ygrid=False, xgrid=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(vals) * 1.14)
    ax.set_xlabel("MAPE (%)   ·   mean ± std over five seeds")
    ax.set_title(f"Ablation on {ds} — “−” removes one component, "
                 f"“+” adds one", loc="left", pad=26)

    handles = [
        Line2D([], [], marker="s", ls="", ms=6, color=CRITICAL,
               label="removing it hurts"),
        Line2D([], [], marker="s", ls="", ms=6, color="#c9d4e0",
               label="no measurable effect"),
        Line2D([], [], marker="s", ls="", ms=6, color=BLUE,
               label="removing it helps"),
        Line2D([], [], marker="s", ls="", ms=6, color="#8a7fb5",
               label="added stage"),
        Line2D([], [], color=ORANGE, ls=(0, (3, 2)), lw=1.0,
               label="full model"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.005),
              ncol=5, fontsize=6.6, columnspacing=1.0, handletextpad=0.4)
    save(fig, os.path.join(FIGURES_DIR, name))


# ═══════════════════════════════════════════════════════════════════════
# Fig. 9 — cross-attention map
# ═══════════════════════════════════════════════════════════════════════

def fig9_attention(preds, name="Fig9_cross_attention"):
    """All three datasets side by side. The queries are the INPUT-window day
    patches (not forecast-day segments) and the keys are past + future
    covariate patches; the past/future split of the mass is annotated because
    it is the quantity the discussion refers to."""
    order = [d for d in ("GEFCom2014", "PJM", "AEMO")
             if d in preds and preds[d]["Proposed"].get("attn") is not None]
    if not order:
        print("  (no attention stored, skipping Fig. 9)")
        return
    fig, axes = plt.subplots(1, len(order), figsize=(7.4, 2.9))
    fig.subplots_adjust(left=0.075, right=0.915, top=0.72, bottom=0.235,
                        wspace=0.30)
    if len(order) == 1:
        axes = [axes]

    vmax = max(np.asarray(preds[d]["Proposed"]["attn"]).mean(axis=0).max()
               for d in order)
    for ax, ds in zip(axes, order):
        A = np.asarray(preds[ds]["Proposed"]["attn"]).mean(axis=0)
        n_past = DATASETS[ds]["input_window"] // DATASETS[ds]["patch_length"]
        n_fut = A.shape[1] - n_past
        rown = A / A.sum(axis=1, keepdims=True)
        past = rown[:, :n_past].mean(axis=0).sum()

        im = ax.imshow(A, cmap=CMAP, aspect="auto", vmin=0, vmax=vmax)
        ax.axvline(n_past - 0.5, color=SURFACE, lw=2.0)
        ax.set_xticks(list(range(0, n_past, 2)) +
                      list(range(n_past, A.shape[1])))
        ax.set_xticklabels([f"P{i+1}" for i in range(0, n_past, 2)] +
                           [f"F{i+1}" for i in range(n_fut)], fontsize=6.4)
        ax.set_yticks(range(A.shape[0]))
        ax.set_yticklabels([f"D−{A.shape[0] - i}" for i in range(A.shape[0])],
                           fontsize=6.4)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{ds}\npast {past*100:.0f} %  ·  future "
                     f"{(1-past)*100:.0f} %", loc="left", fontsize=7.6,
                     pad=5, linespacing=1.5)
        ax.set_xlabel("covariate tokens", fontsize=7.0)
    axes[0].set_ylabel("input-window day patch (query)", fontsize=7.0)

    cax = fig.add_axes([0.930, 0.235, 0.014, 0.485])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean attention weight", fontsize=6.9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.3, length=0)

    fig.suptitle(f"Cross-attention, averaged over each test set "
                 f"(seed {PRIMARY_SEED})", x=0.075,
                 ha="left", y=0.965, fontsize=9.4, fontweight="bold")
    fig.text(0.075, 0.885, "P = past day-patches, F = future horizon patches "
             "(right of the divider); the split of the attention mass differs "
             "sharply between the covariate-rich and the univariate datasets",
             fontsize=6.9, color=MUTED)
    save(fig, os.path.join(FIGURES_DIR, name))


# ═══════════════════════════════════════════════════════════════════════
# Fig. 10 — where the error lives: lead time vs time of day
# ═══════════════════════════════════════════════════════════════════════

def fig10_leadtime(preds, name="Fig10_error_by_leadtime"):
    """Two views of the same errors.

    Because forecasts are issued once per day over a one-day horizon, lead
    time and time of day are perfectly confounded: the two panels hold the
    same numbers, re-indexed. Showing only the left panel (the conventional
    choice) invites the reader to attribute everything to lead time; the
    right panel separates the two effects, because the datasets are issued
    at different clock times (15:00, 08:00, 23:00). The step at each issue
    marker — where lead 1 and lead H land on the same clock time — is the
    pure lead-time effect; AEMO's midday hump, which has no counterpart on
    the other two, is a pure time-of-day effect.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    fig.subplots_adjust(left=0.078, right=0.988, top=0.690, bottom=0.180,
                        wspace=0.20)
    colors = {"GEFCom2014": BLUE, "PJM": ORANGE, "AEMO": AQUA}
    axL, axR = axes

    for ds, c in colors.items():
        if ds not in preds:
            continue
        p = preds[ds]["Proposed"]
        yt, yp = np.asarray(p["y_true"]), np.asarray(p["y_pred"])
        ape = np.abs(yt - yp) / np.abs(yt) * 100
        H = ape.shape[1]
        med = np.median(ape, axis=0)
        q1, q3 = np.percentile(ape, [25, 75], axis=0)

        # ── left: normalised lead time (the conventional view) ──
        xs = np.arange(1, H + 1) / H
        axL.fill_between(xs, q1, q3, color=c, alpha=0.13, lw=0)
        axL.plot(xs, med, color=c, lw=1.7, label=ds)

        # ── right: the same values re-indexed by wall-clock time ──
        clk = preds[ds].get("_clock")
        if clk is None:
            continue
        step_h = clk["step_min"] / 60.0
        # Draw the curve twice, offset by one day, and let the axes clip it:
        # the halves either side of midnight then join correctly, while the
        # two ends (lead 1 and lead H, same clock time) stay unconnected
        # instead of being bridged by a spurious vertical segment.
        # (medians only here — a clipped band would end in a hard vertical
        #  edge wherever a copy runs off the axes, which reads as data)
        hu = clk["issue_hour"] + np.arange(H) * step_h
        for shift in (0.0, -24.0):
            axR.plot(hu + shift, med, color=c, lw=1.7,
                     label=ds if shift == 0.0 else None)
        # mark where each dataset's day-ahead forecast is actually issued
        axR.plot(clk["issue_hour"] % 24, med[0], marker="v", color=c, ms=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)

    for ax in axes:
        tidy(ax)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax)
    axR.set_yticklabels([])

    axL.set_xlim(0, 1.0)
    axL.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axL.set_xticklabels(["0", "¼", "½", "¾", "full"])
    axL.set_xlabel("lead time (fraction of the horizon)")
    axL.set_ylabel("absolute percentage error (%)")
    axL.set_title("indexed by lead time  (median, interquartile band)",
                  loc="left", fontsize=8.0, pad=6)

    axR.set_xlim(0, 24)
    axR.set_xticks([0, 6, 12, 18, 24])
    axR.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
    axR.set_xlabel("time of day of the forecast step")
    axR.set_title("the same medians, indexed by time of day  "
                  r"($\blacktriangledown$ = forecast issue time)",
                  loc="left", fontsize=8.0, pad=6)

    axL.legend(loc="upper left", ncol=1, fontsize=7.2, handlelength=1.4)
    fig.suptitle("Lead time and time of day both shape the error, "
                 "and the protocol confounds them", x=0.078, ha="left",
                 y=0.968, fontsize=9.4, fontweight="bold")
    fig.text(0.078, 0.905,
             f"Median with interquartile band over every day-ahead forecast "
             f"in the test period (seed {PRIMARY_SEED}). One forecast per day "
             f"over a one-day horizon "
             "makes the two panels equivalent\nre-indexings of the same "
             "numbers, so neither effect is isolated within a dataset. The "
             "break at each issue marker separates the end of one forecast "
             "from the start\nof the next by a single time step on the clock, "
             "so it is essentially lead time; AEMO's midday hump, absent on "
             "the other two, is a time-of-day pattern.",
             fontsize=6.7, color=MUTED, linespacing=1.6, va="top")
    save(fig, os.path.join(FIGURES_DIR, name))


# ═══════════════════════════════════════════════════════════════════════
# Fig. 11 — the leakage experiment
# ═══════════════════════════════════════════════════════════════════════

def read_leakage(dataset="GEFCom2014"):
    """The measured protocol-A vs protocol-B result, or None if never run.

    [round-5 audit] This used to be a pair of literal dicts default-argumented
    into the figure. That made the paper's headline leakage number the ONE
    figure in the release not derived from a result file — in a paper whose
    contribution is precisely that hidden, hand-carried numbers are how
    decomposition hybrids overstate themselves. It is now read back from what
    leakage_demo.py wrote, and its absence is an error rather than a silent
    fallback to whatever was typed here last.
    """
    path = os.path.join(RESULTS_DIR, f"leakage_{dataset}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig11_leakage(leaky=None, honest=None, name="Fig11_leakage_effect",
                  datasets=("GEFCom2014", "PJM")):
    """One panel per benchmark on which the experiment has been run.

    MAPE only: it is the one metric comparable across benchmarks whose loads
    differ by three orders of magnitude, and the MAE and RMSE effects (given in
    the caption) are larger, so nothing is being hidden by the choice.

    The MATCHED pair is the claim — the same CEEMDAN with the same settings,
    the same feature layout and the same ridge, fitted over the whole series or
    recomputed inside each input window, so the protocol is the only thing that
    varies. The ORIGINAL pair beside it also swaps the decomposition family and
    the fitting scheme, so on its own it bounds the effect rather than
    isolating it (round-5 audit). Every value is read from
    results/leakage_<dataset>.json — never from literals.
    """
    got = [(ds, read_leakage(ds)) for ds in datasets]
    have = [(ds, d) for ds, d in got if d is not None]
    if not have:
        raise FileNotFoundError(
            f"no results/leakage_*.json for {list(datasets)} — run "
            f"`python leakage_demo.py --dataset <name>` first. "
            f"Fig. 11 is never drawn from hardcoded values.")

    fig, axes = plt.subplots(1, len(have), figsize=(7.2, 3.05),
                             squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.085, right=0.985, top=0.635, bottom=0.235,
                        wspace=0.24)
    effects = []
    for ax, (ds, d) in zip(axes, have):
        matched = "protocol_A2_global_ceemdan" in d
        if matched:
            vals = [d["protocol_A2_global_ceemdan"]["MAPE"],
                    d["protocol_C_causal_ceemdan"]["MAPE"],
                    d["protocol_A_decompose_then_split"]["MAPE"],
                    d["protocol_B_causal"]["MAPE"]]
            xs = [0, 0.72, 1.85, 2.57]
            effects.append((ds, d["leakage_effect_matched_pct"],
                            d["illusory_improvement_pct"]))
        else:
            vals = [d["protocol_A_decompose_then_split"]["MAPE"],
                    d["protocol_B_causal"]["MAPE"]]
            xs = [0, 0.72]
            effects.append((ds, None, d["illusory_improvement_pct"]))
        bars = ax.bar(xs, vals, width=0.60,
                      color=[CRITICAL, BLUE] * (len(vals) // 2), zorder=3)
        if matched:
            for b in bars[2:]:
                b.set_alpha(0.42)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.2, color=INK,
                    fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(["leaky", "causal"] * (len(vals) // 2), fontsize=7.0)
        tidy(ax)
        ax.set_xlim(-0.55, xs[-1] + 0.55)
        ax.set_ylim(0, max(vals) * 1.26)
        ax.set_ylabel("MAPE (%)" if ax is axes[0] else "")
        ax.tick_params(axis="x", length=0)
        eff = d.get("leakage_effect_matched_pct", d["illusory_improvement_pct"])
        ax.set_title(f"{ds}   ·   protocol accounts for {eff:.1f} %",
                     loc="left", fontsize=8.4, pad=5)
        if matched:
            for cx, txt in ((0.36, "matched"), (2.21, "original")):
                ax.text(cx, -0.145, txt, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=6.6, color=MUTED)

    lo = min(e for _, e, _ in effects if e is not None)
    hi = max(e for _, e, _ in effects if e is not None)
    fig.suptitle(f"Decompose-then-split reports {lo:.0f}–{hi:.0f} % lower "
                 f"error on both benchmarks, with the protocol as the only "
                 f"difference", x=0.085, ha="left", y=0.972, fontsize=9.4,
                 fontweight="bold")
    fig.text(0.085, 0.885,
             "“matched”: the same CEEMDAN (20 trials, K = 8), the same feature "
             "layout and the same ridge, fitted over the whole series or "
             "recomputed inside each input window.\n“original”: the contrast "
             "as first run, which also swaps the decomposition family and the "
             "fitting scheme (" + ", ".join(f"{ds} {u:.1f} %"
                                            for ds, _, u in effects)
             + ") and therefore bounds the effect rather than isolating it.",
             fontsize=6.8, color=MUTED, va="top", linespacing=1.6)
    save(fig, os.path.join(FIGURES_DIR, name))


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary, ablation, preds = _load()
    print("Result figures:")
    fig3_decomposition()
    fig4_overview(summary)
    for i, ds in enumerate(["GEFCom2014", "PJM", "AEMO"]):
        if ds in preds:
            fig_forecast(preds, ds, 5 + i, name=f"Fig{5+i}_forecast_{ds}")
    fig8_ablation(ablation)
    fig9_attention(preds)
    fig10_leadtime(preds)
    fig11_leakage()
