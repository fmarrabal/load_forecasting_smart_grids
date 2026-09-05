#!/usr/bin/env python3
"""Graphical abstract for the Applied Energy submission.

Elsevier: minimum 531 x 1328 px (h x w), ratio about 1:2.5, readable at
5 x 13 cm on a 96 dpi screen; TIFF, EPS, PDF or Office formats; Arial/Times.
This draws 13.5 x 5.4 cm at 300 dpi (1594 x 638 px, exact canvas) with the
paper's shared figure style and reads every number from the result files:

  * panel 1 from results/leakage_{GEFCom2014,PJM}.json (matched contrast A2 vs C),
  * panel 3 from the ablations in results/summary_v4.json (unrounded means).

    python scripts/graphical_abstract.py   ->  latex/figures/graphical_abstract.{pdf,png,tif}
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
RESULTS = os.path.join(CODE, "results")
# manuscript layout (this file under scripts/): write into <project>/latex/figures;
# public repository layout (this file under python/): write into <repo>/figures
if os.path.basename(HERE) == "scripts":
    _OUT_DIR = os.path.join(ROOT, "latex", "figures")
else:
    _OUT_DIR = os.path.join(CODE, "figures")
OUT = os.path.join(_OUT_DIR, "graphical_abstract")
sys.path[:0] = [CODE, HERE]  # figstyle lives next to the code in both layouts

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

import figstyle as fs  # noqa: E402

fs.apply_style()
# the shared style saves with a tight bounding box; the graphical abstract must
# keep its exact canvas, because Elsevier sizes it by pixels
matplotlib.rcParams["savefig.bbox"] = None
matplotlib.rcParams["savefig.pad_inches"] = 0.0
W_CM, H_CM, DPI = 13.5, 5.4, 300


def leak(ds):
    with open(os.path.join(RESULTS, f"leakage_{ds}.json"), encoding="utf-8") as f:
        J = json.load(f)
    return (J["protocol_A2_global_ceemdan"]["MAPE"], J["protocol_C_causal_ceemdan"]["MAPE"],
            J["leakage_effect_matched_pct"])


def ablation(ds):
    """Change in MAPE from the unrounded five-seed means in summary_v4.json,
    the same source the ablation figure and the manuscript table use."""
    with open(os.path.join(RESULTS, "summary_v4.json"), encoding="utf-8") as f:
        a = json.load(f)["ablations"][ds]
    full = a["Full (Proposed)"]["mean"]["MAPE"]
    return {k: v["mean"]["MAPE"] - full for k, v in a.items()
            if isinstance(v, dict) and "mean" in v}


def box(ax, xy, w, h, text, fill, edge, size=5.4, weight="normal"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc=fill, ec=edge, lw=0.8, transform=ax.transAxes,
                                clip_on=False))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight,
            color=fs.INK, transform=ax.transAxes, linespacing=1.15)


def arrow(ax, p, q, color=fs.MUTED):
    ax.add_patch(FancyArrowPatch(p, q, transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=6, lw=0.8, color=color, clip_on=False))


def main():
    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54), dpi=DPI)
    fig.patch.set_facecolor("white")

    # ── title band ───────────────────────────────────────────────────────────
    fig.text(0.02, 0.935, "Leakage, not decomposition", fontsize=10, weight="bold",
             color=fs.INK, va="center")
    fig.text(0.98, 0.935, "auditing decompose-then-split hybrids for short-term load forecasting",
             fontsize=5.6, color=fs.MUTED, ha="right", va="center")
    fig.add_artist(Line2D([0.02, 0.98], [0.878, 0.878], color=fs.AXIS, lw=0.6,
                          transform=fig.transFigure))

    def header(x, num, text):
        fig.text(x, 0.825, num, fontsize=7.0, weight="bold", color="white", va="center",
                 ha="center", bbox=dict(boxstyle="circle,pad=0.22", fc=fs.INK, ec="none"))
        fig.text(x + 0.022, 0.825, text, fontsize=6.6, weight="bold", color=fs.INK, va="center")

    header(0.034, "1", "The protocol inflates accuracy")
    header(0.348, "2", "Close the leak, then ablate")
    header(0.695, "3", "The decomposition is inert")

    # ── panel 1: matched leakage contrast ────────────────────────────────────
    ax1 = fig.add_axes([0.065, 0.31, 0.215, 0.43])
    xs, labels = [0.0, 1.0], ["GEFCom2014", "PJM"]
    for x, ds in zip(xs, ("GEFCom2014", "PJM")):
        a2, c, eff = leak(ds)
        ax1.bar(x - 0.19, a2, 0.34, color=fs.CRITICAL, zorder=3)
        ax1.bar(x + 0.19, c, 0.34, color=fs.BLUE, zorder=3)
        ax1.text(x - 0.19, a2 + 0.2, f"{a2:.1f}", ha="center", va="bottom", fontsize=5.0, color=fs.INK)
        ax1.text(x + 0.19, c + 0.2, f"{c:.1f}", ha="center", va="bottom", fontsize=5.0, color=fs.INK)
        ax1.text(x, max(a2, c) + 1.35, f"−{eff:.0f} %", ha="center", va="bottom",
                 fontsize=6.4, weight="bold", color=fs.CRITICAL)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=5.4)
    ax1.set_ylim(0, 11.6)
    ax1.set_yticks([0, 4, 8])
    ax1.tick_params(axis="y", labelsize=5.0, length=2, pad=1.5)
    ax1.tick_params(axis="x", length=0, pad=2)
    ax1.set_ylabel("MAPE (%)", fontsize=5.4, labelpad=2)
    fs.tidy(ax1)
    ax1.text(-0.12, -0.20, "■ decompose-then-split (leaky)", color=fs.CRITICAL, fontsize=4.8,
             transform=ax1.transAxes, ha="left", va="center")
    ax1.text(-0.12, -0.31, "■ same CEEMDAN inside each window", color=fs.BLUE, fontsize=4.8,
             transform=ax1.transAxes, ha="left", va="center")
    ax1.text(-0.12, -0.42, "same features and learner; only the protocol differs",
             transform=ax1.transAxes, ha="left", va="center", fontsize=4.6, color=fs.MUTED)

    # ── panel 2: the causal analogue ─────────────────────────────────────────
    ax2 = fig.add_axes([0.335, 0.215, 0.32, 0.565])
    ax2.set_axis_off()
    box(ax2, (0.5, 0.93), 0.72, 0.11, "input window (one week of load)", fs.FILL_SOFT, fs.AXIS, size=5.2)
    arrow(ax2, (0.5, 0.875), (0.5, 0.80))
    box(ax2, (0.5, 0.69), 0.96, 0.20,
        "causal dual-stage decomposition\nmulti-scale split → learnable FIR bank → gate\n"
        "inside the window, exact reconstruction",
        fs.FILL_AQUA, fs.AQUA, size=4.9)
    arrow(ax2, (0.26, 0.585), (0.26, 0.515))
    arrow(ax2, (0.75, 0.585), (0.75, 0.515))
    box(ax2, (0.26, 0.39), 0.48, 0.23,
        "patch Transformer\n+ cross-attention to\nknown-future inputs\n+ BiGRU",
        fs_fill := fs.FILL_BLUE, fs.BLUE, size=4.8)
    box(ax2, (0.75, 0.39), 0.42, 0.23,
        "linear base\ninitialised at\nweekly persistence",
        fs.FILL_ORANGE, fs.ORANGE, size=4.8)
    arrow(ax2, (0.30, 0.27), (0.44, 0.175))
    arrow(ax2, (0.71, 0.27), (0.56, 0.175))
    box(ax2, (0.5, 0.105), 0.56, 0.11, "day-ahead forecast", fs.FILL_SOFT, fs.AXIS, size=5.4, weight="bold")
    ax2.text(0.5, -0.05, "nothing sees the future, so an ablation\nmeasures the decomposition and not the leak",
             ha="center", va="top", fontsize=4.8, color=fs.MUTED, transform=ax2.transAxes,
             linespacing=1.25)

    # ── panel 3: ablation contrast ───────────────────────────────────────────
    g, a = ablation("GEFCom2014"), ablation("AEMO")
    rows = [("Multi-scale split", "w/o Stage 1 (multi-scale split)"),
            ("Filter bank", "w/o Stage 2 (filter bank)"),
            ("Adaptive gate", "w/o adaptive gating"),
            ("Linear base", "w/o linear skip"),
            ("Covariate path", "w/o covariate skip")]
    ax3 = fig.add_axes([0.815, 0.27, 0.16, 0.47])
    ys = [4, 3, 2, 0.9, -0.1]
    ax3.axvspan(-0.11, 0.11, color=fs.FILL_SOFT, zorder=0)
    ax3.axvline(0, color=fs.AXIS, lw=0.6, zorder=1)
    h = 0.34
    for y, (lab, key) in zip(ys, rows):
        vg, va = g.get(key), a.get(key)
        if vg is not None:
            ax3.barh(y + h / 2, vg, h, color=fs.BLUE, zorder=3)
            ax3.text(max(vg, 0) + 0.03, y + h / 2, f"{vg:+.2f}", va="center", fontsize=4.6, color=fs.INK)
        if va is not None:
            ax3.barh(y - h / 2, va, h, color=fs.ORANGE, zorder=3)
            ax3.text(max(va, 0) + 0.03, y - h / 2, f"{va:+.2f}", va="center", fontsize=4.6, color=fs.INK)
        else:
            ax3.text(0.03, y - h / 2, "n/a", va="center", fontsize=4.4, color=fs.MUTED)
        ax3.text(-0.16, y, lab, ha="right", va="center", fontsize=5.0, color=fs.INK)
    ax3.set_xlim(-0.14, 0.95)
    ax3.set_ylim(-0.6, 4.6)
    ax3.set_yticks([])
    ax3.set_xticks([0, 0.4, 0.8])
    ax3.tick_params(axis="x", labelsize=5.0, length=2, pad=1.5)
    ax3.set_xlabel("ΔMAPE when removed (pp)", fontsize=5.2, labelpad=2)
    fs.tidy(ax3, ygrid=False)
    ax3.text(0.0, 4.45, "seed\nspread", ha="center", va="bottom", fontsize=4.0, color=fs.MUTED,
             linespacing=1.0)
    ax3.text(0.94, 4.5, "GEFCom2014", ha="right", va="center", fontsize=4.8, color=fs.BLUE, weight="bold")
    ax3.text(0.94, 4.15, "AEMO", ha="right", va="center", fontsize=4.8, color=fs.ORANGE, weight="bold")

    # ── bottom line ──────────────────────────────────────────────────────────
    fig.add_artist(Line2D([0.02, 0.98], [0.10, 0.10], color=fs.AXIS, lw=0.6,
                          transform=fig.transFigure))
    fig.text(0.02, 0.048,
             "Accuracy is carried by a weekly-persistence linear base and by covariates, not by decomposition.\n"
             "No baseline beats the model at the Holm-corrected 5 % level and it is in the 90 % model confidence set on all three benchmarks.",
             fontsize=5.2, color=fs.INK, va="center", linespacing=1.25)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # exact canvas: Elsevier sizes the graphical abstract by pixels, so no tight bbox
    fig.savefig(OUT + ".pdf", dpi=DPI, bbox_inches=None)
    fig.savefig(OUT + ".png", dpi=DPI, bbox_inches=None)
    fig.savefig(OUT + ".tif", dpi=DPI, bbox_inches=None, pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image
    im = Image.open(OUT + ".png")
    print("written", OUT + ".{pdf,png,tif}", "pixels:", im.size)
    assert im.size[0] >= 1328 and im.size[1] >= 531, im.size


if __name__ == "__main__":
    main()
