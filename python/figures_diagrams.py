"""
Schematic figures for the manuscript (Fig. 1 and Fig. 2).

These are drawn programmatically so they are reproducible and restyle with the
rest of the figures:

  Fig. 1  The protocol contrast: conventional decompose-then-split (invalid,
          information flows backwards from the test period) vs the proposed
          causal protocol (decomposition lives inside the input window).
  Fig. 2  The CPTB architecture with tensor shapes.

Run:  python figures_diagrams.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from figstyle import (apply_style, save, SURFACE, INK, INK2, MUTED, GRID, AXIS,
                      BLUE, ORANGE, AQUA, CRITICAL, GOOD,
                      FILL_SOFT, FILL_BLUE, FILL_ORANGE, FILL_AQUA)
from config import FIGURES_DIR

apply_style()

MONO = {"family": "monospace", "size": 6.8}


# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

def box(ax, x, y, w, h, label, sub=None, fill=FILL_SOFT, edge=AXIS,
        lw=0.8, fs=8.0, fw="bold", tc=INK, sub_fs=6.9, radius=0.014):
    """Rounded block. The title anchors to the top and the secondary text to
    the bottom, so multi-line content never collides."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h - 0.011, label, ha="center", va="top",
                fontsize=fs, fontweight=fw, color=tc, zorder=3)
        ax.text(cx, y + 0.010, sub, ha="center", va="bottom",
                fontsize=sub_fs, color=INK2, zorder=3, linespacing=1.4)
    else:
        ax.text(cx, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=3)


def arrow(ax, p0, p1, color=AXIS, lw=1.0, style="-|>", ls="-",
          rad=0.0, zorder=1, mut=7):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=mut, linewidth=lw,
        linestyle=ls, color=color, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))


def shape(ax, x, y, txt, ha="center", color=MUTED):
    ax.text(x, y, txt, ha=ha, va="center", color=color, zorder=4, **MONO)


# ═══════════════════════════════════════════════════════════════════════
# Fig. 1 — why the conventional protocol is invalid
# ═══════════════════════════════════════════════════════════════════════

def fig1_protocol(name="Fig1_protocol_contrast"):
    """Two stacked panels on ONE axes so every element is placed on a single
    explicit coordinate grid (no inter-axes drift, no collisions)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.995, bottom=0.005)

    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 620)
    trace = (0.5 + 0.16 * np.sin(2 * np.pi * 9 * t) +
             0.07 * np.sin(2 * np.pi * 2.2 * t) + 0.02 * rng.standard_normal(620))

    X0, X1 = 0.045, 0.955                      # ribbon extent
    SPLITS = [(0.00, 0.70, "train", FILL_BLUE, "#2f5f96"),
              (0.70, 0.85, "validation", "#eef3f8", "#6b8299"),
              (0.85, 1.00, "test", FILL_ORANGE, "#b84a1e")]
    RH = 0.118                                  # ribbon height

    def ribbon(y0, labels=True):
        """train/val/test bands with the load trace; names sit BELOW."""
        for f0, f1, lab, fc, ec in SPLITS:
            x0, x1 = X0 + f0 * (X1 - X0), X0 + f1 * (X1 - X0)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, RH, facecolor=fc,
                                   edgecolor="none", zorder=1))
            ax.plot([x1, x1], [y0, y0 + RH], color=SURFACE, lw=1.4, zorder=3)
            if labels:
                ax.text((x0 + x1) / 2, y0 - 0.016, lab, ha="center", va="top",
                        fontsize=6.8, color=ec, fontweight="bold", zorder=3)
        ax.plot(X0 + t * (X1 - X0), y0 + 0.024 + trace * (RH - 0.048),
                color=INK2, lw=0.5, zorder=4, alpha=0.8)

    # ══ panel (a) ══════════════════════════════════════════════════════
    ax.text(0.012, 0.968, "(a)  Conventional pipeline — decompose, then split",
            fontsize=8.8, fontweight="bold", color=INK, va="center")
    ax.text(0.988, 0.968, "INVALID", fontsize=7.4, fontweight="bold",
            color=CRITICAL, va="center", ha="right")

    ax.add_patch(FancyBboxPatch(
        (X0, 0.855), X1 - X0, 0.068,
        boxstyle="round,pad=0,rounding_size=0.010",
        facecolor="#fdeaea", edgecolor=CRITICAL, linewidth=0.9, zorder=2))
    ax.text((X0 + X1) / 2, 0.889, "CEEMDAN / VMD fitted on the complete series",
            ha="center", va="center", fontsize=7.6, fontweight="bold",
            color=CRITICAL, zorder=3)

    ribbon(0.640)

    # one unambiguous backwards arrow, bowing DOWN into the clear gap
    x_test = X0 + 0.93 * (X1 - X0)
    x_train = X0 + 0.28 * (X1 - X0)
    arrow(ax, (x_test, 0.851), (x_train, 0.851), color=CRITICAL, lw=1.1,
          rad=-0.26, zorder=5, mut=8)
    ax.text((x_test + x_train) / 2, 0.836,
            "future information flows back into every training window",
            ha="center", va="top", fontsize=7.2, color=CRITICAL,
            fontstyle="italic", zorder=6)
    # the magnitude is read back from the experiment, never typed here
    from figures_results import read_leakage
    m = read_leakage()
    if m is None:
        raise FileNotFoundError(
            "results/leakage_GEFCom2014.json not found — run "
            "`python leakage_demo.py --dataset GEFCom2014` first. The measured "
            "consequence annotated on this figure is never hardcoded.")
    # prefer the MATCHED contrast, which varies the protocol and nothing else
    infl = m.get("leakage_effect_matched_pct", m["illusory_improvement_pct"])
    ax.text(0.012, 0.585, "measured consequence:  apparent accuracy gain of "
            f"+{infl:.1f} % that vanishes under a causal protocol", ha="left",
            va="center", fontsize=7.3, color=CRITICAL, fontweight="bold")

    ax.plot([0.012, 0.988], [0.520, 0.520], color=GRID, lw=0.8)

    # ══ panel (b) ══════════════════════════════════════════════════════
    ax.text(0.012, 0.468, "(b)  Proposed framework — split, then decompose "
            "causally inside each input window",
            fontsize=8.8, fontweight="bold", color=INK, va="center")
    ax.text(0.988, 0.468, "CAUSAL", fontsize=7.4, fontweight="bold",
            color=GOOD, va="center", ha="right")

    # same split ribbon (bands already named in panel a)
    ribbon(0.150, labels=False)

    wx0 = X0 + 0.450 * (X1 - X0)
    wx1 = X0 + 0.585 * (X1 - X0)
    hx1 = X0 + 0.650 * (X1 - X0)
    ax.add_patch(Rectangle((wx0, 0.142), wx1 - wx0, 0.134, facecolor="none",
                           edgecolor=BLUE, lw=1.2, zorder=6))
    ax.add_patch(Rectangle((wx1, 0.142), hx1 - wx1, 0.134, facecolor="none",
                           edgecolor=ORANGE, lw=1.2,
                           linestyle=(0, (2.5, 1.6)), zorder=6))

    for f0 in (0.170, 0.310):
        sx = X0 + f0 * (X1 - X0)
        ax.add_patch(Rectangle((sx, 0.150), 0.135 * (X1 - X0), RH,
                               facecolor="none", edgecolor=AXIS, lw=0.7,
                               zorder=5))

    # decomposition box directly above its window; arrow lands left-of-centre
    bw = 0.300
    bx = (wx0 + wx1) / 2 - bw / 2
    ax.add_patch(FancyBboxPatch(
        (bx, 0.352), bw, 0.066, boxstyle="round,pad=0,rounding_size=0.010",
        facecolor=FILL_AQUA, edgecolor=AQUA, linewidth=0.9, zorder=2))
    ax.text(bx + bw / 2, 0.385, "decomposition computed inside the window only",
            ha="center", va="center", fontsize=7.0, fontweight="bold",
            color="#0d6b4b", zorder=3)
    arrow(ax, ((wx0 + wx1) / 2, 0.349), ((wx0 + wx1) / 2, 0.280), color=AQUA,
          lw=1.0, zorder=5)

    # issue time marked ABOVE the ribbon, where the space is free
    ax.plot([wx1, wx1], [0.276, 0.302], color=INK2, lw=0.7, zorder=7)
    ax.text(wx1 + 0.005, 0.302, "issue time $t_0$", ha="left", va="center",
            fontsize=6.8, color=INK2, zorder=7)

    # brackets BELOW the ribbon: L and H, clear of the boxes
    def bracket(xa, xb, y, color, label, dy=0.015):
        ax.plot([xa, xa, xb, xb], [y + dy, y, y, y + dy], color=color,
                lw=0.9, zorder=6, solid_joinstyle="miter")
        ax.text((xa + xb) / 2, y - 0.013, label, ha="center", va="top",
                fontsize=6.9, color=color, fontweight="bold")

    bracket(wx0, wx1, 0.104, BLUE, "input window  $L$")
    # horizon label sits to the right of its (narrow) bracket, so the two
    # labels never read as one line
    ax.plot([wx1, wx1, hx1, hx1], [0.119, 0.104, 0.104, 0.119], color=ORANGE,
            lw=0.9, zorder=6, solid_joinstyle="miter")
    ax.text(hx1 + 0.006, 0.104, "horizon  $H$", ha="left", va="center",
            fontsize=6.9, color=ORANGE, fontweight="bold")
    ax.text(X0, 0.088, "sliding windows · one non-overlapping forecast per day",
            ha="left", va="top", fontsize=6.9, color=MUTED)

    save(fig, os.path.join(FIGURES_DIR, name))


# ═══════════════════════════════════════════════════════════════════════
# Fig. 2 — architecture with tensor shapes
# ═══════════════════════════════════════════════════════════════════════

def fig2_architecture(name="Fig2_architecture"):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)

    ax.text(0.0, 1.005, "CPTB — causal-decomposition Patch Transformer–BiGRU",
            fontsize=9.8, fontweight="bold", color=INK, va="bottom")
    ax.text(0.0, 0.963, "shapes for GEFCom2014:  L = 168,  H = 24,  "
            "M = 8 components,  N = 7 patches,  d = 64",
            fontsize=7.2, color=MUTED, va="bottom")

    # Box heights sized to their content: H1 = title only, H2 = title + one
    # sub-line, H3 = title + two sub-lines.
    H1, H2, H3 = 0.052, 0.076, 0.106

    # ── column 1: causal dual-stage decomposition ──
    x0, w = 0.012, 0.268
    box(ax, x0, 0.850, w, H2, "Input window  $x$", "past load, z-scored",
        fill=FILL_BLUE, edge=BLUE, fs=7.8, sub_fs=6.6)
    box(ax, x0, 0.772, w, H1, "RevIN  normalise", fill=FILL_SOFT, fs=7.8)
    box(ax, x0, 0.640, w, H3, "Stage 1 · causal split",
        "moving averages 12 / 24 / 168\n→ HF, intra-day, weekly, trend",
        fill=FILL_AQUA, edge=AQUA, fs=7.8, sub_fs=6.5)
    box(ax, x0, 0.508, w, H3, "Stage 2 · FIR filter bank",
        "K = 4 causal band-pass filters\napplied to HF, + remainder",
        fill=FILL_AQUA, edge=AQUA, fs=7.8, sub_fs=6.5)
    box(ax, x0, 0.428, w, H1, "Complexity gate", fill=FILL_AQUA, edge=AQUA,
        fs=7.8)

    for y0, y1 in [(0.850, 0.824), (0.772, 0.746), (0.640, 0.614),
                   (0.508, 0.480)]:
        arrow(ax, (x0 + w / 2, y0), (x0 + w / 2, y1))

    ax.add_patch(FancyBboxPatch(
        (x0 - 0.007, 0.416), w + 0.014, 0.340,
        boxstyle="round,pad=0,rounding_size=0.012", facecolor="none",
        edgecolor=AQUA, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1))
    ax.text(x0 + w / 2, 0.400, "exact reconstruction · strictly causal",
            ha="center", va="top", fontsize=6.7, color="#0d6b4b",
            fontstyle="italic")
    shape(ax, x0 + w / 2, 0.368, "(B, 8, 168)")

    # ── column 2: patch transformer ──
    x1, w2 = 0.330, 0.250
    box(ax, x1, 0.760, w2, H3, "Patch embedding",
        "M × N day-length patches\nlinear → d, + component id",
        fill=FILL_BLUE, edge=BLUE, fs=7.8, sub_fs=6.5)
    box(ax, x1, 0.628, w2, H3, "Transformer encoder",
        "2 layers, 4 heads\nover the 56-token grid", fill=FILL_BLUE,
        edge=BLUE, fs=7.8, sub_fs=6.5)
    box(ax, x1, 0.548, w2, H1, "Component pooling", fill=FILL_BLUE,
        edge=BLUE, fs=7.8)
    box(ax, x1, 0.416, w2, H3, "Cross-attention",
        "queries: time tokens\nkeys/values: covariate tokens",
        fill=FILL_BLUE, edge=BLUE, fs=7.8, sub_fs=6.5)
    box(ax, x1, 0.336, w2, H1, "BiGRU decoder  (64 × 2)", fill=FILL_BLUE,
        edge=BLUE, fs=7.8)
    box(ax, x1, 0.256, w2, H1, "MLP head  (128, GELU)", fill=FILL_BLUE,
        edge=BLUE, fs=7.8)

    arrow(ax, (x0 + w, 0.454), (x1, 0.800), rad=-0.16, color=AQUA)
    for y0, y1 in [(0.760, 0.734), (0.628, 0.600), (0.548, 0.522),
                   (0.416, 0.388), (0.336, 0.308)]:
        arrow(ax, (x1 + w2 / 2, y0), (x1 + w2 / 2, y1))
    # shapes right-aligned to the column edge, clear of the centred arrows
    shape(ax, x1 + w2, 0.747, "(B, 8, 7, 64)", ha="right")
    shape(ax, x1 + 0.004, 0.535, "(B, 7, 64)", ha="left")

    # ── column 3: covariates + additive paths + output ──
    x2, wc = 0.634, 0.250
    box(ax, x2, 0.850, wc, H2, "Known covariates",
        "calendar · temperature · HDD/CDD", fill=FILL_ORANGE, edge=ORANGE,
        fs=7.8, sub_fs=6.5)
    box(ax, x2, 0.735, wc, H2, "Covariate tokens",
        "past patches + future patches", fill=FILL_ORANGE, edge=ORANGE,
        fs=7.8, sub_fs=6.5)
    arrow(ax, (x2 + wc / 2, 0.850), (x2 + wc / 2, 0.813))
    shape(ax, x2 + wc, 0.828, "past (B,168,C) · future (B,24,C)", ha="right")
    arrow(ax, (x2, 0.766), (x1 + w2 + 0.004, 0.462), rad=0.20, color=ORANGE)

    box(ax, x2, 0.598, wc, H3, "Full-resolution covariate path",
        "future covariates → horizon\nadditive, zero-initialised",
        fill=FILL_ORANGE, edge=ORANGE, fs=7.3, sub_fs=6.5)
    # [round-5 audit] this used to read "enabled where temperature exists" —
    # the very rule §3.7 identifies as selection leakage and replaces
    ax.text(x2 + wc + 0.008, 0.651, "enabled by\nthe validation\nrule of "
            "Eq. (6)", ha="left", va="center", fontsize=6.3, color=MUTED,
            linespacing=1.35)

    box(ax, x2, 0.452, wc, H3, "Per-component linear base",
        "$W_m$: window → horizon\ninit = weekly persistence",
        fill=FILL_AQUA, edge=AQUA, fs=7.6, sub_fs=6.5)
    # route this one UNDER the central column: bowing up made it cut through
    # the component-pooling and cross-attention boxes
    arrow(ax, (x0 + w * 0.86, 0.392), (x2, 0.470), rad=0.34, color=AQUA)

    # summation node
    cx, cy, r = x2 + wc / 2, 0.372, 0.020
    ax.add_patch(plt.Circle((cx, cy), r, facecolor=SURFACE, edgecolor=INK2,
                            lw=0.9, zorder=4))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=10.5,
            color=INK, zorder=5, fontweight="bold")
    arrow(ax, (cx - 0.050, 0.598), (cx - 0.014, cy + r), color=ORANGE, rad=0.12)
    arrow(ax, (cx, 0.452), (cx, cy + r), color=AQUA)
    arrow(ax, (x1 + w2, 0.284), (cx - r, cy), rad=-0.18, color=BLUE)

    box(ax, x2, 0.268, wc, H1, "RevIN$^{-1}$  denormalise", fill=FILL_SOFT,
        fs=7.8)
    arrow(ax, (cx, cy - r), (cx, 0.322))
    box(ax, x2, 0.160, wc, H2, "Day-ahead forecast  $\\hat{y}$",
        "24 steps, original scale", fill="#eaeaf6", edge="#4a3aa7", fs=7.8,
        sub_fs=6.5)
    arrow(ax, (cx, 0.268), (cx, 0.238))
    shape(ax, cx, 0.144, "(B, 24)")

    # legend strip
    ly = 0.055
    for i, (c, fc, lab) in enumerate([
            (AQUA, FILL_AQUA, "causal decomposition & linear base"),
            (BLUE, FILL_BLUE, "patch Transformer – BiGRU"),
            (ORANGE, FILL_ORANGE, "covariate pathways")]):
        lx = 0.012 + i * 0.250
        ax.add_patch(FancyBboxPatch((lx, ly), 0.020, 0.018,
                                    boxstyle="round,pad=0,rounding_size=0.005",
                                    facecolor=fc, edgecolor=c, lw=0.8))
        ax.text(lx + 0.028, ly + 0.009, lab, fontsize=6.8, color=INK2,
                va="center")

    save(fig, os.path.join(FIGURES_DIR, name))


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Schematic figures:")
    fig1_protocol()
    fig2_architecture()
