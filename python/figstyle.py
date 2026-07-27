"""
Shared publication figure style — STLF V5.

One place that defines the look of every figure in the paper: a validated,
colour-blind-safe categorical palette, hairline chrome, thin marks and a
consistent type scale. Colours are NOT eyeballed: the categorical slots below
pass the lightness-band, chroma-floor, CVD-separation (OKLab dE >= 8 under
simulated protanopia/deuteranopia) and normal-vision floor checks on the
all-pairs list for the first three slots, which is the most any figure here
uses simultaneously.

    slot 1  blue    #2a78d6   reference / actual / "valid protocol"
    slot 2  orange  #eb6834   the proposed model (the hero series)
    slot 3  aqua    #1baf7a   strongest baseline
    (worst all-pairs CVD dE 9.2, normal-vision dE 24.0, light surface)

Aqua sits at 2.74:1 against the surface, below the 3:1 bar, so every figure
using it also carries a direct label or a legend entry — never colour alone.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── palette ────────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text
MUTED = "#898781"        # axis labels, annotations
GRID = "#e1e0d9"         # hairline gridlines
AXIS = "#c3c2b7"         # baseline / spines

BLUE = "#2a78d6"         # slot 1
ORANGE = "#eb6834"       # slot 2
AQUA = "#1baf7a"         # slot 3
VIOLET = "#4a3aa7"       # slot 7 (4th series, adjacent-pairs contexts only)
CRITICAL = "#d03b3b"     # status: invalid / leaky protocol
GOOD = "#0ca30c"         # status: valid

SERIES = [BLUE, ORANGE, AQUA, VIOLET]

# Sequential ramp (one hue, light -> dark) for heatmaps
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
            "#256abf", "#184f95", "#0d366b"]

# Neutral fills for diagram blocks
FILL_SOFT = "#f2f1ed"
FILL_BLUE = "#e8f1fc"
FILL_ORANGE = "#fdefe8"
FILL_AQUA = "#e6f7f1"


def apply_style():
    """Global rcParams: hairline chrome, sans type, vector-friendly output."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 400,
        "figure.dpi": 150,
        # Type: system sans, no serif/display faces
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Segoe UI", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        # Chrome: recessive, solid hairlines, no top/right spines
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",          # never dashed
        "axes.axisbelow": True,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 4,
        "ytick.major.pad": 3,
        # Marks: thin
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "patch.linewidth": 0,
        # Legend: no box, it is chrome not a mark
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.4,
        "legend.borderaxespad": 0.2,
        # Vector output with real text (editable, searchable)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, path_noext, formats=("pdf", "png")):
    """Save one figure as vector PDF + 400-dpi PNG."""
    import os
    for ext in formats:
        fig.savefig(f"{path_noext}.{ext}", format=ext)
    plt.close(fig)
    print(f"  saved {os.path.basename(path_noext)}.{'/'.join(formats)}")


def tidy(ax, ygrid=True, xgrid=False):
    """Apply the recessive-chrome conventions to one axes."""
    ax.set_axisbelow(True)
    # passing line properties alongside visible=False would silently turn the
    # grid back on, so only style the axis that is actually being drawn
    for on, axis in ((ygrid, "y"), (xgrid, "x")):
        ax.grid(on, axis=axis, **({"color": GRID, "lw": 0.6} if on else {}))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.6)
    return ax


def label_end(ax, x, y, text, color, dx=6, dy=0, weight="bold", size=8):
    """Direct-label a series at its endpoint (used instead of dense labels)."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=size, fontweight=weight,
                va="center", ha="left", annotation_clip=False)
