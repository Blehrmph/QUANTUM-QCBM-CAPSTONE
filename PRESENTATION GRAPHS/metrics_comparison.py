"""
QCBM Capstone – Aer Simulator vs IBM Hardware Metrics Comparison
Run from project root OR inside the 'PRESENTATION GRAPHS' folder:
    python "PRESENTATION GRAPHS/metrics_comparison.py"
Saves 15_metrics_comparison.png to the 'PRESENTATION GRAPHS' directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ── data from provided table ──────────────────────────────────────────────────
METRICS = [
    ("ROC-AUC",      0.9671, 0.8629, False),
    ("PR-AUC",       0.8931, 0.3834, False),
    ("F1",           0.9015, 0.5300, False),
    ("Recall (DR)",  0.9080, 0.8263, False),
    ("Precision",    0.8952, 0.3901, False),
    ("FAR",          0.0131, 0.1598, True ),   # lower is better
    ("MCC",          0.8893, 0.4934, False),
]

LABELS   = [m[0] for m in METRICS]
SIM_VALS = [m[1] for m in METRICS]
IBM_VALS = [m[2] for m in METRICS]
LOW_BETTER = [m[3] for m in METRICS]

# ── light theme ───────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
PANEL   = "#F8F8F8"
DARK    = "#111111"
SUBTEXT = "#555555"
BORDER  = "#CCCCCC"
GRIDCOL = "#E8E8E8"

from matplotlib.colors import LinearSegmentedColormap

# red(0) -> yellow(0.5) -> green(1) gradient for bar colors
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_grad",
    ["#D32F2F", "#FDD835", "#2E7D32"],  # red -> yellow -> green
    N=256,
)

def bar_color(value, force_green=False):
    if force_green:
        return "#2E7D32"
    return SCORE_CMAP(float(np.clip(value, 0, 1)))

STYLE = {
    "figure.facecolor": BG,     "axes.facecolor":  PANEL,
    "axes.edgecolor":   BORDER, "axes.labelcolor": DARK,
    "xtick.color":      DARK,   "ytick.color":     DARK,
    "text.color":       DARK,
    "grid.color":       GRIDCOL,"grid.linestyle":  "--",
    "grid.alpha":       1.0,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
}
plt.rcParams.update(STYLE)


def main():
    n  = len(METRICS)
    x  = np.arange(n)
    w  = 0.34

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    colors = [
        bar_color(v, force_green=low)
        for v, low in zip(SIM_VALS, LOW_BETTER)
    ]

    bars_sim = ax.bar(x, SIM_VALS, w + 0.1, color=colors,
                      zorder=3, clip_on=False)

    # value labels on top of each bar
    for bar in bars_sim:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012,
            f"{h:.4f}",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=DARK,
        )

    # x-axis labels — append note for lower-is-better metrics
    x_labels = [
        f"{lbl}\n(lower is better)" if low else lbl
        for lbl, low in zip(LABELS, LOW_BETTER)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=12, color=DARK)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "QCBM Aer Simulator – Performance Metrics",
        fontsize=14, fontweight="bold", color=DARK, pad=14,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

    plt.tight_layout()
    out = os.path.join(HERE, "15_metrics_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
