"""
QCBM Capstone – Confusion Matrices
Run from project root OR inside the 'PRESENTATION GRAPHS' folder:
    python "PRESENTATION GRAPHS/confusion_matrices.py"
Saves 14_confusion_matrices.png to the 'PRESENTATION GRAPHS' directory.

Three matrices from saved metrics:
  1. QCBM Simulator – F1-optimal threshold
  2. QCBM Simulator – LR + Isotonic calibrated (best operating point)
  3. IBM Hardware
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
BEST_RUN = os.path.join(ROOT, "artifacts", "best_run")

# ── white → dark navy sequential colormap (matches reference image) ───────────
CMAP = LinearSegmentedColormap.from_list(
    "navy_seq",
    ["#FFFFFF", "#D0E4F2", "#6BAED6", "#2171B5", "#08306B"],
    N=256,
)

BG      = "#FFFFFF"
DARK    = "#111111"
SUBTEXT = "#555555"
BORDER  = "#CCCCCC"

STYLE = {
    "figure.facecolor": BG,     "axes.facecolor":  BG,
    "axes.edgecolor":   BORDER, "axes.labelcolor": DARK,
    "xtick.color":      DARK,   "ytick.color":     DARK,
    "text.color":       DARK,   "grid.alpha":      0.0,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
}
plt.rcParams.update(STYLE)

# ── load metrics ──────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

stage1   = load_json(os.path.join(BEST_RUN, "hier_stage1_metrics.json"))
ibm_meta = load_json(os.path.join(BEST_RUN, "ibm_score_metrics.json"))

MATRICES = [
    {
        "title":   "QCBM Simulator\nF1-Optimal Threshold",
        "tp": stage1["f1_threshold_metrics"]["tp"],
        "fp": stage1["f1_threshold_metrics"]["fp"],
        "fn": stage1["f1_threshold_metrics"]["fn"],
        "tn": stage1["f1_threshold_metrics"]["tn"],
        "metrics": {
            "F1":        stage1["f1_threshold_metrics"]["f1"],
            "Precision": stage1["f1_threshold_metrics"]["precision"],
            "Recall":    stage1["f1_threshold_metrics"]["recall_dr"],
            "FAR":       stage1["f1_threshold_metrics"]["far"],
        },
    },
    {
        "title":   "QCBM Simulator\nLR + Isotonic Calibrated",
        "tp": stage1["isotonic_calibration_metrics"]["tp"],
        "fp": stage1["isotonic_calibration_metrics"]["fp"],
        "fn": stage1["isotonic_calibration_metrics"]["fn"],
        "tn": stage1["isotonic_calibration_metrics"]["tn"],
        "metrics": {
            "F1":        stage1["isotonic_calibration_metrics"]["f1"],
            "Precision": stage1["isotonic_calibration_metrics"]["precision"],
            "Recall":    stage1["isotonic_calibration_metrics"]["recall_dr"],
            "FAR":       stage1["isotonic_calibration_metrics"]["far"],
        },
    },
    {
        "title":   "IBM Hardware\n(ibm_fez, 5-member ensemble)",
        "tp": ibm_meta["ibm_metrics"]["tp"],
        "fp": ibm_meta["ibm_metrics"]["fp"],
        "fn": ibm_meta["ibm_metrics"]["fn"],
        "tn": ibm_meta["ibm_metrics"]["tn"],
        "metrics": {
            "F1":        ibm_meta["ibm_metrics"]["f1"],
            "Precision": ibm_meta["ibm_metrics"]["precision"],
            "Recall":    ibm_meta["ibm_metrics"]["recall_dr"],
            "FAR":       ibm_meta["ibm_metrics"]["far"],
        },
    },
]

# display order: row 0 = True Normal, row 1 = True Attack
#                col 0 = Pred Normal, col 1 = Pred Attack
CELL_TERMS = [["TN", "FP"],
              ["FN", "TP"]]

ROW_LABELS = ["True Normal", "True Attack"]
COL_LABELS = ["Pred Normal", "Pred Attack"]


def draw_matrix(ax, m):
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=float)

    vmax = cm.max()
    im   = ax.imshow(cm, cmap=CMAP, vmin=0, vmax=vmax,
                     aspect="equal", interpolation="nearest")

    # white separator lines between cells
    for k in [0.5]:
        ax.axhline(k, color="white", lw=3)
        ax.axvline(k, color="white", lw=3)

    # annotate each cell: term label + count
    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            term  = CELL_TERMS[i][j]
            count = int(val)
            # white text on dark cells, dark text on light cells
            text_col = "white" if val / vmax > 0.45 else DARK
            # term label (bold, smaller)
            ax.text(j, i - 0.13, term,
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_col)
            # count
            ax.text(j, i + 0.17, f"{count:,}",
                    ha="center", va="center",
                    fontsize=11, color=text_col)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(COL_LABELS, fontsize=10)
    ax.set_yticklabels(ROW_LABELS, fontsize=10)
    ax.tick_params(length=0)

    ax.set_title(m["title"], color=DARK, fontsize=11,
                 fontweight="bold", pad=12)

    met = m["metrics"]
    footer = (
        f"F1={met['F1']:.3f}   Prec={met['Precision']:.3f}   "
        f"Recall={met['Recall']:.3f}   FAR={met['FAR']:.4f}"
    )
    ax.set_xlabel(footer, fontsize=8.5, color=SUBTEXT, labelpad=10)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

    return im


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Confusion Matrices – QCBM Anomaly Detection (test set)",
        color=DARK, fontsize=14, fontweight="bold", y=1.04,
    )

    for ax, m in zip(axes, MATRICES):
        im = draw_matrix(ax, m)
        # individual colorbar per matrix
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, color=SUBTEXT)
        cbar.outline.set_edgecolor(BORDER)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTEXT)

    plt.tight_layout()
    out = os.path.join(HERE, "14_confusion_matrices.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
