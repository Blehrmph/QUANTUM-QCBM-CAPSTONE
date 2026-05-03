"""
QCBM Capstone – Correlation Matrix
Run from inside the 'PRESENTATION GRAPHS' folder:
    python correlation_matrix.py
Saves 11_correlation_matrix.png to the same directory.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "datasets", "UNSW-NB15_cleaned.csv")

# ── light theme matching reference image ───────────────────────────────────────
BG      = "#FFFFFF"
PANEL   = "#F8F8F8"
DARK    = "#111111"
SUBTEXT = "#444444"
BORDER  = "#CCCCCC"

# diverging: dark red -> white -> dark navy (matches reference image exactly)
CMAP = LinearSegmentedColormap.from_list(
    "rdwbu",
    ["#053061", "#2166AC", "#92C5DE", "#FFFFFF", "#F4A582", "#B2182B", "#67001F"],
    N=512,
)

STYLE = {
    "figure.facecolor":  BG,     "axes.facecolor":   PANEL,
    "axes.edgecolor":    BORDER, "axes.labelcolor":  DARK,
    "xtick.color":       DARK,   "ytick.color":      DARK,
    "text.color":        DARK,   "grid.alpha":       0.0,
    "font.family":       "DejaVu Sans",
}
plt.rcParams.update(STYLE)

FEATURES = ["dur", "sbytes", "dbytes", "Sload", "Dload",
            "Spkts", "Dpkts", "tcprtt", "sttl"]

LABELS = ["dur", "sbytes", "dbytes", "Sload", "Dload",
          "Spkts", "Dpkts", "tcprtt", "sttl"]

def load_data(n_per_class=60_000, seed=42):
    cols = FEATURES + ["label"]
    df   = pd.read_csv(DATASET, usecols=cols)
    for col in ["sbytes", "dbytes", "Sload", "Dload", "dur"]:
        df[col] = np.log1p(df[col].clip(lower=0))

    normal = df[df["label"] == 0].drop(columns="label")
    attack = df[df["label"] == 1].drop(columns="label")

    if len(normal) > n_per_class:
        normal = normal.sample(n=n_per_class, random_state=seed)
    if len(attack) > n_per_class:
        attack = attack.sample(n=n_per_class, random_state=seed)

    full = df.drop(columns="label").sample(
        n=min(len(df), 2 * n_per_class), random_state=seed
    )
    return normal, attack, full

def draw_heatmap(ax, corr, title):
    n = len(corr)
    im = ax.imshow(corr.values, cmap=CMAP, vmin=-1, vmax=1,
                   aspect="equal", interpolation="nearest")

    # grid lines between cells
    for k in range(n + 1):
        ax.axhline(k - 0.5, color=BG, lw=0.8)
        ax.axvline(k - 0.5, color=BG, lw=0.8)

    # annotate each cell
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            # white text on dark cells, dark text on light cells
            text_color = "white" if abs(val) > 0.45 else DARK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=9, color=DARK)
    ax.set_yticklabels(LABELS, fontsize=9, color=DARK)
    ax.set_title(title, color=DARK, fontsize=11, fontweight="bold", pad=10)
    ax.set_facecolor(PANEL)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

    return im

def main():
    print("Loading dataset ...")
    normal, attack, full = load_data()
    print(f"  Normal: {len(normal):,}  |  Attack: {len(attack):,}  |  Full sample: {len(full):,}")

    corr_normal = normal.corr()
    corr_attack = attack.corr()
    corr_full   = full.corr()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "UNSW-NB15  -  Pearson Correlation Matrix  (log1p-transformed features)",
        color=DARK, fontsize=13, fontweight="bold", y=1.02,
    )

    im = None
    for ax, corr, title in zip(
        axes,
        [corr_normal, corr_attack, corr_full],
        ["Normal Traffic", "Attack Traffic", "All Traffic (combined)"],
    ):
        im = draw_heatmap(ax, corr, title)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.yaxis.set_tick_params(color=SUBTEXT, labelsize=8)
    cbar.outline.set_edgecolor(BORDER)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTEXT)
    cbar.set_label("Pearson r", color=SUBTEXT, fontsize=9, rotation=270, labelpad=14)

    plt.tight_layout()
    out = os.path.join(HERE, "11_correlation_matrix.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  Saved: {out}")

if __name__ == "__main__":
    main()
