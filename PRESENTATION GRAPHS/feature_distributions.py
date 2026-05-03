"""
QCBM Capstone – Feature Distribution Graph
Run from inside the 'PRESENTATION GRAPHS' folder:
    python feature_distributions.py
Saves 10_feature_distributions.png to the same directory.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "datasets", "UNSW-NB15_cleaned.csv")

# ── colour palette (matches generate_graphs.py) ────────────────────────────────
BG    = "#0F1117"
CARD  = "#1A1D27"
VIOL  = "#6C63FF"
GOLD  = "#FFD700"
TEAL  = "#00C9A7"
CORAL = "#FF6B6B"
MUTED = "#8B95A8"
WHITE = "#F0F4FF"
GRID  = "#2A2D3A"

STYLE = {
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": GRID, "axes.labelcolor": WHITE,
    "xtick.color": MUTED,  "ytick.color": MUTED,
    "text.color": WHITE,   "grid.color": GRID,
    "grid.linestyle": "--","grid.alpha": 0.55,
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(STYLE)

# ── features and display metadata ──────────────────────────────────────────────
FEATURES = [
    ("dur",    "Flow Duration",         True),
    ("sbytes", "Source Bytes",          True),
    ("dbytes", "Dest Bytes",            True),
    ("Sload",  "Source Load",           True),
    ("Dload",  "Dest Load",             True),
    ("Spkts",  "Source Packets",        False),
    ("Dpkts",  "Dest Packets",          False),
    ("tcprtt", "TCP Round-Trip Time",   True),
    ("sttl",   "Source TTL",            False),
]

def load_data(n_normal=80_000, n_attack=40_000, seed=42):
    rng = np.random.default_rng(seed)
    cols = [f for f, _, _ in FEATURES] + ["label"]
    df   = pd.read_csv(DATASET, usecols=cols)

    normal = df[df["label"] == 0]
    attack = df[df["label"] == 1]

    if len(normal) > n_normal:
        normal = normal.sample(n=n_normal, random_state=seed)
    if len(attack) > n_attack:
        attack = attack.sample(n=n_attack, random_state=seed)

    return normal, attack

def kde_line(values, x_grid, bw="scott"):
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(values, bw_method=bw)
        return kde(x_grid)
    except Exception:
        return np.zeros_like(x_grid)

def plot_feature(ax, col, title, log_transform, norm_vals, att_vals):
    if log_transform:
        norm_vals = np.log1p(np.maximum(norm_vals, 0))
        att_vals  = np.log1p(np.maximum(att_vals,  0))
        xlabel    = f"log(1 + {col})"
    else:
        xlabel = col

    all_vals = np.concatenate([norm_vals, att_vals])
    all_vals = all_vals[np.isfinite(all_vals)]
    lo, hi   = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
    if lo >= hi:
        lo, hi = all_vals.min(), all_vals.max()
    x_grid = np.linspace(lo, hi, 300)

    # histograms (density)
    bins = np.linspace(lo, hi, 50)
    ax.hist(norm_vals, bins=bins, density=True,
            color=VIOL,  alpha=0.25, label="Normal")
    ax.hist(att_vals,  bins=bins, density=True,
            color=CORAL, alpha=0.25, label="Attack")

    # KDE lines
    norm_kde = kde_line(norm_vals, x_grid)
    att_kde  = kde_line(att_vals,  x_grid)
    ax.plot(x_grid, norm_kde, color=VIOL,  lw=1.8, label="Normal KDE")
    ax.plot(x_grid, att_kde,  color=CORAL, lw=1.8, label="Attack KDE")

    ax.set_title(title, color=WHITE, fontsize=10, pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density",  fontsize=8)
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=7)
    ax.grid(True)
    ax.set_facecolor(CARD)

def main():
    print("Loading dataset …")
    normal, attack = load_data()
    print(f"  Normal: {len(normal):,}  |  Attack: {len(attack):,}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "UNSW-NB15  –  Feature Distributions: Normal vs Attack",
        color=WHITE, fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (col, title, log_t) in zip(axes.flat, FEATURES):
        norm_vals = normal[col].dropna().values.astype(float)
        att_vals  = attack[col].dropna().values.astype(float)
        plot_feature(ax, col, title, log_t, norm_vals, att_vals)

    # shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=VIOL,  lw=2, label="Normal"),
        Line2D([0], [0], color=CORAL, lw=2, label="Attack"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper right",
        fontsize=10, framealpha=0.3,
        facecolor=CARD, edgecolor=GRID, labelcolor=WHITE,
    )

    plt.tight_layout()
    out = os.path.join(HERE, "10_feature_distributions.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  Saved: {out}")

if __name__ == "__main__":
    main()
