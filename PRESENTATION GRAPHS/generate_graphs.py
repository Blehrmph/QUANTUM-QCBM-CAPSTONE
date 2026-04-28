"""
QCBM Capstone – Presentation Graphs
Run from inside the 'PRESENTATION GRAPHS' folder:
    python generate_graphs.py
All PNGs are written to the same directory.
"""

import json, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)           # one level up = project root
BEST_RUN = os.path.join(ROOT, "artifacts", "best_run")
ARTS     = os.path.join(ROOT, "artifacts")

def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)

# ── colour palette ─────────────────────────────────────────────────────────────
BG    = "#0F1117"
CARD  = "#1A1D27"
VIOL  = "#6C63FF"
GOLD  = "#FFD700"
TEAL  = "#00C9A7"
CORAL = "#FF6B6B"
MUTED = "#8B95A8"
WHITE = "#F0F4FF"
GRID  = "#2A2D3A"
BLUE  = "#4A90D9"

STYLE = {
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": GRID, "axes.labelcolor": WHITE,
    "xtick.color": MUTED,  "ytick.color": MUTED,
    "text.color": WHITE,   "grid.color": GRID,
    "grid.linestyle": "--","grid.alpha": 0.55,
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(STYLE)


def save(name):
    path = os.path.join(HERE, name)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 01  SIMULATOR vs IBM HARDWARE – key metrics bar chart
# ══════════════════════════════════════════════════════════════════════════════
def graph_01():
    ibm  = load("artifacts/best_run/ibm_score_metrics.json")
    sim  = ibm["simulator_metrics_saved"]
    hw   = ibm["ibm_metrics"]

    labels = ["ROC-AUC", "F1",          "Recall",         "Precision",    "MCC"]
    s_vals = [sim["roc_auc"], sim["f1"], sim["recall_dr"], 0.8952,         0.8893]
    h_vals = [hw["roc_auc"],  hw["f1"], hw["recall_dr"],  hw["precision"], hw["mcc"]]

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    b1 = ax.bar(x - w/2, s_vals, w, color=VIOL,  label="Aer Simulator",        zorder=3)
    b2 = ax.bar(x + w/2, h_vals, w, color=CORAL, label="IBM Hardware (5-member avg)", zorder=3)

    for bar in (*b1, *b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.013,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=10, color=WHITE, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("QCBM Performance: Aer Simulator vs IBM Quantum Hardware",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=12, framealpha=0.15)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    save("01_metrics_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 02  IBM HARDWARE CONDITIONS  (Table 14)
# ══════════════════════════════════════════════════════════════════════════════
def graph_02():
    conditions = [
        "Aer Sim\n(best)",
        "ibm_fez\n3-layer\nsingle",
        "ibm_kingston\n3-layer\nsingle",
        "ibm_fez\n1-layer\nsingle",
        "ibm_fez\n3-layer\nensemble avg",
    ]
    tvd     = [0.00,  0.60,  0.60,  0.82,  0.60 ]
    roc_auc = [0.967, 0.519, 0.486, 0.469, 0.8629]
    roc_cols = [VIOL if v >= 0.80 else CORAL for v in roc_auc]
    roc_cols[-1] = GOLD

    x = np.arange(len(conditions)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx(); ax2.set_facecolor(CARD)

    bars1 = ax1.bar(x - w/2, roc_auc, w, color=roc_cols, zorder=3)
    bars2 = ax2.bar(x + w/2, tvd,     w, color=TEAL, alpha=0.75, zorder=3)

    for bar in bars1:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 fontsize=10, color=WHITE, fontweight="bold")
    for bar in bars2:
        if bar.get_height() > 0:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f"{bar.get_height():.2f}", ha="center", va="bottom",
                     fontsize=10, color=TEAL, fontweight="bold")

    ax1.set_xticks(x); ax1.set_xticklabels(conditions, fontsize=10.5)
    ax1.set_ylim(0, 1.15); ax1.set_ylabel("ROC-AUC", fontsize=13)
    ax2.set_ylim(0, 1.15); ax2.set_ylabel("TVD  (0 = identical, 1 = fully different)", fontsize=12, color=TEAL)
    ax2.tick_params(axis="y", colors=TEAL)

    ax1.set_title("IBM Hardware Conditions: ROC-AUC and Distribution Fidelity (TVD)",
                  fontsize=14, fontweight="bold", pad=14)
    handles = [
        mpatches.Patch(color=VIOL,  label="ROC-AUC ≥ 0.80"),
        mpatches.Patch(color=CORAL, label="ROC-AUC < 0.80"),
        mpatches.Patch(color=GOLD,  label="★ Ensemble-avg (best hardware)"),
        mpatches.Patch(color=TEAL,  label="TVD (right axis)"),
    ]
    ax1.legend(handles=handles, fontsize=10.5, framealpha=0.15, loc="upper right")
    ax1.yaxis.grid(True, zorder=0); ax1.set_axisbelow(True)
    fig.tight_layout()
    save("02_hardware_conditions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 03  SOTA COMPARISON  (horizontal bar, F1)
# ══════════════════════════════════════════════════════════════════════════════
def graph_03():
    sota = load("artifacts/sota_comparison.json")
    names, f1s, colours = [], [], []
    for entry in sota:
        f = entry.get("f1")
        if f is None:
            continue
        names.append(f"{entry['method']}\n({entry['type']})")
        f1s.append(f)
        if entry.get("this_work"):
            colours.append(GOLD)
        elif entry["type"] == "Supervised":
            colours.append(BLUE)
        elif "Quantum" in entry["type"]:
            colours.append(TEAL)
        else:
            colours.append(MUTED)

    # Add our best hardware result manually
    names.append("QCBM IBM Hardware (ours)\n(Quantum-Unsupervised)")
    f1s.append(0.530)
    colours.append(CORAL)

    order = np.argsort(f1s)
    names  = [names[i]   for i in order]
    f1s    = [f1s[i]     for i in order]
    colours = [colours[i] for i in order]

    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(names))
    bars = ax.barh(y, f1s, color=colours, height=0.65, zorder=3)

    for bar, v in zip(bars, f1s):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=10.5, color=WHITE, fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("F1 Score", fontsize=13)
    ax.set_title("QCBM vs State-of-the-Art on UNSW-NB15", fontsize=15,
                 fontweight="bold", pad=14)
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=BLUE,  label="Supervised"),
        mpatches.Patch(color=MUTED, label="Classical Unsupervised"),
        mpatches.Patch(color=TEAL,  label="Other Quantum"),
        mpatches.Patch(color=GOLD,  label="QCBM Simulator (ours)"),
        mpatches.Patch(color=CORAL, label="QCBM IBM Hardware (ours)"),
    ]
    ax.legend(handles=handles, fontsize=10.5, framealpha=0.15, loc="lower right")
    fig.tight_layout()
    save("03_sota_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 04  BINNING ABLATION
# ══════════════════════════════════════════════════════════════════════════════
def graph_04():
    abl = load("artifacts/binning_ablation.json")
    methods = ["IsoForest", "Autoencoder", "KDE", "QCBM (Ours)"]
    quant = [
        abl["classical_quantile_bins"]["IsoForest"]["roc_auc"],
        abl["classical_quantile_bins"]["Autoencoder"]["roc_auc"],
        abl["classical_quantile_bins"]["KDE"]["roc_auc"],
        abl["qcbm_quantile"]["roc_auc"],
    ]
    anomaly = [
        abl["classical_anomaly_bins"]["IsoForest"]["roc_auc"],
        abl["classical_anomaly_bins"]["Autoencoder"]["roc_auc"],
        abl["classical_anomaly_bins"]["KDE"]["roc_auc"],
        0.9671,
    ]

    x = np.arange(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, quant,   w, color=MUTED,  label="Quantile Bins (baseline)", zorder=3)
    b2 = ax.bar(x + w/2, anomaly, w, color=VIOL,   label="Anomaly-Aware Bins (ours)", zorder=3)

    for bar in (*b1, *b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=10.5, color=WHITE, fontweight="bold")

    delta = anomaly[-1] - quant[-1]
    ax.annotate(f"+{delta:.3f} lift",
                xy=(x[-1]+w/2, anomaly[-1]),
                xytext=(x[-1]+w/2+0.4, anomaly[-1]-0.06),
                fontsize=11, color=GOLD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8))

    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=13)
    ax.set_ylim(0, 1.12); ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.set_title("Anomaly-Aware Binning vs Standard Quantile Bins\n(ROC-AUC, one-class setting)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=12, framealpha=0.15)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    save("04_binning_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 05  ENSEMBLE MEMBER GAPS & PIE WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def graph_05():
    members    = ["M1", "M2", "M3", "M4", "M5"]
    normal_kl  = [3.2,  4.1,  2.8,  3.9,  3.4 ]
    anomaly_kl = [11.8, 14.9, 21.1, 9.9,  12.16]
    gaps       = [8.6,  10.8, 18.3, 6.0,  8.76 ]
    total      = sum(gaps)
    weights    = [g / total for g in gaps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(members)); w = 0.55
    pal = [VIOL, TEAL, GOLD, CORAL, MUTED]

    ax1.bar(x, normal_kl,  w, color=VIOL,  label="Normal KL",          zorder=3)
    ax1.bar(x, anomaly_kl, w, color=CORAL, bottom=normal_kl,
            label="Anomaly KL (extra)", zorder=3, alpha=0.85)
    for i, g in enumerate(gaps):
        ax1.text(i, normal_kl[i]+anomaly_kl[i]+0.5,
                 f"gap = {g:.1f}", ha="center", fontsize=11,
                 color=GOLD, fontweight="bold")

    ax1.set_xticks(x); ax1.set_xticklabels(members, fontsize=13)
    ax1.set_ylabel("KL Divergence", fontsize=12)
    ax1.set_title("Per-Member KL Divergence\n(Normal vs Anomaly)", fontsize=12,
                  fontweight="bold")
    ax1.legend(fontsize=11, framealpha=0.15)
    ax1.yaxis.grid(True, zorder=0); ax1.set_axisbelow(True)

    wedges, texts, auto = ax2.pie(
        weights, labels=members, colors=pal,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": WHITE, "fontsize": 12},
        wedgeprops={"linewidth": 1.5, "edgecolor": BG},
    )
    for a in auto:
        a.set_fontsize(11); a.set_color(BG); a.set_fontweight("bold")
    ax2.set_facecolor(CARD)
    ax2.set_title("Gap-Weighted Ensemble Allocation", fontsize=12,
                  fontweight="bold")

    fig.suptitle("QCBM Ensemble: Anomaly Gap and Weight Distribution",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save("05_ensemble_gaps.png")


# ══════════════════════════════════════════════════════════════════════════════
# 06  CONFIDENCE INTERVALS  (5 seeds)
# ══════════════════════════════════════════════════════════════════════════════
def graph_06():
    ci = load("artifacts/confidence_intervals.json")
    m  = ci["metrics"]
    display = {
        "ROC-AUC": "roc_auc",
        "F1":      "f1",
        "Recall":  "recall_dr",
        "MCC":     "mcc",
    }

    keys   = list(display.keys())
    means  = [m[display[k]]["mean"]    for k in keys]
    lo_err = [m[display[k]]["mean"] - m[display[k]]["ci_95_lo"] for k in keys]
    hi_err = [m[display[k]]["ci_95_hi"] - m[display[k]]["mean"] for k in keys]
    seed_vals = [m[display[k]]["values"] for k in keys]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, means, 0.55, color=VIOL, zorder=3, label="Mean (5 seeds)")
    ax.errorbar(x, means, yerr=[lo_err, hi_err],
                fmt="none", color=GOLD, capsize=9, capthick=2.5,
                elinewidth=2.5, zorder=4, label="95% CI")

    for i, (k, sv) in enumerate(zip(keys, seed_vals)):
        for v in sv:
            ax.scatter(i, v, color=WHITE, s=30, zorder=5, alpha=0.75)
        ax.text(i, means[i] + hi_err[i] + 0.018,
                f"{means[i]:.3f}", ha="center", fontsize=11,
                color=WHITE, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=13)
    ax.set_ylim(0.55, 1.08)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Stability: 95% Confidence Intervals Across 5 Random Seeds",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11, framealpha=0.15)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    save("06_confidence_intervals.png")


# ══════════════════════════════════════════════════════════════════════════════
# 07  CALIBRATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def graph_07():
    s1 = load("artifacts/best_run/hier_stage1_metrics.json")

    stages   = ["Raw QCBM\nStage 1", "LR\nCalibration", "LR + Isotonic\n(Final)"]
    roc_auc  = [s1["roc_auc"],
                s1["two_stage_lr_metrics"]["roc_auc"],
                s1["isotonic_calibration_metrics"]["roc_auc"]]
    f1_vals  = [s1["f1"],
                s1["two_stage_lr_metrics"]["f1"],
                s1["isotonic_calibration_metrics"]["f1"]]
    far_vals = [s1["far"],
                s1["two_stage_lr_metrics"]["far"],
                s1["isotonic_calibration_metrics"]["far"]]

    x = np.arange(len(stages)); w = 0.25
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx(); ax2.set_facecolor(CARD)

    b1 = ax1.bar(x - w, roc_auc,  w, color=VIOL,  label="ROC-AUC", zorder=3)
    b2 = ax1.bar(x,     f1_vals,  w, color=GOLD,  label="F1",      zorder=3)
    b3 = ax2.bar(x + w, far_vals, w, color=CORAL, label="FAR",     zorder=3, alpha=0.85)

    for bar in (*b1, *b2):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{bar.get_height():.4f}", ha="center", va="bottom",
                 fontsize=9.5, color=WHITE, fontweight="bold")
    for bar in b3:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003,
                 f"{bar.get_height():.4f}", ha="center", va="bottom",
                 fontsize=9.5, color=CORAL, fontweight="bold")

    ax1.set_xticks(x); ax1.set_xticklabels(stages, fontsize=12)
    ax1.set_ylim(0.85, 1.01); ax1.set_ylabel("ROC-AUC / F1",            fontsize=13)
    ax2.set_ylim(0, 0.05);    ax2.set_ylabel("FAR (lower = better)",     fontsize=13, color=CORAL)
    ax2.tick_params(axis="y", colors=CORAL)

    ax1.set_title("Two-Stage Calibration Pipeline: Step-by-Step Improvement",
                  fontsize=13, fontweight="bold", pad=14)
    handles = [mpatches.Patch(color=VIOL, label="ROC-AUC"),
               mpatches.Patch(color=GOLD, label="F1"),
               mpatches.Patch(color=CORAL, label="FAR (right axis)")]
    ax1.legend(handles=handles, fontsize=11, framealpha=0.15)
    ax1.yaxis.grid(True, zorder=0); ax1.set_axisbelow(True)
    fig.tight_layout()
    save("07_calibration_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# 08  HARDWARE NOISE MITIGATION  (single vs ensemble)
# ══════════════════════════════════════════════════════════════════════════════
def graph_08():
    ibm = load("artifacts/best_run/ibm_score_metrics.json")
    sim = ibm["simulator_metrics_saved"]
    hw  = ibm["ibm_metrics"]

    categories = ["ROC-AUC", "Recall"]
    s_vals  = [sim["roc_auc"], sim["recall_dr"]]
    h1_vals = [0.519, 0.75]           # single-member approx from ibm_run.log
    h5_vals = [hw["roc_auc"], hw["recall_dr"]]

    x = np.arange(len(categories)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w,   s_vals,  w, color=VIOL,  label="Aer Simulator",              zorder=3)
    b2 = ax.bar(x,       h1_vals, w, color=CORAL, label="IBM Hardware (1 member)",     zorder=3, alpha=0.85)
    b3 = ax.bar(x + w,   h5_vals, w, color=GOLD,  label="IBM Hardware (5-member avg)", zorder=3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.013,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=11, color=WHITE, fontweight="bold")

    # Recovery arrow on ROC-AUC
    ax.annotate("",
                xy=(x[0]+w, h5_vals[0]),
                xytext=(x[0], h1_vals[0]),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=2.2))
    ax.text(x[0]+w/2+0.02, (h5_vals[0]+h1_vals[0])/2 + 0.05,
            "+0.344 recovered", ha="center", fontsize=10.5,
            color=GOLD, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Ensemble Averaging as Implicit Hardware Noise Mitigation",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11, framealpha=0.15)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    save("08_hardware_noise_mitigation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 09  DEPOLARISING NOISE SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def graph_09():
    noise = load("artifacts/noise_simulation.json")
    ibm   = load("artifacts/best_run/ibm_score_metrics.json")
    hw_roc = ibm["ibm_metrics"]["roc_auc"]

    labels  = [n["label"]             for n in noise]
    roc_auc = [n["metrics"]["roc_auc"] for n in noise]
    f1_vals = [n["metrics"]["f1"]      for n in noise]

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx(); ax2.set_facecolor(CARD)

    ax1.plot(x, roc_auc, "o-", color=VIOL, lw=2.5, ms=9, label="ROC-AUC", zorder=4)
    ax2.plot(x, f1_vals, "s--", color=GOLD, lw=2.5, ms=9, label="F1",     zorder=4)
    ax1.fill_between(x, [v-0.0015 for v in roc_auc], [v+0.0015 for v in roc_auc],
                     color=VIOL, alpha=0.13)

    for i, (r, f) in enumerate(zip(roc_auc, f1_vals)):
        ax1.text(i, r+0.002,  f"{r:.4f}", ha="center", fontsize=9.5,
                 color=VIOL, fontweight="bold")
        ax2.text(i, f-0.005, f"{f:.4f}", ha="center", fontsize=9.5,
                 color=GOLD, fontweight="bold")

    ax1.axhline(hw_roc, color=CORAL, lw=1.8, ls=":",
                label=f"Actual IBM hardware ({hw_roc:.3f})")
    ax1.text(len(labels)-0.98, hw_roc+0.003,
             "actual IBM hardware", color=CORAL, fontsize=9.5)

    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0.82, 0.98);  ax1.set_ylabel("ROC-AUC",  fontsize=13)
    ax2.set_ylim(0.65, 0.78);  ax2.set_ylabel("F1 Score",  fontsize=13, color=GOLD)
    ax2.tick_params(axis="y", colors=GOLD)
    ax1.set_title("Depolarising Noise Sweep: ROC-AUC and F1 vs Gate Error Rate",
                  fontsize=13, fontweight="bold", pad=14)

    handles = [mpatches.Patch(color=VIOL,  label="ROC-AUC (left)"),
               mpatches.Patch(color=GOLD,  label="F1 (right)"),
               mpatches.Patch(color=CORAL, label="Actual IBM hardware ROC-AUC")]
    ax1.legend(handles=handles, fontsize=11, framealpha=0.15)
    ax1.yaxis.grid(True, zorder=0); ax1.set_axisbelow(True)
    fig.tight_layout()
    save("09_noise_simulation.png")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating presentation graphs -> PRESENTATION GRAPHS/\n")
    graph_01()
    graph_02()
    graph_03()
    graph_04()
    graph_05()
    graph_06()
    graph_07()
    graph_08()
    graph_09()
    print(f"\nAll 9 graphs saved.")
