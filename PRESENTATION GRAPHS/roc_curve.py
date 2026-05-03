"""
QCBM Capstone – ROC Curve Figure
Run from project root OR inside the 'PRESENTATION GRAPHS' folder:
    python "PRESENTATION GRAPHS/roc_curve.py"
Saves 12_roc_curve.png to the 'PRESENTATION GRAPHS' directory.

Three curves:
  1. QCBM Simulator  (raw -log p scores)
  2. QCBM + LR+Isotonic calibration  (reported best operating point)
  3. IBM Hardware     (real quantum device)
"""

from __future__ import annotations

import os, sys, warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

BEST_RUN  = os.path.join(ROOT, "artifacts", "best_run")
CFG_PATH  = os.path.join(ROOT, "best_config.json")
SIM_DIST  = os.path.join(BEST_RUN, "hier_qcbm_model_dist.npy")
IBM_DIST  = os.path.join(BEST_RUN, "ibm_dist.npy")

# ── light-theme style matching reference image ─────────────────────────────────
BG      = "#FFFFFF"
PANEL   = "#F8F8F8"
DARK    = "#111111"
SUBTEXT = "#555555"
BORDER  = "#CCCCCC"
GRIDCOL = "#E0E0E0"

C_BLUE   = "#4B92C8"   # steel blue  (curve 1)
C_ORANGE = "#E07B39"   # burnt orange (curve 2, calibrated)
C_RED    = "#C43B2C"   # dark red     (curve 3, IBM)
C_GREY   = "#9E9E9E"   # chance line

STYLE = {
    "figure.facecolor":  BG,     "axes.facecolor":   PANEL,
    "axes.edgecolor":    BORDER, "axes.labelcolor":  DARK,
    "xtick.color":       DARK,   "ytick.color":      DARK,
    "text.color":        DARK,
    "grid.color":        GRIDCOL,"grid.linestyle":   "--",
    "grid.alpha":        1.0,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
}
plt.rcParams.update(STYLE)


# ── preprocessing (mirrors ibm_score.py exactly) ──────────────────────────────

def run_preprocessing(cfg: dict):
    from src.data.preprocessing import (
        add_categorical_features, apply_log1p,
        select_features, DEFAULT_LOG1P_COLS, Scaler,
    )
    from src.training_setup import train_val_test_split, filter_normal
    from src.discretize import auto_mixed_precision_map, fit_bins, transform_bins, encode_bits

    input_path = cfg.get("input", "datasets/UNSW-NB15_cleaned.csv")
    label_col  = cfg.get("label_col",  "label")
    seed       = cfg.get("seed", 42)

    print("  Loading dataset ...")
    df = pd.read_csv(os.path.join(ROOT, input_path), low_memory=False)

    for col in ["proto", "state", "service"]:
        if col not in df.columns:
            extra = pd.read_csv(
                os.path.join(ROOT, "datasets/UNSW-NB15_cleaned.csv"),
                usecols=[col], low_memory=False,
            )
            df[col] = extra[col].values

    df = add_categorical_features(df)

    features = [f.strip() for f in cfg["features"].split(",") if f.strip()]
    print(f"  Features ({len(features)}): {', '.join(features)}")

    X = select_features(df, features)
    y = df[label_col]

    print("  Splitting ...")
    splits = train_val_test_split(
        X, y,
        test_frac=cfg.get("test_frac", 0.2),
        val_frac=cfg.get("val_frac", 0.1),
        seed=seed, stratify=True,
    )

    if cfg.get("log1p", True):
        splits.X_train = apply_log1p(splits.X_train, DEFAULT_LOG1P_COLS)
        splits.X_val   = apply_log1p(splits.X_val,   DEFAULT_LOG1P_COLS)
        splits.X_test  = apply_log1p(splits.X_test,  DEFAULT_LOG1P_COLS)

    print("  Scaling ...")
    scaler  = Scaler(mode=cfg.get("scaler", "standard")).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
    X_val   = scaler.transform(splits.X_val,   features)
    X_test  = scaler.transform(splits.X_test,  features)

    use_amp = cfg.get("auto_mixed_precision", False)
    if use_amp:
        bits_map, bins_map = auto_mixed_precision_map(
            X_train, features,
            continuous_bits=cfg.get("bits_per_feature", 2),
            continuous_bins=cfg.get("n_bins", 4),
        )
    else:
        bits_map, bins_map = None, None

    y_train_reset = splits.y_train.reset_index(drop=True)
    anomaly_mask  = (y_train_reset.to_numpy() == 1)
    X_train_anom  = X_train.iloc[anomaly_mask] if anomaly_mask.any() else None

    print("  Fitting bins ...")
    edges  = fit_bins(X_train, features,
                      n_bins=cfg.get("n_bins", 4),
                      strategy=cfg.get("bin_strategy", "quantile"),
                      n_bins_map=bins_map,
                      df_anomaly=X_train_anom)

    enc_kw = dict(
        bits_per_feature=cfg.get("bits_per_feature", 2),
        encoding=cfg.get("encoding", "binary"),
        n_bins=cfg.get("n_bins", 4),
        bits_per_feature_map=bits_map,
    )
    bit_val  = encode_bits(transform_bins(X_val,  edges), **enc_kw)
    bit_test = encode_bits(transform_bins(X_test, edges), **enc_kw)

    return (
        bit_val,  bit_test,
        splits.y_val.reset_index(drop=True),
        splits.y_test.reset_index(drop=True),
    )


# ── scoring helpers ────────────────────────────────────────────────────────────

def raw_scores(bits, dist):
    from src.score_eval import score_samples
    return score_samples(bits, dist)


def calibrate_lr_isotonic(scores_val, y_val, scores_test):
    """Two-stage LR + Isotonic calibration (mirrors stage1.py)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    X_v = scores_val.reshape(-1, 1)
    lr  = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X_v, y_val.to_numpy())
    p_val  = lr.predict_proba(X_v)[:, 1]
    p_test = lr.predict_proba(scores_test.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val.to_numpy())
    return iso.predict(p_test)


def youden_point(fpr, tpr):
    """Return the (fpr, tpr) pair maximising Youden's J = TPR - FPR."""
    j   = tpr - fpr
    idx = np.argmax(j)
    return float(fpr[idx]), float(tpr[idx])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with open(CFG_PATH) as f:
        cfg = json.load(f)

    sim_dist = np.load(SIM_DIST)
    if sim_dist.ndim == 2:          # (n_ensemble, 32768) -> average members
        sim_dist = sim_dist.mean(axis=0)
    ibm_dist = np.load(IBM_DIST)
    if ibm_dist.ndim == 2:
        ibm_dist = ibm_dist.mean(axis=0)

    print("Preprocessing ...")
    bit_val, bit_test, y_val, y_test = run_preprocessing(cfg)

    print("Scoring ...")
    sim_raw  = raw_scores(bit_test, sim_dist)
    ibm_raw  = raw_scores(bit_test, ibm_dist)

    print("Calibrating (LR + Isotonic) ...")
    val_raw  = raw_scores(bit_val, sim_dist)
    sim_cal  = calibrate_lr_isotonic(val_raw, y_val, sim_raw)

    y_arr = y_test.to_numpy()
    from sklearn.metrics import roc_curve, auc

    fpr_sim, tpr_sim, _ = roc_curve(y_arr, sim_raw)
    fpr_cal, tpr_cal, _ = roc_curve(y_arr, sim_cal)
    fpr_ibm, tpr_ibm, _ = roc_curve(y_arr, ibm_raw)

    auc_sim = auc(fpr_sim, tpr_sim)
    auc_cal = auc(fpr_cal, tpr_cal)
    auc_ibm = auc(fpr_ibm, tpr_ibm)

    print(f"  Simulator (raw)       AUC = {auc_sim:.4f}")
    print(f"  Simulator (LR+Iso)    AUC = {auc_cal:.4f}")
    print(f"  IBM Hardware          AUC = {auc_ibm:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    ax.plot(fpr_sim, tpr_sim, color=C_BLUE,   lw=2.0,
            label=f"QCBM Simulator  AUC={auc_sim:.4f}")
    ax.plot(fpr_cal, tpr_cal, color=C_ORANGE, lw=2.0,
            label=f"QCBM + LR+Isotonic  AUC={auc_cal:.4f}")
    ax.plot(fpr_ibm, tpr_ibm, color=C_RED,    lw=2.0,
            label=f"IBM Hardware  AUC={auc_ibm:.4f}")
    ax.plot([0, 1], [0, 1], color=C_GREY, lw=1.2, linestyle="--", label="Chance")

    # operating-point stars (Youden's J)
    for fpr_c, tpr_c, color in [
        (fpr_sim, tpr_sim, C_BLUE),
        (fpr_cal, tpr_cal, C_ORANGE),
        (fpr_ibm, tpr_ibm, C_RED),
    ]:
        ox, oy = youden_point(fpr_c, tpr_c)
        ax.plot(ox, oy, marker="*", markersize=14, color=color,
                markeredgecolor="black", markeredgewidth=0.6, zorder=5)

    ax.set_xlabel("False Positive Rate", fontsize=12, color=DARK)
    ax.set_ylabel("True Positive Rate",  fontsize=12, color=DARK)
    ax.set_title("ROC Curve – QCBM Anomaly Detection (test set)",
                 fontsize=13, fontweight="bold", color=DARK, pad=10)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

    ax.legend(
        loc="lower right", fontsize=10,
        framealpha=0.9, facecolor=BG,
        edgecolor=BORDER, labelcolor=DARK,
    )

    plt.tight_layout()
    out = os.path.join(HERE, "12_roc_curve.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
