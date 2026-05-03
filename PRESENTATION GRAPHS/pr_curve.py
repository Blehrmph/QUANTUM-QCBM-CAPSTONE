"""
QCBM Capstone – Precision-Recall Curve Figure
Run from project root OR inside the 'PRESENTATION GRAPHS' folder:
    python "PRESENTATION GRAPHS/pr_curve.py"
Saves 13_pr_curve.png to the 'PRESENTATION GRAPHS' directory.

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

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

BEST_RUN = os.path.join(ROOT, "artifacts", "best_run")
CFG_PATH = os.path.join(ROOT, "best_config.json")
SIM_DIST = os.path.join(BEST_RUN, "hier_qcbm_model_dist.npy")
IBM_DIST = os.path.join(BEST_RUN, "ibm_dist.npy")

BG      = "#FFFFFF"
PANEL   = "#F8F8F8"
DARK    = "#111111"
SUBTEXT = "#555555"
BORDER  = "#CCCCCC"
GRIDCOL = "#E0E0E0"

C_BLUE   = "#4B92C8"
C_ORANGE = "#E07B39"
C_RED    = "#C43B2C"
C_GREY   = "#9E9E9E"

STYLE = {
    "figure.facecolor":  BG,      "axes.facecolor":  PANEL,
    "axes.edgecolor":    BORDER,  "axes.labelcolor": DARK,
    "xtick.color":       DARK,    "ytick.color":     DARK,
    "text.color":        DARK,
    "grid.color":        GRIDCOL, "grid.linestyle":  "--",
    "grid.alpha":        1.0,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
}
plt.rcParams.update(STYLE)


# ── preprocessing (identical to roc_curve.py / ibm_score.py) ──────────────────

def run_preprocessing(cfg: dict):
    from src.data.preprocessing import (
        add_categorical_features, apply_log1p,
        select_features, DEFAULT_LOG1P_COLS, Scaler,
    )
    from src.training_setup import train_val_test_split
    from src.discretize import auto_mixed_precision_map, fit_bins, transform_bins, encode_bits

    input_path = cfg.get("input", "datasets/UNSW-NB15_cleaned.csv")
    label_col  = cfg.get("label_col", "label")
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
    edges = fit_bins(X_train, features,
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
        bit_val, bit_test,
        splits.y_val.reset_index(drop=True),
        splits.y_test.reset_index(drop=True),
    )


def raw_scores(bits, dist):
    from src.score_eval import score_samples
    return score_samples(bits, dist)


def calibrate_lr_isotonic(scores_val, y_val, scores_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(scores_val.reshape(-1, 1), y_val.to_numpy())
    p_val  = lr.predict_proba(scores_val.reshape(-1, 1))[:, 1]
    p_test = lr.predict_proba(scores_test.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val.to_numpy())
    return iso.predict(p_test)


def best_f1_point(precision, recall):
    """Return (recall, precision) at the threshold with the highest F1."""
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    idx = np.argmax(f1)
    return float(recall[idx]), float(precision[idx])


def main():
    with open(CFG_PATH) as f:
        cfg = json.load(f)

    sim_dist = np.load(SIM_DIST)
    if sim_dist.ndim == 2:
        sim_dist = sim_dist.mean(axis=0)
    ibm_dist = np.load(IBM_DIST)
    if ibm_dist.ndim == 2:
        ibm_dist = ibm_dist.mean(axis=0)

    print("Preprocessing ...")
    bit_val, bit_test, y_val, y_test = run_preprocessing(cfg)

    print("Scoring ...")
    sim_raw = raw_scores(bit_test, sim_dist)
    ibm_raw = raw_scores(bit_test, ibm_dist)

    print("Calibrating (LR + Isotonic) ...")
    val_raw = raw_scores(bit_val, sim_dist)
    sim_cal = calibrate_lr_isotonic(val_raw, y_val, sim_raw)

    y_arr      = y_test.to_numpy()
    prevalence = y_arr.mean()

    from sklearn.metrics import precision_recall_curve, average_precision_score

    prec_sim, rec_sim, _ = precision_recall_curve(y_arr, sim_raw)
    prec_cal, rec_cal, _ = precision_recall_curve(y_arr, sim_cal)
    prec_ibm, rec_ibm, _ = precision_recall_curve(y_arr, ibm_raw)

    ap_sim = average_precision_score(y_arr, sim_raw)
    ap_cal = average_precision_score(y_arr, sim_cal)
    ap_ibm = average_precision_score(y_arr, ibm_raw)

    print(f"  Simulator (raw)       AP = {ap_sim:.4f}")
    print(f"  Simulator (LR+Iso)    AP = {ap_cal:.4f}")
    print(f"  IBM Hardware          AP = {ap_ibm:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    ax.plot(rec_sim, prec_sim, color=C_BLUE,   lw=2.0,
            label=f"QCBM Simulator  AP={ap_sim:.4f}")
    ax.plot(rec_cal, prec_cal, color=C_ORANGE, lw=2.0,
            label=f"QCBM + LR+Isotonic  AP={ap_cal:.4f}")
    ax.plot(rec_ibm, prec_ibm, color=C_RED,    lw=2.0,
            label=f"IBM Hardware  AP={ap_ibm:.4f}")

    # no-skill baseline
    ax.axhline(prevalence, color=C_GREY, lw=1.2, linestyle="--",
               label=f"No-skill (prevalence={prevalence:.3f})")

    # best-F1 operating-point stars
    for rec_c, prec_c, color in [
        (rec_sim, prec_sim, C_BLUE),
        (rec_cal, prec_cal, C_ORANGE),
        (rec_ibm, prec_ibm, C_RED),
    ]:
        ox, oy = best_f1_point(prec_c, rec_c)
        ax.plot(ox, oy, marker="*", markersize=14, color=color,
                markeredgecolor="black", markeredgewidth=0.6, zorder=5)

    ax.set_xlabel("Recall",    fontsize=12, color=DARK)
    ax.set_ylabel("Precision", fontsize=12, color=DARK)
    ax.set_title("Precision-Recall Curve – QCBM Anomaly Detection (test set)",
                 fontsize=12, fontweight="bold", color=DARK, pad=10)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

    ax.legend(
        loc="upper right", fontsize=10,
        framealpha=0.9, facecolor=BG,
        edgecolor=BORDER, labelcolor=DARK,
    )

    plt.tight_layout()
    out = os.path.join(HERE, "13_pr_curve.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
