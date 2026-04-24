"""
ibm_score.py  —  Compute ROC-AUC and full metrics using the IBM hardware distribution.

Reruns the deterministic preprocessing pipeline (same config + seed = identical
bin edges and test bitstrings), then scores the test set using the distribution
fetched from real IBM quantum hardware instead of the Aer simulator.

No IBM jobs are submitted — uses the ibm_dist.npy already saved by ibm_inference.py.

Usage
-----
python ibm_score.py
python ibm_score.py --config best_config.json --ibm-dist artifacts/best_run/ibm_dist.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score test set with IBM hardware distribution and compare vs simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="best_config.json",
                   help="JSON config used to train the best run.")
    p.add_argument("--ibm-dist", default="artifacts/best_run/ibm_dist.npy",
                   help="IBM hardware distribution saved by ibm_inference.py.")
    p.add_argument("--sim-dist", default="artifacts/best_run/hier_qcbm_model_dist.npy",
                   help="Aer simulator distribution for side-by-side comparison.")
    p.add_argument("--sim-metrics", default="artifacts/best_run/hier_stage1_metrics.json",
                   help="Saved simulator metrics for comparison.")
    p.add_argument("--output", default="artifacts/best_run/ibm_score_metrics.json",
                   help="Where to save the IBM scoring results.")
    return p


# ---------------------------------------------------------------------------
# Preprocessing — mirrors main.py exactly so bin edges are identical
# ---------------------------------------------------------------------------

def run_preprocessing(cfg: dict):
    """Return (bit_val, bit_test, y_val, y_test, features) using the saved config."""
    from src.data.preprocessing import (
        add_categorical_features, apply_log1p, select_features,
        DEFAULT_LOG1P_COLS, Scaler,
    )
    from src.training_setup import train_val_test_split
    from src.discretize import auto_mixed_precision_map, fit_bins, transform_bins, encode_bits
    from src.training_setup import filter_normal

    input_path = cfg.get("input", "datasets/UNSW-NB15_cleaned.csv")
    label_col  = cfg.get("label_col",  "label")
    attack_col = cfg.get("attack_col", "attack_cat")
    seed       = cfg.get("seed", 42)

    print(f"  Loading dataset: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    # Pull label/attack columns from the same file if present
    for col in [label_col, attack_col, "proto", "state", "service"]:
        if col not in df.columns:
            label_file = cfg.get("label_input", "datasets/UNSW-NB15_cleaned.csv")
            if label_file != input_path:
                extra = pd.read_csv(label_file, usecols=[col], low_memory=False)
                df[col] = extra[col].values

    print("  Engineering categorical features...")
    df = add_categorical_features(df)

    features = [f.strip() for f in cfg["features"].split(",") if f.strip()]
    print(f"  Features ({len(features)}): {', '.join(features)}")

    X = select_features(df, features)
    y = df[label_col]

    print("  Splitting train / val / test...")
    splits = train_val_test_split(
        X, y,
        test_frac=cfg.get("test_frac", 0.2),
        val_frac=cfg.get("val_frac", 0.1),
        seed=seed,
        stratify=True,
    )

    if cfg.get("log1p", True):
        print("  Applying log1p...")
        splits.X_train = apply_log1p(splits.X_train, DEFAULT_LOG1P_COLS)
        splits.X_val   = apply_log1p(splits.X_val,   DEFAULT_LOG1P_COLS)
        splits.X_test  = apply_log1p(splits.X_test,  DEFAULT_LOG1P_COLS)

    print(f"  Scaling ({cfg.get('scaler', 'standard')})...")
    scaler = Scaler(mode=cfg.get("scaler", "standard")).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
    X_val   = scaler.transform(splits.X_val,   features)
    X_test  = scaler.transform(splits.X_test,  features)

    # Auto mixed precision — must match exactly what stage1.py computed
    use_amp = cfg.get("auto_mixed_precision", False)
    if use_amp:
        bits_map, bins_map = auto_mixed_precision_map(
            X_train, features,
            continuous_bits=cfg.get("bits_per_feature", 2),
            continuous_bins=cfg.get("n_bins", 4),
        )
    else:
        bits_map, bins_map = None, None

    # Anomaly-aware binning needs anomaly training rows
    y_train_reset = splits.y_train.reset_index(drop=True)
    anomaly_mask  = (y_train_reset.to_numpy() == 1)
    X_train_anom  = X_train.iloc[anomaly_mask] if anomaly_mask.any() else None

    print(f"  Fitting bins (strategy={cfg.get('bin_strategy', 'quantile')})...")
    edges  = fit_bins(X_train, features,
                      n_bins=cfg.get("n_bins", 4),
                      strategy=cfg.get("bin_strategy", "quantile"),
                      n_bins_map=bins_map,
                      df_anomaly=X_train_anom)
    btrain = transform_bins(X_train, edges)
    bval   = transform_bins(X_val,   edges)
    btest  = transform_bins(X_test,  edges)

    enc_kwargs = dict(
        bits_per_feature=cfg.get("bits_per_feature", 2),
        encoding=cfg.get("encoding", "binary"),
        n_bins=cfg.get("n_bins", 4),
        bits_per_feature_map=bits_map,
    )
    bit_train = encode_bits(btrain, **enc_kwargs)
    bit_val   = encode_bits(bval,   **enc_kwargs)
    bit_test  = encode_bits(btest,  **enc_kwargs)

    # Normal training bitstrings (needed for Hamming smooth, kept for completeness)
    btrain_df, _ = filter_normal(pd.DataFrame(bit_train), y_train_reset)
    bit_train_normal = btrain_df.to_numpy()

    return (
        bit_val, bit_test,
        splits.y_val.reset_index(drop=True),
        splits.y_test.reset_index(drop=True),
        bit_train_normal,
        features,
    )


# ---------------------------------------------------------------------------
# Scoring & metrics
# ---------------------------------------------------------------------------

def score_and_eval(bit_data, y_true, dist: np.ndarray, label: str) -> dict:
    from src.score_eval import score_samples, evaluate

    scores = score_samples(bit_data, dist)
    metrics = evaluate(y_true.to_numpy(), scores)

    # Find F1-optimal threshold on the same split (mirrors stage1 behaviour)
    thresholds = np.unique(scores)
    if len(thresholds) > 200:
        thresholds = np.quantile(scores, np.linspace(0, 1, 201))
    best_t, best_f1 = thresholds[0], -1.0
    y_arr = y_true.to_numpy()
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_arr == 1))
        fp = np.sum((y_pred == 1) & (y_arr == 0))
        fn = np.sum((y_pred == 0) & (y_arr == 1))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t

    full = evaluate(y_arr, scores, threshold=best_t)

    print(f"\n  [{label}]")
    print(f"    ROC-AUC : {full['roc_auc']:.4f}")
    print(f"    PR-AUC  : {full['pr_auc']:.4f}")
    print(f"    F1      : {full.get('f1', float('nan')):.4f}  (threshold={best_t:.4f})")
    print(f"    Precision: {full.get('precision', float('nan')):.4f}")
    print(f"    Recall  : {full.get('recall_dr', float('nan')):.4f}")
    print(f"    FAR     : {full.get('far', float('nan')):.4f}")
    print(f"    MCC     : {full.get('mcc', float('nan')):.4f}")

    return full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Load IBM distribution
    ibm_dist_path = Path(args.ibm_dist)
    if not ibm_dist_path.exists():
        sys.exit(
            f"ERROR: {ibm_dist_path} not found.\n"
            "Run ibm_inference.py first to fetch the IBM hardware distribution."
        )
    ibm_dist = np.load(str(ibm_dist_path))
    print(f"Loaded IBM distribution  : {ibm_dist_path}  ({len(ibm_dist):,} states, sum={ibm_dist.sum():.4f})")

    # Load simulator distribution for comparison
    sim_dist = None
    sim_dist_path = Path(args.sim_dist)
    if sim_dist_path.exists():
        raw = np.load(str(sim_dist_path))
        sim_dist = raw.mean(axis=0) if raw.ndim == 2 else raw
        print(f"Loaded simulator baseline: {sim_dist_path}")

    # Load saved simulator metrics
    sim_metrics = {}
    sim_metrics_path = Path(args.sim_metrics)
    if sim_metrics_path.exists():
        with sim_metrics_path.open() as f:
            sim_metrics = json.load(f)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"ERROR: {args.config} not found.")
    with config_path.open() as f:
        cfg = json.load(f)

    print(f"\nReproducing preprocessing from: {args.config}")
    bit_val, bit_test, y_val, y_test, bit_train_normal, features = run_preprocessing(cfg)
    print(f"  Test set : {len(y_test):,} samples  "
          f"({int(y_test.sum()):,} attacks, {int((y_test==0).sum()):,} normal)")

    print(f"\n{'=' * 60}")
    print("  SCORING RESULTS")
    print(f"{'=' * 60}")

    # Score with IBM hardware distribution
    ibm_metrics = score_and_eval(bit_test, y_test, ibm_dist, "IBM hardware")

    # Score with simulator distribution for direct comparison
    if sim_dist is not None:
        sim_live = score_and_eval(bit_test, y_test, sim_dist, "Aer simulator (live recompute)")

    # Side-by-side summary
    print(f"\n{'=' * 60}")
    print("  COMPARISON: IBM hardware vs Aer simulator")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<12} {'IBM hardware':>14} {'Sim (saved)':>14} {'Delta':>10}")
    print(f"  {'-'*52}")

    saved_roc = sim_metrics.get("isotonic_calibration_metrics", {}).get("roc_auc",
                sim_metrics.get("roc_auc", None))
    saved_f1  = sim_metrics.get("isotonic_calibration_metrics", {}).get("f1",
                sim_metrics.get("f1", None))
    saved_far = sim_metrics.get("isotonic_calibration_metrics", {}).get("far",
                sim_metrics.get("far", None))
    saved_rec = sim_metrics.get("isotonic_calibration_metrics", {}).get("recall_dr",
                sim_metrics.get("recall_dr", None))

    def row(name, ibm_val, sim_val, fmt=".4f"):
        ibm_s = f"{ibm_val:{fmt}}" if ibm_val is not None else "  n/a"
        sim_s = f"{sim_val:{fmt}}" if sim_val is not None else "  n/a"
        delta = f"{ibm_val - sim_val:+.4f}" if (ibm_val is not None and sim_val is not None) else ""
        print(f"  {name:<12} {ibm_s:>14} {sim_s:>14} {delta:>10}")

    row("ROC-AUC",  ibm_metrics.get("roc_auc"),  saved_roc)
    row("PR-AUC",   ibm_metrics.get("pr_auc"),    sim_metrics.get("pr_auc"))
    row("F1",       ibm_metrics.get("f1"),        saved_f1)
    row("Recall",   ibm_metrics.get("recall_dr"), saved_rec)
    row("FAR",      ibm_metrics.get("far"),       saved_far)
    row("MCC",      ibm_metrics.get("mcc"),       sim_metrics.get("mcc"))
    print(f"{'=' * 60}")

    print(
        "\n  Interpretation: delta shows the hardware noise penalty.\n"
        "  TVD=0.60 between distributions does not map linearly to ROC-AUC drop\n"
        "  because anomaly scoring uses relative log-probabilities, not raw probs."
    )

    # Save results
    output = {
        "ibm_dist_path": str(ibm_dist_path),
        "config": args.config,
        "ibm_metrics": ibm_metrics,
        "simulator_metrics_saved": {
            "roc_auc": saved_roc,
            "f1": saved_f1,
            "far": saved_far,
            "recall_dr": saved_rec,
        },
        "delta": {
            "roc_auc": round(ibm_metrics["roc_auc"] - saved_roc, 4) if saved_roc else None,
            "f1": round(ibm_metrics.get("f1", 0) - saved_f1, 4) if saved_f1 else None,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
