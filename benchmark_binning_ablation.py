"""
benchmark_binning_ablation.py  --  Binning strategy ablation for classical baselines

Key research question:
    Is the QCBM performance improvement due to anomaly-aware binning alone,
    or does the quantum generative model contribute beyond the binning strategy?

Protocol:
    1. For each classical baseline (IsoForest, Autoencoder, KDE, RBM):
       a. Train + evaluate with QUANTILE binning
       b. Train + evaluate with ANOMALY-AWARE binning (same bins as QCBM)
    2. Load QCBM results for both binning strategies
    3. Print full comparison table + save to artifacts/

If IsoForest + anomaly-aware binning ~= QCBM + anomaly-aware binning:
    -> The binning drives the gain; quantum contribution is marginal
If QCBM >> IsoForest under SAME anomaly-aware binning:
    -> The quantum generative model adds real discriminative value

Usage:
    python -u benchmark_binning_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    DEFAULT_LOG1P_COLS, Scaler, add_categorical_features, apply_log1p, select_features,
)
from src.training_setup import train_val_test_split, filter_normal
from src.discretize import auto_mixed_precision_map, encode_bits, fit_bins, transform_bins
from src.classical_baseline import (
    train_kde, score_kde,
    train_rbm, score_rbm,
    train_isolation_forest, score_isolation_forest,
    train_autoencoder, score_autoencoder,
)
from src.score_eval import evaluate
from STAGES.stage1 import find_best_threshold


FEATURES = "sbytes,Sload,dbytes,Dload,Dpkts,is_not_tcp,is_int_state,is_con_state"
N_BINS = 4
BITS_PER_FEATURE = 2
ENCODING = "binary"
LOG1P = True
SCALER = "standard"
TEST_FRAC = 0.2
VAL_FRAC = 0.1
SEED = 42
AMP = True


def load_data():
    df = pd.read_csv("datasets/UNSW-NB15_core_features.csv", low_memory=False)
    need = ["label", "attack_cat"]
    for col in ["proto", "state", "service"]:
        if col not in df.columns:
            need.append(col)
    missing = [c for c in need if c not in df.columns]
    if missing:
        ldf = pd.read_csv("datasets/UNSW-NB15_cleaned.csv",
                          usecols=[c for c in missing
                                   if c in pd.read_csv("datasets/UNSW-NB15_cleaned.csv",
                                                       nrows=0).columns.tolist()],
                          low_memory=False)
        for c in ldf.columns:
            df[c] = ldf[c]
    df = add_categorical_features(df)
    features = [c.strip() for c in FEATURES.split(",") if c.strip()]
    X = select_features(df, features)
    y = df["label"]
    splits = train_val_test_split(X, y, test_frac=TEST_FRAC, val_frac=VAL_FRAC,
                                  seed=SEED, stratify=True)
    if LOG1P:
        splits.X_train = apply_log1p(splits.X_train, DEFAULT_LOG1P_COLS)
        splits.X_val   = apply_log1p(splits.X_val,   DEFAULT_LOG1P_COLS)
        splits.X_test  = apply_log1p(splits.X_test,  DEFAULT_LOG1P_COLS)
    splits.X_train = splits.X_train[features]
    splits.X_val   = splits.X_val[features]
    splits.X_test  = splits.X_test[features]
    scaler = Scaler(mode=SCALER).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
    X_val   = scaler.transform(splits.X_val,   features)
    X_test  = scaler.transform(splits.X_test,  features)
    return X_train, X_val, X_test, splits.y_train, splits.y_val, splits.y_test, features


def build_bitstrings(X_train, X_val, X_test, y_train, features, strategy: str):
    """Build bitstring encodings using the specified binning strategy."""
    bits_map, bins_map = None, None
    if AMP:
        bits_map, bins_map = auto_mixed_precision_map(
            X_train, features,
            continuous_bits=BITS_PER_FEATURE,
            continuous_bins=N_BINS,
        )

    # Build anomaly data for anomaly-aware binning
    df_anomaly = None
    if strategy == "anomaly_aware":
        y_tr = y_train.reset_index(drop=True).to_numpy()
        anom_mask = (y_tr == 1)
        if anom_mask.any():
            df_anomaly = X_train.iloc[anom_mask]
            print(f"    Anomaly-aware bins: {anom_mask.sum():,} anomaly training samples used")

    edges = fit_bins(X_train, features, n_bins=N_BINS, strategy=strategy,
                     n_bins_map=bins_map, df_anomaly=df_anomaly)

    kw = dict(bits_per_feature=BITS_PER_FEATURE, encoding=ENCODING,
              n_bins=N_BINS, bits_per_feature_map=bits_map)
    bit_train = encode_bits(transform_bins(X_train, edges), **kw)
    bit_val   = encode_bits(transform_bins(X_val,   edges), **kw)
    bit_test  = encode_bits(transform_bins(X_test,  edges), **kw)

    y_tr = y_train.reset_index(drop=True)
    normal_df, _ = filter_normal(pd.DataFrame(bit_train), y_tr)
    bit_train_normal = normal_df.to_numpy()
    return bit_train, bit_val, bit_test, bit_train_normal


def run_baselines(bit_train_normal, bit_val, bit_test, y_val, y_test,
                  strategy_label: str, max_kde_samples: int = 50_000):
    N = len(bit_train_normal)
    rng = np.random.default_rng(42)
    if N > max_kde_samples:
        idx = rng.choice(N, size=max_kde_samples, replace=False)
        X_fit = bit_train_normal[idx]
    else:
        X_fit = bit_train_normal

    results = {}

    # IsoForest (full data)
    print(f"  [{strategy_label}] IsoForest (full {N:,} samples)...")
    iso = train_isolation_forest(bit_train_normal, max_samples=256)
    val_s  = score_isolation_forest(bit_val,  iso)
    test_s = score_isolation_forest(bit_test, iso)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["IsoForest"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["IsoForest"]["train_n"] = N
    print(f"    IsoForest  ROC-AUC={results['IsoForest']['roc_auc']:.4f}  "
          f"F1={results['IsoForest']['f1']:.4f}  "
          f"Prec={results['IsoForest']['precision']:.4f}  "
          f"FAR={results['IsoForest']['far']:.4f}")

    # Autoencoder (full data)
    print(f"  [{strategy_label}] Autoencoder 13->6->13 (full {N:,} samples)...")
    ae = train_autoencoder(bit_train_normal, hidden_dim=6, max_iter=50, batch_size=1024)
    val_s  = score_autoencoder(bit_val,  ae)
    test_s = score_autoencoder(bit_test, ae)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["Autoencoder"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["Autoencoder"]["train_n"] = N
    print(f"    Autoencoder  ROC-AUC={results['Autoencoder']['roc_auc']:.4f}  "
          f"F1={results['Autoencoder']['f1']:.4f}  "
          f"Prec={results['Autoencoder']['precision']:.4f}  "
          f"FAR={results['Autoencoder']['far']:.4f}")

    # KDE (subsampled)
    print(f"  [{strategy_label}] KDE ({max_kde_samples:,} subsample)...")
    best_bw, best_roc = 0.5, 0.0
    for bw in [0.1, 0.3, 0.5, 1.0]:
        kde = train_kde(X_fit, bandwidth=bw)
        m = evaluate(y_val.to_numpy(), score_kde(bit_val, kde))
        if m["roc_auc"] > best_roc:
            best_roc, best_bw = m["roc_auc"], bw
    kde = train_kde(X_fit, bandwidth=best_bw)
    val_s  = score_kde(bit_val,  kde)
    test_s = score_kde(bit_test, kde)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["KDE"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["KDE"]["train_n"] = max_kde_samples
    print(f"    KDE (bw={best_bw})  ROC-AUC={results['KDE']['roc_auc']:.4f}  "
          f"F1={results['KDE']['f1']:.4f}  "
          f"Prec={results['KDE']['precision']:.4f}  "
          f"FAR={results['KDE']['far']:.4f}")

    # RBM (subsampled)
    print(f"  [{strategy_label}] RBM n_components=5 ({max_kde_samples:,} subsample)...")
    rbm = train_rbm(X_fit, n_components=5, n_iter=200)
    val_s  = score_rbm(bit_val,  rbm)
    test_s = score_rbm(bit_test, rbm)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["RBM"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["RBM"]["train_n"] = max_kde_samples
    print(f"    RBM  ROC-AUC={results['RBM']['roc_auc']:.4f}  "
          f"F1={results['RBM']['f1']:.4f}  "
          f"Prec={results['RBM']['precision']:.4f}  "
          f"FAR={results['RBM']['far']:.4f}")

    return results


def print_ablation_table(results_quantile: dict, results_anomaly: dict,
                         qcbm_quantile: dict, qcbm_anomaly: dict):
    cols = ["ROC-AUC", "PR-AUC", "F1", "Prec", "Recall", "FAR", "MCC"]

    def row(name, m):
        return (f"  {name:<28} "
                f"{m.get('roc_auc',0):>9.4f} "
                f"{m.get('pr_auc',0):>8.4f} "
                f"{m.get('f1',0):>8.4f} "
                f"{m.get('precision',0):>8.4f} "
                f"{m.get('recall_dr',0):>8.4f} "
                f"{m.get('far',0):>8.4f} "
                f"{m.get('mcc',0):>8.4f}")

    header = (f"  {'Model':<28} {'ROC-AUC':>9} {'PR-AUC':>8} {'F1':>8} "
              f"{'Prec':>8} {'Recall':>8} {'FAR':>8} {'MCC':>8}")
    sep = "  " + "=" * 96

    print("\n" + sep)
    print(f"  BINNING ABLATION: Does anomaly-aware binning explain QCBM gains?")
    print(sep)
    print(f"\n  QUANTILE BINNING (standard)")
    print("  " + "-" * 96)
    print(header)
    print("  " + "-" * 96)
    print(row("QCBM (quantum) [quantile]", qcbm_quantile) + "  <-- quantum")
    print("  " + "-" * 96)
    for name, m in results_quantile.items():
        delta = m.get("roc_auc", 0) - qcbm_quantile.get("roc_auc", 0)
        print(row(f"{name} [quantile]", m) + f"  ({delta:+.4f})")

    print(f"\n  ANOMALY-AWARE BINNING")
    print("  " + "-" * 96)
    print(header)
    print("  " + "-" * 96)
    print(row("QCBM (quantum) [anom-bins]", qcbm_anomaly) + "  <-- quantum")
    print("  " + "-" * 96)
    for name, m in results_anomaly.items():
        delta = m.get("roc_auc", 0) - qcbm_anomaly.get("roc_auc", 0)
        print(row(f"{name} [anom-bins]", m) + f"  ({delta:+.4f})")

    print(f"\n  BINNING DELTA (anomaly-aware minus quantile per model)")
    print("  " + "-" * 96)
    all_names = ["QCBM (quantum)"] + list(results_quantile.keys())
    all_q = [qcbm_quantile] + list(results_quantile.values())
    all_a = [qcbm_anomaly] + list(results_anomaly.values())
    for name, mq, ma in zip(all_names, all_q, all_a):
        droc = ma.get("roc_auc", 0)  - mq.get("roc_auc", 0)
        dpr  = ma.get("pr_auc",  0)  - mq.get("pr_auc",  0)
        df1  = ma.get("f1",      0)  - mq.get("f1",      0)
        dpre = ma.get("precision",0) - mq.get("precision",0)
        dfar = ma.get("far",     0)  - mq.get("far",      0)
        dmcc = ma.get("mcc",     0)  - mq.get("mcc",      0)
        print(f"  {name:<28} "
              f"{'dROC':>5}={droc:+.4f}  "
              f"{'dPR':>4}={dpr:+.4f}  "
              f"{'dF1':>4}={df1:+.4f}  "
              f"{'dPrec':>6}={dpre:+.4f}  "
              f"{'dFAR':>5}={dfar:+.4f}  "
              f"{'dMCC':>5}={dmcc:+.4f}")

    print("\n" + sep)

    # Verdict
    qcbm_anom_roc = qcbm_anomaly.get("roc_auc", 0)
    best_classical_anom_roc = max(m.get("roc_auc", 0) for m in results_anomaly.values())
    gap = qcbm_anom_roc - best_classical_anom_roc
    print(f"\n  VERDICT:")
    print(f"    QCBM (anom-bins) ROC-AUC:              {qcbm_anom_roc:.4f}")
    print(f"    Best classical  (anom-bins) ROC-AUC:   {best_classical_anom_roc:.4f}")
    print(f"    Quantum advantage margin:               {gap:+.4f}")

    qcbm_anom_f1 = qcbm_anomaly.get("f1", 0)
    best_cl_f1   = max(m.get("f1", 0) for m in results_anomaly.values())
    qcbm_anom_prec = qcbm_anomaly.get("precision", 0)
    best_cl_prec   = max(m.get("precision", 0) for m in results_anomaly.values())
    qcbm_anom_far = qcbm_anomaly.get("far", 0)
    best_cl_far   = min(m.get("far", 0) for m in results_anomaly.values())
    print(f"    QCBM F1={qcbm_anom_f1:.4f}  vs best classical F1={best_cl_f1:.4f}  (gap={qcbm_anom_f1-best_cl_f1:+.4f})")
    print(f"    QCBM Prec={qcbm_anom_prec:.4f}  vs best classical Prec={best_cl_prec:.4f}  (gap={qcbm_anom_prec-best_cl_prec:+.4f})")
    print(f"    QCBM FAR={qcbm_anom_far:.4f}   vs best classical FAR={best_cl_far:.4f}   (gap={qcbm_anom_far-best_cl_far:+.4f})")

    if gap > 0.02:
        print("\n  -> Quantum model adds REAL discriminative value beyond binning alone.")
    elif gap > 0.005:
        print("\n  -> Marginal quantum advantage; both binning and model contribute.")
    else:
        print("\n  -> Binning strategy is the primary driver; quantum advantage is minimal.")
    print(sep)


def main():
    print("=" * 60)
    print("BINNING ABLATION EXPERIMENT")
    print("Q: Does anomaly-aware binning alone explain QCBM gains?")
    print("=" * 60)

    print("\nLoading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data()
    print(f"  {len(X_train):,} train  |  {len(X_val):,} val  |  {len(X_test):,} test")

    # --- Quantile binning ---
    print("\n[1/2] Building QUANTILE bitstrings...")
    bit_tr_q, bit_val_q, bit_te_q, bit_normal_q = build_bitstrings(
        X_train, X_val, X_test, y_train, features, strategy="quantile"
    )
    print(f"  Unique normal bitstrings (train): "
          f"{len(np.unique(bit_normal_q, axis=0))}")

    print("\n  Running classical baselines on QUANTILE bins...")
    results_q = run_baselines(bit_normal_q, bit_val_q, bit_te_q, y_val, y_test,
                              strategy_label="quantile")

    # --- Anomaly-aware binning ---
    print("\n[2/2] Building ANOMALY-AWARE bitstrings...")
    bit_tr_a, bit_val_a, bit_te_a, bit_normal_a = build_bitstrings(
        X_train, X_val, X_test, y_train, features, strategy="anomaly_aware"
    )
    print(f"  Unique normal bitstrings (train): "
          f"{len(np.unique(bit_normal_a, axis=0))}")

    print("\n  Running classical baselines on ANOMALY-AWARE bins...")
    results_a = run_baselines(bit_normal_a, bit_val_a, bit_te_a, y_val, y_test,
                              strategy_label="anomaly_aware")

    # Load QCBM results
    print("\nLoading QCBM results...")
    # Quantile-binned QCBM: original best_config (pre-anomaly-aware)
    qcbm_q = {
        "roc_auc":   0.9398,
        "pr_auc":    0.5302,
        "f1":        0.6376,
        "precision": 0.5496,
        "recall_dr": 0.7714,
        "far":       0.0782,
        "mcc":       0.6003,
    }
    # Anomaly-aware QCBM: load from artifacts
    try:
        with open("artifacts/anomaly_binning/hier_stage1_metrics.json") as f:
            ab_metrics = json.load(f)
        # Use isotonic calibration metrics (best threshold)
        iso = ab_metrics.get("isotonic_calibration_metrics", ab_metrics)
        qcbm_a = {
            "roc_auc":   ab_metrics["roc_auc"],
            "pr_auc":    ab_metrics["pr_auc"],
            "f1":        iso.get("f1",        ab_metrics["f1"]),
            "precision": iso.get("precision", ab_metrics["precision"]),
            "recall_dr": iso.get("recall_dr", ab_metrics["recall_dr"]),
            "far":       iso.get("far",       ab_metrics["far"]),
            "mcc":       iso.get("mcc",       ab_metrics["mcc"]),
        }
    except FileNotFoundError:
        print("  Warning: anomaly_binning metrics not found, using known values")
        qcbm_a = {
            "roc_auc":   0.9395,
            "pr_auc":    0.7813,
            "f1":        0.7708,
            "precision": 0.9463,
            "recall_dr": 0.6503,
            "far":       0.0046,
            "mcc":       0.7643,
        }

    print_ablation_table(results_q, results_a, qcbm_q, qcbm_a)

    # Save
    out = {
        "qcbm_quantile":     qcbm_q,
        "qcbm_anomaly_bins": qcbm_a,
        "classical_quantile_bins": {
            k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
            for k, v in results_q.items()
        },
        "classical_anomaly_bins": {
            k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
            for k, v in results_a.items()
        },
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/binning_ablation.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: artifacts/binning_ablation.json")


if __name__ == "__main__":
    main()
