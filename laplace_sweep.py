"""
Laplace smoothing alpha ablation study.

Sweeps alpha in {0.5, 1.0, 2.0, 3.0} and records:
  - ROC-AUC, PR-AUC, F1, Precision, Recall, FAR, MCC
  - FAR floor (from bitstring coverage analysis)

Each run uses best_config.json but overrides laplace_alpha.
Results saved to artifacts/laplace_sweep.json and printed as a table.

Usage:
    python -u laplace_sweep.py
"""
from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.preprocessing import (
    DEFAULT_LOG1P_COLS, Scaler, add_categorical_features, apply_log1p, select_features,
)
from src.training_setup import train_val_test_split, filter_normal
from src.discretize import auto_mixed_precision_map, encode_bits, fit_bins, transform_bins
from src.qcbm_train import QCBMConfig, train_qcbm
from src.score_eval import evaluate, score_samples
from src.bitstring_coverage import compute_bitstring_coverage
from STAGES.stage1 import find_best_threshold, find_youden_threshold, zscore

ALPHAS = [0.5, 1.0, 2.0, 3.0]


def load_config(path="best_config.json"):
    with open(path) as f:
        cfg = json.load(f)
    class Args: pass
    args = Args()
    defaults = dict(
        input="datasets/UNSW-NB15_core_features.csv",
        label_input="datasets/UNSW-NB15_cleaned.csv",
        label_col="label", attack_col="attack_cat",
        features="sbytes,Sload,dbytes,Dload,Dpkts,is_not_tcp,is_int_state,is_con_state",
        log1p=True, scaler="standard", n_bins=4, bits_per_feature=2,
        bin_strategy="quantile", encoding="binary", test_frac=0.2, val_frac=0.1,
        seed=42, auto_mixed_precision=True, qcbm_layers=3, qcbm_iter=1500,
        qcbm_ensemble=3, warmstart_layers=True, optimizer="adam", adam_lr=0.003,
        adam_beta1=0.9, adam_beta2=0.999, lambda_contrast=0.5, contrast_margin=10.0,
        laplace_alpha=1.0, spsa_a=0.2, spsa_c=0.1, tail_percentile=0.99,
    )
    for k, v in defaults.items():
        setattr(args, k, v)
    for k, v in cfg.items():
        if not k.startswith("_"):
            setattr(args, k.replace("-","_"), v)
    return args


def load_data(args):
    df = pd.read_csv(args.input, low_memory=False)
    need = [args.label_col, args.attack_col]
    for col in ["proto", "state", "service"]:
        if col not in df.columns:
            need.append(col)
    missing = [c for c in need if c not in df.columns]
    if missing:
        ldf = pd.read_csv(args.label_input, usecols=[c for c in missing
                          if c in pd.read_csv(args.label_input, nrows=0).columns],
                          low_memory=False)
        for c in ldf.columns:
            df[c] = ldf[c]
    df = add_categorical_features(df)
    features = [c.strip() for c in args.features.split(",") if c.strip()]
    X = select_features(df, features)
    y = df[args.label_col]
    splits = train_val_test_split(X, y, test_frac=args.test_frac,
                                  val_frac=args.val_frac, seed=args.seed, stratify=True)
    for attr in ["X_train", "X_val", "X_test"]:
        setattr(splits, attr, apply_log1p(getattr(splits, attr), DEFAULT_LOG1P_COLS))
        setattr(splits, attr, getattr(splits, attr)[features])
    scaler = Scaler(mode=args.scaler).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
    X_val   = scaler.transform(splits.X_val,   features)
    X_test  = scaler.transform(splits.X_test,  features)
    return X_train, X_val, X_test, splits.y_train, splits.y_val, splits.y_test, features


def get_bitstrings(X_train, X_val, X_test, y_train, features, args):
    bits_map, bins_map = None, None
    if getattr(args, "auto_mixed_precision", False):
        bits_map, bins_map = auto_mixed_precision_map(
            X_train, features, continuous_bits=args.bits_per_feature,
            continuous_bins=args.n_bins,
        )
    edges = fit_bins(X_train, features, n_bins=args.n_bins,
                     strategy=args.bin_strategy, n_bins_map=bins_map)
    kw = dict(bits_per_feature=args.bits_per_feature, encoding=args.encoding,
              n_bins=args.n_bins, bits_per_feature_map=bits_map)
    bit_train = encode_bits(transform_bins(X_train, edges), **kw)
    bit_val   = encode_bits(transform_bins(X_val,   edges), **kw)
    bit_test  = encode_bits(transform_bins(X_test,  edges), **kw)
    y_train_r = y_train.reset_index(drop=True)
    normal_df, _ = filter_normal(pd.DataFrame(bit_train), y_train_r)
    bit_train_normal = normal_df.to_numpy()
    anomaly_mask = (y_train_r.to_numpy() == 1)
    bit_train_anomaly = bit_train[anomaly_mask]
    return bit_train, bit_val, bit_test, bit_train_normal, bit_train_anomaly


def run_one_alpha(alpha, args, bit_train_normal, bit_train_anomaly,
                  bit_val, bit_test, y_val, y_test, n_qubits):
    print(f"\n{'='*60}")
    print(f"  alpha = {alpha}")
    print(f"{'='*60}")

    ensemble = max(1, int(args.qcbm_ensemble))
    model_scores_val, model_scores_test, gaps = [], [], []

    for i in range(ensemble):
        seed = args.seed + i * 97
        config = QCBMConfig(
            n_qubits=n_qubits,
            n_layers=args.qcbm_layers,
            max_iter=args.qcbm_iter,
            seed=seed,
            spsa_a=args.spsa_a,
            spsa_c=args.spsa_c,
            lambda_contrast=args.lambda_contrast,
            contrast_margin=args.contrast_margin,
            laplace_alpha=alpha,
            warmstart_layers=getattr(args, "warmstart_layers", False),
            optimizer=getattr(args, "optimizer", "adam"),
            adam_lr=getattr(args, "adam_lr", 0.003),
            adam_beta1=getattr(args, "adam_beta1", 0.9),
            adam_beta2=getattr(args, "adam_beta2", 0.999),
        )
        out = train_qcbm(bit_train_normal, config, anomaly_bitstrings=bit_train_anomaly)
        model_scores_val.append(score_samples(bit_val, out["model_dist"],
                                              normal_bitstrings=bit_train_normal))
        model_scores_test.append(score_samples(bit_test, out["model_dist"],
                                               normal_bitstrings=bit_train_normal))
        nkl = float(out["loss"])
        akl = out.get("anomaly_kl")
        gap = float(akl - nkl) if akl is not None else 1.0
        gaps.append(max(0.0, gap))
        akl_str = f"{akl:.4f}" if akl is not None else "N/A"
        print(f"  Model {i+1}: normal_kl={nkl:.4f}  anomaly_kl={akl_str}  gap={gap:.4f}")

    total_gap = sum(gaps)
    weights = [g / total_gap for g in gaps] if total_gap > 1e-8 else [1.0/ensemble]*ensemble
    val_scores  = sum(w * s for w, s in zip(weights, model_scores_val))
    test_scores = sum(w * s for w, s in zip(weights, model_scores_test))

    # Z-score normalise using val normal scores
    y_val_np  = y_val.reset_index(drop=True).to_numpy()
    y_test_np = y_test.to_numpy()
    normal_mask_val = (y_val_np == 0)
    mu    = float(np.mean(val_scores[normal_mask_val]))
    sigma = float(np.std(val_scores[normal_mask_val]))
    val_z  = zscore(val_scores,  mu, sigma)
    test_z = zscore(test_scores, mu, sigma)

    m_base = evaluate(y_test_np, test_z)
    if m_base["roc_auc"] < 0.5:
        val_z, test_z = -val_z, -test_z

    f1_t, _ = find_best_threshold(y_val_np, val_z)
    youden_t, _ = find_youden_threshold(y_val_np, val_z)
    m_f1     = evaluate(y_test_np, test_z, threshold=f1_t)
    m_youden = evaluate(y_test_np, test_z, threshold=youden_t)

    # FAR floor from coverage analysis
    cov = compute_bitstring_coverage(bit_train_normal, bit_test, y_test_np, n_qubits)
    far_floor = cov["far_floor_empirical"]

    result = {
        "alpha": alpha,
        "roc_auc":   m_base["roc_auc"],
        "pr_auc":    m_base["pr_auc"],
        # F1-threshold operating point
        "f1":        m_f1.get("f1", 0),
        "precision": m_f1.get("precision", 0),
        "recall":    m_f1.get("recall_dr", 0),
        "far_f1":    m_f1.get("far", 0),
        "mcc_f1":    m_f1.get("mcc", 0),
        # Youden operating point
        "recall_youden": m_youden.get("recall_dr", 0),
        "far_youden":    m_youden.get("far", 0),
        "mcc_youden":    m_youden.get("mcc", 0),
        # FAR floor
        "far_floor": far_floor,
        "far_floor_pct": far_floor * 100,
        "train_coverage_pct": cov["train_coverage_pct"],
    }

    print(f"\n  alpha={alpha}  ROC-AUC={result['roc_auc']:.4f}  "
          f"Prec={result['precision']:.4f}  Recall={result['recall']:.4f}  "
          f"FAR={result['far_f1']*100:.2f}%  FAR-floor={result['far_floor_pct']:.2f}%  "
          f"MCC={result['mcc_f1']:.4f}")
    return result


def main():
    print("Laplace Alpha Ablation Study")
    print(f"Alphas: {ALPHAS}")

    args = load_config()
    print("\nLoading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data(args)
    print("Building bitstrings...")
    bit_train, bit_val, bit_test, bit_train_normal, bit_train_anomaly = get_bitstrings(
        X_train, X_val, X_test, y_train, features, args
    )
    n_qubits = bit_train_normal.shape[1]
    print(f"  {n_qubits} qubits | train_normal={len(bit_train_normal):,}")

    results = []
    for alpha in ALPHAS:
        r = run_one_alpha(alpha, args, bit_train_normal, bit_train_anomaly,
                          bit_val, bit_test, y_val, y_test, n_qubits)
        results.append(r)

    # Print summary table
    print("\n\n" + "="*90)
    print("  LAPLACE ALPHA ABLATION RESULTS")
    print("="*90)
    print(f"  {'Alpha':>7} {'ROC-AUC':>9} {'PR-AUC':>8} {'Prec':>8} {'Recall':>8} "
          f"{'FAR(F1t)':>10} {'FAR-floor':>11} {'MCC':>8}")
    print("  " + "-"*80)
    for r in results:
        marker = " <-- best" if r["roc_auc"] == max(x["roc_auc"] for x in results) else ""
        print(f"  {r['alpha']:>7.1f} {r['roc_auc']:>9.4f} {r['pr_auc']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['far_f1']*100:>8.2f}% {r['far_floor_pct']:>9.2f}%  "
              f"{r['mcc_f1']:>8.4f}{marker}")
    print("="*90)

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/laplace_sweep.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: artifacts/laplace_sweep.json")


if __name__ == "__main__":
    main()
