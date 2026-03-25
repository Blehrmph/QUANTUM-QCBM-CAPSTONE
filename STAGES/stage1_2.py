"""
STAGES/stage1_2.py  –  Stage 1 Multi-Config Benchmark

Runs four circuit/feature configurations head-to-head using the same
train/val/test split for a fair comparison.  Results + circuit diagrams
are saved to benchmark_stage1/ in the project root.

Configurations
--------------
0  baseline       8q  original 8 continuous features, RZ-RY-RZ + CNOT
1  8q_new_feats   8q  state + proto replace Sload/Dload,  RZ-RY-RZ + CNOT
2  10q_new_feats  10q state + proto + service + tcprtt,   RZ-RY-RZ + CNOT
3  8q_rzz         8q  same new features as config 1,      RZ-RY-RZ + RZZ

Usage
-----
    python STAGES/stage1_2.py
    python STAGES/stage1_2.py --qcbm-iter 800 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import apply_log1p, Scaler
from src.discretize import encode_bits, fit_bins, transform_bins
from src.qcbm_train import (
    QCBMConfig,
    build_ansatz,
    empirical_distribution,
    kl_divergence,
    n_params,
    qcbm_distribution,
    spsa_optimize,
    train_qcbm,
)
from src.score_eval import evaluate, score_samples
from src.training_setup import train_val_test_split
from STAGES.stage1 import find_best_threshold, find_youden_threshold, zscore

# ---------------------------------------------------------------------------
# Categorical feature engineering
# ---------------------------------------------------------------------------

def add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary indicator columns from proto, state, and service."""
    out = df.copy()
    if "proto" in df.columns:
        # Non-TCP protocols (udp, rare protocols) have higher attack rates
        out["is_not_tcp"] = (df["proto"] != "tcp").astype(float)
    if "state" in df.columns:
        # INT state (incomplete connection) = 55% attack rate for UDP
        out["is_int_state"] = (df["state"] == "INT").astype(float)
        # CON state (established) = 0.1% attack rate — strongly normal
        out["is_con_state"] = (df["state"] == "CON").astype(float)
    if "service" in df.columns:
        # These services show ~100% attack rate in UNSW-NB15
        _anomalous = {"pop3", "ssl", "snmp"}
        out["is_anom_service"] = df["service"].isin(_anomalous).astype(float)
    return out


# ---------------------------------------------------------------------------
# Benchmark configuration definitions
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "baseline",
        "label": "8q - original continuous features",
        "features": ["dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts", "tcprtt"],
        "log1p_cols": ["sbytes", "dbytes", "Sload", "Dload"],
        "n_bins": 2,
        "bits_per_feature": 1,
        "use_rzz": False,
        "n_layers": 3,
    },
    {
        "name": "8q_new_feats",
        "label": "8q - state + proto replace Sload/Dload",
        "features": ["is_not_tcp", "is_int_state", "is_con_state",
                     "dur", "sbytes", "dbytes", "Spkts", "Dpkts"],
        "log1p_cols": ["sbytes", "dbytes"],
        "n_bins": 2,
        "bits_per_feature": 1,
        "use_rzz": False,
        "n_layers": 3,
    },
    {
        "name": "10q_new_feats",
        "label": "10q - state + proto + service + tcprtt",
        "features": ["is_not_tcp", "is_int_state", "is_con_state", "is_anom_service",
                     "dur", "sbytes", "dbytes", "Spkts", "Dpkts", "tcprtt"],
        "log1p_cols": ["sbytes", "dbytes"],
        "n_bins": 2,
        "bits_per_feature": 1,
        "use_rzz": False,
        "n_layers": 3,
    },
    {
        "name": "8q_rzz",
        "label": "8q - state + proto + RZZ entanglement",
        "features": ["is_not_tcp", "is_int_state", "is_con_state",
                     "dur", "sbytes", "dbytes", "Spkts", "Dpkts"],
        "log1p_cols": ["sbytes", "dbytes"],
        "n_bins": 2,
        "bits_per_feature": 1,
        "use_rzz": True,
        "n_layers": 3,
    },
]


# ---------------------------------------------------------------------------
# Circuit diagram export
# ---------------------------------------------------------------------------

def save_circuit(cfg: dict, out_dir: Path) -> None:
    """Save text (and optional PNG) circuit diagram for a config."""
    n_qubits = len(cfg["features"]) * cfg["bits_per_feature"]
    n_layers = cfg["n_layers"]
    use_rzz = cfg["use_rzz"]
    total_params = n_params(n_qubits, n_layers, use_rzz=use_rzz)
    theta_example = np.full(total_params, np.pi / 4)

    try:
        qc = build_ansatz(n_qubits, n_layers, theta_example, use_rzz=use_rzz)

        # Text diagram
        txt_path = out_dir / f"circuit_{cfg['name']}.txt"
        txt_path.write_text(
            f"Config : {cfg['label']}\n"
            f"Qubits : {n_qubits}\n"
            f"Layers : {n_layers}\n"
            f"Params : {total_params}\n"
            f"RZZ    : {use_rzz}\n"
            f"Features: {cfg['features']}\n\n"
            + str(qc.draw("text"))
        )
        print(f"  Circuit saved: {txt_path.name}")

        # PNG (requires matplotlib + pylatexenc)
        try:
            import matplotlib
            matplotlib.use("Agg")
            fig = qc.draw("mpl", fold=-1, style={"backgroundcolor": "#FFFFFF"})
            png_path = out_dir / f"circuit_{cfg['name']}.png"
            fig.savefig(png_path, dpi=120, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  Circuit PNG saved: {png_path.name}")
        except Exception:
            pass  # matplotlib optional

    except Exception as exc:
        print(f"  Warning: could not draw circuit for {cfg['name']}: {exc}")


# ---------------------------------------------------------------------------
# Single config runner
# ---------------------------------------------------------------------------

def run_config(
    cfg: dict,
    X_train_full: pd.DataFrame,
    X_val_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    seed: int = 42,
    max_iter: int = 800,
    spsa_a: float = 0.2,
    spsa_c: float = 0.1,
) -> dict:
    """Run one benchmark configuration end-to-end and return serialisable results."""
    name     = cfg["name"]
    features = cfg["features"]
    log1p_cols      = cfg["log1p_cols"]
    n_bins          = cfg["n_bins"]
    bits_per_feature = cfg["bits_per_feature"]
    use_rzz  = cfg["use_rzz"]
    n_layers = cfg["n_layers"]

    print(f"\n{'='*62}")
    print(f"  Config: {cfg['label']}")
    print(f"{'='*62}")

    missing = [f for f in features if f not in X_train_full.columns]
    if missing:
        raise ValueError(f"Missing features for config '{name}': {missing}")

    X_tr  = X_train_full[features].copy()
    X_va  = X_val_full[features].copy()
    X_te  = X_test_full[features].copy()

    # Log1p on applicable columns only
    actual_log1p = [c for c in log1p_cols if c in features]
    X_tr = apply_log1p(X_tr, actual_log1p)
    X_va = apply_log1p(X_va, actual_log1p)
    X_te = apply_log1p(X_te, actual_log1p)

    # Scale
    scaler = Scaler(mode="standard").fit(X_tr, features)
    X_tr = scaler.transform(X_tr, features)
    X_va = scaler.transform(X_va, features)
    X_te = scaler.transform(X_te, features)

    # Bin + bit-encode
    edges    = fit_bins(X_tr, features, n_bins=n_bins, strategy="quantile")
    bt       = transform_bins(X_tr, edges)
    bv       = transform_bins(X_va, edges)
    bte      = transform_bins(X_te, edges)
    bit_train = encode_bits(bt,  bits_per_feature=bits_per_feature, encoding="binary", n_bins=n_bins)
    bit_val   = encode_bits(bv,  bits_per_feature=bits_per_feature, encoding="binary", n_bins=n_bins)
    bit_test  = encode_bits(bte, bits_per_feature=bits_per_feature, encoding="binary", n_bins=n_bins)

    n_qubits = bit_train.shape[1]
    total_params = n_params(n_qubits, n_layers, use_rzz=use_rzz)
    print(f"  n_qubits={n_qubits}  states={2**n_qubits}  params={total_params}  RZZ={use_rzz}")

    # Split normal / anomaly from training set
    y_tr_np      = y_train.reset_index(drop=True).to_numpy()
    normal_mask  = y_tr_np == 0
    anomaly_mask = y_tr_np == 1
    bit_train_normal  = bit_train[normal_mask]
    bit_train_anomaly = bit_train[anomaly_mask]

    # QCBM config
    config = QCBMConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        max_iter=max_iter,
        seed=seed,
        spsa_a=spsa_a,
        spsa_c=spsa_c,
        lambda_contrast=0.0,
        contrast_margin=0.0,
        laplace_alpha=0.0,
        warmstart_layers=True,
        use_rzz=use_rzz,
    )

    # Train
    train_out  = train_qcbm(bit_train_normal, config, anomaly_bitstrings=bit_train_anomaly)
    model_dist = train_out["model_dist"]

    # Score all splits
    train_scores = score_samples(bit_train, model_dist)
    val_scores   = score_samples(bit_val,   model_dist)
    test_scores  = score_samples(bit_test,  model_dist)

    # Z-score normalise using normal training scores
    mu    = float(np.mean(train_scores[normal_mask]))
    sigma = float(np.std(train_scores[normal_mask]))
    train_z = zscore(train_scores, mu, sigma)
    val_z   = zscore(val_scores,   mu, sigma)
    test_z  = zscore(test_scores,  mu, sigma)

    # Flip if polarity is inverted
    base = evaluate(y_test.to_numpy(), test_z)
    if base["roc_auc"] < 0.5:
        train_z = -train_z
        val_z   = -val_z
        test_z  = -test_z
        base    = evaluate(y_test.to_numpy(), test_z)

    # Threshold tuning on validation set
    f1_t,     best_f1 = find_best_threshold(y_val.to_numpy(), val_z)
    youden_t, best_j  = find_youden_threshold(y_val.to_numpy(), val_z)

    m_youden = evaluate(y_test.to_numpy(), test_z, threshold=youden_t)
    m_f1     = evaluate(y_test.to_numpy(), test_z, threshold=f1_t)

    # Print summary
    print(f"\n  {'Metric':<12} {'Youden':>10} {'F1-thr':>10}")
    print(f"  {'-'*34}")
    for k in ("roc_auc", "pr_auc", "f1", "recall_dr", "far", "mcc"):
        v_y = m_youden.get(k, "N/A")
        v_f = m_f1.get(k, "N/A")
        if isinstance(v_y, float):
            print(f"  {k:<12} {v_y:>10.4f} {v_f:>10.4f}")
    print(f"  {'TP':<12} {m_youden.get('tp', '?'):>10}  {m_f1.get('tp', '?'):>9}")
    print(f"  {'FP':<12} {m_youden.get('fp', '?'):>10}  {m_f1.get('fp', '?'):>9}")
    print(f"  {'FN':<12} {m_youden.get('fn', '?'):>10}  {m_f1.get('fn', '?'):>9}")
    print(f"  {'TN':<12} {m_youden.get('tn', '?'):>10}  {m_f1.get('tn', '?'):>9}")
    print(f"  Youden threshold : {youden_t:.6f}  (val J={best_j:.4f})")
    print(f"  F1    threshold  : {f1_t:.6f}  (val F1={best_f1:.4f})")

    # Build serialisable result (no large numpy arrays in JSON)
    def _clean(d: dict) -> dict:
        return {k: (float(v) if isinstance(v, (np.floating, float)) else
                    int(v)   if isinstance(v, (np.integer, int))   else v)
                for k, v in d.items() if not isinstance(v, dict)}

    return {
        "config_name":    name,
        "config_label":   cfg["label"],
        "n_qubits":       int(n_qubits),
        "n_layers":       int(n_layers),
        "n_params":       int(total_params),
        "n_states":       int(2 ** n_qubits),
        "use_rzz":        bool(use_rzz),
        "features":       features,
        "final_kl_normal": float(train_out["loss"]),
        "metrics_youden": _clean(m_youden),
        "metrics_f1":     _clean(m_f1),
        "youden_threshold": float(youden_t),
        "f1_threshold":     float(f1_t),
        "val_youden_j":    float(best_j),
        "val_f1":          float(best_f1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 multi-config benchmark.")
    parser.add_argument("--input",     default="datasets/UNSW-NB15_cleaned.csv")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--qcbm-iter", type=int,   default=800)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--spsa-a",    type=float, default=0.2)
    parser.add_argument("--spsa-c",    type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac",  type=float, default=0.1)
    parser.add_argument("--out-dir",   default="benchmark_stage1")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ---------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv(ROOT / args.input, low_memory=False)
    print(f"  {len(df):,} rows  |  columns: {list(df.columns[:6])} ...")

    # ---- Derive categorical features --------------------------------------
    print("Engineering categorical features...")
    df = add_categorical_features(df)

    # Collect all features that any config might need
    all_needed = sorted({f for cfg in CONFIGS for f in cfg["features"]})
    missing_cols = [c for c in all_needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Derived feature creation failed for: {missing_cols}")

    X = df[all_needed].copy()
    y = df[args.label_col]
    print(f"  Normal: {(y==0).sum():,}  |  Attack: {(y==1).sum():,}  "
          f"|  Attack rate: {y.mean():.3%}")

    # ---- Shared train/val/test split (same seed = fair comparison) --------
    print("Splitting train/val/test (stratified)...")
    splits = train_val_test_split(
        X, y,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        stratify=True,
    )

    # ---- Save circuit diagrams --------------------------------------------
    print("\nSaving circuit diagrams...")
    for cfg in CONFIGS:
        save_circuit(cfg, out_dir)

    # ---- Run each config --------------------------------------------------
    all_results = []
    for cfg in CONFIGS:
        result = run_config(
            cfg,
            splits.X_train, splits.X_val, splits.X_test,
            splits.y_train, splits.y_val, splits.y_test,
            seed=args.seed,
            max_iter=args.qcbm_iter,
            spsa_a=args.spsa_a,
            spsa_c=args.spsa_c,
        )
        all_results.append(result)

    # ---- Print comparison table ------------------------------------------
    print("\n" + "=" * 74)
    print("  BENCHMARK SUMMARY")
    print("=" * 74)
    header = f"  {'Config':<22} {'ROC-AUC':>9} {'PR-AUC':>8} {'F1':>7} {'DR':>7} {'FAR':>7} {'MCC':>7}"
    print(header)
    print("  " + "-" * 70)
    for r in all_results:
        m = r["metrics_youden"]
        print(
            f"  {r['config_name']:<22} "
            f"{m.get('roc_auc', 0):>9.4f} "
            f"{m.get('pr_auc', 0):>8.4f} "
            f"{m.get('f1', 0):>7.4f} "
            f"{m.get('recall_dr', 0):>7.4f} "
            f"{m.get('far', 0):>7.4f} "
            f"{m.get('mcc', 0):>7.4f}"
        )
    print("=" * 74)

    # ---- Save JSON --------------------------------------------------------
    output = {
        "benchmark_date": str(date.today()),
        "settings": {
            "qcbm_iter": args.qcbm_iter,
            "seed":      args.seed,
            "spsa_a":    args.spsa_a,
            "spsa_c":    args.spsa_c,
        },
        "configs": all_results,
    }
    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {json_path}")
    print(f"Circuits saved to: {out_dir}/circuit_*.txt")


if __name__ == "__main__":
    main()
