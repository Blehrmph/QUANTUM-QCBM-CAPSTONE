"""
Classical baseline + quantum metrics benchmark for Stage 1.

Usage:
    python -u benchmark_classical.py [--config best_config.json]

Outputs:
  - Console: comparison table (QCBM vs KDE vs RBM vs IsoForest)
  - Console: expressibility KL and entanglement entropy per qubit
  - artifacts/classical_baseline_comparison.json
  - PHASES_METRICS/classical_vs_qcbm.png
  - PHASES_METRICS/quantum_metrics.png
"""
from __future__ import annotations

import argparse
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
from src.classical_baseline import (
    train_kde, score_kde,
    train_rbm, score_rbm,
    train_isolation_forest, score_isolation_forest,
    train_autoencoder, score_autoencoder,
)
from src.score_eval import evaluate
from STAGES.stage1 import find_best_threshold


def load_args(config_path: str | None = None):
    defaults = dict(
        input="datasets/UNSW-NB15_core_features.csv",
        label_input="datasets/UNSW-NB15_cleaned.csv",
        label_col="label",
        attack_col="attack_cat",
        features="sbytes,Sload,dbytes,Dload,Dpkts,is_not_tcp,is_int_state,is_con_state",
        log1p=True,
        scaler="standard",
        n_bins=4,
        bits_per_feature=2,
        bin_strategy="quantile",
        encoding="binary",
        test_frac=0.2,
        val_frac=0.1,
        seed=42,
        auto_mixed_precision=True,
        var_threshold=0.0,
        mi_top_k=8,
    )
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if not k.startswith("_"):
                defaults[k.replace("-", "_")] = v

    class Args:
        pass
    args = Args()
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def load_data(args):
    df = pd.read_csv(args.input, low_memory=False)
    need = [args.label_col, args.attack_col]
    for col in ["proto", "state", "service"]:
        if col not in df.columns:
            need.append(col)
    missing = [c for c in need if c not in df.columns]
    if missing:
        available = pd.read_csv(args.label_input, nrows=0).columns.tolist()
        to_load = [c for c in missing if c in available]
        if to_load:
            ldf = pd.read_csv(args.label_input, usecols=to_load, low_memory=False)
            for c in to_load:
                df[c] = ldf[c]
    df = add_categorical_features(df)
    features = [c.strip() for c in args.features.split(",") if c.strip()]
    X = select_features(df, features)
    y = df[args.label_col]
    splits = train_val_test_split(X, y, test_frac=args.test_frac,
                                  val_frac=args.val_frac, seed=args.seed, stratify=True)
    if args.log1p:
        splits.X_train = apply_log1p(splits.X_train, DEFAULT_LOG1P_COLS)
        splits.X_val   = apply_log1p(splits.X_val,   DEFAULT_LOG1P_COLS)
        splits.X_test  = apply_log1p(splits.X_test,  DEFAULT_LOG1P_COLS)
    splits.X_train = splits.X_train[features]
    splits.X_val   = splits.X_val[features]
    splits.X_test  = splits.X_test[features]
    scaler = Scaler(mode=args.scaler).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
    X_val   = scaler.transform(splits.X_val,   features)
    X_test  = scaler.transform(splits.X_test,  features)
    return X_train, X_val, X_test, splits.y_train, splits.y_val, splits.y_test, features


def get_bitstrings(X_train, X_val, X_test, y_train, features, args):
    use_amp = getattr(args, "auto_mixed_precision", False)
    bits_map, bins_map = None, None
    if use_amp:
        bits_map, bins_map = auto_mixed_precision_map(
            X_train, features,
            continuous_bits=args.bits_per_feature,
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
    return bit_train, bit_val, bit_test, bit_train_normal


def run_classical_baselines(bit_train_normal, bit_val, bit_test, y_val, y_test,
                            max_train_samples: int = 50_000):
    N = len(bit_train_normal)
    # Subsample for KDE/RBM — O(N²) scoring makes full-data infeasible
    rng = np.random.default_rng(42)
    if N > max_train_samples:
        idx = rng.choice(N, size=max_train_samples, replace=False)
        X_fit = bit_train_normal[idx]
        print(f"  KDE/RBM subsampled: {max_train_samples:,} / {N:,} normal samples")
    else:
        X_fit = bit_train_normal

    results = {}

    # ── KDE (bandwidth tuned on val, subsampled) ─────────────────────────────
    print("  Training KDE (50K subsample — O(N²) scoring constraint)...")
    best_bw, best_roc = 0.5, 0.0
    for bw in [0.1, 0.3, 0.5, 1.0]:
        kde = train_kde(X_fit, bandwidth=bw)
        val_s = score_kde(bit_val, kde)
        m = evaluate(y_val.to_numpy(), val_s)
        if m["roc_auc"] > best_roc:
            best_roc, best_bw = m["roc_auc"], bw
    kde = train_kde(X_fit, bandwidth=best_bw)
    val_s  = score_kde(bit_val,  kde)
    test_s = score_kde(bit_test, kde)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["KDE"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["KDE"]["bandwidth"] = best_bw
    results["KDE"]["train_n"] = max_train_samples
    print(f"    KDE best bandwidth={best_bw}  ROC-AUC={results['KDE']['roc_auc']:.4f}")

    # ── RBM (n_components=5 -> 85 params, closest to QCBM budget, subsampled) ─
    print("  Training RBM (n_components=5, ~85 params, 50K subsample)...")
    rbm = train_rbm(X_fit, n_components=5, n_iter=200)
    val_s  = score_rbm(bit_val,  rbm)
    test_s = score_rbm(bit_test, rbm)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["RBM_5"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["RBM_5"]["n_params"] = 13 * 5 + 5 + 13
    results["RBM_5"]["train_n"] = max_train_samples
    print(f"    RBM ROC-AUC={results['RBM_5']['roc_auc']:.4f}")

    # ── RBM (n_components=26, larger, subsampled) ────────────────────────────
    print("  Training RBM (n_components=26, larger, 50K subsample)...")
    rbm26 = train_rbm(X_fit, n_components=26, n_iter=200)
    val_s  = score_rbm(bit_val,  rbm26)
    test_s = score_rbm(bit_test, rbm26)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["RBM_26"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["RBM_26"]["n_params"] = 13 * 26 + 26 + 13
    results["RBM_26"]["train_n"] = max_train_samples
    print(f"    RBM-26 ROC-AUC={results['RBM_26']['roc_auc']:.4f}")

    # ── Isolation Forest (FULL data, max_samples=256 per tree) ───────────────
    print(f"  Training Isolation Forest (100 trees, full {N:,} samples, max_samples=256)...")
    iso = train_isolation_forest(bit_train_normal, max_samples=256)
    val_s  = score_isolation_forest(bit_val,  iso)
    test_s = score_isolation_forest(bit_test, iso)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["IsoForest"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["IsoForest"]["train_n"] = N
    print(f"    IsoForest ROC-AUC={results['IsoForest']['roc_auc']:.4f}")

    # ── Autoencoder (FULL data, 13->6->13, mini-batch Adam) ─────────────────
    print(f"  Training Autoencoder (13->6->13, full {N:,} samples, batch=1024)...")
    ae = train_autoencoder(bit_train_normal, hidden_dim=6, max_iter=50, batch_size=1024)
    val_s  = score_autoencoder(bit_val,  ae)
    test_s = score_autoencoder(bit_test, ae)
    t, _ = find_best_threshold(y_val.to_numpy(), val_s)
    results["Autoencoder"] = evaluate(y_test.to_numpy(), test_s, threshold=t)
    results["Autoencoder"]["n_params"] = 13 * 6 + 6 + 6 * 13 + 13  # ~175
    results["Autoencoder"]["train_n"] = N
    print(f"    Autoencoder ROC-AUC={results['Autoencoder']['roc_auc']:.4f}")

    return results


def run_quantum_metrics(theta_path: str, config_path: str, n_expr_samples: int = 200):
    from src.quantum_metrics import expressibility, entanglement_entropy
    from src.qcbm_train import QCBMConfig

    print("  Loading saved QCBM theta...")
    thetas = np.load(theta_path)  # shape (ensemble, n_params)
    with open(config_path) as f:
        cfg_d = json.load(f)

    n_qubits = cfg_d["n_qubits"]
    n_layers = cfg_d["n_layers"]

    # Use first ensemble member for quantum metrics
    theta = thetas[0]

    print(f"  Computing expressibility ({n_expr_samples} sample pairs)...")
    expr_kl, fidelities = expressibility(
        n_qubits=n_qubits, n_layers=n_layers, theta=theta,
        n_samples=n_expr_samples, seed=42,
    )
    print(f"    Expressibility KL(empirical||Haar) = {expr_kl:.4f}")
    print(f"    (lower = more expressive; random circuit ~0)")

    print("  Computing entanglement entropy...")
    ent = entanglement_entropy(theta=theta, n_qubits=n_qubits, n_layers=n_layers)
    print(f"    Mean S = {ent['mean']:.4f} bits  Max S = {ent['max']:.4f}  Min S = {ent['min']:.4f}")
    for i, s in enumerate(ent["per_qubit"]):
        print(f"    Qubit {i:2d}: S = {s:.4f}")

    return expr_kl, fidelities, ent


def print_comparison_table(classical_results: dict, qcbm_metrics: dict):
    qcbm = {
        "ROC-AUC":   qcbm_metrics.get("roc_auc", 0),
        "PR-AUC":    qcbm_metrics.get("pr_auc",  0),
        "F1":        qcbm_metrics.get("f1", qcbm_metrics.get("f1_threshold_metrics", {}).get("f1", 0)),
        "Precision": qcbm_metrics.get("precision", qcbm_metrics.get("f1_threshold_metrics", {}).get("precision", 0)),
        "Recall":    qcbm_metrics.get("recall_dr", qcbm_metrics.get("f1_threshold_metrics", {}).get("recall_dr", 0)),
        "FAR":       qcbm_metrics.get("far", qcbm_metrics.get("f1_threshold_metrics", {}).get("far", 0)),
        "MCC":       qcbm_metrics.get("mcc", qcbm_metrics.get("f1_threshold_metrics", {}).get("mcc", 0)),
    }

    header = f"  {'Model':<16} {'Train N':>10} {'ROC-AUC':>9} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'FAR':>8} {'MCC':>8}"
    sep    = "  " + "-" * 90
    print("\n" + sep)
    print(header)
    print(sep)
    print(f"  {'QCBM (ours)':<16} {'1,582,625':>10} {qcbm['ROC-AUC']:>9.4f} {qcbm['PR-AUC']:>8.4f} "
          f"{qcbm['F1']:>8.4f} {qcbm['Precision']:>8.4f} "
          f"{qcbm['Recall']:>8.4f} {qcbm['FAR']:>8.4f} {qcbm['MCC']:>8.4f}  <-- quantum")
    print(sep)
    for name, m in classical_results.items():
        roc   = m.get("roc_auc", 0)
        pr    = m.get("pr_auc",  0)
        f1    = m.get("f1",      0)
        prec  = m.get("precision", 0)
        rec   = m.get("recall_dr", 0)
        far   = m.get("far",      0)
        mcc   = m.get("mcc",      0)
        n_tr  = m.get("train_n",   0)
        delta = roc - qcbm["ROC-AUC"]
        delta_str = f"({delta:+.4f})"
        n_str = f"{n_tr:,}" if n_tr else "—"
        print(f"  {name:<16} {n_str:>10} {roc:>9.4f} {pr:>8.4f} {f1:>8.4f} {prec:>8.4f} "
              f"{rec:>8.4f} {far:>8.4f} {mcc:>8.4f}  {delta_str}")
    print(sep)


def save_comparison_chart(classical_results: dict, qcbm_roc: float, qcbm_pr: float,
                          fidelities, ent: dict, n_qubits: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path("PHASES_METRICS")
    out.mkdir(exist_ok=True)

    # ── Chart 1: ROC-AUC and PR-AUC comparison bar chart ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#f8f8f8")
    fig.suptitle("Stage 1: QCBM vs Classical Baselines", fontsize=14, fontweight="bold")

    models = ["QCBM\n(ours)"] + list(classical_results.keys())
    roc_vals = [qcbm_roc] + [classical_results[k]["roc_auc"] for k in classical_results]
    pr_vals  = [qcbm_pr]  + [classical_results[k]["pr_auc"]  for k in classical_results]
    colors = ["#4e79a7"] + ["#e15759"] * len(classical_results)

    for ax, vals, ylabel, title in [
        (axes[0], roc_vals, "ROC-AUC", "ROC-AUC Comparison"),
        (axes[1], pr_vals,  "PR-AUC",  "PR-AUC Comparison"),
    ]:
        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_facecolor("#f8f8f8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if val == vals[0] else "normal")
        ax.axhline(vals[0], color="#4e79a7", linestyle="--", lw=1.2, alpha=0.5)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#4e79a7", label="QCBM (quantum)"),
                         Patch(color="#e15759", label="Classical baselines")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(out / "classical_vs_qcbm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: PHASES_METRICS/classical_vs_qcbm.png")

    # ── Chart 2: Quantum metrics ──────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f8f8f8")
    fig2.suptitle("Stage 1: Quantum Circuit Metrics", fontsize=14, fontweight="bold")

    # Fidelity distribution vs Haar
    ax = axes2[0]
    ax.set_facecolor("#f8f8f8")
    bins = np.linspace(0, 1, 50)
    ax.hist(fidelities, bins=bins, density=True, alpha=0.7,
            color="#4e79a7", label="Empirical fidelities")
    dim = 2 ** n_qubits
    F = np.linspace(0, 1, 300)
    haar = (dim - 1) * (1 - F) ** (dim - 2)
    ax.plot(F, haar / haar.max() * ax.get_ylim()[1] if haar.max() > 0 else haar,
            color="#e15759", lw=2, label="Haar-random (ideal)")
    ax.set_xlabel("State Fidelity |<ψ₁|ψ₂>|²")
    ax.set_ylabel("Density")
    ax.set_title("Expressibility: Fidelity Distribution\nvs Haar-Random Measure", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Entanglement entropy per qubit
    ax2 = axes2[1]
    ax2.set_facecolor("#f8f8f8")
    qubits = list(range(n_qubits))
    entropies = ent["per_qubit"]
    bars2 = ax2.bar(qubits, entropies, color="#59a14f", edgecolor="white", linewidth=1)
    ax2.axhline(ent["mean"], color="#f28e2b", lw=2, linestyle="--",
                label=f"Mean S = {ent['mean']:.3f}")
    ax2.axhline(1.0, color="gray", lw=1, linestyle=":", alpha=0.5,
                label="Max (maximally entangled)")
    ax2.set_xlabel("Qubit Index")
    ax2.set_ylabel("Von Neumann Entropy S (bits)")
    ax2.set_title("Entanglement Entropy per Qubit\n(Higher = more quantum correlation)",
                  fontsize=11)
    ax2.set_xticks(qubits)
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="#e0e0e0", linewidth=0.8)

    fig2.tight_layout()
    fig2.savefig(out / "quantum_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved: PHASES_METRICS/quantum_metrics.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="best_config.json")
    parser.add_argument("--expr-samples", type=int, default=200,
                        help="Number of random parameter pairs for expressibility.")
    parser.add_argument("--skip-quantum-metrics", action="store_true")
    args_cli = parser.parse_args()

    args = load_args(args_cli.config)
    features = [c.strip() for c in args.features.split(",") if c.strip()]

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data(args)

    print("Building bitstring encodings...")
    bit_train, bit_val, bit_test, bit_train_normal = get_bitstrings(
        X_train, X_val, X_test, y_train, features, args
    )
    print(f"  {bit_train_normal.shape[1]} qubits  |  "
          f"train_normal={len(bit_train_normal)}  val={len(bit_val)}  test={len(bit_test)}")

    print("\n[1/3] Bitstring coverage analysis & FAR floor derivation...")
    try:
        from src.bitstring_coverage import compute_bitstring_coverage, print_coverage_report
        coverage_stats = compute_bitstring_coverage(
            bit_train_normal, bit_test, y_test.to_numpy(), bit_train_normal.shape[1]
        )
        print_coverage_report(coverage_stats)
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/bitstring_coverage.json").write_text(
            json.dumps(coverage_stats, indent=2)
        )
        print("  Saved: artifacts/bitstring_coverage.json")
    except Exception as e:
        print(f"  Coverage analysis failed: {e}")
        coverage_stats = None

    print("\n[2/4] Classical baselines...")
    classical = run_classical_baselines(
        bit_train_normal, bit_val, bit_test, y_val, y_test
    )

    print("\n[3/4] Loading QCBM metrics...")
    qcbm_metrics_path = Path("artifacts/hier_stage1_metrics.json")
    with open(qcbm_metrics_path) as f:
        qcbm_metrics = json.load(f)
    qcbm_roc = qcbm_metrics.get("roc_auc", 0.9350)
    qcbm_pr  = qcbm_metrics.get("pr_auc",  0.5230)

    print_comparison_table(classical, qcbm_metrics)

    expr_kl, fidelities, ent = None, None, None
    if not args_cli.skip_quantum_metrics:
        print("\n[4/4] Quantum metrics (no retraining)...")
        try:
            expr_kl, fidelities, ent = run_quantum_metrics(
                theta_path="artifacts/hier_qcbm_theta.npy",
                config_path="artifacts/hier_qcbm_config.json",
                n_expr_samples=args_cli.expr_samples,
            )
        except Exception as e:
            print(f"  Warning: quantum metrics failed: {e}")

    # Save comparison chart
    if fidelities is not None and ent is not None:
        print("\nGenerating charts...")
        save_comparison_chart(classical, qcbm_roc, qcbm_pr, fidelities, ent,
                              n_qubits=13)

    # Save JSON results
    out = {
        "qcbm": {"roc_auc": qcbm_roc, "pr_auc": qcbm_pr},
        "classical": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
                      for k, v in classical.items()},
    }
    if expr_kl is not None:
        out["quantum_metrics"] = {
            "expressibility_kl": expr_kl,
            "entanglement_entropy": ent,
        }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/classical_baseline_comparison.json").write_text(
        json.dumps(out, indent=2)
    )
    print("\nSaved: artifacts/classical_baseline_comparison.json")
    print("Done.")


if __name__ == "__main__":
    main()
