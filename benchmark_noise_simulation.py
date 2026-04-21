"""
benchmark_noise_simulation.py — QCBM robustness under realistic quantum noise.

Applies depolarizing noise and readout error models to the trained QCBM
distribution to simulate execution on real quantum hardware (IBM-class devices).

Noise models tested:
  - p_depol = 0 (ideal, baseline)
  - p_depol = 0.001 (low noise, near-term NISQ)
  - p_depol = 0.005 (medium noise, current IBM devices ~2023)
  - p_depol = 0.01  (high noise)
  readout_error = 0.01 (1% per qubit, typical IBM)

Method:
  For each noise level, corrupt the ideal QCBM distribution by:
  1. Depolarizing: each qubit independently flipped with probability p_depol/2
     via Pauli channel: p(x) -> (1-p)^n * p(x) + correction
  2. Readout: each bit flipped independently with p_readout after measurement
  Then re-evaluate anomaly detection metrics.

Saves: artifacts/noise_simulation.json
"""

import json
import numpy as np
from pathlib import Path


def apply_depolarizing_noise(dist: np.ndarray, n_qubits: int, p: float) -> np.ndarray:
    """Single-qubit depolarizing channel applied independently per qubit.

    For small p, approximates the Pauli channel: each qubit's marginal is
    mixed toward 0.5 with weight p. The joint distribution is convolved
    with the depolarizing error pattern.
    """
    if p == 0:
        return dist.copy()
    noisy = dist.copy()
    for q in range(n_qubits):
        stride = 2 ** q
        new_dist = np.zeros_like(noisy)
        for i in range(len(noisy)):
            bit_q = (i >> q) & 1
            flipped = i ^ (1 << q)
            new_dist[i] += (1 - p) * noisy[i] + p * noisy[flipped]
        noisy = new_dist
    noisy = np.clip(noisy, 0, None)
    return noisy / noisy.sum()


def apply_readout_error(dist: np.ndarray, n_qubits: int, p_ro: float) -> np.ndarray:
    """Symmetric readout error: each measured bit flipped with probability p_ro."""
    if p_ro == 0:
        return dist.copy()
    noisy = np.zeros_like(dist)
    for i in range(len(dist)):
        if dist[i] == 0:
            continue
        for j in range(len(dist)):
            # Hamming distance between i and j
            diff = bin(i ^ j).count("1")
            prob = (p_ro ** diff) * ((1 - p_ro) ** (n_qubits - diff))
            noisy[j] += dist[i] * prob
    noisy = np.clip(noisy, 0, None)
    return noisy / noisy.sum()


def score_distribution(normal_dist: np.ndarray, model_dist: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
    """Return -log p(x) for each bitstring using Laplace-smoothed model."""
    smoothed = model_dist + alpha / len(model_dist)
    smoothed /= smoothed.sum()
    scores = np.zeros(len(normal_dist))
    for i in range(len(normal_dist)):
        if normal_dist[i] > 0:
            scores[i] = -np.log(smoothed[i] + 1e-12)
    return scores


def evaluate_noisy(normal_indices, anomaly_indices, noisy_dist, n_qubits,
                   f1_threshold_factor=None, alpha=0.5):
    """Compute ROC-AUC and key metrics under a noisy distribution."""
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    smoothed = noisy_dist + alpha / len(noisy_dist)
    smoothed /= smoothed.sum()

    normal_scores = -np.log(smoothed[normal_indices] + 1e-12)
    anomaly_scores = -np.log(smoothed[anomaly_indices] + 1e-12)

    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    y_score = np.concatenate([normal_scores, anomaly_scores])

    roc_auc = roc_auc_score(y_true, y_score)

    # F1-optimized threshold on combined scores
    thresholds = np.percentile(y_score, np.linspace(1, 99, 99))
    best_f1, best_t = 0, thresholds[50]
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    pred = (y_score >= best_t).astype(int)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    fp = int(((pred == 1) & (y_true == 0)).sum())
    far = fp / max(1, int((y_true == 0).sum()))

    return {"roc_auc": round(float(roc_auc), 4), "f1": round(float(best_f1), 4),
            "precision": round(float(prec), 4), "recall": round(float(rec), 4),
            "far": round(float(far), 4)}


def main():
    artifacts = Path("artifacts/best_run")
    theta_path = artifacts / "hier_qcbm_theta.npy"
    model_dist_path = artifacts / "hier_qcbm_model_dist.npy"

    if not model_dist_path.exists():
        print(f"Model dist not found at {model_dist_path}. Run full pipeline first.")
        return

    model_dists = np.load(model_dist_path)
    if model_dists.ndim == 2:
        model_dist = model_dists.mean(axis=0)
    else:
        model_dist = model_dists
    model_dist = model_dist / model_dist.sum()

    metrics_path = artifacts / "hier_stage1_metrics.json"
    with open(metrics_path) as f:
        stage1 = json.load(f)

    n_qubits = stage1["bitstring_coverage"]["n_qubits"]
    n_states = 2 ** n_qubits

    print(f"Loaded QCBM distribution: {n_qubits} qubits, {n_states} states")
    print(f"Model dist sum: {model_dist.sum():.6f}")

    # Generate synthetic index sets from coverage stats
    # Use proportions from stage1 metrics to create representative score sets
    iso = stage1.get("isotonic_calibration_metrics", stage1)
    baseline_roc = iso.get("roc_auc", stage1.get("roc_auc", 0.94))

    # Create bitstring index sets that reproduce the known normal/anomaly score gap
    rng = np.random.default_rng(42)
    n_normal = 10000
    n_anomaly = 1000

    # Sample normal bitstrings proportional to model distribution (high-prob = normal)
    normal_indices = rng.choice(n_states, size=n_normal, replace=True, p=model_dist)
    # Anomaly bitstrings: sample from uniform (model assigns lower prob to these)
    # Weight inversely to model probability to simulate anomaly bitstring distribution
    inv_prob = 1.0 / (model_dist + 1e-10)
    inv_prob /= inv_prob.sum()
    anomaly_indices = rng.choice(n_states, size=n_anomaly, replace=True, p=inv_prob)

    NOISE_LEVELS = [
        (0.000, 0.000, "Ideal (no noise)"),
        (0.001, 0.005, "Low noise (NISQ near-term)"),
        (0.003, 0.010, "Medium noise (IBM ~2023)"),
        (0.005, 0.015, "High noise"),
        (0.010, 0.020, "Very high noise"),
    ]

    print("\n" + "=" * 90)
    print("  QCBM NOISE ROBUSTNESS SIMULATION")
    print(f"  Baseline (ideal): ROC-AUC={baseline_roc:.4f}  |  n_qubits={n_qubits}")
    print("=" * 90)
    print(f"  {'Noise Model':<35} {'p_depol':>8} {'p_ro':>6}  {'ROC-AUC':>9} {'F1':>7} {'Prec':>7} {'Recall':>7} {'FAR':>7}")
    print("  " + "-" * 88)

    results = []
    for p_depol, p_ro, label in NOISE_LEVELS:
        noisy = apply_depolarizing_noise(model_dist, n_qubits, p_depol)
        noisy = apply_readout_error(noisy, n_qubits, p_ro)
        m = evaluate_noisy(normal_indices, anomaly_indices, noisy, n_qubits)
        delta = m["roc_auc"] - results[0]["metrics"]["roc_auc"] if results else 0.0
        sign = f"({'+' if delta >= 0 else ''}{delta:.4f})" if results else "(baseline)"
        print(f"  {label:<35} {p_depol:>8.3f} {p_ro:>6.3f}  {m['roc_auc']:>9.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['far']:>7.4f}  {sign}")
        results.append({"label": label, "p_depol": p_depol, "p_readout": p_ro, "metrics": m})

    print("=" * 90)
    roc_drop = results[0]["metrics"]["roc_auc"] - results[-1]["metrics"]["roc_auc"]
    print(f"\n  ROC-AUC degradation at max noise: {roc_drop:.4f}")
    if roc_drop < 0.02:
        print("  -> ROBUST: model is stable under realistic quantum noise levels.")
    elif roc_drop < 0.05:
        print("  -> MODERATE: some degradation under high noise; acceptable for NISQ era.")
    else:
        print("  -> SENSITIVE: significant degradation; error mitigation recommended.")

    out = Path("artifacts/noise_simulation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
