"""
ibm_zne.py  —  Zero-Noise Extrapolation for the QCBM on IBM quantum hardware.

Noise mitigation strategy: CX gate folding + Richardson extrapolation.

How it works
------------
1. Scale-1 distribution is already saved from ibm_inference.py (ibm_dist.npy).
2. This script runs each ensemble member at scale-3: every CX gate in the circuit
   is replaced by CX·CX·CX (three repetitions). The ideal output is unchanged
   because CX·CX = I, so CX·CX·CX = CX. But the noise triples because each CX
   execution adds hardware error.
3. Richardson extrapolation reconstructs the zero-noise limit:

       p_ZNE[i] = (3 * p_scale1[i] - p_scale3[i]) / 2

4. Negative values are clipped to 0 and the distribution is renormalized.
5. The test set is scored with the ZNE distribution and metrics are reported.

Usage
-----
python ibm_zne.py
python ibm_zne.py --artifact-dir artifacts/best_run --shots 32768 --backend ibm_fez
python ibm_zne.py --ensemble-member 0   # run only one member (saves quota)

Requirements
------------
pip install qiskit-ibm-runtime python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zero-Noise Extrapolation: recover QCBM accuracy from noisy IBM hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--artifact-dir", default="artifacts/best_run")
    p.add_argument("--config", default="best_config.json")
    p.add_argument("--shots", type=int, default=32768)
    p.add_argument("--backend", default=None,
                   help="Pin a specific IBM backend (e.g. ibm_fez). Omit = least-busy.")
    p.add_argument("--token", default=None)
    p.add_argument("--ensemble-member", type=int, default=None,
                   help="Run only this member index (0-based). Omit = run all members.")
    p.add_argument("--output", default=None,
                   help="JSON output path (default: <artifact-dir>/ibm_zne_metrics.json).")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value


def resolve_token(cli_token: str | None) -> str | None:
    return cli_token or os.environ.get("IBM_QUANTUM_TOKEN")


def load_artifacts(artifact_dir: str):
    d = Path(artifact_dir)
    theta_path  = d / "hier_qcbm_theta.npy"
    config_path = d / "hier_qcbm_config.json"
    if not theta_path.exists():
        sys.exit(f"ERROR: {theta_path} not found. Run main.py first.")
    if not config_path.exists():
        sys.exit(f"ERROR: {config_path} not found.")
    theta = np.load(str(theta_path))
    with config_path.open() as f:
        cfg = json.load(f)
    return theta, cfg


# ---------------------------------------------------------------------------
# CX gate folding
# ---------------------------------------------------------------------------

def fold_cx(qc_abstract, scale_factor: int):
    """Return a new circuit where every CX is replaced by CX repeated scale_factor times.

    scale_factor must be odd so the ideal unitary is unchanged (CX^2 = I).
    This amplifies two-qubit gate noise by ~scale_factor while leaving the
    ideal output distribution identical.
    """
    assert scale_factor % 2 == 1, "scale_factor must be odd (1, 3, 5, ...)"
    from qiskit import QuantumCircuit

    qc_folded = QuantumCircuit(qc_abstract.num_qubits)
    for instr in qc_abstract.data:
        gate    = instr.operation
        qubits  = instr.qubits
        qc_folded.append(gate, qubits)
        if gate.name == "cx" and scale_factor > 1:
            n_extra_pairs = (scale_factor - 1) // 2
            for _ in range(n_extra_pairs):
                qc_folded.append(gate, qubits)  # CX (extra #1 — noise)
                qc_folded.append(gate, qubits)  # CX (extra #2 — noise, cancels ideal)
    return qc_folded


# ---------------------------------------------------------------------------
# Run one ensemble member on IBM at a given scale factor
# ---------------------------------------------------------------------------

def run_member_ibm(
    theta: np.ndarray,
    config_dict: dict,
    scale_factor: int,
    shots: int,
    backend_name: str | None,
    token: str | None,
) -> np.ndarray:
    """Run one ensemble member at the given CX noise scale, return probability array."""
    import os as _os
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit import transpile
    from src.qcbm_train import build_ansatz, QCBMConfig

    config = QCBMConfig(
        n_qubits=config_dict["n_qubits"],
        n_layers=config_dict["n_layers"],
        use_rzz=config_dict.get("use_rzz", False),
        entanglement_pairs=config_dict.get("entanglement_pairs"),
    )

    # Connect to IBM
    resolved_token = token or _os.environ.get("IBM_QUANTUM_TOKEN")
    if resolved_token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=resolved_token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            operational=True, simulator=False,
            min_num_qubits=config.n_qubits,
        )

    # Build abstract ansatz (no measurements yet)
    qc = build_ansatz(
        config.n_qubits, config.n_layers, theta,
        use_rzz=config.use_rzz,
        entanglement_pairs=config.entanglement_pairs,
    )

    # Fold CX gates at requested scale
    qc_folded = fold_cx(qc, scale_factor)
    qc_folded.measure_all()

    # Transpile the folded circuit to native backend gates
    # optimization_level=1: maps to native gates without cancelling the folded CX pairs
    qc_t = transpile(qc_folded, backend=backend, optimization_level=1)
    cx_count_orig = sum(1 for i in qc.data if i.operation.name == "cx")
    cx_count_fold = sum(1 for i in qc_folded.data if i.operation.name == "cx")
    print(f"  [IBM scale={scale_factor}] Backend: {backend.name}  "
          f"depth={qc_t.depth()}  CX: {cx_count_orig}→{cx_count_fold}")

    sampler = Sampler(mode=backend)
    job = sampler.run([qc_t], shots=shots)
    print(f"  [IBM scale={scale_factor}] Job ID: {job.job_id()}  waiting...")

    result = job.result()
    counts = result[0].data.meas.get_counts()

    n_states = 2 ** config.n_qubits
    probs = np.zeros(n_states)
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        idx = int(cleaned[::-1], 2)
        probs[idx] = count / shots
    return probs


# ---------------------------------------------------------------------------
# ZNE extrapolation
# ---------------------------------------------------------------------------

def richardson_extrapolate(p1: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Linear Richardson extrapolation to zero noise using scale factors 1 and 3.

    Assumes p(lambda) = p_0 + c * lambda.
    Solving for p_0 with lambda=1 and lambda=3:
        p_ZNE = (3 * p1 - p3) / 2
    """
    p_zne = (3.0 * p1 - p3) / 2.0
    p_zne = np.clip(p_zne, 0.0, None)       # clip any negative artefacts
    total = p_zne.sum()
    if total > 0:
        p_zne /= total                        # renormalize to a valid distribution
    return p_zne


# ---------------------------------------------------------------------------
# Scoring & metrics (mirrors ibm_score.py)
# ---------------------------------------------------------------------------

def score_distribution(dist: np.ndarray, artifact_dir: str, cfg: dict, label: str) -> dict:
    """Reproduce test bitstrings and evaluate the given distribution."""
    from src.data.preprocessing import (
        add_categorical_features, apply_log1p, select_features,
        DEFAULT_LOG1P_COLS, Scaler,
    )
    from src.training_setup import train_val_test_split, filter_normal
    from src.discretize import auto_mixed_precision_map, fit_bins, transform_bins, encode_bits
    from src.score_eval import score_samples, evaluate
    import pandas as pd

    print(f"\n  Reproducing preprocessing to score [{label}]...")
    df = pd.read_csv(cfg.get("input", "datasets/UNSW-NB15_cleaned.csv"), low_memory=False)
    df = add_categorical_features(df)
    features = [f.strip() for f in cfg["features"].split(",") if f.strip()]
    X = select_features(df, features)
    y = df[cfg.get("label_col", "label")]

    splits = train_val_test_split(
        X, y,
        test_frac=cfg.get("test_frac", 0.2),
        val_frac=cfg.get("val_frac", 0.1),
        seed=cfg.get("seed", 42),
        stratify=True,
    )
    if cfg.get("log1p", True):
        splits.X_train = apply_log1p(splits.X_train, DEFAULT_LOG1P_COLS)
        splits.X_test  = apply_log1p(splits.X_test,  DEFAULT_LOG1P_COLS)

    scaler = Scaler(mode=cfg.get("scaler", "standard")).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
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
    anomaly_mask  = y_train_reset.to_numpy() == 1
    X_train_anom  = X_train.iloc[anomaly_mask] if anomaly_mask.any() else None

    edges  = fit_bins(X_train, features,
                      n_bins=cfg.get("n_bins", 4),
                      strategy=cfg.get("bin_strategy", "quantile"),
                      n_bins_map=bins_map, df_anomaly=X_train_anom)
    btest  = transform_bins(X_test, edges)
    enc_kw = dict(bits_per_feature=cfg.get("bits_per_feature", 2),
                  encoding=cfg.get("encoding", "binary"),
                  n_bins=cfg.get("n_bins", 4),
                  bits_per_feature_map=bits_map)
    bit_test = encode_bits(btest, **enc_kw)

    y_test = splits.y_test.reset_index(drop=True)
    scores = score_samples(bit_test, dist)

    # F1-optimal threshold
    thresholds = np.unique(scores)
    if len(thresholds) > 200:
        thresholds = np.quantile(scores, np.linspace(0, 1, 201))
    best_t, best_f1 = thresholds[0], -1.0
    y_arr = y_test.to_numpy()
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

    return evaluate(y_arr, scores, threshold=best_t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args  = build_parser().parse_args()
    token = resolve_token(args.token)

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    theta_raw, config_dict = load_artifacts(args.artifact_dir)
    is_ensemble = theta_raw.ndim == 2
    if is_ensemble:
        n_members = theta_raw.shape[0]
        if args.ensemble_member is not None:
            if not (0 <= args.ensemble_member < n_members):
                sys.exit(f"ERROR: --ensemble-member must be 0..{n_members - 1}")
            theta_list = [theta_raw[args.ensemble_member]]
            print(f"Using ensemble member {args.ensemble_member} only.")
        else:
            theta_list = [theta_raw[i] for i in range(n_members)]
            print(f"Using all {n_members} ensemble members.")
    else:
        theta_list = [theta_raw]

    # Load existing scale-1 distribution (from ibm_inference.py)
    p1_path = Path(args.artifact_dir) / "ibm_dist.npy"
    if not p1_path.exists():
        sys.exit(
            f"ERROR: {p1_path} not found.\n"
            "Run ibm_inference.py first to get the scale-1 IBM distribution."
        )
    p_scale1 = np.load(str(p1_path))
    print(f"\nLoaded scale-1 IBM distribution: {p1_path}  (sum={p_scale1.sum():.4f})")

    # ------------------------------------------------------------------
    # Run scale-3 circuits on IBM hardware
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  RUNNING SCALE-3 CIRCUITS (CX gate folding)")
    print(f"  Each CX → CX·CX·CX  (~3× two-qubit noise, same ideal output)")
    print(f"  Jobs to submit: {len(theta_list)}")
    print(f"{'=' * 60}\n")

    scale3_dists = []
    for i, theta in enumerate(theta_list):
        print(f"--- Ensemble member {i + 1}/{len(theta_list)} ---")
        d = run_member_ibm(
            theta, config_dict,
            scale_factor=3,
            shots=args.shots,
            backend_name=args.backend,
            token=token,
        )
        scale3_dists.append(d)

    p_scale3 = np.mean(scale3_dists, axis=0)
    print(f"\nScale-3 distribution computed (averaged {len(scale3_dists)} members).")

    # ------------------------------------------------------------------
    # ZNE extrapolation
    # ------------------------------------------------------------------
    p_zne = richardson_extrapolate(p_scale1, p_scale3)

    n_positive = int(np.sum(p_zne > 0))
    n_clipped  = int(np.sum((3.0 * p_scale1 - p_scale3) / 2.0 < 0))
    print(f"\nZNE distribution:")
    print(f"  States with positive prob : {n_positive:,} / {len(p_zne):,}")
    print(f"  States clipped (negative) : {n_clipped:,}  (expected for noisy extrapolation)")
    print(f"  Distribution sum after norm: {p_zne.sum():.6f}")

    # Save ZNE distribution
    out_dir = Path(args.artifact_dir)
    zne_dist_path = out_dir / "ibm_zne_dist.npy"
    scale3_dist_path = out_dir / "ibm_scale3_dist.npy"
    np.save(str(zne_dist_path), p_zne)
    np.save(str(scale3_dist_path), p_scale3)
    print(f"  Saved ZNE distribution   : {zne_dist_path}")
    print(f"  Saved scale-3 distribution: {scale3_dist_path}")

    # ------------------------------------------------------------------
    # Score all three distributions
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = json.load(f)

    print(f"\n{'=' * 60}")
    print("  SCORING RESULTS")
    print(f"{'=' * 60}")

    metrics_zne    = score_distribution(p_zne,    args.artifact_dir, cfg, "ZNE (extrapolated)")
    metrics_scale1 = score_distribution(p_scale1, args.artifact_dir, cfg, "IBM scale-1 (raw)")
    metrics_scale3 = score_distribution(p_scale3, args.artifact_dir, cfg, "IBM scale-3 (more noise)")

    # Load simulator baseline for comparison
    sim_metrics_path = Path(args.artifact_dir) / "hier_stage1_metrics.json"
    sim_roc = sim_f1 = sim_far = None
    if sim_metrics_path.exists():
        with sim_metrics_path.open() as f:
            sm = json.load(f)
        iso = sm.get("isotonic_calibration_metrics", sm)
        sim_roc = iso.get("roc_auc")
        sim_f1  = iso.get("f1")
        sim_far = iso.get("far")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  FULL COMPARISON")
    print(f"{'=' * 65}")
    print(f"  {'Method':<28} {'ROC-AUC':>9} {'F1':>8} {'FAR':>8} {'MCC':>8}")
    print(f"  {'-' * 55}")

    def row(name, m):
        roc = f"{m.get('roc_auc', 0):.4f}"
        f1  = f"{m.get('f1', 0):.4f}"
        far = f"{m.get('far', 0):.4f}"
        mcc = f"{m.get('mcc', 0):.4f}"
        print(f"  {name:<28} {roc:>9} {f1:>8} {far:>8} {mcc:>8}")

    if sim_roc:
        print(f"  {'Aer simulator (ideal)':<28} {sim_roc:>9.4f} {sim_f1:>8.4f} {sim_far:>8.4f} {'—':>8}")
    row("IBM scale-1 (raw, noisy)", metrics_scale1)
    row("IBM scale-3 (more noise)", metrics_scale3)
    row("IBM ZNE (extrapolated) ←", metrics_zne)
    print(f"{'=' * 65}")

    if sim_roc:
        delta = metrics_zne.get("roc_auc", 0) - sim_roc
        recov = metrics_zne.get("roc_auc", 0) - metrics_scale1.get("roc_auc", 0)
        print(f"\n  ZNE vs simulator  : {delta:+.4f} ROC-AUC")
        print(f"  ZNE vs raw IBM    : {recov:+.4f} ROC-AUC  (noise mitigation gain)")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    out_path = Path(args.output) if args.output else out_dir / "ibm_zne_metrics.json"
    results = {
        "zne_metrics":    metrics_zne,
        "scale1_metrics": metrics_scale1,
        "scale3_metrics": metrics_scale3,
        "simulator_roc_auc": sim_roc,
        "delta_zne_vs_sim":  round(metrics_zne.get("roc_auc", 0) - (sim_roc or 0), 4),
        "delta_zne_vs_raw":  round(metrics_zne.get("roc_auc", 0) - metrics_scale1.get("roc_auc", 0), 4),
        "n_states_clipped":  n_clipped,
        "shots": args.shots,
        "ensemble_members_run": len(theta_list),
    }
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
