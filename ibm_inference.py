"""
ibm_inference.py  —  Run a trained QCBM on real IBM quantum hardware.

Loads a previously trained theta from the artifacts directory, submits the
circuit to an IBM quantum device, and compares the resulting shot-sampled
distribution against the local Aer simulator baseline.

Usage
-----
# Quick start — uses least-busy device, reads token from .env
python ibm_inference.py

# Specify artifact directory and shot count
python ibm_inference.py --artifact-dir artifacts/best_run --shots 32768

# Pin a specific backend
python ibm_inference.py --backend ibm_brisbane --shots 65536

# List available backends without running a job
python ibm_inference.py --list-backends

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
        description="Run trained QCBM on IBM quantum hardware and compare with simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--artifact-dir", default="artifacts",
        help="Directory containing hier_qcbm_theta.npy and hier_qcbm_config.json.",
    )
    p.add_argument(
        "--shots", type=int, default=32768,
        help="Number of measurement shots. More = better probability estimate.",
    )
    p.add_argument(
        "--backend", default=None,
        help="IBM backend name (e.g. ibm_brisbane). Omit to use least-busy device.",
    )
    p.add_argument(
        "--token", default=None,
        help="IBM Quantum API token. Overrides .env / IBM_QUANTUM_TOKEN env var.",
    )
    p.add_argument(
        "--list-backends", action="store_true",
        help="Print available IBM backends and exit (no job submitted).",
    )
    p.add_argument(
        "--ensemble-member", type=int, default=None,
        help="Run only this ensemble member index (0-based). "
             "Omit to run all members and average (default behaviour).",
    )
    p.add_argument(
        "--output", default=None,
        help="JSON file to save results (default: <artifact-dir>/ibm_results.json).",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader — sets IBM_QUANTUM_TOKEN if found."""
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value


def resolve_token(cli_token: str | None) -> str | None:
    """Return token: CLI arg > IBM_QUANTUM_TOKEN env var."""
    return cli_token or os.environ.get("IBM_QUANTUM_TOKEN")


def load_artifacts(artifact_dir: str) -> tuple[np.ndarray, dict]:
    """Load theta array and config dict from the artifact directory."""
    d = Path(artifact_dir)
    theta_path = d / "hier_qcbm_theta.npy"
    config_path = d / "hier_qcbm_config.json"

    if not theta_path.exists():
        sys.exit(
            f"ERROR: {theta_path} not found.\n"
            "Run main.py first to train the QCBM and save artifacts."
        )
    if not config_path.exists():
        sys.exit(f"ERROR: {config_path} not found.")

    theta = np.load(str(theta_path))
    with config_path.open() as f:
        config_dict = json.load(f)

    return theta, config_dict


def load_simulator_dist(artifact_dir: str) -> np.ndarray | None:
    """Load the saved Aer simulator distribution for comparison (may not exist)."""
    path = Path(artifact_dir) / "hier_qcbm_model_dist.npy"
    if path.exists():
        return np.load(str(path))
    return None


def list_backends(token: str | None) -> None:
    """Print all operational real IBM backends and exit."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        sys.exit("ERROR: qiskit-ibm-runtime not installed. Run: pip install qiskit-ibm-runtime")

    if token:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")

    backends = service.backends(operational=True, simulator=False)
    print(f"\n{'Backend':<25} {'Qubits':>6}  {'Status'}")
    print("-" * 45)
    for b in sorted(backends, key=lambda x: x.num_qubits):
        print(f"{b.name:<25} {b.num_qubits:>6}  operational")
    print()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TVD = 0.5 * sum |p_i - q_i|. Range [0, 1]; 0 = identical distributions."""
    return float(0.5 * np.sum(np.abs(p - q)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q). Returns inf if q assigns 0 probability to states p visits."""
    q = np.clip(q, eps, 1.0)
    p_safe = np.clip(p, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q)))


def top_k_overlap(p: np.ndarray, q: np.ndarray, k: int = 20) -> float:
    """Fraction of the top-k states in p that also appear in top-k states of q."""
    top_p = set(np.argsort(p)[-k:])
    top_q = set(np.argsort(q)[-k:])
    return len(top_p & top_q) / k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()  # populate IBM_QUANTUM_TOKEN from .env if present

    args = build_parser().parse_args()
    token = resolve_token(args.token)

    if not token:
        print(
            "WARNING: No IBM Quantum token found.\n"
            "  Option 1 (recommended): Edit .env → set IBM_QUANTUM_TOKEN=your_token\n"
            "  Option 2: pass --token YOUR_TOKEN on the command line\n"
            "  Option 3: run QiskitRuntimeService.save_account(...) once in Python\n"
        )

    if args.list_backends:
        list_backends(token)
        return

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    print(f"Loading artifacts from: {args.artifact_dir}")
    theta, config_dict = load_artifacts(args.artifact_dir)
    sim_dist_raw = load_simulator_dist(args.artifact_dir)

    from src.qcbm_train import QCBMConfig, qcbm_distribution_ibm

    config = QCBMConfig(
        n_qubits=config_dict["n_qubits"],
        n_layers=config_dict["n_layers"],
        use_rzz=config_dict.get("use_rzz", False),
        entanglement_pairs=config_dict.get("entanglement_pairs"),
    )

    # theta may be 1D (single model) or 2D (ensemble), shape (n_members, n_params)
    is_ensemble = theta.ndim == 2
    if is_ensemble:
        n_members = theta.shape[0]
        n_params_each = theta.shape[1]
        if args.ensemble_member is not None:
            if not (0 <= args.ensemble_member < n_members):
                sys.exit(f"ERROR: --ensemble-member must be 0..{n_members - 1}")
            theta_list = [theta[args.ensemble_member]]
            print(f"\nEnsemble: running member {args.ensemble_member} only (of {n_members})")
        else:
            theta_list = [theta[i] for i in range(n_members)]
            print(f"\nEnsemble: {n_members} members — will run all and average distributions")
    else:
        theta_list = [theta]
        n_params_each = len(theta)
        print()

    # Average ensemble simulator baseline if needed
    if sim_dist_raw is not None and sim_dist_raw.ndim == 2:
        sim_dist = sim_dist_raw.mean(axis=0)
    else:
        sim_dist = sim_dist_raw

    print(
        f"Circuit: {config.n_qubits} qubits  |  {config.n_layers} layers  |  "
        f"{n_params_each} parameters per member"
    )
    print(f"Shots  : {args.shots:,}")
    print(f"Backend: {args.backend or 'least-busy'}\n")

    # ------------------------------------------------------------------
    # Run on IBM hardware — one job per ensemble member, then average
    # ------------------------------------------------------------------
    member_dists = []
    for i, th in enumerate(theta_list):
        if len(theta_list) > 1:
            print(f"--- Ensemble member {i + 1}/{len(theta_list)} ---")
        member_dist = qcbm_distribution_ibm(
            th,
            config,
            shots=args.shots,
            backend_name=args.backend,
            token=token,
        )
        member_dists.append(member_dist)

    ibm_dist = np.mean(member_dists, axis=0) if len(member_dists) > 1 else member_dists[0]

    # ------------------------------------------------------------------
    # Compare distributions
    # ------------------------------------------------------------------
    n_states = 2 ** config.n_qubits
    coverage = float(np.sum(ibm_dist > 0)) / n_states

    print(f"\n{'=' * 55}")
    print("  Distribution comparison: IBM hardware vs Aer simulator")
    print(f"{'=' * 55}")
    print(f"  States visited (IBM)  : {int(np.sum(ibm_dist > 0)):,} / {n_states:,}  ({coverage:.1%})")

    results: dict = {
        "artifact_dir": str(args.artifact_dir),
        "shots": args.shots,
        "n_qubits": config.n_qubits,
        "n_layers": config.n_layers,
        "n_states": n_states,
        "ibm_coverage": coverage,
    }

    if sim_dist is not None:
        tvd = total_variation_distance(ibm_dist, sim_dist)
        kl  = kl_divergence(sim_dist, ibm_dist)
        ov  = top_k_overlap(sim_dist, ibm_dist, k=min(50, n_states))
        print(f"  TVD  (hardware vs sim): {tvd:.4f}  (0=identical, 1=completely different)")
        print(f"  KL   (sim || hardware): {kl:.4f}")
        print(f"  Top-50 overlap        : {ov:.1%}")
        results.update({"tvd": tvd, "kl_sim_vs_ibm": kl, "top50_overlap": ov})
    else:
        print("  (No Aer simulator baseline found — skipping comparison)")

    # Top-10 most probable states on hardware
    top10_idx = np.argsort(ibm_dist)[-10:][::-1]
    top10 = [
        {"state": format(int(i), f"0{config.n_qubits}b"), "prob": float(ibm_dist[i])}
        for i in top10_idx
    ]
    print(f"\n  Top-10 states on IBM hardware:")
    for entry in top10:
        bar = "#" * int(entry["prob"] * 200)
        print(f"    |{entry['state']}>  {entry['prob']:.5f}  {bar}")

    results["top10_states_ibm"] = top10

    if sim_dist is not None:
        top10_sim_idx = np.argsort(sim_dist)[-10:][::-1]
        top10_sim = [
            {"state": format(int(i), f"0{config.n_qubits}b"), "prob": float(sim_dist[i])}
            for i in top10_sim_idx
        ]
        print(f"\n  Top-10 states on Aer simulator (reference):")
        for entry in top10_sim:
            bar = "#" * int(entry["prob"] * 200)
            print(f"    |{entry['state']}>  {entry['prob']:.5f}  {bar}")
        results["top10_states_sim"] = top10_sim

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = Path(args.output) if args.output else Path(args.artifact_dir) / "ibm_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    ibm_dist_path = out_path.with_name("ibm_dist.npy")
    np.save(str(ibm_dist_path), ibm_dist)

    print(f"\n  Results saved to : {out_path}")
    print(f"  Distribution saved: {ibm_dist_path}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
