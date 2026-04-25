"""
ibm_topo.py — Topology-matched QCBM for IBM quantum hardware.

Workflow
--------
1. Fetch the IBM backend coupling map (or load a saved topology from --topo-file).
2. Greedily select a 15-qubit connected subgraph (max-connectivity frontier).
3. Extract entanglement_pairs from the subgraph edges.
4. Retrain the QCBM with hardware-native CNOT topology on Aer simulator.
5. Transpile with initial_layout to eliminate SWAP overhead.
6. Submit to IBM hardware; compare circuit depth and ROC-AUC vs circular baseline.

Usage
-----
# Full pipeline: topology → retrain → IBM job → score
python ibm_topo.py --backend ibm_fez

# Fetch and save topology only (no training, no IBM job)
python ibm_topo.py --topology-only --backend ibm_fez

# Retrain only, using a previously saved topology (no IBM job)
python ibm_topo.py --train-only

# Submit saved theta to IBM without retraining
python ibm_topo.py --submit-only --backend ibm_fez

# Score only (ibm_dist.npy already saved from a prior run)
python ibm_topo.py --score-only

Requirements
------------
pip install qiskit qiskit-ibm-runtime qiskit-aer python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARTIFACT_DIR      = Path("artifacts/topo_circuit")
BEST_CONFIG       = Path("best_config.json")
BEST_QCBM_CONFIG  = Path("artifacts/best_run/hier_qcbm_config.json")
BASE_IBM_METRICS  = Path("artifacts/best_run/ibm_score_metrics.json")
BASE_IBM_RESULTS  = Path("artifacts/best_run/ibm_results.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Topology-matched QCBM: hardware-native entanglement for IBM devices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", default=None,
                   help="IBM backend name (e.g. ibm_fez, ibm_kingston). "
                        "Required for --topology-only and IBM job steps.")
    p.add_argument("--shots", type=int, default=32768,
                   help="Measurement shots for the IBM hardware job.")
    p.add_argument("--token", default=None,
                   help="IBM Quantum API token (overrides .env / IBM_QUANTUM_TOKEN env var).")
    p.add_argument("--topo-file", default=str(ARTIFACT_DIR / "topology.json"),
                   help="Saved topology JSON (used when --backend is absent).")
    p.add_argument("--config", default=str(BEST_CONFIG),
                   help="Preprocessing config JSON (features, bins, scaler, etc.).")
    p.add_argument("--qcbm-config", default=str(BEST_QCBM_CONFIG),
                   help="QCBM training hyperparams JSON (n_layers, spsa_a, etc.).")
    p.add_argument("--max-iter", type=int, default=None,
                   help="Override max training iterations (default: from --qcbm-config).")
    p.add_argument("--n-qubits", type=int, default=15,
                   help="Number of qubits for the subgraph search.")
    # Mode flags
    p.add_argument("--topology-only", action="store_true",
                   help="Fetch topology and exit (no training, no IBM job).")
    p.add_argument("--train-only", action="store_true",
                   help="Retrain QCBM with saved topology, then exit.")
    p.add_argument("--submit-only", action="store_true",
                   help="Skip training; submit saved topo_theta.npy to IBM hardware.")
    p.add_argument("--score-only", action="store_true",
                   help="Skip everything; score saved ibm_dist.npy and print comparison.")
    p.add_argument("--list-backends", action="store_true",
                   help="Print available IBM backends and exit.")
    return p


# ---------------------------------------------------------------------------
# IBM helpers
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


def get_service(token: str | None):
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        sys.exit("ERROR: qiskit-ibm-runtime not installed. Run: pip install qiskit-ibm-runtime")
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    return QiskitRuntimeService(channel="ibm_quantum_platform")


# ---------------------------------------------------------------------------
# Step 1: Topology discovery
# ---------------------------------------------------------------------------

def find_connected_subgraph(backend, n_qubits: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Greedy max-connectivity subgraph of n_qubits from the IBM heavy-hex coupling map.

    Strategy: start at the highest-degree qubit, then repeatedly add the frontier
    qubit with the most edges into the already-selected set.  This keeps the region
    contiguous and maximises internal connectivity.

    Returns
    -------
    phys_qubits : sorted list of physical qubit indices.
                  phys_qubits[logical_i] = physical_i → use as initial_layout.
    pairs       : (logical_ctrl, logical_tgt) tuples, one per undirected edge.
    """
    coupling = list(backend.coupling_map)  # [[a, b], ...] directed edges
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in coupling:
        adj[a].add(b)
        adj[b].add(a)

    all_nodes = set(adj.keys())
    start = max(all_nodes, key=lambda q: len(adj[q]))
    selected: set[int] = {start}
    frontier: set[int] = set(adj[start])

    while len(selected) < n_qubits and frontier:
        best = max(frontier, key=lambda q: len(adj[q] & selected))
        selected.add(best)
        frontier.discard(best)
        frontier |= (adj[best] - selected)

    if len(selected) < n_qubits:
        raise RuntimeError(
            f"Only {len(selected)}-qubit connected subgraph found (need {n_qubits}). "
            f"Backend '{backend.name}' has {backend.num_qubits} qubits."
        )

    phys_qubits = sorted(selected)
    p2l = {p: l for l, p in enumerate(phys_qubits)}

    pairs: list[tuple[int, int]] = []
    for p in phys_qubits:
        for nb in adj[p]:
            if nb in selected:
                la, lb = p2l[p], p2l[nb]
                if la < lb:
                    pairs.append((la, lb))
    pairs.sort()
    return phys_qubits, pairs


def save_topology(phys_qubits: list[int], pairs: list[tuple[int, int]],
                  backend_name: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    topo = {
        "backend": backend_name,
        "n_qubits": len(phys_qubits),
        "physical_qubits": phys_qubits,
        "entanglement_pairs": [list(p) for p in pairs],
        "n_edges_topo": len(pairs),
        "n_edges_circular": len(phys_qubits),
    }
    out = ARTIFACT_DIR / "topology.json"
    with out.open("w") as f:
        json.dump(topo, f, indent=2)
    print(f"  Saved topology : {out}")
    print(f"  Physical qubits: {phys_qubits}")
    print(f"  Topo edges     : {len(pairs)}  (circular ring has {len(phys_qubits)})")


def load_topology(path: str) -> tuple[list[int], list[tuple[int, int]], str]:
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"ERROR: Topology file not found: {p}\n"
            "Fetch it first: python ibm_topo.py --topology-only --backend <name>"
        )
    with p.open() as f:
        topo = json.load(f)
    phys = topo["physical_qubits"]
    pairs = [tuple(e) for e in topo["entanglement_pairs"]]
    backend_name = topo.get("backend", "unknown")
    print(f"  Loaded topology : {p}  (backend={backend_name}, edges={len(pairs)})")
    return phys, pairs, backend_name


# ---------------------------------------------------------------------------
# Step 2: Preprocessing
# ---------------------------------------------------------------------------

def run_preprocessing(cfg: dict):
    """Reproduce the training split and bitstring encoding.

    Returns
    -------
    bit_train_normal  : np.ndarray  normal training bitstrings
    bit_train_anomaly : np.ndarray  anomaly training bitstrings
    bit_val           : np.ndarray
    bit_test          : np.ndarray
    y_val             : pd.Series
    y_test            : pd.Series
    features          : list[str]
    """
    import pandas as pd
    from src.data.preprocessing import (
        add_categorical_features, apply_log1p, select_features,
        DEFAULT_LOG1P_COLS, Scaler,
    )
    from src.training_setup import train_val_test_split, filter_normal
    from src.discretize import auto_mixed_precision_map, fit_bins, transform_bins, encode_bits

    input_path = cfg.get("input", "datasets/UNSW-NB15_cleaned.csv")
    label_col  = cfg.get("label_col", "label")
    seed       = cfg.get("seed", 42)

    print(f"    Loading dataset: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    df = add_categorical_features(df)

    features = [f.strip() for f in cfg["features"].split(",") if f.strip()]
    print(f"    Features ({len(features)}): {', '.join(features[:5])}"
          f"{'...' if len(features) > 5 else ''}")

    X = select_features(df, features)
    y = df[label_col]

    splits = train_val_test_split(
        X, y,
        test_frac=cfg.get("test_frac", 0.2),
        val_frac=cfg.get("val_frac", 0.1),
        seed=seed,
        stratify=True,
    )

    if cfg.get("log1p", True):
        splits.X_train = apply_log1p(splits.X_train, DEFAULT_LOG1P_COLS)
        splits.X_val   = apply_log1p(splits.X_val,   DEFAULT_LOG1P_COLS)
        splits.X_test  = apply_log1p(splits.X_test,  DEFAULT_LOG1P_COLS)

    scaler = Scaler(mode=cfg.get("scaler", "standard")).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
    X_val   = scaler.transform(splits.X_val,   features)
    X_test  = scaler.transform(splits.X_test,  features)

    use_amp  = cfg.get("auto_mixed_precision", False)
    bits_map = bins_map = None
    if use_amp:
        bits_map, bins_map = auto_mixed_precision_map(
            X_train, features,
            continuous_bits=cfg.get("bits_per_feature", 2),
            continuous_bins=cfg.get("n_bins", 4),
        )

    y_train_reset = splits.y_train.reset_index(drop=True)
    anomaly_mask  = (y_train_reset.to_numpy() == 1)
    X_train_anom  = X_train.iloc[anomaly_mask] if anomaly_mask.any() else None

    edges = fit_bins(
        X_train, features,
        n_bins=cfg.get("n_bins", 4),
        strategy=cfg.get("bin_strategy", "quantile"),
        n_bins_map=bins_map,
        df_anomaly=X_train_anom,
    )

    btrain = transform_bins(X_train, edges)
    bval   = transform_bins(X_val,   edges)
    btest  = transform_bins(X_test,  edges)

    enc_kw = dict(
        bits_per_feature=cfg.get("bits_per_feature", 2),
        encoding=cfg.get("encoding", "binary"),
        n_bins=cfg.get("n_bins", 4),
        bits_per_feature_map=bits_map,
    )
    bit_train = encode_bits(btrain, **enc_kw)
    bit_val   = encode_bits(bval,   **enc_kw)
    bit_test  = encode_bits(btest,  **enc_kw)

    normal_df, anom_df = filter_normal(pd.DataFrame(bit_train), y_train_reset)
    bit_train_normal  = normal_df.to_numpy()
    bit_train_anomaly = anom_df.to_numpy() if len(anom_df) > 0 else np.empty((0, bit_train.shape[1]))

    return (
        bit_train_normal, bit_train_anomaly,
        bit_val, bit_test,
        splits.y_val.reset_index(drop=True),
        splits.y_test.reset_index(drop=True),
        features,
    )


# ---------------------------------------------------------------------------
# Step 3: Train with hardware-native topology
# ---------------------------------------------------------------------------

def train_topology_qcbm(pairs: list[tuple[int, int]], cfg: dict, qcbm_cfg: dict,
                        max_iter_override: int | None) -> dict:
    from src.qcbm_train import QCBMConfig, train_qcbm

    print("\n  Preprocessing data for retraining...")
    bit_normal, bit_anomaly, _, _, _, _, _ = run_preprocessing(cfg)
    print(f"    Normal train samples  : {len(bit_normal):,}")
    print(f"    Anomaly train samples : {len(bit_anomaly):,}")

    n_qubits = qcbm_cfg.get("n_qubits", 15)
    n_layers = qcbm_cfg.get("n_layers", 3)
    max_iter = max_iter_override or qcbm_cfg.get("max_iter", 500)

    config = QCBMConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        max_iter=max_iter,
        seed=qcbm_cfg.get("seed", 42),
        spsa_a=qcbm_cfg.get("spsa_a", 0.2),
        spsa_c=qcbm_cfg.get("spsa_c", 0.1),
        lambda_contrast=qcbm_cfg.get("lambda_contrast", 0.8),
        contrast_margin=qcbm_cfg.get("contrast_margin", 15.0),
        laplace_alpha=qcbm_cfg.get("laplace_alpha", 1.0),
        per_sample_contrast=qcbm_cfg.get("per_sample_contrast", False),
        warmstart_layers=False,
        use_rzz=False,
        optimizer=qcbm_cfg.get("optimizer", "spsa"),
        adam_lr=qcbm_cfg.get("adam_lr", 0.01),
        adam_beta1=qcbm_cfg.get("adam_beta1", 0.9),
        adam_beta2=qcbm_cfg.get("adam_beta2", 0.999),
        entanglement_pairs=pairs,
    )

    anomaly_arg = bit_anomaly if len(bit_anomaly) > 0 else None

    print(f"\n  Training QCBM — topology-matched entanglement")
    print(f"    Qubits   : {n_qubits}    Layers : {n_layers}    Max iter : {max_iter}")
    print(f"    CNOT pairs: {len(pairs)} (topo)  vs {n_qubits} (circular ring)")
    result = train_qcbm(bit_normal, config, anomaly_bitstrings=anomaly_arg)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(ARTIFACT_DIR / "topo_theta.npy"), result["theta"])
    np.save(str(ARTIFACT_DIR / "topo_model_dist.npy"), result["model_dist"])

    saved_cfg = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "use_rzz": False,
        "entanglement_pairs": [list(p) for p in pairs],
        "max_iter": max_iter,
        "optimizer": qcbm_cfg.get("optimizer", "spsa"),
        "final_loss": result["loss"],
    }
    with (ARTIFACT_DIR / "topo_qcbm_config.json").open("w") as f:
        json.dump(saved_cfg, f, indent=2)

    print(f"\n  Final training loss : {result['loss']:.6f}")
    if result.get("anomaly_kl") is not None:
        print(f"  Final anomaly KL    : {result['anomaly_kl']:.6f}")
    print(f"  Artifacts saved to : {ARTIFACT_DIR}/")
    return result


# ---------------------------------------------------------------------------
# Step 4: IBM hardware inference with initial_layout
# ---------------------------------------------------------------------------

def run_ibm_inference(theta: np.ndarray, phys_qubits: list[int],
                      pairs: list[tuple[int, int]], qcbm_cfg: dict,
                      backend_name: str, token: str | None, shots: int) -> np.ndarray:
    """Submit topology-matched circuit to IBM with initial_layout.

    Passing initial_layout=phys_qubits tells the Qiskit transpiler exactly which
    physical qubits to use.  Because our CNOT pairs already match the hardware
    coupling map, the transpiler adds zero SWAP gates for the entanglement layer.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        from qiskit import transpile
    except ImportError:
        sys.exit("ERROR: qiskit-ibm-runtime not installed. Run: pip install qiskit-ibm-runtime")

    from src.qcbm_train import build_ansatz

    n_qubits = qcbm_cfg.get("n_qubits", 15)
    n_layers = qcbm_cfg.get("n_layers", 3)

    # Topology-matched circuit
    qc_topo = build_ansatz(n_qubits, n_layers, theta, entanglement_pairs=pairs)
    qc_topo.measure_all()

    # Circular baseline circuit (same theta, different CNOT topology) — depth compare only
    qc_circ = build_ansatz(n_qubits, n_layers, theta, entanglement_pairs=None)
    qc_circ.measure_all()

    service = get_service(token)
    backend = service.backend(backend_name)
    print(f"  [IBM] Backend        : {backend.name}  ({backend.num_qubits} qubits)")

    # Topology: transpile with initial_layout → SWAPs only for same-device calibration noise
    qc_topo_t = transpile(qc_topo, backend=backend,
                          initial_layout=phys_qubits, optimization_level=3)
    # Circular: no initial_layout → transpiler must route the ring through heavy-hex
    qc_circ_t = transpile(qc_circ, backend=backend, optimization_level=3)

    depth_topo = qc_topo_t.depth()
    depth_circ = qc_circ_t.depth()
    reduction  = (depth_circ - depth_topo) / depth_circ * 100 if depth_circ > 0 else 0.0

    print(f"  [IBM] Raw circuit depth      : topo={qc_topo.depth()}  circ={qc_circ.depth()}")
    print(f"  [IBM] Transpiled depth       : topo={depth_topo}  circ={depth_circ}")
    print(f"  [IBM] Depth reduction        : {reduction:+.1f}%  ({'improved' if reduction > 0 else 'no improvement'})")

    depth_comparison = {
        "backend": backend.name,
        "initial_layout": phys_qubits,
        "n_cnot_pairs_topo": len(pairs),
        "n_cnot_pairs_circular": n_qubits,
        "topo_raw_depth": qc_topo.depth(),
        "circular_raw_depth": qc_circ.depth(),
        "topo_transpiled_depth": depth_topo,
        "circular_transpiled_depth": depth_circ,
        "depth_reduction_pct": round(reduction, 1),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_DIR / "depth_comparison.json").open("w") as f:
        json.dump(depth_comparison, f, indent=2)

    print(f"  [IBM] Submitting {shots:,}-shot job (topology circuit)...")
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_topo_t], shots=shots)
    print(f"  [IBM] Job ID         : {job.job_id()}")
    print(f"  [IBM] Waiting for result...")

    result = job.result()
    counts = result[0].data.meas.get_counts()

    # SamplerV2: bit 0 is at the LEFT (little-endian) → reverse to match Qiskit ordering
    n_states = 2 ** n_qubits
    probs = np.zeros(n_states)
    for bitstring, count in counts.items():
        idx = int(bitstring.replace(" ", "")[::-1], 2)
        probs[idx] = count / shots

    np.save(str(ARTIFACT_DIR / "ibm_dist.npy"), probs)
    with (ARTIFACT_DIR / "ibm_job_id.txt").open("w") as f:
        f.write(f"{job.job_id()}\n")

    coverage = float(np.sum(probs > 0)) / n_states
    print(f"  [IBM] States visited : {int(np.sum(probs > 0)):,} / {n_states:,}  ({coverage:.1%})")
    print(f"  [IBM] Distribution saved to {ARTIFACT_DIR}/ibm_dist.npy")
    return probs


# ---------------------------------------------------------------------------
# Step 5: Score and compare
# ---------------------------------------------------------------------------

def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def score_and_compare(cfg: dict) -> None:
    from src.score_eval import score_samples, evaluate

    ibm_dist_path = ARTIFACT_DIR / "ibm_dist.npy"
    if not ibm_dist_path.exists():
        sys.exit(f"ERROR: {ibm_dist_path} not found. Run the IBM job step first.")

    ibm_dist = np.load(str(ibm_dist_path))
    print(f"\n  Loaded IBM topo distribution: {len(ibm_dist):,} states")

    print("\n  Reproducing preprocessing for scoring...")
    _, _, _, bit_test, _, y_test, _ = run_preprocessing(cfg)

    scores = score_samples(bit_test, ibm_dist)
    ibm_metrics = evaluate(y_test.to_numpy(), scores)

    # Compare with simulator distribution if saved
    sim_dist_path = ARTIFACT_DIR / "topo_model_dist.npy"
    tvd_hw_vs_sim = None
    if sim_dist_path.exists():
        sim_dist = np.load(str(sim_dist_path))
        tvd_hw_vs_sim = total_variation_distance(ibm_dist, sim_dist)

    # Load circular baseline IBM metrics (from ibm_score.py run)
    baseline_roc = baseline_tvd = None
    if BASE_IBM_METRICS.exists():
        with BASE_IBM_METRICS.open() as f:
            base = json.load(f)
        baseline_roc = base.get("ibm_metrics", {}).get("roc_auc")
    if BASE_IBM_RESULTS.exists():
        with BASE_IBM_RESULTS.open() as f:
            base_res = json.load(f)
        baseline_tvd = base_res.get("tvd")

    # Load depth comparison
    depth_path = ARTIFACT_DIR / "depth_comparison.json"
    depth_info = {}
    if depth_path.exists():
        with depth_path.open() as f:
            depth_info = json.load(f)

    backend_name = depth_info.get("backend", "IBM hardware")

    print(f"\n{'=' * 65}")
    print(f"  Topology-matched QCBM vs Circular baseline — {backend_name}")
    print(f"{'=' * 65}")

    if depth_info:
        print(f"\n  Circuit depth (transpiled):")
        print(f"    Circular (baseline) : {depth_info.get('circular_transpiled_depth', 'n/a')}")
        print(f"    Topology-matched    : {depth_info.get('topo_transpiled_depth', 'n/a')}")
        print(f"    Reduction           : {depth_info.get('depth_reduction_pct', 'n/a'):+}%")
        print(f"    CNOT pairs — topo   : {depth_info.get('n_cnot_pairs_topo', 'n/a')}")
        print(f"    CNOT pairs — circ   : {depth_info.get('n_cnot_pairs_circular', 'n/a')}")

    print(f"\n  Distribution fidelity:")
    if tvd_hw_vs_sim is not None:
        print(f"    TVD (topo IBM vs sim) : {tvd_hw_vs_sim:.4f}")
    if baseline_tvd is not None:
        print(f"    TVD (circ IBM vs sim) : {baseline_tvd:.4f}  (baseline)")

    print(f"\n  {'Metric':<14} {'Topo IBM':>10} {'Circ IBM':>10} {'Delta':>8}")
    print(f"  {'-' * 46}")

    def row(name, topo_val, base_val):
        t = f"{topo_val:.4f}" if topo_val is not None else "     n/a"
        b = f"{base_val:.4f}" if base_val is not None else "     n/a"
        d = f"{topo_val - base_val:+.4f}" if (topo_val is not None and base_val is not None) else ""
        print(f"  {name:<14} {t:>10} {b:>10} {d:>8}")

    row("ROC-AUC", ibm_metrics.get("roc_auc"), baseline_roc)
    row("PR-AUC",  ibm_metrics.get("pr_auc"),  None)
    print(f"{'=' * 65}")

    out = {
        "ibm_topo_metrics": ibm_metrics,
        "baseline_circular_metrics": {
            "roc_auc": baseline_roc,
            "tvd": baseline_tvd,
        },
        "topo_tvd_hw_vs_sim": tvd_hw_vs_sim,
        "delta_roc_auc": (
            round(ibm_metrics["roc_auc"] - baseline_roc, 4)
            if baseline_roc is not None else None
        ),
        "depth_comparison": depth_info,
    }
    out_path = ARTIFACT_DIR / "ibm_score_metrics.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    token = resolve_token(args.token)

    if args.list_backends:
        service = get_service(token)
        backends = service.backends(operational=True, simulator=False)
        print(f"\n{'Backend':<25} {'Qubits':>6}  Status")
        print("-" * 40)
        for b in sorted(backends, key=lambda x: x.num_qubits):
            print(f"{b.name:<25} {b.num_qubits:>6}  operational")
        return

    # Load preprocessing config
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"ERROR: Preprocessing config not found: {config_path}")
    with config_path.open() as f:
        cfg = json.load(f)

    # Load QCBM training hyperparams
    qcbm_config_path = Path(args.qcbm_config)
    if not qcbm_config_path.exists():
        print(f"  WARNING: QCBM config not found at {qcbm_config_path}; using defaults.")
        qcbm_cfg: dict = {}
    else:
        with qcbm_config_path.open() as f:
            qcbm_cfg = json.load(f)

    # Override n_qubits from CLI
    qcbm_cfg["n_qubits"] = args.n_qubits

    # ------------------------------------------------------------------
    # Step 1: Topology
    # ------------------------------------------------------------------
    phys_qubits: list[int] | None = None
    pairs: list[tuple[int, int]] | None = None

    if args.score_only:
        # Scoring doesn't need topology to be loaded (it uses ibm_dist.npy)
        pass
    elif args.train_only or args.submit_only:
        phys_qubits, pairs, _ = load_topology(args.topo_file)
    else:
        # Fetch live topology
        if args.backend is None:
            sys.exit(
                "ERROR: --backend is required to fetch topology.\n"
                "Use --train-only / --score-only if you have saved artifacts."
            )
        print(f"\n[1] Fetching coupling map from {args.backend}...")
        service = get_service(token)
        backend_obj = service.backend(args.backend)
        phys_qubits, pairs = find_connected_subgraph(backend_obj, args.n_qubits)
        save_topology(phys_qubits, pairs, args.backend)

        if args.topology_only:
            print("\nTopology saved. Exiting (--topology-only).")
            return

    # ------------------------------------------------------------------
    # Step 2: Retrain
    # ------------------------------------------------------------------
    if not args.submit_only and not args.score_only:
        print(f"\n[2] Training QCBM with topology-matched entanglement...")
        train_topology_qcbm(pairs, cfg, qcbm_cfg, args.max_iter)
        if args.train_only:
            print("\nTraining complete. Exiting (--train-only).")
            return

    # ------------------------------------------------------------------
    # Step 3: IBM inference
    # ------------------------------------------------------------------
    if not args.score_only:
        theta_path = ARTIFACT_DIR / "topo_theta.npy"
        if not theta_path.exists():
            sys.exit(
                f"ERROR: {theta_path} not found.\n"
                "Train first: python ibm_topo.py --train-only"
            )
        theta = np.load(str(theta_path))

        if args.backend is None:
            sys.exit("ERROR: --backend is required to submit an IBM job.")

        if phys_qubits is None or pairs is None:
            phys_qubits, pairs, _ = load_topology(args.topo_file)

        backend_name = args.backend
        print(f"\n[3] Submitting topology-matched circuit to IBM hardware ({backend_name})...")
        run_ibm_inference(theta, phys_qubits, pairs, qcbm_cfg, backend_name, token, args.shots)

    # ------------------------------------------------------------------
    # Step 4: Score
    # ------------------------------------------------------------------
    print(f"\n[4] Scoring test set with IBM topology distribution...")
    score_and_compare(cfg)


if __name__ == "__main__":
    main()
