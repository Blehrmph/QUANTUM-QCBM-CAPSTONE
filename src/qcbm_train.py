from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.discretize import bitstrings_to_indices


@dataclass
class QCBMConfig:
    n_qubits: int
    n_layers: int = 2
    max_iter: int = 200
    seed: int = 42
    spsa_a: float = 0.628  # calibrated: first step ~0.3 rad
    spsa_c: float = 0.1


def _import_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except Exception as exc:
        raise ImportError(
            "Qiskit is required for QCBM training. Install qiskit to proceed."
        ) from exc
    return QuantumCircuit, Statevector


def n_params(n_qubits: int, n_layers: int) -> int:
    """Total number of trainable parameters for the ansatz.

    Each layer contributes 3 params per qubit (RZ-RY-RZ).
    A final RY layer adds n_qubits more params.
    """
    return n_qubits * n_layers * 3 + n_qubits


def build_ansatz(n_qubits: int, n_layers: int, theta: np.ndarray):
    """Hardware-efficient ansatz: RZ-RY-RZ rotations + circular CNOT entanglement.

    Improvements over the previous RY-only linear-CNOT circuit:
    - RZ-RY-RZ per qubit covers the full Bloch sphere (vs. single-axis RY).
    - Circular CNOT (q_last -> q_0 wrap-around) adds long-range correlations.
    - Final RY layer after the last entanglement block captures post-entanglement
      rotations that the old circuit was missing.
    """
    QuantumCircuit, _ = _import_qiskit()
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        # RZ-RY-RZ per qubit: 3 parameters each, spans full single-qubit SU(2)
        for q in range(n_qubits):
            qc.rz(theta[idx], q); idx += 1
            qc.ry(theta[idx], q); idx += 1
            qc.rz(theta[idx], q); idx += 1
        # Circular CNOT entanglement: linear chain + wrap-around
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)
    # Final RY layer (was missing in the old ansatz)
    for q in range(n_qubits):
        qc.ry(theta[idx], q)
        idx += 1
    return qc


def qcbm_distribution(theta: np.ndarray, config: QCBMConfig) -> np.ndarray:
    _, Statevector = _import_qiskit()
    qc = build_ansatz(config.n_qubits, config.n_layers, theta)
    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2
    return probs


def empirical_distribution(bitstrings: np.ndarray, n_qubits: int) -> np.ndarray:
    indices = bitstrings_to_indices(bitstrings)
    counts = np.bincount(indices, minlength=2**n_qubits).astype(float)
    if counts.sum() == 0:
        raise ValueError("No data to build empirical distribution.")
    return counts / counts.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    q = np.clip(q, eps, 1.0)
    p = np.clip(p, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def spsa_optimize(
    loss_fn,
    theta0: np.ndarray,
    max_iter: int,
    a: float,
    c: float,
    seed: int,
    patience: int = 50,
) -> tuple[np.ndarray, list[float]]:
    """SPSA optimizer with momentum and early stopping.

    Improvements over plain SPSA:
    - Momentum (beta=0.9): accumulates gradient direction across iterations,
      smoothing noisy SPSA estimates and speeding up convergence.
    - Best-theta tracking: returns the parameters at the lowest observed loss,
      not just the final iterate (which may have drifted due to noise).
    - Early stopping: halts when loss hasn't improved by >1e-5 for `patience`
      consecutive iterations, avoiding wasted computation on a plateau.
    - Progress logging every 50 iterations to monitor convergence.

    Returns
    -------
    best_theta : np.ndarray
        Parameters at the lowest observed training loss.
    loss_history : list[float]
        Per-iteration loss estimates for diagnostics.
    """
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    m = np.zeros_like(theta)          # momentum accumulator
    best_loss = np.inf
    best_theta = theta.copy()
    no_improve = 0
    loss_history: list[float] = []

    for k in range(1, max_iter + 1):
        ak = a / (k ** 0.602)
        ck = c / (k ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        loss_plus  = loss_fn(theta + ck * delta)
        loss_minus = loss_fn(theta - ck * delta)
        ghat = (loss_plus - loss_minus) / (2.0 * ck * delta)

        # Momentum: blend current gradient estimate with running average
        m = 0.9 * m + 0.1 * ghat
        theta = theta - ak * m

        current_loss = (loss_plus + loss_minus) / 2.0
        loss_history.append(current_loss)

        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            best_theta = theta.copy()
            no_improve = 0
        else:
            no_improve += 1

        if k % 50 == 0:
            print(f"  SPSA iter {k}/{max_iter}: loss={current_loss:.6f}  best={best_loss:.6f}")

        if no_improve >= patience:
            print(f"  Early stop at iter {k}  best_loss={best_loss:.6f}")
            break

    return best_theta, loss_history


def train_qcbm(bitstrings: np.ndarray, config: QCBMConfig) -> dict:
    n_qubits = config.n_qubits
    data_dist = empirical_distribution(bitstrings, n_qubits)
    rng = np.random.default_rng(config.seed)

    # Uniform [0, 2π] initialization breaks the near-zero symmetry that causes
    # barren plateaus with small-normal initialization.
    n_theta = n_params(n_qubits, config.n_layers)
    theta0 = rng.uniform(0, 2 * np.pi, size=n_theta)

    def loss_fn(theta: np.ndarray) -> float:
        model_dist = qcbm_distribution(theta, config)
        return kl_divergence(data_dist, model_dist)

    theta, loss_history = spsa_optimize(
        loss_fn,
        theta0=theta0,
        max_iter=config.max_iter,
        a=config.spsa_a,
        c=config.spsa_c,
        seed=config.seed,
    )

    model_dist = qcbm_distribution(theta, config)
    final_loss = kl_divergence(data_dist, model_dist)

    return {
        "theta": theta,
        "data_dist": data_dist,
        "model_dist": model_dist,
        "loss": final_loss,
        "loss_history": loss_history,
    }
