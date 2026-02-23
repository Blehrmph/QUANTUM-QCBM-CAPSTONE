from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discretize import bitstrings_to_indices


@dataclass
class QCBMConfig:
    n_qubits: int
    n_layers: int = 2
    max_iter: int = 200
    seed: int = 42
    spsa_a: float = 0.2
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


def build_ansatz(n_qubits: int, n_layers: int, theta: np.ndarray):
    QuantumCircuit, _ = _import_qiskit()
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
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
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    for k in range(1, max_iter + 1):
        ak = a / (k ** 0.602)
        ck = c / (k ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        loss_plus = loss_fn(theta + ck * delta)
        loss_minus = loss_fn(theta - ck * delta)
        ghat = (loss_plus - loss_minus) / (2.0 * ck * delta)
        theta = theta - ak * ghat
    return theta


def train_qcbm(bitstrings: np.ndarray, config: QCBMConfig) -> dict:
    n_qubits = config.n_qubits
    data_dist = empirical_distribution(bitstrings, n_qubits)
    rng = np.random.default_rng(config.seed)
    theta0 = rng.normal(scale=0.1, size=config.n_qubits * config.n_layers)

    def loss_fn(theta: np.ndarray) -> float:
        model_dist = qcbm_distribution(theta, config)
        return kl_divergence(data_dist, model_dist)

    theta = spsa_optimize(
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
    }
