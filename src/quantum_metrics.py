"""
Quantum-specific metrics for the QCBM circuit:
  - Expressibility: how uniformly the circuit samples the Hilbert space
  - Entanglement entropy: average von Neumann entropy across qubit bipartitions
"""
from __future__ import annotations

import numpy as np


def expressibility(
    n_qubits: int,
    n_layers: int,
    theta: np.ndarray,
    n_samples: int = 1000,
    use_rzz: bool = False,
    entanglement_pairs=None,
    seed: int = 42,
) -> float:
    """Expressibility of the ansatz via KL divergence from the Haar measure.

    Samples n_samples random parameter vectors, computes the resulting state
    fidelity distribution, and compares it to the Haar-random fidelity
    distribution (which is known analytically: P_Haar(F) = (2^n - 1)(1-F)^(2^n - 2)).

    A lower KL divergence = more expressive (closer to Haar-random).
    Returns the KL(empirical || Haar) value.

    Reference: Sim et al., "Expressibility and entangling capability of
    parameterized quantum circuits for hybrid quantum-classical algorithms",
    Advanced Quantum Technologies 2019.
    """
    from src.qcbm_train import build_ansatz, _get_aer_simulator, _import_qiskit
    from qiskit import transpile

    rng = np.random.default_rng(seed)
    n_params_per = len(theta)
    simulator = _get_aer_simulator()
    QuantumCircuit, Statevector = _import_qiskit()

    def get_statevector(params):
        qc = build_ansatz(n_qubits, n_layers, params, use_rzz=use_rzz,
                          entanglement_pairs=entanglement_pairs)
        if simulator is not None:
            qc_sv = qc.copy(); qc_sv.save_statevector()
            tqc = transpile(qc_sv, simulator, optimization_level=0)
            result = simulator.run(tqc).result()
            return np.array(result.get_statevector())
        else:
            sv = Statevector.from_instruction(qc)
            return np.array(sv.data)

    # Sample random parameter pairs and compute fidelities
    fidelities = []
    for _ in range(n_samples):
        t1 = rng.uniform(0, 2 * np.pi, size=n_params_per)
        t2 = rng.uniform(0, 2 * np.pi, size=n_params_per)
        sv1 = get_statevector(t1)
        sv2 = get_statevector(t2)
        F = float(np.abs(np.dot(sv1.conj(), sv2)) ** 2)
        fidelities.append(np.clip(F, 0.0, 1.0))

    fidelities = np.array(fidelities)
    dim = 2 ** n_qubits

    # Empirical fidelity histogram
    n_bins = 75
    bins = np.linspace(0, 1, n_bins + 1)
    emp_counts, _ = np.histogram(fidelities, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    # Haar-random distribution: P(F) = (d-1)(1-F)^(d-2), d = 2^n_qubits
    haar = (dim - 1) * (1 - bin_centers) ** (dim - 2)
    haar = np.clip(haar, 1e-12, None)
    haar /= haar.sum()  # normalise to sum-to-1 over bins

    emp = np.clip(emp_counts * bin_width, 1e-12, None)
    emp /= emp.sum()

    # KL(empirical || Haar)
    kl = float(np.sum(emp * np.log(emp / haar)))
    return kl, fidelities


def entanglement_entropy(
    theta: np.ndarray,
    n_qubits: int,
    n_layers: int,
    use_rzz: bool = False,
    entanglement_pairs=None,
) -> dict:
    """Compute average von Neumann entanglement entropy across all bipartitions.

    For each qubit i, computes the entanglement entropy S(rho_A) where A = {i}
    and B = all other qubits. A high entropy means qubit i is strongly entangled
    with the rest of the system — a genuinely quantum property.

    Returns:
        dict with keys:
          'per_qubit': list of S(rho_i) for each qubit
          'mean': average across all qubits
          'max': maximum single-qubit entropy (log2 scale, max=1.0 for maximally entangled)
    """
    from src.qcbm_train import build_ansatz, _get_aer_simulator, _import_qiskit
    from qiskit import transpile

    simulator = _get_aer_simulator()
    QuantumCircuit, Statevector = _import_qiskit()

    qc = build_ansatz(n_qubits, n_layers, theta, use_rzz=use_rzz,
                      entanglement_pairs=entanglement_pairs)

    if simulator is not None:
        qc_sv = qc.copy(); qc_sv.save_statevector()
        tqc = transpile(qc_sv, simulator, optimization_level=0)
        result = simulator.run(tqc).result()
        sv = np.array(result.get_statevector())
    else:
        sv_obj = Statevector.from_instruction(qc)
        sv = np.array(sv_obj.data)

    # Reshape into tensor: (2, 2, ..., 2) with n_qubits indices
    psi = sv.reshape([2] * n_qubits)

    entropies = []
    for qubit in range(n_qubits):
        # Move target qubit to axis 0, reshape into (2, 2^(n-1))
        axes = [qubit] + [i for i in range(n_qubits) if i != qubit]
        psi_T = np.transpose(psi, axes).reshape(2, -1)

        # Singular value decomposition -> Schmidt coefficients
        _, s, _ = np.linalg.svd(psi_T, full_matrices=False)
        lambdas = s ** 2
        lambdas = lambdas[lambdas > 1e-15]

        # Von Neumann entropy in bits (log2)
        S = float(-np.sum(lambdas * np.log2(lambdas + 1e-15)))
        entropies.append(S)

    return {
        "per_qubit": entropies,
        "mean": float(np.mean(entropies)),
        "max":  float(np.max(entropies)),
        "min":  float(np.min(entropies)),
    }
