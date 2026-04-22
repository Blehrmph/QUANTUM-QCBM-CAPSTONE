"""
CIRCUIT/generate_circuits.py  —  Generate final QCBM circuit diagram

Final configuration:
  - 15 qubits (Auto Mixed Precision)
      q0-q1   : sbytes   (2 bits / 4 bins)
      q2-q3   : Sload    (2 bits / 4 bins)
      q4-q5   : dbytes   (2 bits / 4 bins)
      q6-q7   : Dload    (2 bits / 4 bins)
      q8-q9   : Dpkts    (2 bits / 4 bins)
      q10-q11 : sttl     (2 bits / 4 bins)
      q12     : is_not_tcp    (1 bit / 2 bins)
      q13     : is_int_state  (1 bit / 2 bins)
      q14     : is_con_state  (1 bit / 2 bins)
  - 3 layers: RZ-RY-RZ per qubit + circular CNOT entanglement
  - Anomaly-aware binning + contrastive loss (lambda=0.8, margin=15)
  - ADAM lr=0.003, 1500 iterations, ensemble=5, warm-start 2->3

Usage:
    python CIRCUIT/generate_circuits.py
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit

OUT = os.path.dirname(os.path.abspath(__file__))

N_QUBITS = 15
N_LAYERS = 3

QUBIT_LABELS = [
    "sbytes[0]", "sbytes[1]",
    "Sload[0]",  "Sload[1]",
    "dbytes[0]", "dbytes[1]",
    "Dload[0]",  "Dload[1]",
    "Dpkts[0]",  "Dpkts[1]",
    "sttl[0]",   "sttl[1]",
    "is_!tcp",   "is_int_st", "is_con_st",
]


def build_final_circuit(param_val: float = 0.3) -> QuantumCircuit:
    """Build the final 15-qubit, 3-layer QCBM ansatz for visualisation."""
    n = N_QUBITS
    qc = QuantumCircuit(n)

    # Fixed representative angles
    theta = np.full(n * 3 * N_LAYERS + n, param_val)
    idx = 0

    for layer in range(N_LAYERS):
        qc.barrier(label=f"L{layer + 1}")
        # RZ-RY-RZ block per qubit
        for q in range(n):
            qc.rz(theta[idx],     q); idx += 1
            qc.ry(theta[idx],     q); idx += 1
            qc.rz(theta[idx],     q); idx += 1
        # Circular CNOT entanglement
        for q in range(n):
            qc.cx(q, (q + 1) % n)

    # Final measurement layer (RY)
    qc.barrier(label="Meas")
    for q in range(n):
        qc.ry(theta[idx], q); idx += 1

    return qc


def generate():
    print("Generating final QCBM circuit diagram...")

    qc = build_final_circuit()

    fig = qc.draw(
        "mpl",
        style="iqp",
        fold=80,
        initial_state=False,
        wire_order=list(range(N_QUBITS)),
    )

    fig.suptitle(
        "Final QCBM Ansatz — 15 Qubits · 3 Layers · Circular CNOT",
        fontsize=13, fontweight="bold", y=1.02,
    )

    subtitle_lines = (
        "Auto Mixed Precision: continuous features (incl. sttl) -> 2 bits/4 bins  |  "
        "binary flags -> 1 bit/2 bins  |  15 qubits / 32,768 states\n"
        "Loss = KL(normal || model) + 0.8 * max(0, 15 - KL(anomaly || model))  |  "
        "Anomaly-aware binning  |  ADAM lr=3e-3  1500 iter  ensemble=5  warm-start 2->3\n"
        "Best results (LR+Isotonic):  ROC-AUC 0.9671  PR-AUC 0.8931  F1 0.9015  "
        "Recall 90.8%  FAR 1.31%  MCC 0.8893"
    )
    fig.text(0.5, -0.04, subtitle_lines, ha="center", fontsize=8.5,
             style="italic", color="#333333")

    fig.tight_layout()

    out_path = os.path.join(OUT, "qcbm_final_circuit.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    generate()
