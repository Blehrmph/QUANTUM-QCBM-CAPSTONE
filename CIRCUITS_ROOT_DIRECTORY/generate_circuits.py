"""
Generate circuit diagrams for each phase of the QCBM Capstone project.
Run from the project root: python CIRCUITS_ROOT_DIRECTORY/generate_circuits.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
OUT = os.path.dirname(os.path.abspath(__file__))


def save_circuit(qc: QuantumCircuit, filename: str, title: str, subtitle: str = ""):
    fig = qc.draw("mpl", style="iqp", fold=60)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    if subtitle:
        fig.text(0.5, -0.02, subtitle, ha="center", fontsize=9, color="#444444",
                 wrap=True, style="italic")
    fig.tight_layout()
    path = os.path.join(OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────────────────────
# Helper: RZ-RY-RZ block for one qubit
def _rzryrz(qc, q, idx, theta):
    qc.rz(theta[idx],   q); idx += 1
    qc.ry(theta[idx],   q); idx += 1
    qc.rz(theta[idx],   q); idx += 1
    return idx


def build_circuit(n_qubits, n_layers, use_rzz=False, entanglement_pairs=None,
                  param_val=0.3):
    """Build a representative ansatz with fixed numeric angles for visualisation."""
    theta = np.full(
        n_qubits * 3 * n_layers + n_qubits + (n_qubits * n_layers if use_rzz else 0),
        param_val
    )
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        qc.barrier(label=f"L{layer+1}")
        for q in range(n_qubits):
            idx = _rzryrz(qc, q, idx, theta)
        if entanglement_pairs is not None:
            for ctrl, tgt in entanglement_pairs:
                qc.cx(ctrl, tgt)
        elif use_rzz:
            for q in range(n_qubits):
                qc.rzz(theta[idx], q, (q + 1) % n_qubits)
                idx += 1
        else:
            for q in range(n_qubits):
                qc.cx(q, (q + 1) % n_qubits)
    qc.barrier(label="Final")
    for q in range(n_qubits):
        qc.ry(theta[idx], q)
        idx += 1
    return qc


# ─────────────────────────────────────────────────────────────
# PHASE 1: Classical baseline — no quantum circuit
# We draw a simple diagram explaining the pipeline instead.
def phase1_diagram():
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    stages = [
        ("Raw\nFeatures", "#cfe2f3"),
        ("log1p +\nStandardScale", "#d9ead3"),
        ("Logistic\nRegression", "#fce5cd"),
        ("Anomaly\nScore", "#ead1dc"),
    ]
    x = 0.08
    for label, color in stages:
        box = mpatches.FancyBboxPatch((x, 0.25), 0.17, 0.5,
                                       boxstyle="round,pad=0.02",
                                       linewidth=1.5, edgecolor="#333",
                                       facecolor=color)
        ax.add_patch(box)
        ax.text(x + 0.085, 0.5, label, ha="center", va="center",
                fontsize=10, fontweight="bold")
        if x + 0.17 < 0.9:
            ax.annotate("", xy=(x + 0.2, 0.5), xytext=(x + 0.17, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
        x += 0.22
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Phase 1 — Classical Baseline (Logistic Regression)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02,
             "No quantum component yet. Establishes preprocessing pipeline and evaluation framework.",
             ha="center", fontsize=9, style="italic", color="#444")
    fig.tight_layout()
    path = os.path.join(OUT, "phase1_classical_baseline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: phase1_classical_baseline.png")


# ─────────────────────────────────────────────────────────────
# PHASE 2: Basic 8-qubit QCBM, 2 layers, circular CNOT, binary encoding
def phase2():
    qc = build_circuit(n_qubits=8, n_layers=2)
    qc.name = "Phase 2 — 8q QCBM"
    save_circuit(
        qc,
        "phase2_8q_qcbm_basic.png",
        "Phase 2 — 8-Qubit QCBM, 2 Layers, Circular CNOT",
        "Binary encoding (1 bit/feature). 8 features → 8 qubits → 256 states. "
        "SPSA optimizer. ROC-AUC ≈ 0.73"
    )


# ─────────────────────────────────────────────────────────────
# PHASE 3: 8-qubit QCBM, 3 layers + contrastive loss annotation
def phase3():
    qc = build_circuit(n_qubits=8, n_layers=3)
    # Add a visual annotation for contrastive loss
    qc.name = "Phase 3 — Contrastive QCBM"

    fig = qc.draw("mpl", style="iqp", fold=60)
    fig.suptitle("Phase 3 — 8-Qubit QCBM + Contrastive Loss", fontsize=13,
                 fontweight="bold", y=1.01)

    # Annotate contrastive loss formula
    fig.text(0.5, -0.04,
             "Loss = KL(normal ‖ model) + λ · max(0, margin − KL(anomaly ‖ model))   "
             "| Laplace smoothing α=1.0 | Youden threshold",
             ha="center", fontsize=8.5, style="italic", color="#333")
    fig.tight_layout()
    path = os.path.join(OUT, "phase3_8q_contrastive.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: phase3_8q_contrastive.png")


# ─────────────────────────────────────────────────────────────
# PHASE 4: Warm-start — show 2-layer warm circuit → expanded 3-layer circuit
def phase4():
    # Warm-start shallow
    qc_warm = build_circuit(n_qubits=8, n_layers=2)
    qc_warm.name = "Warm (2L)"
    # Full depth
    qc_full = build_circuit(n_qubits=8, n_layers=3)
    qc_full.name = "Full (3L)"

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Phase 4 — ADAM Optimizer + GPU + Warm-Start Layers",
                 fontsize=13, fontweight="bold")

    circ_fig_warm = qc_warm.draw("mpl", style="iqp", fold=60)
    circ_fig_full = qc_full.draw("mpl", style="iqp", fold=60)

    # Re-draw onto subplots using canvas trick
    circ_fig_warm.savefig(os.path.join(OUT, "_tmp_warm.png"), dpi=120, bbox_inches="tight")
    circ_fig_full.savefig(os.path.join(OUT, "_tmp_full.png"), dpi=120, bbox_inches="tight")
    plt.close(circ_fig_warm); plt.close(circ_fig_full)

    from PIL import Image
    img_warm = np.array(Image.open(os.path.join(OUT, "_tmp_warm.png")))
    img_full = np.array(Image.open(os.path.join(OUT, "_tmp_full.png")))

    axes[0].imshow(img_warm); axes[0].axis("off")
    axes[0].set_title("Step 1 — Warm-start: pre-train 2-layer circuit (no contrastive loss)",
                      fontsize=10, style="italic")
    axes[1].imshow(img_full); axes[1].axis("off")
    axes[1].set_title("Step 2 — Expand to 3-layer circuit (copy warm weights + random new layer)",
                      fontsize=10, style="italic")

    fig.text(0.5, 0.01,
             "ADAM + exact parameter-shift gradients | Batched GPU evaluation (2N+1 circuits/job) | "
             "Gap-weighted ensemble (3 models) | ROC-AUC 0.9077 · F1 0.5911 · FAR 10.8%",
             ha="center", fontsize=8.5, style="italic", color="#333")
    fig.tight_layout()
    path = os.path.join(OUT, "phase4_adam_warmstart.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    for tmp in ["_tmp_warm.png", "_tmp_full.png"]:
        try:
            os.remove(os.path.join(OUT, tmp))
        except Exception:
            pass
    print("  Saved: phase4_adam_warmstart.png")


# ─────────────────────────────────────────────────────────────
# PHASE 5: FAR-constrained — same 8-qubit circuit, annotated with operating points
def phase5():
    qc = build_circuit(n_qubits=8, n_layers=3)
    fig = qc.draw("mpl", style="iqp", fold=60)
    fig.suptitle("Phase 5 — FAR-Constrained Operating Points (Same Circuit)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.05,
             "No circuit change — new evaluation: scan ROC curve for max-recall threshold at each FAR budget\n"
             "FAR floor diagnosed: 10.7% of normal traffic maps to unseen bitstrings → irreducible with 8q binary\n"
             "1% / 2% / 5% FAR budgets unreachable  |  @ 10% FAR → Recall = 80.6%",
             ha="center", fontsize=8.5, style="italic", color="#333")
    fig.tight_layout()
    path = os.path.join(OUT, "phase5_far_constrained.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: phase5_far_constrained.png")


# ─────────────────────────────────────────────────────────────
# PHASE 6: Auto Mixed Precision — 13-qubit circuit
# Feature layout: sbytes[0,1] Sload[2,3] dbytes[4,5] Dload[6,7] Dpkts[8,9]
#                 is_int_state[10] is_not_tcp[11] is_con_state[12]
def phase6():
    qc = build_circuit(n_qubits=13, n_layers=3)

    # Custom qubit labels
    labels = [
        "sbytes[0]", "sbytes[1]",
        "Sload[0]",  "Sload[1]",
        "dbytes[0]", "dbytes[1]",
        "Dload[0]",  "Dload[1]",
        "Dpkts[0]",  "Dpkts[1]",
        "is_int_st", "is_!tcp",   "is_con_st",
    ]
    qc.name = "Phase 6 — AMP 13q QCBM"

    fig = qc.draw("mpl", style="iqp", fold=80,
                  initial_state=False,
                  wire_order=list(range(13)))
    fig.suptitle("Phase 6 — Auto Mixed Precision (13 Qubits, Circular CNOT)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.04,
             "Binary features (is_not_tcp, is_int_state, is_con_state) → 1 bit/2 bins  |  "
             "Continuous features → 2 bits/4 bins\n"
             "13 qubits / 8,192 states  |  FAR floor BROKEN: 0.0%  |  "
             "ROC-AUC 0.9058 · F1 0.6340 · FAR 8.2%",
             ha="center", fontsize=8.5, style="italic", color="#333")
    fig.tight_layout()
    path = os.path.join(OUT, "phase6_amp_13q.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: phase6_amp_13q.png")


# ─────────────────────────────────────────────────────────────
# PHASE 7: Best result — AMP 13q + contrast-margin=10
def phase7():
    qc = build_circuit(n_qubits=13, n_layers=3)
    fig = qc.draw("mpl", style="iqp", fold=80)
    fig.suptitle("Phase 7 — BEST: AMP 13q + Contrastive Margin=10 (Current Best)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.06,
             "Same circuit as Phase 6 — contrast-margin raised 5.0 → 10.0\n"
             "Contrastive loss ACTIVATED: anomaly_kl jumped 5.38 → 11.62  |  KL gap: 3.2 → 8.1\n"
             "ROC-AUC 0.9350 (+0.03)  |  PR-AUC 0.5230 (+0.35)  |  F1 0.6376  |  FAR 7.6%  |  "
             "@ 10% FAR → Recall 80.6%",
             ha="center", fontsize=8.5, style="italic", color="#333")
    fig.tight_layout()
    path = os.path.join(OUT, "phase7_amp_contrast10_best.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: phase7_amp_contrast10_best.png")


# ─────────────────────────────────────────────────────────────
# PHASE 8: Domain-informed entanglement — 13 qubits, custom CNOT pairs
def phase8():
    # Exact pairs from the run:
    domain_pairs = [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9),   # within-feature
        (0, 2), (4, 6), (4, 8), (6, 8),             # source/dest side
        (0, 4), (2, 6),                              # cross-direction
        (11, 10), (11, 12), (10, 12),                # protocol flags
        (11, 0), (11, 4),                            # protocol → volume
    ]
    qc = build_circuit(n_qubits=13, n_layers=3, entanglement_pairs=domain_pairs)

    fig = qc.draw("mpl", style="iqp", fold=80)
    fig.suptitle("Phase 8 — Domain-Informed Entanglement (16 CNOT pairs, 13 Qubits)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.08,
             "Entanglement encodes known network traffic correlations:\n"
             "  Source side: sbytes↔Sload  |  Dest side: dbytes↔Dload, dbytes↔Dpkts\n"
             "  Cross-direction: sbytes↔dbytes, Sload↔Dload  |  "
             "Protocol: is_not_tcp↔is_int_state↔is_con_state  |  is_not_tcp→sbytes,dbytes\n"
             "KL gap improved: 8.1 → 8.9  |  BUT ROC-AUC dropped: 0.9350 → 0.9025 "
             "(uneven expressibility)\n"
             "Phase 7 (circular) remains best overall",
             ha="center", fontsize=8.5, style="italic", color="#333")
    fig.tight_layout()
    path = os.path.join(OUT, "phase8_domain_entanglement.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: phase8_domain_entanglement.png")


# ─────────────────────────────────────────────────────────────
# Summary comparison chart
def summary_chart():
    phases = [
        "Ph1\nClassical",
        "Ph2\n8q SPSA",
        "Ph3\n+Contrast",
        "Ph4\nADAM+GPU",
        "Ph5\nFAR pts",
        "Ph6\nAMP 13q",
        "Ph7\nMargin=10",
        "Ph8\nDomain Ent",
    ]
    roc_auc = [0.65, 0.73, 0.77, 0.9077, 0.9077, 0.9058, 0.9350, 0.9025]
    f1      = [0.30, 0.42, 0.50, 0.5911, 0.5911, 0.6340, 0.6376, 0.6376]
    far     = [0.40, 0.30, 0.25, 0.108,  0.108,  0.082,  0.076,  0.076]
    pr_auc  = [0.10, 0.15, 0.18, 0.20,   0.20,   0.17,   0.5230, 0.4887]

    x = np.arange(len(phases))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*width, roc_auc, width, label="ROC-AUC", color="#4e79a7")
    ax.bar(x - 0.5*width, pr_auc,  width, label="PR-AUC",  color="#f28e2b")
    ax.bar(x + 0.5*width, f1,      width, label="F1",       color="#59a14f")
    ax.bar(x + 1.5*width, far,     width, label="FAR",      color="#e15759", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("QCBM Capstone — Stage 1 Metric Progression Across All Phases",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(3.6, 1.01, "ADAM + GPU introduced", fontsize=7.5, color="gray")
    ax.axvline(x=5.5, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(5.6, 1.01, "AMP introduced", fontsize=7.5, color="green")

    # Annotate best
    ax.annotate("BEST\n0.9350", xy=(6, 0.9350), xytext=(6.3, 0.97),
                fontsize=8, color="#4e79a7",
                arrowprops=dict(arrowstyle="->", color="#4e79a7", lw=1.2))

    fig.tight_layout()
    path = os.path.join(OUT, "summary_metric_progression.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: summary_metric_progression.png")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating circuit diagrams...")

    # PIL needed for phase 4 stacked image — gracefully skip if missing
    try:
        import PIL
        has_pil = True
    except ImportError:
        has_pil = False
        print("  [PIL not found] Phase 4 warm-start stacked image will be skipped.")

    phase1_diagram()
    phase2()
    phase3()
    if has_pil:
        phase4()
    else:
        # Save each circuit individually if PIL is unavailable
        qc_warm = build_circuit(n_qubits=8, n_layers=2)
        save_circuit(qc_warm, "phase4a_warmstart_2layer.png",
                     "Phase 4 — Warm-Start: 2-Layer Pre-training",
                     "ADAM + GPU | Pre-train without contrastive loss")
        qc_full = build_circuit(n_qubits=8, n_layers=3)
        save_circuit(qc_full, "phase4b_warmstart_3layer.png",
                     "Phase 4 — Full 3-Layer Circuit (after warm-start expansion)",
                     "ROC-AUC 0.9077 · F1 0.5911 · FAR 10.8% · FAR floor 10.7%")
    phase5()
    phase6()
    phase7()
    phase8()
    summary_chart()

    print("\nDone. All circuits saved to CIRCUITS_ROOT_DIRECTORY/")
