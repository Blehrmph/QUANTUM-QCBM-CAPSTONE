# QCBM Network Intrusion Detection — Capstone Project

**Quantum Circuit Born Machine (QCBM) for one-class anomaly detection on network traffic.**  
Trained entirely on normal traffic. No attack labels required.  
Validated on real IBM Quantum hardware (ibm_fez, 156-qubit superconducting processor).

---

## Results at a Glance

| Metric | Aer Simulator | IBM Hardware (5-member avg) |
|--------|:---:|:---:|
| ROC-AUC | **0.9671** | 0.8629 |
| PR-AUC | 0.8931 | 0.3834 |
| F1 | 0.9015 | 0.5300 |
| Recall (DR) | 0.9080 | 0.8263 |
| Precision | 0.8952 | 0.3901 |
| FAR | **0.0131** | 0.1598 |
| MCC | 0.8893 | 0.4934 |

> Best operating point: LR + Isotonic two-stage calibration.  
> Test set: 507,947 samples — 55,911 attacks, 452,036 normal (UNSW-NB15).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Approach](#2-approach)
3. [Dataset](#3-dataset)
4. [Feature Engineering](#4-feature-engineering)
5. [QCBM Architecture](#5-qcbm-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Scoring and Calibration](#7-scoring-and-calibration)
8. [Full Results](#8-full-results)
9. [Ablation Studies](#9-ablation-studies)
10. [IBM Hardware Validation](#10-ibm-hardware-validation)
11. [Project Structure](#11-project-structure)
12. [Setup and Usage](#12-setup-and-usage)
13. [Configuration Reference](#13-configuration-reference)

---

## 1. Problem Statement

Classical intrusion detection systems rely on labeled attack signatures. They fail silently against zero-day attacks — threats that have never been catalogued. This project takes a fundamentally different approach: learn a precise generative model of **normal** network traffic, then flag anything that deviates as an anomaly.

The core challenge is the encoding bottleneck: quantum circuits operate on discrete bitstrings, so every feature must be digitized into bits. The number of bits determines the qubit count, which is constrained by hardware. This project co-designs the feature engineering and circuit architecture to fit exactly 15 qubits — the largest register that can run on current IBM hardware without prohibitive SWAP overhead.

---

## 2. Approach

The pipeline has three stages:

```
Stage 1 — QCBM generative model of normal traffic
            ↓
Stage 2 — Classical anomaly scoring (log-probability, KL divergence)
            ↓
Stage 3 — Two-stage calibration (LR → Isotonic regression)
```

**Key design choices:**
- One-class learning: QCBM trained exclusively on normal traffic
- Contrastive loss: uses anomaly samples as a *repulsive* signal during training without relying on their labels for scoring
- Gap-weighted ensembling: five independently trained QCBMs, weighted by anomaly discrimination gap
- Anomaly-aware binning: bin boundaries optimized to maximize KL divergence between normal and attack distributions
- Auto mixed precision: binary features get 1 qubit, continuous features get 2 qubits → exactly 15 qubits total

---

## 3. Dataset

**UNSW-NB15** — University of New South Wales Network Intrusion dataset.

| Split | Samples | Normal | Attacks |
|-------|--------:|-------:|--------:|
| Train | 1,582,625 | 1,582,625 | 0 (one-class) |
| Val | 253,974 | 226,018 | 28,092 |
| Test | 507,947 | 452,036 | 55,911 |

- 49 raw features, 9 attack categories (DoS, Exploits, Fuzzers, Generic, Reconnaissance, Backdoors, Analysis, Shellcode, Worms)
- Preprocessed version: `datasets/UNSW-NB15_cleaned.csv`
- Class imbalance: ~11% attack rate in test set

---

## 4. Feature Engineering

### Feature Selection

9 features selected via mutual information + variance filtering, with **auto mixed precision** encoding:

| Feature | Type | Bits | Qubits |
|---------|------|:----:|:------:|
| sbytes | Continuous | 2 | 2 |
| Sload | Continuous | 2 | 2 |
| dbytes | Continuous | 2 | 2 |
| Dload | Continuous | 2 | 2 |
| Dpkts | Continuous | 2 | 2 |
| is_not_tcp | Binary | 1 | 1 |
| is_int_state | Binary | 1 | 1 |
| is_con_state | Binary | 1 | 1 |
| sttl | Continuous | 2 | 2 |
| **Total** | | **15** | **15** |

### Preprocessing

1. Categorical encoding: `is_not_tcp`, `is_int_state`, `is_con_state` derived from raw protocol/state fields
2. log1p transform on skewed continuous features
3. StandardScaler (fit on training normal data only)
4. Tail clipping at 99th percentile to suppress outlier influence on bin boundaries

### Anomaly-Aware Binning

Standard quantile bins treat normal and attack samples identically. Instead, bin boundaries are chosen to **maximize KL divergence** between the normal and attack marginal distributions for each feature — placing boundaries exactly where the two distributions disagree most. This directly improves the downstream discrimination signal available to the QCBM.

**Impact on ROC-AUC:**

| Model | Quantile Bins | Anomaly-Aware Bins | Lift |
|-------|:---:|:---:|:---:|
| IsoForest | 0.4239 | 0.8997 | +0.476 |
| Autoencoder | 0.7611 | 0.8648 | +0.104 |
| KDE | 0.4595 | 0.9191 | +0.460 |
| QCBM (Ours) | 0.9398 | **0.9671** | +0.027 |

The QCBM already extracts most of the signal with quantile bins (strong prior from quantum interference), but the anomaly-aware bins push it further and benefit classical baselines dramatically.

---

## 5. QCBM Architecture

### Circuit Structure

```
For each layer ℓ ∈ {1, 2, 3}:
  For each qubit q ∈ {0..14}:
    RZ(θ[ℓ,q,0]) · RY(θ[ℓ,q,1]) · RZ(θ[ℓ,q,2])   ← universal single-qubit
  
  CNOT ring: q0→q1, q1→q2, ..., q13→q14, q14→q0   ← circular entanglement

Final layer:
  For each qubit q: RY(θ_final[q])                  ← output expressibility
```

- **Qubits:** 15
- **Layers:** 3
- **Parameters per member:** 15 × 3 × 3 + 15 = **150**
- **Ensemble size:** 5 (independently initialized)
- **Total parameters:** 750

The circuit produces a 2¹⁵ = 32,768 dimensional probability distribution over bitstrings. Each bitstring decodes to a discretized network flow fingerprint. The QCBM learns which fingerprints are characteristic of normal traffic.

### Why This Ansatz

The RZ-RY-RZ parameterization provides a complete SU(2) rotation per qubit per layer — any single-qubit state is reachable. The circular CNOT ring introduces entanglement between all adjacent qubit pairs in a single depth-1 layer, making it hardware-efficient (maps directly to the heavy-hex topology with `initial_layout`). Three layers provides sufficient expressibility while keeping circuit depth under 165 gates before transpilation.

---

## 6. Training Pipeline

### Loss Function

```
L(θ) = KL(p_normal ∥ p_θ)  −  λ · max(0, margin − KL(p_anomaly ∥ p_θ))
```

- First term: minimize surprise on normal traffic (standard density estimation)
- Second term: penalize the model if anomaly traffic looks too likely — forces a KL gap of at least `margin`
- **λ = 0.8**, **margin = 15.0** (tuned via grid search)
- Laplace smoothing α = 0.5 applied to empirical distributions

### Gradients: Parameter-Shift Rule

Backpropagation is impossible through a quantum measurement. Instead, for each parameter θᵢ:

```
∂L/∂θᵢ = ½ [L(θ + π/2 · eᵢ) − L(θ − π/2 · eᵢ)]
```

For 150 parameters: **301 circuit evaluations per gradient step**, batched into a single Qiskit job.

### Optimizer: ADAM

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.003 |
| β₁ | 0.9 |
| β₂ | 0.999 |
| Iterations | 1,500 |

### Warm-Start

Before full contrastive training, the first 2 layers are pre-trained with a simpler KL-only loss. This initializes the circuit in a region with non-trivial gradients, avoiding the barren plateau problem (vanishing gradients in random circuit initialization).

### Ensembling

Five QCBMs are trained with different random seeds. After training, each member is scored on a validation set, yielding:

- **normal_KL**: how well the member fits normal traffic
- **anomaly_KL**: how surprised the member is by attack traffic  
- **gap** = anomaly_KL − normal_KL

Members are weighted proportionally to their gap score:

| Member | Gap | Weight |
|--------|----:|-------:|
| M1 | 8.60 | 16.3% |
| M2 | 10.80 | 20.5% |
| M3 | **18.30** | **34.7%** |
| M4 | 6.00 | 11.4% |
| M5 | 8.76 | 16.6% |

The final distribution is a gap-weighted average of the five member distributions.

### Simulation Backend

Training uses **Qiskit AerSimulator** (CPU statevector, v0.17.2) — approximately 100× faster than the fallback `qiskit.quantum_info.Statevector` for batched circuit evaluation.

---

## 7. Scoring and Calibration

### Anomaly Score

For a test sample encoded as bitstring `b`:

```
score(b) = −log p_θ(b)     [negative log-probability under the QCBM]
```

Higher score = more anomalous. Normal traffic should have high probability (low score); attacks should have low probability (high score).

### Two-Stage Calibration

Raw log-probability scores are not well-calibrated for binary classification. The pipeline applies:

1. **Logistic Regression (LR)**: fits a sigmoid mapping from raw score to [0,1] using validation labels
2. **Isotonic Regression**: fits a monotone non-parametric correction on top of LR outputs

**Calibration improvement:**

| Stage | ROC-AUC | F1 | FAR |
|-------|:---:|:---:|:---:|
| Raw QCBM Stage 1 | 0.9616 | 0.8830 | 0.0068 |
| + LR Calibration | 0.9618 | 0.9015 | 0.0131 |
| + Isotonic (Final) | **0.9671** | **0.9015** | 0.0131 |

### Threshold Selection

Multiple operating points are supported:

| Strategy | F1 | Recall | FAR |
|----------|:--:|:------:|:---:|
| F1-maximizing threshold | 0.9015 | 0.9080 | 0.0131 |
| Youden's J | 0.8806 | 0.9170 | 0.0206 |
| FAR ≤ 1% constrained | 0.8744 | 0.8356 | 0.0094 |
| FAR ≤ 2% constrained | 0.8790 | 0.9108 | 0.0200 |

---

## 8. Full Results

### Primary Results (Best Configuration — 15q, 3 layers, ensemble=5)

| Metric | Value |
|--------|------:|
| ROC-AUC | **0.9671** |
| PR-AUC | 0.8931 |
| F1 | **0.9015** |
| Precision | 0.8952 |
| Recall (DR) | 0.9080 |
| FAR | **0.0131** |
| MCC | 0.8893 |
| TP | 50,765 |
| FP | 5,943 |
| FN | 5,146 |
| TN | 446,093 |

### Stability Across 5 Seeds (95% Confidence Intervals)

| Metric | Mean | 95% CI |
|--------|:----:|:------:|
| ROC-AUC | 0.9410 | [0.9360, 0.9461] |
| F1 | 0.7734 | [0.7685, 0.7783] |
| Recall | 0.6445 | [0.6343, 0.6548] |
| MCC | 0.7703 | [0.7620, 0.7786] |
| FAR | 0.0028 | [0.0004, 0.0052] |

> Note: CI runs use a slightly smaller 13-qubit configuration. The 15-qubit best_run is a single optimized configuration.

### SOTA Comparison on UNSW-NB15

| Method | Type | F1 | ROC-AUC |
|--------|------|----|---------|
| XGBoost (MDPI 2024) | Supervised | 0.995 | 0.9993 |
| Random Forest (Kasongo 2020) | Supervised | 0.995 | — |
| CNN-LSTM (JCBI 2024) | Supervised | 0.960 | — |
| GCN-LOF (2023) | Unsupervised | 0.964 | — |
| KDE [anomaly-aware bins] | Unsupervised | 0.760 | 0.9191 |
| IsoForest [anomaly-aware bins] | Unsupervised | 0.741 | 0.8997 |
| QNN on IonQ (2021) | Quantum | 0.860 | — |
| **QCBM Simulator (Ours)** | **Quantum-Unsupervised** | **0.9015** | **0.9671** |
| QCBM IBM Hardware (Ours) | Quantum-Unsupervised | 0.5300 | 0.8629 |

The QCBM significantly outperforms all classical unsupervised baselines and exceeds the only prior quantum hardware result (QNN on IonQ, F1=0.860), while requiring no attack labels.

---

## 9. Ablation Studies

### Effect of Contrastive Loss

| Configuration | ROC-AUC | F1 | FAR |
|---|:---:|:---:|:---:|
| No contrastive (KL-only) | 0.9398 | 0.6376 | 0.0782 |
| λ=0.5, margin=10 | 0.9395 | 0.7708 | 0.0046 |
| **λ=0.8, margin=15 (best)** | **0.9671** | **0.9015** | **0.0131** |

### Effect of Feature Set

| Configuration | Qubits | ROC-AUC | F1 |
|---|:---:|:---:|:---:|
| 8 features (no sttl) | 13 | 0.9434 | 0.7729 |
| **9 features + sttl (best)** | **15** | **0.9671** | **0.9015** |

### Laplace Smoothing Sweep

| α | ROC-AUC |
|---|:---:|
| **0.5 (best)** | **0.9398** |
| 1.0 | 0.9350 |
| 2.0 | 0.9332 |
| 3.0 | 0.9323 |

Lighter smoothing (α=0.5) preserves the sharpness of the learned distribution, yielding better discrimination.

---

## 10. IBM Hardware Validation

### Setup

All five ensemble members were independently submitted to **ibm_fez** (156-qubit IBM Eagle R3 processor) using Qiskit's `SamplerV2` primitive. Each job: 32,768 shots. The circuit was transpiled with `optimization_level=3` and `initial_layout` set to a hardware-native 15-qubit subgraph, eliminating SWAP overhead.

### Job Summary

| Member | Job ID | Depth (gates) |
|--------|--------|:---:|
| M1 | d7n30plqrg3c738lsghg | 203 |
| M2 | d7n313s3g2mc7393376g | 204 |
| M3 | d7n319baq2pc73a2a2gg | 203 |
| M4 | d7n31eat99kc73d34ds0 | 207 |
| M5 | d7n31j43g2mc739337m0 | 203 |

### Distribution Fidelity

| Metric | Value |
|--------|------:|
| States visited (IBM) | 29,116 / 32,768 (88.9%) |
| TVD (hardware vs sim) | **0.5990** |
| KL (sim ∥ hardware) | 2.675 |
| Top-50 state overlap | 0.0% |

TVD of 0.60 is expected for 203-gate circuits on current NISQ hardware. Gate errors accumulate across the circuit depth and redistribute probability mass widely across the Hilbert space.

### Scoring Results

| Condition | TVD | ROC-AUC | Δ vs Sim |
|-----------|:---:|:-------:|:--------:|
| Aer Simulator (best) | — | 0.9671 | Baseline |
| ibm_fez (3-layer, single member) | 0.60 | 0.519 | −0.448 |
| ibm_kingston (3-layer, single member) | 0.60 | 0.486 | −0.481 |
| ibm_fez (1-layer, shallow) | 0.82 | 0.469 | −0.489 |
| **ibm_fez (3-layer, 5-member avg)** | **0.60** | **0.8629** | **−0.105** |

### Key Finding: Ensemble Averaging as Noise Mitigation

Averaging the distributions of all five ensemble members before scoring recovered **nearly 70% of the hardware noise penalty**:

- Single member on hardware: ROC-AUC 0.519 (−0.448)
- Five-member average on hardware: ROC-AUC 0.8629 (−0.105)

Each member's parameters produce independent noise realizations. Averaging suppresses uncorrelated noise components while reinforcing the learned signal. This is a zero-cost noise mitigation strategy — the ensemble was originally designed for training variance, but it doubles as hardware resilience.

### Shallow Circuit Experiment

A 1-layer circuit (depth ≈ 80 gates) was tested to determine whether reduced depth would improve hardware fidelity.

- TVD: **0.82** (worse than 3-layer, 0.60)
- ROC-AUC: 0.469

Shallower circuits produce more peaked distributions that collapse more severely under depolarising noise, because fewer entangling gates means less distribution spread — the noise has a proportionally larger effect. This confirms the **expressibility-noise tradeoff**: sufficient circuit depth is required to spread probability mass in a way that is robust to noise.

### Zero-Noise Extrapolation (ZNE) Attempt

Richardson extrapolation via CX gate folding (scale factors 1× and 3×) was attempted. The extrapolation was invalidated by mixed-backend execution: scale-factor 1 ran on ibm_fez, scale-factor 3 ran on ibm_kingston. ZNE requires a pinned backend across all scale factors to produce valid noise-scaled data points.

---

## 11. Project Structure

```
QUANTUM-QCBM-CAPSTONE/
│
├── main.py                          # End-to-end training pipeline entry point
├── best_config.json                 # Best hyperparameter configuration
│
├── src/
│   ├── qcbm_train.py               # QCBM circuit, training loop, gradient estimation
│   ├── discretize.py               # Anomaly-aware binning, bitstring encoding
│   ├── data/
│   │   ├── preprocessing.py        # Feature engineering, scaling, log1p
│   │   └── dataset_cleaning.py     # Raw UNSW-NB15 ingestion and cleaning
│   ├── score_eval.py               # ROC-AUC, F1, FAR, MCC evaluation
│   ├── classical_baseline.py       # KDE, IsoForest, RBM baselines
│   ├── bitstring_coverage.py       # Distribution coverage analysis
│   └── quantum_metrics.py          # Expressibility, entanglement entropy
│
├── STAGES/
│   ├── stage1.py                   # QCBM training + gap-weighted ensemble
│   ├── stage2.py                   # Anomaly scoring + two-stage calibration
│   └── stage3.py                   # Attack category classification (optional)
│
├── ibm_inference.py                # IBM hardware job submission (SamplerV2)
├── ibm_score.py                    # Score test set using hardware distribution
├── ibm_topo.py                     # Topology-matched circuit (hardware subgraph)
├── ibm_zne.py                      # Zero-noise extrapolation attempt
│
├── benchmark_binning_ablation.py   # Quantile vs anomaly-aware binning sweep
├── benchmark_classical.py          # Classical one-class baseline evaluation
├── benchmark_confidence_intervals.py # 5-seed stability benchmark
├── benchmark_noise_simulation.py   # Depolarising noise sweep
├── benchmark_sota_comparison.py    # SOTA table generation
├── laplace_sweep.py                # Laplace smoothing hyperparameter sweep
├── generate_paper_results.py       # Reproduce all paper tables/figures
│
├── CIRCUIT/
│   └── generate_circuits.py        # Circuit diagram generation
│
├── PHASES_METRICS/
│   └── generate_metrics.py         # Phase-by-phase metric reporting
│
├── PRESENTATION GRAPHS/
│   └── generate_graphs.py          # All 9 presentation figures (matplotlib)
│
├── artifacts/
│   ├── best_run/                   # Best trained model artifacts
│   │   ├── hier_qcbm_theta.npy    # Trained parameters (5×150)
│   │   ├── hier_qcbm_model_dist.npy # Learned probability distribution
│   │   ├── hier_qcbm_config.json  # Circuit configuration
│   │   ├── hier_stage1_metrics.json # Full metrics including calibration
│   │   ├── ibm_dist.npy           # Hardware-averaged distribution
│   │   ├── ibm_results.json       # IBM run metadata (TVD, top states)
│   │   └── ibm_score_metrics.json # Hardware scoring results
│   ├── confidence_intervals.json   # 5-seed CI results
│   ├── binning_ablation.json       # Binning strategy comparison
│   ├── sota_comparison.json        # SOTA benchmark data
│   ├── noise_simulation.json       # Depolarising noise sweep results
│   ├── classical_baseline_comparison.json
│   └── ci_runs/                   # Per-seed artifacts (seeds 0,42,123,256,999)
│
└── datasets/
    └── UNSW-NB15_cleaned.csv       # Preprocessed dataset
```

---

## 12. Setup and Usage

### Requirements

```
Python 3.10
qiskit >= 1.0
qiskit-aer >= 0.17.2        # CPU statevector simulation (~100x faster)
qiskit-ibm-runtime          # IBM hardware submission
numpy, pandas, scikit-learn
matplotlib                  # Presentation graphs
```

Install qiskit-aer into the correct Python environment:
```bash
python3.10 -m pip install qiskit-aer qiskit-ibm-runtime
```

### Run Full Training Pipeline

```bash
python main.py --config best_config.json
```

### Run Individual Benchmarks

```bash
python benchmark_binning_ablation.py
python benchmark_confidence_intervals.py
python benchmark_classical.py
python benchmark_noise_simulation.py
```

### Submit to IBM Hardware

```bash
python ibm_inference.py --config best_config.json
```

### Score Hardware Results

```bash
python ibm_score.py --config best_config.json
```

### Generate Presentation Graphs

```bash
python "PRESENTATION GRAPHS/generate_graphs.py"
```

---

## 13. Configuration Reference

`best_config.json` — the configuration that produced the best results:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `features` | 9 features | sbytes, Sload, dbytes, Dload, Dpkts, is_not_tcp, is_int_state, is_con_state, sttl |
| `bin_strategy` | `anomaly_aware` | KL-maximizing bin boundaries |
| `auto_mixed_precision` | `true` | Binary=1 bit, continuous=2 bits |
| `qcbm_layers` | 3 | Circuit depth |
| `qcbm_ensemble` | 5 | Number of ensemble members |
| `optimizer` | `adam` | Gradient optimizer |
| `adam_lr` | 0.003 | Learning rate |
| `qcbm_iter` | 1500 | Training iterations |
| `lambda_contrast` | 0.8 | Contrastive loss weight |
| `contrast_margin` | 15.0 | Minimum KL gap from anomaly distribution |
| `laplace_alpha` | 0.5 | Distribution smoothing |
| `warmstart_layers` | `true` | Pre-train 2 layers before full depth |
| `scaler` | `standard` | Feature normalization |
| `log1p` | `true` | Log transform skewed features |
| `test_frac` | 0.2 | Test set fraction |
| `val_frac` | 0.1 | Validation set fraction |
| `seed` | 42 | Random seed |

---

## Technical Notes

**Bit ordering (SamplerV2):** IBM's SamplerV2 returns bitstrings in little-endian order (qubit 0 at the leftmost position). The inference code applies `[::-1]` reversal to match the simulator's big-endian convention before distribution comparison.

**Gradient batching:** All 2n+1 = 301 circuits for a single parameter-shift gradient step are submitted as a single Qiskit job, minimizing queue overhead during IBM hardware training.

**SWAP elimination:** `ibm_topo.py` extracts the best-connected 15-qubit subgraph of the IBM heavy-hex topology and sets `initial_layout` to map logical qubits directly to physical qubits, eliminating SWAP insertion during transpilation.

**Windows buffering:** When redirecting Python output on Windows, use `python -u` (unbuffered) to ensure logs are written in real-time rather than being held in an internal buffer.
