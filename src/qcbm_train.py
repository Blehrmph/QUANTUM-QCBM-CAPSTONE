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
    lambda_contrast: float = 0.5   # weight of contrastive term
    contrast_margin: float = 0.3   # min KL distance from anomaly distribution
    laplace_alpha: float = 1.0     # Laplace smoothing on empirical distribution
    warmstart_layers: bool = False  # pre-train with n_layers-1, then expand
    use_rzz: bool = False           # parametrised RZZ entanglement instead of fixed CNOT
    optimizer: str = "spsa"         # "spsa" or "adam"
    adam_lr: float = 0.01           # ADAM learning rate
    adam_beta1: float = 0.9         # ADAM first moment decay
    adam_beta2: float = 0.999       # ADAM second moment decay


def _import_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except Exception as exc:
        raise ImportError(
            "Qiskit is required for QCBM training. Install qiskit to proceed."
        ) from exc
    return QuantumCircuit, Statevector


_aer_simulator = None  # cached simulator instance


def _get_aer_simulator():
    """Return a cached AerSimulator. Tries GPU first, falls back to CPU.

    Returns None if qiskit-aer is not installed (falls back to Statevector).
    """
    global _aer_simulator
    if _aer_simulator is not None:
        return _aer_simulator
    try:
        from qiskit_aer import AerSimulator
        try:
            sim = AerSimulator(method="statevector", device="GPU")
            # Smoke-test: run a trivial circuit to confirm GPU actually works
            from qiskit import QuantumCircuit
            _test = QuantumCircuit(1)
            _test.h(0)
            _test.save_statevector()
            sim.run(_test).result()
            print("  [AerSimulator] Using GPU-accelerated statevector simulation")
            _aer_simulator = sim
        except Exception:
            sim = AerSimulator(method="statevector", device="CPU")
            print("  [AerSimulator] GPU not available — using CPU statevector simulation")
            _aer_simulator = sim
    except ImportError:
        print("  [AerSimulator] qiskit-aer not found — falling back to qiskit.quantum_info.Statevector")
        _aer_simulator = None
    return _aer_simulator


def n_params(n_qubits: int, n_layers: int, use_rzz: bool = False) -> int:
    """Total number of trainable parameters for the ansatz.

    Each layer contributes 3 params per qubit (RZ-RY-RZ) plus, when use_rzz=True,
    1 RZZ param per qubit pair (circular entanglement).
    A final RY layer adds n_qubits more params.
    """
    per_layer = n_qubits * 3 + (n_qubits if use_rzz else 0)
    return per_layer * n_layers + n_qubits


def build_ansatz(n_qubits: int, n_layers: int, theta: np.ndarray, use_rzz: bool = False):
    """Hardware-efficient ansatz: RZ-RY-RZ rotations + circular entanglement.

    Entanglement options:
    - use_rzz=False (default): fixed CNOT circular pattern (no extra params).
    - use_rzz=True: parametrised RZZ(theta) circular pattern — the optimizer
      learns the strength/direction of each qubit-pair correlation rather than
      using an all-or-nothing fixed gate.
    """
    QuantumCircuit, _ = _import_qiskit()
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.rz(theta[idx], q); idx += 1
            qc.ry(theta[idx], q); idx += 1
            qc.rz(theta[idx], q); idx += 1
        for q in range(n_qubits):
            if use_rzz:
                qc.rzz(theta[idx], q, (q + 1) % n_qubits); idx += 1
            else:
                qc.cx(q, (q + 1) % n_qubits)
    for q in range(n_qubits):
        qc.ry(theta[idx], q)
        idx += 1
    return qc


def qcbm_distribution(theta: np.ndarray, config: QCBMConfig) -> np.ndarray:
    qc = build_ansatz(config.n_qubits, config.n_layers, theta, use_rzz=config.use_rzz)
    simulator = _get_aer_simulator()
    if simulator is not None:
        from qiskit import transpile
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        tqc = transpile(qc_sv, simulator, optimization_level=0)
        result = simulator.run(tqc).result()
        sv_data = np.array(result.get_statevector())
        probs = np.abs(sv_data) ** 2
    else:
        _, Statevector = _import_qiskit()
        sv = Statevector.from_instruction(qc)
        probs = np.abs(sv.data) ** 2
    return probs


def empirical_distribution(
    bitstrings: np.ndarray,
    n_qubits: int,
    alpha: float = 0.0,
) -> np.ndarray:
    """Build empirical probability distribution over 2^n_qubits bitstring states.

    Parameters
    ----------
    alpha : float
        Laplace smoothing parameter. Adds `alpha` pseudo-counts to every state
        before normalising. Benefits:
        - States unseen in training get probability alpha/(N + alpha*2^n) instead
          of 0, preventing -log(eps) score spikes on rare-but-legitimate normal
          traffic (reduces false alarm rate).
        - Smooths the target distribution, making KL divergence gradients more
          stable during SPSA optimisation.
        alpha=0 disables smoothing (raw counts). alpha=1 is add-one smoothing.
    """
    indices = bitstrings_to_indices(bitstrings)
    counts = np.bincount(indices, minlength=2**n_qubits).astype(float)
    if counts.sum() == 0:
        raise ValueError("No data to build empirical distribution.")
    counts += alpha
    return counts / counts.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Forward KL divergence KL(p||q). Mode-seeking: heavily penalises the model
    for assigning near-zero probability to states the data visits, producing sharp
    concentrated distributions — ideal for anomaly detection."""
    q = np.clip(q, eps, 1.0)
    p = np.clip(p, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between p and q.

    Symmetric alternative to KL divergence. Bounded in [0, log(2)] ≈ [0, 0.693].

    Advantages over KL for QCBM training:
    - Symmetric: penalises the model both for missing normal modes AND for
      assigning probability mass to states the data never visits.
    - Bounded: gradient magnitudes stay stable throughout training (KL can
      blow up when model assigns near-zero prob to data states).
    - More balanced fit: the model is less likely to collapse to a degenerate
      distribution that covers only the most common bitstrings.
    """
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    q = np.clip(q, eps, 1.0); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    )


def spsa_optimize(
    loss_fn,
    theta0: np.ndarray,
    max_iter: int,
    a: float,
    c: float,
    seed: int,
    patience: int = 50,
) -> tuple[np.ndarray, list[float]]:
    """SPSA optimizer with momentum and early stopping."""
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    m = np.zeros_like(theta)
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


def compute_gradient_param_shift(
    loss_fn,
    theta: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Exact gradient via the parameter-shift rule.

    For each parameter theta_i:
        dL/dtheta_i = [ L(theta + shift*e_i) - L(theta - shift*e_i) ] / 2

    Requires 2 * len(theta) circuit evaluations per call — exact but expensive.
    For our 80-param circuit: 160 evaluations per gradient step vs SPSA's 2.
    """
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        t_plus  = theta.copy(); t_plus[i]  += shift
        t_minus = theta.copy(); t_minus[i] -= shift
        grad[i] = (loss_fn(t_plus) - loss_fn(t_minus)) / 2.0
    return grad


def adam_optimize(
    loss_fn,
    theta0: np.ndarray,
    max_iter: int,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    patience: int = 15,
) -> tuple[np.ndarray, list[float]]:
    """ADAM optimizer with exact parameter-shift gradients and early stopping.

    Each iteration computes 2*n_params circuit evaluations (exact gradient)
    plus 1 evaluation for the current loss — 161 evaluations for 80 params.
    Exact gradients mean significantly fewer iterations are needed vs SPSA.
    """
    theta = theta0.copy()
    m = np.zeros_like(theta)  # first moment
    v = np.zeros_like(theta)  # second moment
    best_loss = np.inf
    best_theta = theta.copy()
    no_improve = 0
    loss_history: list[float] = []

    for t in range(1, max_iter + 1):
        grad = compute_gradient_param_shift(loss_fn, theta)

        # Bias-corrected ADAM update
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

        current_loss = loss_fn(theta)
        loss_history.append(current_loss)

        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            best_theta = theta.copy()
            no_improve = 0
        else:
            no_improve += 1

        if t % 10 == 0:
            print(f"  ADAM iter {t}/{max_iter}: loss={current_loss:.6f}  best={best_loss:.6f}")

        if no_improve >= patience:
            print(f"  Early stop at iter {t}  best_loss={best_loss:.6f}")
            break

    return best_theta, loss_history


def train_qcbm(
    bitstrings: np.ndarray,
    config: QCBMConfig,
    anomaly_bitstrings: np.ndarray | None = None,
) -> dict:
    """Train the QCBM with KL divergence loss and optional contrastive term.

    Parameters
    ----------
    bitstrings : np.ndarray
        Bitstrings of NORMAL training samples only.
    config : QCBMConfig
        Training configuration.
    anomaly_bitstrings : np.ndarray, optional
        Bitstrings of known anomaly samples from training data.
        When provided, a contrastive penalty is added to the loss:

            L = KL(normal || model) + λ * max(0, margin - KL(anomaly || model))

        This pushes the model to assign LOW probability to anomaly bitstrings.
        λ = config.lambda_contrast, margin = config.contrast_margin.
        Set lambda_contrast=0 to disable.
    """
    n_qubits = config.n_qubits
    data_dist = empirical_distribution(bitstrings, n_qubits, alpha=config.laplace_alpha)
    if config.laplace_alpha > 0:
        print(f"  Laplace smoothing alpha={config.laplace_alpha}")

    # Always build anomaly_dist for diagnostics; only use in loss if λ > 0
    anomaly_dist = None
    _anomaly_dist_diag = None
    if anomaly_bitstrings is not None and len(anomaly_bitstrings) > 0:
        _anomaly_dist_diag = empirical_distribution(anomaly_bitstrings, n_qubits, alpha=config.laplace_alpha)
        if config.lambda_contrast > 0:
            anomaly_dist = _anomaly_dist_diag
            print(f"  Contrastive loss enabled  lambda={config.lambda_contrast}  margin={config.contrast_margin}")

    rng = np.random.default_rng(config.seed)
    n_theta = n_params(n_qubits, config.n_layers, use_rzz=config.use_rzz)

    # Warm-start: pre-train a shallower circuit, then expand to full depth
    params_per_layer = n_qubits * 3 + (n_qubits if config.use_rzz else 0)
    if config.warmstart_layers and config.n_layers > 1:
        print(f"  Warm-start: pre-training {config.n_layers - 1} layers -> {config.n_layers} layers")
        warm_config = QCBMConfig(
            n_qubits=n_qubits,
            n_layers=config.n_layers - 1,
            max_iter=config.max_iter // 3,
            seed=config.seed,
            spsa_a=config.spsa_a,
            spsa_c=config.spsa_c,
            lambda_contrast=0.0,
            contrast_margin=config.contrast_margin,
            laplace_alpha=0.0,
            warmstart_layers=False,
            use_rzz=config.use_rzz,
            optimizer=config.optimizer,
            adam_lr=config.adam_lr,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
        )
        warm_n_theta = n_params(n_qubits, warm_config.n_layers, use_rzz=config.use_rzz)
        warm_theta0 = rng.uniform(0, 2 * np.pi, size=warm_n_theta)

        def warm_loss(theta: np.ndarray) -> float:
            model_dist = qcbm_distribution(theta, warm_config)
            return kl_divergence(data_dist, model_dist)

        if config.optimizer == "adam":
            warm_theta, _ = adam_optimize(
                warm_loss,
                theta0=warm_theta0,
                max_iter=warm_config.max_iter,
                lr=config.adam_lr,
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
            )
        else:
            warm_theta, _ = spsa_optimize(
                warm_loss,
                theta0=warm_theta0,
                max_iter=warm_config.max_iter,
                a=warm_config.spsa_a,
                c=warm_config.spsa_c,
                seed=warm_config.seed,
            )
        # Expand: copy warm theta into first (n_layers-1) layers, randomly init new layer
        theta0 = np.zeros(n_theta)
        warm_layer_params = (config.n_layers - 1) * params_per_layer
        theta0[:warm_layer_params] = warm_theta[:warm_layer_params]
        theta0[warm_layer_params:warm_layer_params + params_per_layer] = rng.uniform(
            0, 2 * np.pi, size=params_per_layer
        )
        # final RY from warm theta
        theta0[warm_layer_params + params_per_layer:] = warm_theta[warm_layer_params:]
    else:
        theta0 = rng.uniform(0, 2 * np.pi, size=n_theta)

    def loss_fn(theta: np.ndarray) -> float:
        model_dist = qcbm_distribution(theta, config)
        # KL divergence: mode-seeking, creates sharp concentrated distributions
        # which is exactly what anomaly detection needs (low prob on unseen states)
        loss = kl_divergence(data_dist, model_dist)
        # Contrastive term: penalise if model is too close to anomaly distribution
        if anomaly_dist is not None:
            anom_kl = kl_divergence(anomaly_dist, model_dist)
            loss += config.lambda_contrast * max(0.0, config.contrast_margin - anom_kl)
        return loss

    if config.optimizer == "adam":
        print(f"  Optimizer: ADAM  lr={config.adam_lr}  beta1={config.adam_beta1}  beta2={config.adam_beta2}")
        print(f"  Gradient: parameter-shift rule  ({2 * n_theta} evals/step)")
        theta, loss_history = adam_optimize(
            loss_fn,
            theta0=theta0,
            max_iter=config.max_iter,
            lr=config.adam_lr,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
        )
    else:
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

    # Always report anomaly KL — critical for calibrating --contrast-margin
    if _anomaly_dist_diag is not None:
        final_anomaly_kl = kl_divergence(_anomaly_dist_diag, model_dist)
        print(f"  Final KL(normal || model) : {final_loss:.6f}")
        print(f"  Final KL(anomaly || model): {final_anomaly_kl:.6f}"
              f"  (set --contrast-margin above this to activate contrastive loss)")

    return {
        "theta": theta,
        "data_dist": data_dist,
        "model_dist": model_dist,
        "loss": final_loss,
        "loss_history": loss_history,
    }
