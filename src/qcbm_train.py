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
    per_sample_contrast: bool = False  # per-bitstring contrastive instead of aggregate KL
    warmstart_layers: bool = False  # pre-train with n_layers-1, then expand
    use_rzz: bool = False           # parametrised RZZ entanglement instead of fixed CNOT
    optimizer: str = "spsa"         # "spsa" or "adam"
    adam_lr: float = 0.01           # ADAM learning rate
    adam_beta1: float = 0.9         # ADAM first moment decay
    adam_beta2: float = 0.999       # ADAM second moment decay
    entanglement_pairs: list | None = None  # explicit CNOT pairs [(ctrl, tgt), ...]; None = circular


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


def build_ansatz(
    n_qubits: int,
    n_layers: int,
    theta: np.ndarray,
    use_rzz: bool = False,
    entanglement_pairs: list | None = None,
):
    """Hardware-efficient ansatz: RZ-RY-RZ rotations + entanglement layer.

    Entanglement options:
    - entanglement_pairs=None, use_rzz=False (default): fixed CNOT circular pattern.
    - entanglement_pairs=None, use_rzz=True: parametrised RZZ circular pattern.
    - entanglement_pairs=[(ctrl, tgt), ...]: domain-informed CNOT topology.
      Each pair is a (control, target) qubit index. Applied once per layer.
      use_rzz is ignored when entanglement_pairs is provided.
    """
    QuantumCircuit, _ = _import_qiskit()
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.rz(theta[idx], q); idx += 1
            qc.ry(theta[idx], q); idx += 1
            qc.rz(theta[idx], q); idx += 1
        if entanglement_pairs is not None:
            for ctrl, tgt in entanglement_pairs:
                qc.cx(ctrl, tgt)
        else:
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
    qc = build_ansatz(config.n_qubits, config.n_layers, theta, use_rzz=config.use_rzz,
                      entanglement_pairs=config.entanglement_pairs)
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


def qcbm_distribution_batch(thetas: list, config: QCBMConfig) -> np.ndarray:
    """Evaluate multiple theta vectors in a single AerSimulator job.

    Submits all circuits together so the GPU processes them as one batch,
    amortising kernel-launch and Python overhead across the whole set.

    Returns
    -------
    np.ndarray, shape (len(thetas), 2**n_qubits)
        Probability distributions for each theta vector.
    """
    simulator = _get_aer_simulator()
    if simulator is not None:
        from qiskit import transpile
        circuits = []
        for theta in thetas:
            qc = build_ansatz(config.n_qubits, config.n_layers, theta, use_rzz=config.use_rzz,
                              entanglement_pairs=config.entanglement_pairs)
            qc.save_statevector()
            circuits.append(qc)
        tqcs = transpile(circuits, simulator, optimization_level=0)
        result = simulator.run(tqcs).result()
        return np.stack([
            np.abs(np.array(result.get_statevector(i))) ** 2
            for i in range(len(circuits))
        ])
    else:
        # No Aer: fall back to sequential Statevector evaluation
        return np.stack([qcbm_distribution(theta, config) for theta in thetas])


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


def _loss_from_dist(
    model_probs: np.ndarray,
    data_dist: np.ndarray,
    anomaly_dist: np.ndarray | None,
    lambda_contrast: float,
    contrast_margin: float,
    anomaly_per_sample: np.ndarray | None = None,
) -> float:
    """Compute the QCBM loss given a pre-evaluated model probability distribution.

    Separates the cheap loss arithmetic from the expensive circuit simulation so
    that batched circuit results can be reused without re-running the simulator.

    If anomaly_per_sample is provided (shape: n_unique_anomaly_bitstrings,),
    uses per-sample contrastive loss: mean over individual bitstring log-prob penalties.
    Otherwise uses aggregate KL(anomaly_dist || model).
    """
    loss = kl_divergence(data_dist, model_probs)
    if lambda_contrast > 0:
        if anomaly_per_sample is not None:
            # Per-sample: penalise each unique anomaly bitstring individually
            # score = -log p(x_anom); we want score >> margin -> p(x_anom) << exp(-margin)
            scores = -np.log(np.clip(model_probs[anomaly_per_sample], 1e-12, 1.0))
            loss += lambda_contrast * float(np.mean(np.maximum(0.0, contrast_margin - scores)))
        elif anomaly_dist is not None:
            anom_kl = kl_divergence(anomaly_dist, model_probs)
            loss += lambda_contrast * max(0.0, contrast_margin - anom_kl)
    return loss


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


def compute_gradient_param_shift_batched(
    theta: np.ndarray,
    config: QCBMConfig,
    data_dist: np.ndarray,
    anomaly_dist: np.ndarray | None,
    lambda_contrast: float,
    contrast_margin: float,
    shift: float = np.pi / 2,
) -> tuple[np.ndarray, float]:
    """Parameter-shift gradient computed in a single batched GPU job.

    Builds all 2*n_params + 1 shifted circuits upfront and submits them as one
    batch to AerSimulator, replacing 321 sequential circuit submissions with a
    single GPU job. The loss arithmetic (KL, contrastive term) is then done on
    CPU using the returned probability arrays.

    Returns
    -------
    grad : np.ndarray
        Exact parameter-shift gradient, same as compute_gradient_param_shift.
    current_loss : float
        Loss evaluated at the un-shifted theta (used for early stopping).
    """
    n = len(theta)
    thetas_batch = []
    for i in range(n):
        t_plus  = theta.copy(); t_plus[i]  += shift
        t_minus = theta.copy(); t_minus[i] -= shift
        thetas_batch.append(t_plus)
        thetas_batch.append(t_minus)
    thetas_batch.append(theta.copy())  # index -1: current theta for loss eval

    all_probs = qcbm_distribution_batch(thetas_batch, config)

    grad = np.zeros(n)
    for i in range(n):
        loss_plus  = _loss_from_dist(all_probs[2 * i],     data_dist, anomaly_dist, lambda_contrast, contrast_margin)
        loss_minus = _loss_from_dist(all_probs[2 * i + 1], data_dist, anomaly_dist, lambda_contrast, contrast_margin)
        grad[i] = (loss_plus - loss_minus) / 2.0

    current_loss = _loss_from_dist(all_probs[-1], data_dist, anomaly_dist, lambda_contrast, contrast_margin)
    return grad, current_loss


def adam_optimize(
    loss_fn,
    theta0: np.ndarray,
    max_iter: int,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    patience: int = 50,
    grad_fn=None,
) -> tuple[np.ndarray, list[float]]:
    """ADAM optimizer with exact parameter-shift gradients and early stopping.

    Parameters
    ----------
    grad_fn : callable(theta) -> (grad, loss), optional
        When provided, called instead of compute_gradient_param_shift + loss_fn.
        Use compute_gradient_param_shift_batched here to submit all circuits in
        one GPU job per iteration instead of 2*n_params sequential submissions.
        If None, falls back to the original sequential parameter-shift approach.
    """
    theta = theta0.copy()
    m = np.zeros_like(theta)  # first moment
    v = np.zeros_like(theta)  # second moment
    best_loss = np.inf
    best_theta = theta.copy()
    no_improve = 0
    loss_history: list[float] = []

    for t in range(1, max_iter + 1):
        if grad_fn is not None:
            grad, current_loss = grad_fn(theta)
        else:
            grad = compute_gradient_param_shift(loss_fn, theta)
            current_loss = loss_fn(theta)

        # Bias-corrected ADAM update
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

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
    config: QCBMConfig
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
    anomaly_indices = None  # unique bitstring indices for per-sample contrastive
    _anomaly_dist_diag = None
    if anomaly_bitstrings is not None and len(anomaly_bitstrings) > 0:
        _anomaly_dist_diag = empirical_distribution(anomaly_bitstrings, n_qubits, alpha=config.laplace_alpha)
        if config.lambda_contrast > 0:
            if config.per_sample_contrast:
                # Pre-compute unique anomaly bitstring indices once
                anomaly_indices = np.unique(bitstrings_to_indices(anomaly_bitstrings))
                print(f"  Per-sample contrastive loss  lambda={config.lambda_contrast}  "
                      f"margin={config.contrast_margin}  unique_anomaly_bitstrings={len(anomaly_indices)}")
            else:
                anomaly_dist = _anomaly_dist_diag
                print(f"  Contrastive loss enabled  lambda={config.lambda_contrast}  margin={config.contrast_margin}")

    rng = np.random.default_rng(config.seed)
    n_theta = n_params(n_qubits, config.n_layers, use_rzz=config.use_rzz)

    # Warm-start: chain-expand from 2 layers up to n_layers to avoid barren plateaus.
    # For n_layers=3: 2->3.  For n_layers=4: 2->3->4 (chained).
    params_per_layer = n_qubits * 3 + (n_qubits if config.use_rzz else 0)

    def _run_warm_stage(n_layers_warm, theta0_warm):
        """Train a warm-start stage and return its best theta."""
        wc = QCBMConfig(
            n_qubits=n_qubits,
            n_layers=n_layers_warm,
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
        def _wloss(t):
            return kl_divergence(data_dist, qcbm_distribution(t, wc))
        if config.optimizer == "adam":
            _wgrad = None
            if _get_aer_simulator() is not None:
                _wgrad = lambda t: compute_gradient_param_shift_batched(
                    t, wc, data_dist, None, 0.0, wc.contrast_margin
                )
            wt, _ = adam_optimize(_wloss, theta0=theta0_warm, max_iter=wc.max_iter,
                                  lr=config.adam_lr, beta1=config.adam_beta1,
                                  beta2=config.adam_beta2, grad_fn=_wgrad)
        else:
            wt, _ = spsa_optimize(_wloss, theta0=theta0_warm, max_iter=wc.max_iter,
                                  a=wc.spsa_a, c=wc.spsa_c, seed=wc.seed)
        return wt

    def _expand_theta(warm_theta, from_layers, to_layers):
        """Insert a randomly initialised layer after the existing layers."""
        full = np.zeros(n_params(n_qubits, to_layers, use_rzz=config.use_rzz))
        used = from_layers * params_per_layer
        full[:used] = warm_theta[:used]
        full[used:used + params_per_layer] = rng.uniform(0, 2 * np.pi, size=params_per_layer)
        full[used + params_per_layer:] = warm_theta[used:]
        return full

    if config.warmstart_layers and config.n_layers > 1:
        # Chain: 2 -> 3 -> ... -> n_layers
        start_layers = min(2, config.n_layers - 1)
        print(f"  Warm-start: pre-training {start_layers} layers", end="")
        warm_theta = _run_warm_stage(
            start_layers,
            rng.uniform(0, 2 * np.pi, size=n_params(n_qubits, start_layers, use_rzz=config.use_rzz))
        )
        for intermediate in range(start_layers + 1, config.n_layers):
            print(f" -> {intermediate}", end="", flush=True)
            warm_theta = _expand_theta(warm_theta, intermediate - 1, intermediate)
            warm_theta = _run_warm_stage(intermediate, warm_theta)
        print(f" -> {config.n_layers} layers")
        theta0 = _expand_theta(warm_theta, config.n_layers - 1, config.n_layers)
    else:
        theta0 = rng.uniform(0, 2 * np.pi, size=n_theta)

    def loss_fn(theta: np.ndarray) -> float:
        model_dist = qcbm_distribution(theta, config)
        return _loss_from_dist(model_dist, data_dist, anomaly_dist,
                               config.lambda_contrast, config.contrast_margin,
                               anomaly_per_sample=anomaly_indices)

    if config.optimizer == "adam":
        print(f"  Optimizer: ADAM  lr={config.adam_lr}  beta1={config.adam_beta1}  beta2={config.adam_beta2}")
        batched = _get_aer_simulator() is not None
        mode = f"batched GPU ({2 * n_theta + 1} circuits/job)" if batched else f"sequential ({2 * n_theta} evals/step)"
        print(f"  Gradient: parameter-shift rule  [{mode}]")
        grad_fn = None
        if batched:
            grad_fn = lambda t: compute_gradient_param_shift_batched(
                t, config, data_dist, anomaly_dist, config.lambda_contrast, config.contrast_margin
            )
        theta, loss_history = adam_optimize(
            loss_fn,
            theta0=theta0,
            max_iter=config.max_iter,
            lr=config.adam_lr,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            grad_fn=grad_fn,
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
    final_anomaly_kl = None
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
        "anomaly_kl": final_anomaly_kl,
        "loss_history": loss_history,
    }
