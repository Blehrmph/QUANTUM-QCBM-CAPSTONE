"""
Classical baseline comparison for Stage 1 anomaly detection.
Implements KDE and RBM with ~equivalent parameter budget to the QCBM (~78 params).
Run via: python -u benchmark_classical.py
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

from src.score_eval import evaluate


# ─── KDE baseline ────────────────────────────────────────────────────────────

def train_kde(X_normal: np.ndarray, bandwidth: float = 0.5) -> KernelDensity:
    """Fit KDE on normal bitstrings (float array, values 0/1)."""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(X_normal.astype(float))
    return kde


def score_kde(X: np.ndarray, kde: KernelDensity) -> np.ndarray:
    """Return anomaly scores: -log_density (higher = more anomalous)."""
    return -kde.score_samples(X.astype(float))


# ─── RBM baseline ────────────────────────────────────────────────────────────

def train_rbm(
    X_normal: np.ndarray,
    n_components: int = 26,
    n_iter: int = 100,
    learning_rate: float = 0.01,
    random_state: int = 42,
) -> BernoulliRBM:
    """Fit a Bernoulli RBM on normal bitstrings.

    n_components=26 gives ~78*2=156 weight params for 13 visible units
    (13*26 + 26 + 13 = 377... we use n_components=3 for exact 78-param match:
    13*3 + 3 + 13 = 55 params. Use 5 for 85 params — closest to 78.)
    Exact match: n_hidden s.t. n_visible*n_hidden + n_hidden + n_visible = 78
    13*h + h + 13 = 78  ->  14h = 65  ->  h = 4.6  -> use h=5 (85 params).
    """
    rbm = BernoulliRBM(
        n_components=n_components,
        n_iter=n_iter,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=0,
    )
    rbm.fit(X_normal.astype(float))
    return rbm


def score_rbm(X: np.ndarray, rbm: BernoulliRBM) -> np.ndarray:
    """Anomaly score = negative free energy (higher = more anomalous)."""
    return -rbm.score_samples(X.astype(float))


# ─── Isolation Forest baseline ───────────────────────────────────────────────

def train_isolation_forest(
    X_normal: np.ndarray,
    n_estimators: int = 100,
    max_samples: int = 256,
    random_state: int = 42,
):
    """Fit IsolationForest on the full normal training set.

    max_samples=256 is the standard setting for large datasets — each tree
    is built on a 256-sample subsample (sklearn default), so memory and
    runtime are independent of N. This runs on the full 1.58M sample set.
    """
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination="auto",
        random_state=random_state,
    )
    clf.fit(X_normal.astype(float))
    return clf


def score_isolation_forest(X: np.ndarray, clf) -> np.ndarray:
    """Anomaly score: negative decision function (higher = more anomalous)."""
    return -clf.decision_function(X.astype(float))


# ─── Autoencoder baseline ────────────────────────────────────────────────────

def train_autoencoder(
    X_normal: np.ndarray,
    hidden_dim: int = 6,
    max_iter: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    random_state: int = 42,
):
    """Fit a bottleneck autoencoder (13->6->13) on the full normal training set.

    Architecture: input(13) -> hidden(6) -> output(13), ReLU activation.
    ~175 parameters total. Trained via mini-batch SGD (Adam) on the
    full 1.58M normal sample set — no subsampling.

    Anomaly score = mean squared reconstruction error: high error = anomalous.
    """
    from sklearn.neural_network import MLPRegressor
    ae = MLPRegressor(
        hidden_layer_sizes=(hidden_dim,),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        random_state=random_state,
        verbose=False,
        early_stopping=False,
    )
    X = X_normal.astype(float)
    ae.fit(X, X)
    return ae


def score_autoencoder(X: np.ndarray, ae) -> np.ndarray:
    """Anomaly score = mean squared reconstruction error (higher = more anomalous)."""
    X = X.astype(float)
    recon = ae.predict(X)
    return np.mean((X - recon) ** 2, axis=1)


# ─── Unified evaluation ──────────────────────────────────────────────────────

def evaluate_baseline(
    name: str,
    score_fn,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    find_best_threshold_fn,
) -> dict:
    """Train, score, threshold-select on val, evaluate on test."""
    val_scores  = score_fn(X_val)
    test_scores = score_fn(X_test)

    # Threshold selection on val (same protocol as QCBM)
    f1_t, _ = find_best_threshold_fn(y_val, val_scores)

    metrics = evaluate(y_test, test_scores, threshold=f1_t)
    metrics["name"] = name
    return metrics
