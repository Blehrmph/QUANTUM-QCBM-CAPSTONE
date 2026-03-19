from __future__ import annotations

import numpy as np

from src.discretize import bitstrings_to_indices


def score_samples(
    bitstrings: np.ndarray,
    model_dist: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Score samples as anomaly candidates via negative log probability.

    High score = low probability under the learned normal distribution = likely anomaly.
    """
    indices = bitstrings_to_indices(bitstrings)
    probs = np.clip(model_dist[indices], eps, 1.0)
    return -np.log(probs)


def _roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tp_rate = tp / (tp[-1] if tp[-1] else 1.0)
    fp_rate = fp / (fp[-1] if fp[-1] else 1.0)
    return float(np.trapz(tp_rate, fp_rate))


def _pr_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / (tp[-1] if tp[-1] else 1.0)
    return float(np.trapz(precision, recall))


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Compute evaluation metrics for Stage 1 anomaly detection."""
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        roc_auc = float(roc_auc_score(y_true, scores))
        pr_auc  = float(average_precision_score(y_true, scores))
    except Exception:
        roc_auc = _roc_auc_score(y_true, scores)
        pr_auc  = _pr_auc_score(y_true, scores)

    metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc}

    if threshold is not None:
        try:
            from sklearn.metrics import (
                f1_score, precision_score, recall_score,
                matthews_corrcoef, confusion_matrix,
            )
            y_pred = (scores >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics.update({
                "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall_dr": float(recall_score(y_true, y_pred, zero_division=0)),
                "far":       far,
                "mcc":       float(matthews_corrcoef(y_true, y_pred)),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            })
        except Exception:
            pass

    return metrics
