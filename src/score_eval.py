from __future__ import annotations

import numpy as np

from src.discretize import bitstrings_to_indices


def score_samples(bitstrings: np.ndarray, model_dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        return {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)),
        }
    except Exception:
        return {
            "roc_auc": _roc_auc_score(y_true, scores),
            "pr_auc": _pr_auc_score(y_true, scores),
        }
