from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def _shuffle_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> SplitData:
    if not (0 < test_frac < 1 and 0 < val_frac < 1):
        raise ValueError("test_frac and val_frac must be in (0, 1).")

    n = len(X)
    if stratify:
        idx0 = np.where(y.to_numpy() == 0)[0]
        idx1 = np.where(y.to_numpy() == 1)[0]
        idx0 = _shuffle_indices(len(idx0), seed)
        idx1 = _shuffle_indices(len(idx1), seed + 1)
        def split_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            n_total = len(indices)
            n_test = int(n_total * test_frac)
            n_val = int(n_total * val_frac)
            test_idx = indices[:n_test]
            val_idx = indices[n_test : n_test + n_val]
            train_idx = indices[n_test + n_val :]
            return train_idx, val_idx, test_idx
        tr0, v0, te0 = split_indices(idx0)
        tr1, v1, te1 = split_indices(idx1)
        train_idx = np.concatenate([idx0[tr0], idx1[tr1]])
        val_idx = np.concatenate([idx0[v0], idx1[v1]])
        test_idx = np.concatenate([idx0[te0], idx1[te1]])
    else:
        idx = _shuffle_indices(n, seed)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def filter_normal(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask = y == 0
    return X.loc[mask], y.loc[mask]
