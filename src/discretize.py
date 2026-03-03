from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class BinEdges:
    edges: dict[str, list[float]]

    def to_dict(self) -> dict:
        return {"edges": self.edges}

    @classmethod
    def from_dict(cls, data: dict) -> "BinEdges":
        return cls(edges=data["edges"])


def fit_bins(
    df: pd.DataFrame,
    columns: Iterable[str],
    n_bins: int = 4,
    strategy: str = "quantile",
) -> BinEdges:
    edges: dict[str, list[float]] = {}
    for c in columns:
        values = df[c].to_numpy()
        if strategy == "quantile":
            qs = np.linspace(0, 1, n_bins + 1)
            bins = np.quantile(values, qs).tolist()
        elif strategy == "uniform":
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            bins = np.linspace(vmin, vmax, n_bins + 1).tolist()
        else:
            raise ValueError(f"Unknown binning strategy: {strategy}")
        bins = list(dict.fromkeys(bins))
        if len(bins) < 2:
            bins = [-np.inf, np.inf]
        bins[0] = -np.inf
        bins[-1] = np.inf
        edges[c] = bins
    return BinEdges(edges=edges)


def transform_bins(df: pd.DataFrame, edges: BinEdges) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c, bins in edges.edges.items():
        if len(bins) == 2:
            out[c] = 0
            continue
        out[c] = pd.cut(
            df[c],
            bins=bins,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        ).astype(int)
    return out


def _int_to_bits(values: np.ndarray, bits: int) -> np.ndarray:
    out = np.zeros((len(values), bits), dtype=np.int8)
    for i in range(bits):
        out[:, bits - 1 - i] = (values >> i) & 1
    return out


def encode_bits(
    binned: pd.DataFrame,
    bits_per_feature: int = 2,
    encoding: str = "binary",
    n_bins: int | None = None,
) -> np.ndarray:
    arrays = []
    for c in binned.columns:
        values = binned[c].to_numpy()
        if encoding == "gray":
            if n_bins != 3 or bits_per_feature != 2:
                raise ValueError("Gray encoding currently supports only n_bins=3 and bits_per_feature=2.")
            lut = np.asarray([[0, 0], [0, 1], [1, 1]], dtype=np.int8)
            arrays.append(lut[values])
        else:
            arrays.append(_int_to_bits(values, bits_per_feature))
    return np.concatenate(arrays, axis=1)


def bitstrings_to_indices(bitstrings: np.ndarray) -> np.ndarray:
    powers = 2 ** np.arange(bitstrings.shape[1] - 1, -1, -1)
    return (bitstrings * powers).sum(axis=1).astype(int)
