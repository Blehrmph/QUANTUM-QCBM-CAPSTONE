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
    n_bins_map: dict | None = None,
) -> BinEdges:
    """Fit bin edges per column.

    n_bins_map: optional dict mapping column name -> n_bins override.
    Columns not in the map use the global n_bins default.
    """
    edges: dict[str, list[float]] = {}
    for c in columns:
        nb = n_bins_map.get(c, n_bins) if n_bins_map else n_bins
        values = df[c].to_numpy()
        if strategy == "quantile":
            qs = np.linspace(0, 1, nb + 1)
            bins = np.quantile(values, qs).tolist()
        elif strategy == "uniform":
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            bins = np.linspace(vmin, vmax, nb + 1).tolist()
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
    bits_per_feature_map: dict | None = None,
) -> np.ndarray:
    """Encode binned integer columns into concatenated bit arrays.

    bits_per_feature_map: optional dict mapping column name -> bits override.
    Columns not in the map use the global bits_per_feature default.
    """
    arrays = []
    for c in binned.columns:
        values = binned[c].to_numpy()
        bits = bits_per_feature_map.get(c, bits_per_feature) if bits_per_feature_map else bits_per_feature
        if encoding == "gray":
            if n_bins != 3 or bits_per_feature != 2:
                raise ValueError("Gray encoding currently supports only n_bins=3 and bits_per_feature=2.")
            lut = np.asarray([[0, 0], [0, 1], [1, 1]], dtype=np.int8)
            arrays.append(lut[values])
        else:
            arrays.append(_int_to_bits(values, bits))
    return np.concatenate(arrays, axis=1)


def auto_mixed_precision_map(
    df: pd.DataFrame,
    columns: list,
    continuous_bits: int = 2,
    continuous_bins: int = 4,
) -> tuple[dict, dict]:
    """Return (bits_map, bins_map) assigning 1 bit/2 bins to binary features,
    continuous_bits/continuous_bins to all others.

    A feature is considered binary if it has <= 2 distinct values in df.
    """
    bits_map: dict[str, int] = {}
    bins_map: dict[str, int] = {}
    for c in columns:
        n_unique = df[c].nunique()
        if n_unique <= 2:
            bits_map[c] = 1
            bins_map[c] = 2
        else:
            bits_map[c] = continuous_bits
            bins_map[c] = continuous_bins
    return bits_map, bins_map


def bitstrings_to_indices(bitstrings: np.ndarray) -> np.ndarray:
    powers = 2 ** np.arange(bitstrings.shape[1] - 1, -1, -1)
    return (bitstrings * powers).sum(axis=1).astype(int)
