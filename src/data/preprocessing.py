from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "dur",
    "sbytes",
    "dbytes",
    "Spkts",
    "Dpkts",
    "Sload",
    "Dload",
    "tcprtt",
]

DEFAULT_LOG1P_COLS = ["sbytes", "dbytes", "Sload", "Dload"]


@dataclass
class Scaler:
    mode: str = "standard"  # "standard" or "minmax"
    mean_: dict[str, float] | None = None
    std_: dict[str, float] | None = None
    min_: dict[str, float] | None = None
    max_: dict[str, float] | None = None

    def fit(self, df: pd.DataFrame, columns: Iterable[str]) -> "Scaler":
        cols = list(columns)
        if self.mode == "standard":
            self.mean_ = {c: float(df[c].mean()) for c in cols}
            self.std_ = {c: float(df[c].std(ddof=0)) for c in cols}
        elif self.mode == "minmax":
            self.min_ = {c: float(df[c].min()) for c in cols}
            self.max_ = {c: float(df[c].max()) for c in cols}
        else:
            raise ValueError(f"Unknown scaler mode: {self.mode}")
        return self

    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        cols = list(columns)
        out = df.copy()
        if self.mode == "standard":
            if self.mean_ is None or self.std_ is None:
                raise ValueError("Scaler not fitted.")
            for c in cols:
                std = self.std_.get(c, 0.0) or 1.0
                out[c] = (out[c] - self.mean_[c]) / std
        else:
            if self.min_ is None or self.max_ is None:
                raise ValueError("Scaler not fitted.")
            for c in cols:
                denom = (self.max_[c] - self.min_[c]) or 1.0
                out[c] = (out[c] - self.min_[c]) / denom
        return out

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mean_": self.mean_,
            "std_": self.std_,
            "min_": self.min_,
            "max_": self.max_,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scaler":
        return cls(
            mode=data.get("mode", "standard"),
            mean_=data.get("mean_"),
            std_=data.get("std_"),
            min_=data.get("min_"),
            max_=data.get("max_"),
        )


def apply_log1p(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = np.log1p(out[c].clip(lower=0))
    return out


def select_features(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[cols].copy()
