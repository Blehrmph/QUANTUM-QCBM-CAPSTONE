from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "dur",
    "sbytes",
    "dbytes",
    "Sload",
    "Dload",
    "Spkts",
    "Dpkts",
    "tcprtt",
]

DEFAULT_LOG1P_COLS = ["sbytes", "dbytes", "Sload", "Dload", "byte_ratio", "load_ratio", "pkt_ratio"]


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


def add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary indicator and interaction columns.

    Binary indicators (from proto/state/service):
    - is_not_tcp  : non-TCP protocols (udp/rare) have higher attack rates
    - is_int_state: state=INT (incomplete connection) = 55% attack rate for UDP
    - is_con_state: state=CON (established) = 0.1% attack rate -- strongly normal
    - is_ssh      : ssh service = 0% attack rate -- perfect normal indicator
    - is_dns      : dns service = 26.9% attack rate -- strong attack indicator
    - is_http     : http service = 9.1% attack rate -- moderate attack indicator

    Interaction features (domain-motivated ratios):
    - byte_ratio  : sbytes / (dbytes + 1) -- traffic asymmetry; DoS/probe attacks
                    are highly asymmetric (one-directional), normal traffic is not
    - load_ratio  : Sload / (Dload + 1)  -- same asymmetry in load domain
    - pkt_ratio   : Spkts / (Dpkts + 1)  -- packet-count asymmetry

    These break the 99.95% bitstring overlap by creating new feature dimensions
    that separate normal from anomaly traffic even within shared bitstring regions.
    """
    out = df.copy()
    if "proto" in df.columns:
        out["is_not_tcp"] = (df["proto"] != "tcp").astype(float)
    if "state" in df.columns:
        out["is_int_state"] = (df["state"] == "INT").astype(float)
        out["is_con_state"] = (df["state"] == "CON").astype(float)
    if "service" in df.columns:
        out["is_ssh"]  = (df["service"] == "ssh").astype(float)
        out["is_dns"]  = (df["service"] == "dns").astype(float)
        out["is_http"] = (df["service"] == "http").astype(float)
    # Interaction / ratio features
    if "sbytes" in df.columns and "dbytes" in df.columns:
        out["byte_ratio"] = df["sbytes"] / (df["dbytes"].clip(lower=0) + 1.0)
    if "Sload" in df.columns and "Dload" in df.columns:
        out["load_ratio"] = df["Sload"] / (df["Dload"].clip(lower=0) + 1.0)
    if "Spkts" in df.columns and "Dpkts" in df.columns:
        out["pkt_ratio"]  = df["Spkts"] / (df["Dpkts"].clip(lower=0) + 1.0)
    return out


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
