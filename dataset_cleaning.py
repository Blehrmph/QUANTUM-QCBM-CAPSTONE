from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd


CATEGORICAL_COLS = [
    "srcip",
    "dstip",
    "proto",
    "state",
    "service",
    "attack_cat",
]


def _strip_whitespace(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["proto", "state", "service"]:
        if col in df.columns:
            df[col] = _strip_whitespace(df[col])

    if "service" in df.columns:
        df["service"] = df["service"].replace({"-": pd.NA, "": pd.NA})
        df["service"] = df["service"].fillna("Unknown")

    return df


def clean_numerics(
    df: pd.DataFrame,
    categorical_cols: Iterable[str] = CATEGORICAL_COLS,
    rare_nan_frac: float = 0.001,
) -> pd.DataFrame:
    df = df.copy()

    cat_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols = [c for c in df.columns if c not in cat_cols]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not numeric_cols:
        return df

    nan_counts = df[numeric_cols].isna().sum()
    total = len(df)

    rare_cols = [c for c in numeric_cols if total and (nan_counts[c] / total) <= rare_nan_frac]
    if rare_cols:
        df = df.dropna(subset=rare_cols)

    for col in numeric_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)

    return df


def engineer_features(df: pd.DataFrame, apply_log1p: bool = False) -> pd.DataFrame:
    df = df.copy()

    if {"sbytes", "dbytes"}.issubset(df.columns):
        df["total_bytes"] = df["sbytes"] + df["dbytes"]
        df["bytes_ratio"] = df["sbytes"] / (df["dbytes"] + 1)

    if {"Spkts", "Dpkts"}.issubset(df.columns):
        df["total_pkts"] = df["Spkts"] + df["Dpkts"]
        df["pkt_ratio"] = df["Spkts"] / (df["Dpkts"] + 1)

    if {"Stime", "Ltime"}.issubset(df.columns):
        df["recalc_dur"] = df["Ltime"] - df["Stime"]
        if "dur" in df.columns:
            diff = (df["dur"] - df["recalc_dur"]).abs()
            needs_replace = df["dur"].isna() | diff.gt(1e-6)
            df.loc[needs_replace, "dur"] = df.loc[needs_replace, "recalc_dur"]

    if apply_log1p:
        for col in ["sbytes", "dbytes", "Sload", "Dload"]:
            if col in df.columns:
                new_col = f"{col}_log1p"
                df[new_col] = pd.NA
                valid = df[col].notna() & df[col].ge(0)
                df.loc[valid, new_col] = np.log1p(df.loc[valid, col])

    return df


def clean_dataset(
    df: pd.DataFrame,
    rare_nan_frac: float = 0.001,
    apply_log1p: bool = False,
) -> pd.DataFrame:
    df = clean_categoricals(df)
    df = clean_numerics(df, rare_nan_frac=rare_nan_frac)
    df = engineer_features(df, apply_log1p=apply_log1p)
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean UNSW-NB15 dataset.")
    parser.add_argument(
        "--input",
        default="UNSW-NB15_combined.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output",
        default="UNSW-NB15_cleaned.csv",
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--rare-nan-frac",
        type=float,
        default=0.001,
        help="Drop rows with NaN in numeric columns if NaNs are very rare (fraction).",
    )
    parser.add_argument(
        "--log1p-skewed",
        action="store_true",
        help="Add log1p features for sbytes, dbytes, Sload, Dload.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cleaned = clean_dataset(df, rare_nan_frac=args.rare_nan_frac, apply_log1p=args.log1p_skewed)
    cleaned.to_csv(args.output, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(cleaned)}")


if __name__ == "__main__":
    main()
