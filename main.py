import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from discretize import BinEdges, encode_bits, fit_bins, transform_bins
from preprocessing import DEFAULT_FEATURES, DEFAULT_LOG1P_COLS, Scaler, apply_log1p, select_features
from qcbm_train import QCBMConfig, train_qcbm
from score_eval import evaluate, score_samples
from training_setup import filter_normal, train_val_test_split


def build_arg_parser():
    parser = argparse.ArgumentParser(description="QCBM pipeline for UNSW-NB15.")
    parser.add_argument(
        "--input",
        default="datasets/UNSW-NB15_cleaned.csv",
        help="Path to cleaned CSV.",
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory to write artifacts.")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES), help="Comma-separated features.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--log1p", action="store_true", help="Apply log1p to skewed columns.")
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--bits-per-feature", type=int, default=2)
    parser.add_argument("--bin-strategy", choices=["quantile", "uniform"], default="quantile")
    parser.add_argument("--normal-only", action="store_true", help="Train QCBM on normal class only.")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qcbm-layers", type=int, default=2)
    parser.add_argument("--qcbm-iter", type=int, default=200)
    parser.add_argument("--spsa-a", type=float, default=0.2)
    parser.add_argument("--spsa-c", type=float, default=0.1)
    return parser


def main():
    args = build_arg_parser().parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.input, low_memory=False)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    print("Selecting features...")
    features = [c.strip() for c in args.features.split(",") if c.strip()]
    X = select_features(df, features)
    y = df[args.label_col]

    print("Splitting train/val/test...")
    splits = train_val_test_split(
        X,
        y,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        stratify=True,
    )

    if args.log1p:
        print("Applying log1p to skewed features...")
        splits.X_train = apply_log1p(splits.X_train, DEFAULT_LOG1P_COLS)
        splits.X_val = apply_log1p(splits.X_val, DEFAULT_LOG1P_COLS)
        splits.X_test = apply_log1p(splits.X_test, DEFAULT_LOG1P_COLS)

    print(f"Scaling features ({args.scaler})...")
    scaler = Scaler(mode=args.scaler).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
    X_val = scaler.transform(splits.X_val, features)
    X_test = scaler.transform(splits.X_test, features)

    print("Fitting discretization bins...")
    edges = fit_bins(X_train, features, n_bins=args.n_bins, strategy=args.bin_strategy)
    print("Discretizing datasets...")
    btrain = transform_bins(X_train, edges)
    bval = transform_bins(X_val, edges)
    btest = transform_bins(X_test, edges)

    print("Encoding bitstrings...")
    bit_train = encode_bits(btrain, bits_per_feature=args.bits_per_feature)
    bit_val = encode_bits(bval, bits_per_feature=args.bits_per_feature)
    bit_test = encode_bits(btest, bits_per_feature=args.bits_per_feature)

    if args.normal_only:
        print("Filtering normal-only samples for training...")
        btrain_df, ytrain = filter_normal(pd.DataFrame(bit_train), splits.y_train.reset_index(drop=True))
        bit_train = btrain_df.to_numpy()
    else:
        ytrain = splits.y_train

    n_qubits = bit_train.shape[1]
    if n_qubits > 16:
        raise ValueError(
            f"n_qubits={n_qubits} is too large for statevector QCBM. "
            "Reduce features or bits-per-feature."
        )

    print(f"Training QCBM (qubits={n_qubits}, layers={args.qcbm_layers}, iters={args.qcbm_iter})...")
    config = QCBMConfig(
        n_qubits=n_qubits,
        n_layers=args.qcbm_layers,
        max_iter=args.qcbm_iter,
        seed=args.seed,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
    )
    train_out = train_qcbm(bit_train, config)

    print("Scoring test set...")
    scores = score_samples(bit_test, train_out["model_dist"])
    print("Evaluating metrics...")
    metrics = evaluate(splits.y_test.to_numpy(), scores)

    print("Saving artifacts...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2))
    (out_dir / "bins.json").write_text(json.dumps(edges.to_dict(), indent=2))
    (out_dir / "features.json").write_text(json.dumps({"features": features}, indent=2))
    (out_dir / "qcbm_config.json").write_text(json.dumps(config.__dict__, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out_dir / "theta.npy", train_out["theta"])
    np.save(out_dir / "model_dist.npy", train_out["model_dist"])

    print("Training complete.")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
