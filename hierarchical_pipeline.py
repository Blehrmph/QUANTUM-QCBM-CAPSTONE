import argparse
import json

import numpy as np
import pandas as pd

from src.classical_lr import _import_sklearn
from src.data.preprocessing import (
    DEFAULT_FEATURES,
    DEFAULT_LOG1P_COLS,
    Scaler,
    apply_log1p,
    select_features,
)
from src.discretize import encode_bits, fit_bins, transform_bins
from src.qcbm_train import QCBMConfig, train_qcbm
from src.score_eval import evaluate, score_samples
from src.training_setup import filter_normal, train_val_test_split


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Hierarchical IDS pipeline (QCBM + classical).")
    parser.add_argument("--input", default="datasets/UNSW-NB15_core_features.csv")
    parser.add_argument("--label-input", default="datasets/UNSW-NB15_cleaned.csv")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--attack-col", default="attack_cat")
    parser.add_argument("--subtype-col", default="")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--features", default="dur,sbytes,dbytes,tcprtt")
    parser.add_argument("--log1p", action="store_true")
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--bits-per-feature", type=int, default=1)
    parser.add_argument("--bin-strategy", choices=["quantile", "uniform"], default="quantile")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qcbm-layers", type=int, default=3)
    parser.add_argument("--qcbm-iter", type=int, default=400)
    parser.add_argument("--spsa-a", type=float, default=0.2)
    parser.add_argument("--spsa-c", type=float, default=0.1)
    parser.add_argument("--min-subtype-samples", type=int, default=10)
    parser.add_argument("--mi-top-k", type=int, default=8)
    parser.add_argument("--var-threshold", type=float, default=0.0)
    parser.add_argument("--tail-percentile", type=float, default=0.99)
    parser.add_argument("--hybrid-alpha", type=float, default=0.7)
    return parser


def detect_subtype_column(columns):
    candidates = ["attack_subcat", "attack_subtype", "subtype", "subcat"]
    for col in candidates:
        if col in columns:
            return col
    return None


def load_dataset(args):
    df = pd.read_csv(args.input, low_memory=False)
    need_cols = [args.label_col, args.attack_col]
    subtype_col = args.subtype_col.strip()
    if subtype_col:
        need_cols.append(subtype_col)

    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        labels_df = pd.read_csv(args.label_input, usecols=missing, low_memory=False)
        if len(labels_df) != len(df):
            raise ValueError("Label file row count does not match input file.")
        for col in missing:
            df[col] = labels_df[col]
    return df


def find_best_threshold(y_true, scores):
    thresholds = np.unique(scores)
    if len(thresholds) > 200:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 201))

    best_t = thresholds[0]
    best_f1 = -1.0
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t), float(best_f1)


def train_multiclass_logreg(X_train, y_train):
    LogisticRegression = _import_sklearn()
    model = LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=None)
    model.fit(X_train, y_train)
    return model


def train_binary_logreg(X_train, y_train):
    LogisticRegression = _import_sklearn()
    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train, y_train)
    return model


def apply_feature_selection(X_train, y_train, features, top_k, var_threshold):
    if top_k <= 0:
        return X_train, features

    from sklearn.feature_selection import mutual_info_classif

    variances = X_train.var(axis=0, ddof=0)
    keep_mask = variances > var_threshold
    if keep_mask.sum() == 0:
        return X_train, features

    X_var = X_train.loc[:, keep_mask]
    features_var = [f for f, keep in zip(features, keep_mask.tolist()) if keep]
    k = min(top_k, len(features_var))

    mi = mutual_info_classif(X_var, y_train, discrete_features=False, random_state=0)
    order = np.argsort(mi)[::-1]
    keep_idx = order[:k]
    selected_features = [features_var[i] for i in keep_idx]
    X_sel = X_var[selected_features]
    return X_sel, selected_features


def zscore(scores, mu, sigma):
    denom = sigma if sigma else 1.0
    return (scores - mu) / denom


def main():
    args = build_arg_parser().parse_args()

    print("Loading dataset...")
    df = load_dataset(args)
    if args.attack_col not in df.columns:
        raise ValueError(f"Missing attack column: {args.attack_col}")

    subtype_col = args.subtype_col.strip() or detect_subtype_column(df.columns)

    print("Selecting features...")
    features = [c.strip() for c in args.features.split(",") if c.strip()]
    X = select_features(df, features)
    y = df[args.label_col]
    attack_cat = df[args.attack_col]
    attack_sub = df[subtype_col] if subtype_col else None

    print("Splitting train/val/test...")
    splits = train_val_test_split(
        X,
        y,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        stratify=True,
    )

    train_idx = splits.X_train.index
    val_idx = splits.X_val.index
    test_idx = splits.X_test.index
    attack_train = attack_cat.loc[train_idx].reset_index(drop=True)
    attack_val = attack_cat.loc[val_idx].reset_index(drop=True)
    attack_test = attack_cat.loc[test_idx].reset_index(drop=True)
    if attack_sub is not None:
        sub_train = attack_sub.loc[train_idx].reset_index(drop=True)
        sub_val = attack_sub.loc[val_idx].reset_index(drop=True)
        sub_test = attack_sub.loc[test_idx].reset_index(drop=True)
    else:
        sub_train = sub_val = sub_test = None

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

    print("Stage 1: Training QCBM on normal traffic...")
    edges = fit_bins(X_train, features, n_bins=args.n_bins, strategy=args.bin_strategy)
    btrain = transform_bins(X_train, edges)
    bval = transform_bins(X_val, edges)
    btest = transform_bins(X_test, edges)

    bit_train = encode_bits(btrain, bits_per_feature=args.bits_per_feature)
    bit_val = encode_bits(bval, bits_per_feature=args.bits_per_feature)
    bit_test = encode_bits(btest, bits_per_feature=args.bits_per_feature)

    btrain_df, ytrain = filter_normal(pd.DataFrame(bit_train), splits.y_train.reset_index(drop=True))
    bit_train_normal = btrain_df.to_numpy()

    n_qubits = bit_train_normal.shape[1]
    if n_qubits > 16:
        raise ValueError(
            f"n_qubits={n_qubits} is too large for statevector QCBM. "
            "Reduce features or bits-per-feature."
        )

    config = QCBMConfig(
        n_qubits=n_qubits,
        n_layers=args.qcbm_layers,
        max_iter=args.qcbm_iter,
        seed=args.seed,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
    )
    train_out = train_qcbm(bit_train_normal, config)

    val_scores = score_samples(bit_val, train_out["model_dist"])
    test_scores = score_samples(bit_test, train_out["model_dist"])
    train_scores = score_samples(bit_train, train_out["model_dist"])

    best_t, best_f1 = find_best_threshold(splits.y_val.to_numpy(), val_scores)
    stage1_metrics = evaluate(splits.y_test.to_numpy(), test_scores)

    print("Stage 1 metrics:")
    print(f"ROC-AUC: {stage1_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {stage1_metrics['pr_auc']:.4f}")
    print(f"Best val F1: {best_f1:.4f} at threshold {best_t:.6f}")

    pred_anom_train = train_scores >= best_t
    pred_anom_test = test_scores >= best_t

    print("Stage 2: Training attack category classifier on true anomalies...")
    train_anom_mask = (splits.y_train.reset_index(drop=True).to_numpy() == 1)
    X_train_anom = X_train.reset_index(drop=True).iloc[train_anom_mask].reset_index(drop=True)
    y_train_cat = attack_train.reset_index(drop=True).iloc[train_anom_mask].astype(str).str.strip()
    keep = (y_train_cat.notna() & (y_train_cat != "")).to_numpy()
    X_train_anom = X_train_anom.reset_index(drop=True).iloc[keep].reset_index(drop=True)
    y_train_cat = y_train_cat.reset_index(drop=True).iloc[keep].reset_index(drop=True)

    if len(y_train_cat.unique()) < 2:
        print("Not enough attack categories to train Stage 2.")
        stage2_model = None
    else:
        stage2_model = train_multiclass_logreg(X_train_anom, y_train_cat)
        test_anom_mask = (splits.y_test.reset_index(drop=True).to_numpy() == 1)
        X_test_anom = X_test.reset_index(drop=True).iloc[test_anom_mask].reset_index(drop=True)
        y_test_cat = attack_test.reset_index(drop=True).iloc[test_anom_mask].astype(str).str.strip()
        keep_test = (y_test_cat.notna() & (y_test_cat != "")).to_numpy()
        X_test_anom = X_test_anom.reset_index(drop=True).iloc[keep_test].reset_index(drop=True)
        y_test_cat = y_test_cat.reset_index(drop=True).iloc[keep_test].reset_index(drop=True)
        if len(y_test_cat) > 0:
            y_pred_cat = stage2_model.predict(X_test_anom)
            from sklearn.metrics import accuracy_score, f1_score
            print("Stage 2 metrics (true anomalies only):")
            print(f"Accuracy: {accuracy_score(y_test_cat, y_pred_cat):.4f}")
            print(f"Macro-F1: {f1_score(y_test_cat, y_pred_cat, average='macro'):.4f}")
        else:
            print("No labeled anomalies available in test set for Stage 2 evaluation.")

        X_test_pred_anom = X_test.loc[pred_anom_test].reset_index(drop=True)
        if len(X_test_pred_anom) > 0:
            y_pred_cat_all = stage2_model.predict(X_test_pred_anom)
            unique, counts = np.unique(y_pred_cat_all, return_counts=True)
            print("Stage 2 predictions on Stage 1 anomalies:")
            for k, v in zip(unique.tolist(), counts.tolist()):
                print(f"{k}: {int(v)}")
        else:
            print("Stage 1 found no anomalies to pass to Stage 2.")

    if subtype_col:
        print("Stage 3: Training subtype classifiers per attack category...")
        if stage2_model is None:
            print("Skipping Stage 3 because Stage 2 is not trained.")
        else:
            subtype_models = {}
            for attack_type in sorted(y_train_cat.unique().tolist()):
                mask = y_train_cat == attack_type
                sub_y = sub_train.loc[train_anom_mask].reset_index(drop=True)
                sub_y = sub_y.loc[keep].loc[mask].astype(str).str.strip()
                sub_x = X_train_anom.loc[mask].reset_index(drop=True)
                sub_keep = sub_y.notna() & (sub_y != "")
                sub_y = sub_y.loc[sub_keep]
                sub_x = sub_x.loc[sub_keep]
                if len(sub_y) < args.min_subtype_samples or len(sub_y.unique()) < 2:
                    continue
                subtype_models[attack_type] = train_multiclass_logreg(sub_x, sub_y)

            if not subtype_models:
                print("No subtype models trained (insufficient data).")
            else:
                print(f"Trained subtype models: {len(subtype_models)}")
    else:
        print("No subtype column found; skipping Stage 3.")

    print("Saving artifacts...")
    from pathlib import Path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hier_scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2))
    (out_dir / "hier_features.json").write_text(json.dumps({"features": features}, indent=2))
    (out_dir / "hier_stage1_metrics.json").write_text(json.dumps(stage1_metrics, indent=2))
    (out_dir / "hier_qcbm_config.json").write_text(json.dumps(config.__dict__, indent=2))
    np.save(out_dir / "hier_qcbm_theta.npy", train_out["theta"])
    np.save(out_dir / "hier_qcbm_model_dist.npy", train_out["model_dist"])

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
