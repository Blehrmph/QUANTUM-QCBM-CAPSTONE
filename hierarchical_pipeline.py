import argparse
import json

import numpy as np
import pandas as pd
from src.data.preprocessing import (
    DEFAULT_FEATURES,
    DEFAULT_LOG1P_COLS,
    Scaler,
    add_categorical_features,
    apply_log1p,
    select_features,
)
from src.training_setup import train_val_test_split

from STAGES.stage1 import run_stage1, save_stage1_artifacts
from STAGES.stage2 import run_stage2
from STAGES.stage3 import run_stage3


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Hierarchical IDS pipeline (QCBM + classical).")
    parser.add_argument("--input", default="datasets/UNSW-NB15_core_features.csv")
    parser.add_argument("--label-input", default="datasets/UNSW-NB15_cleaned.csv")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--attack-col", default="attack_cat")
    parser.add_argument("--subtype-col", default="")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--features", default="dur,sbytes,dbytes,Sload,Dload,Spkts,Dpkts,tcprtt")
    parser.add_argument("--log1p", action="store_true", default=True)
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--n-bins", type=int, default=2)
    parser.add_argument("--bits-per-feature", type=int, default=1)
    parser.add_argument("--bin-strategy", choices=["quantile", "uniform"], default="quantile")
    parser.add_argument("--encoding", choices=["binary", "gray"], default="binary")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qcbm-layers", type=int, default=3)
    parser.add_argument("--qcbm-iter", type=int, default=800)
    parser.add_argument("--qcbm-ensemble", type=int, default=1)
    parser.add_argument("--spsa-a", type=float, default=0.2)
    parser.add_argument("--spsa-c", type=float, default=0.1)
    parser.add_argument("--lambda-contrast", type=float, default=0.5,
                        help="Weight of contrastive loss term (0 = disabled).")
    parser.add_argument("--contrast-margin", type=float, default=0.3,
                        help="Min KL distance the model must maintain from anomaly distribution.")
    parser.add_argument("--laplace-alpha", type=float, default=1.0,
                        help="Laplace smoothing on empirical distribution (0 = disabled).")
    parser.add_argument("--platt-calibration", action="store_true", default=False,
                        help="Apply Platt scaling to calibrate anomaly scores using validation labels.")
    parser.add_argument("--warmstart-layers", action="store_true", default=False,
                        help="Pre-train with n_layers-1 then expand to full depth (avoids barren plateaus).")
    parser.add_argument("--spsa-a-values", default="",
                        help="Comma-separated spsa_a values to sweep (e.g. '0.3,0.628,1.0').")
    parser.add_argument("--spsa-c-values", default="",
                        help="Comma-separated spsa_c values to sweep (e.g. '0.05,0.1,0.2').")
    parser.add_argument("--min-subtype-samples", type=int, default=10)
    parser.add_argument("--mi-top-k", type=int, default=8)
    parser.add_argument("--var-threshold", type=float, default=0.0)
    parser.add_argument("--tail-percentile", type=float, default=0.99)
    parser.add_argument("--stage1-only", action="store_true", help="Run only Stage 1.")
    parser.add_argument("--sweep", action="store_true", help="Run a small Stage 1 sweep and report best.")
    parser.add_argument("--sweep-bins", default="2,3", help="Comma-separated bins to try.")
    parser.add_argument("--sweep-encodings", default="binary,gray", help="Comma-separated encodings to try.")
    parser.add_argument("--sweep-ensembles", default="1,3", help="Comma-separated ensemble sizes to try.")
    parser.add_argument("--sweep-bits", default="1,2", help="Comma-separated bits per feature to try.")
    parser.add_argument("--sweep-iters", default="300,600", help="Comma-separated QCBM iters to try.")
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


def _parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    args = build_arg_parser().parse_args()

    print("Loading dataset...")
    df = load_dataset(args)
    if args.attack_col not in df.columns:
        raise ValueError(f"Missing attack column: {args.attack_col}")

    print("Engineering categorical features...")
    df = add_categorical_features(df)

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

    print("Feature selection (variance + mutual information)...")
    X_train_fs, selected_features = apply_feature_selection(
        splits.X_train, splits.y_train, features, args.mi_top_k, args.var_threshold
    )
    if len(selected_features) != len(features):
        print(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
    features = selected_features
    splits.X_train = splits.X_train[features]
    splits.X_val = splits.X_val[features]
    splits.X_test = splits.X_test[features]

    print(f"Scaling features ({args.scaler})...")
    scaler = Scaler(mode=args.scaler).fit(splits.X_train, features)
    X_train = scaler.transform(splits.X_train, features)
    X_val = scaler.transform(splits.X_val, features)
    X_test = scaler.transform(splits.X_test, features)

    if args.sweep:
        bins_list = _parse_int_list(args.sweep_bins)
        enc_list = _parse_str_list(args.sweep_encodings)
        ens_list = _parse_int_list(args.sweep_ensembles)
        bits_list = _parse_int_list(args.sweep_bits)
        iter_list = _parse_int_list(args.sweep_iters)

        results = []
        base_bits = args.bits_per_feature
        base_bins = args.n_bins
        base_enc = args.encoding
        base_ens = args.qcbm_ensemble
        base_iter = args.qcbm_iter

        print("Stage 1 sweep...")
        for n_bins in bins_list:
            for enc in enc_list:
                if n_bins == 2 and enc != "binary":
                    continue
                if n_bins == 3 and enc not in ("binary", "gray"):
                    continue
                for bits in bits_list:
                    for ens in ens_list:
                        for iters in iter_list:
                            if enc == "gray" and not (n_bins == 3 and bits == 2):
                                continue
                            args.n_bins = n_bins
                            args.encoding = enc
                            args.bits_per_feature = bits
                            args.qcbm_ensemble = ens
                            args.qcbm_iter = iters

                            print(
                                f"Try: bins={n_bins}, encoding={enc}, bits={bits}, "
                                f"ensemble={ens}, iters={iters}"
                            )
                            out = run_stage1(
                                X_train,
                                X_val,
                                X_test,
                                splits.y_train,
                                splits.y_val,
                                splits.y_test,
                                features,
                                args,
                            )
                            metrics = out["stage1_metrics"]
                            results.append(
                                {
                                    "n_bins": n_bins,
                                    "encoding": enc,
                                    "bits_per_feature": bits,
                                    "ensemble": ens,
                                    "qcbm_iter": iters,
                                    "roc_auc": metrics["roc_auc"],
                                    "pr_auc": metrics["pr_auc"],
                                    "f1": metrics.get("f1"),
                                    "recall_dr": metrics.get("recall_dr"),
                                    "far": metrics.get("far"),
                                    "mcc": metrics.get("mcc"),
                                }
                            )

        args.bits_per_feature = base_bits
        args.n_bins = base_bins
        args.encoding = base_enc
        args.qcbm_ensemble = base_ens
        args.qcbm_iter = base_iter

        if results:
            best = sorted(results, key=lambda r: (r["roc_auc"], r["pr_auc"]), reverse=True)[0]
            print("Best sweep result:")
            print(
                f"ROC-AUC: {best['roc_auc']:.4f} | PR-AUC: {best['pr_auc']:.4f} | "
                f"bins={best['n_bins']} enc={best['encoding']} bits={best['bits_per_feature']} "
                f"ensemble={best['ensemble']} iters={best['qcbm_iter']}"
            )

            print("Saving sweep results...")
            from pathlib import Path
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "hier_stage1_sweep.json").write_text(json.dumps(results, indent=2))

        print("Sweep complete.")
        return

    # SPSA hyperparameter sweep
    if args.spsa_a_values or args.spsa_c_values:
        a_list = [float(v) for v in args.spsa_a_values.split(",") if v.strip()] or [args.spsa_a]
        c_list = [float(v) for v in args.spsa_c_values.split(",") if v.strip()] or [args.spsa_c]
        spsa_results = []
        print(f"SPSA sweep: a={a_list}  c={c_list}")
        base_a, base_c = args.spsa_a, args.spsa_c
        for a_val in a_list:
            for c_val in c_list:
                args.spsa_a = a_val
                args.spsa_c = c_val
                print(f"  spsa_a={a_val}  spsa_c={c_val}")
                out = run_stage1(X_train, X_val, X_test,
                                 splits.y_train, splits.y_val, splits.y_test,
                                 features, args)
                m = out["stage1_metrics"]
                spsa_results.append({
                    "spsa_a": a_val, "spsa_c": c_val,
                    "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                    "f1": m.get("f1"), "recall_dr": m.get("recall_dr"), "far": m.get("far"),
                })
        args.spsa_a, args.spsa_c = base_a, base_c
        best = sorted(spsa_results, key=lambda r: (r["roc_auc"], r["pr_auc"]), reverse=True)[0]
        print(f"\nBest SPSA: a={best['spsa_a']}  c={best['spsa_c']}  "
              f"ROC-AUC={best['roc_auc']:.4f}  PR-AUC={best['pr_auc']:.4f}")
        from pathlib import Path
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "hier_spsa_sweep.json").write_text(json.dumps(spsa_results, indent=2))
        if args.stage1_only:
            print("SPSA sweep complete.")
            return

    stage1_out = run_stage1(
        X_train,
        X_val,
        X_test,
        splits.y_train,
        splits.y_val,
        splits.y_test,
        features,
        args,
    )

    if args.stage1_only:
        print("Saving artifacts...")
        from pathlib import Path
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "hier_scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2))
        (out_dir / "hier_features.json").write_text(json.dumps({"features": features}, indent=2))
        save_stage1_artifacts(out_dir, stage1_out)
        print("Stage 1 complete.")
        return

    stage2_model = run_stage2(
        X_train,
        X_test,
        splits.y_train,
        splits.y_test,
        attack_train,
        attack_test,
        stage1_out["pred_anom_test"],
    )

    if subtype_col:
        if stage2_model is None:
            print("Skipping Stage 3 because Stage 2 is not trained.")
        else:
            train_anom_mask = (splits.y_train.reset_index(drop=True).to_numpy() == 1)
            X_train_anom = X_train.reset_index(drop=True).iloc[train_anom_mask].reset_index(drop=True)
            y_train_cat = attack_train.reset_index(drop=True).iloc[train_anom_mask].astype(str).str.strip()
            keep = (y_train_cat.notna() & (y_train_cat != "")).to_numpy()
            X_train_anom = X_train_anom.reset_index(drop=True).iloc[keep].reset_index(drop=True)
            y_train_cat = y_train_cat.reset_index(drop=True).iloc[keep].reset_index(drop=True)
            sub_train_clean = sub_train.reset_index(drop=True).iloc[train_anom_mask].reset_index(drop=True)
            sub_train_clean = sub_train_clean.reset_index(drop=True).iloc[keep].reset_index(drop=True)
            run_stage3(
                X_train_anom,
                y_train_cat,
                sub_train_clean,
                args.min_subtype_samples,
            )
    else:
        print("No subtype column found; skipping Stage 3.")

    print("Saving artifacts...")
    from pathlib import Path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hier_scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2))
    (out_dir / "hier_features.json").write_text(json.dumps({"features": features}, indent=2))
    save_stage1_artifacts(out_dir, stage1_out)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
