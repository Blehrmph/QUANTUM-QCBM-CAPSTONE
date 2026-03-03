import json

import numpy as np
import pandas as pd
from src.discretize import encode_bits, fit_bins, transform_bins
from src.qcbm_train import QCBMConfig, train_qcbm
from src.score_eval import evaluate, score_samples
from src.training_setup import filter_normal


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


def zscore(scores, mu, sigma):
    denom = sigma if sigma else 1.0
    return (scores - mu) / denom


def run_stage1(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    features,
    args,
):
    print("Stage 1: Training QCBM on normal traffic...")
    edges = fit_bins(X_train, features, n_bins=args.n_bins, strategy=args.bin_strategy)
    btrain = transform_bins(X_train, edges)
    bval = transform_bins(X_val, edges)
    btest = transform_bins(X_test, edges)

    bit_train = encode_bits(
        btrain,
        bits_per_feature=args.bits_per_feature,
        encoding=args.encoding,
        n_bins=args.n_bins,
    )
    bit_val = encode_bits(
        bval,
        bits_per_feature=args.bits_per_feature,
        encoding=args.encoding,
        n_bins=args.n_bins,
    )
    bit_test = encode_bits(
        btest,
        bits_per_feature=args.bits_per_feature,
        encoding=args.encoding,
        n_bins=args.n_bins,
    )

    btrain_df, _ = filter_normal(pd.DataFrame(bit_train), y_train.reset_index(drop=True))
    bit_train_normal = btrain_df.to_numpy()

    n_qubits = bit_train_normal.shape[1]
    if n_qubits > 16:
        raise ValueError(
            f"n_qubits={n_qubits} is too large for statevector QCBM. "
            "Reduce features or bits-per-feature."
        )

    ensemble = max(1, int(args.qcbm_ensemble))
    if ensemble > 1:
        print(f"Ensembling QCBM models: {ensemble}")
    train_scores = np.zeros(len(bit_train), dtype=float)
    val_scores = np.zeros(len(bit_val), dtype=float)
    test_scores = np.zeros(len(bit_test), dtype=float)
    thetas = []
    model_dists = []
    for i in range(ensemble):
        seed = args.seed + i * 7
        config = QCBMConfig(
            n_qubits=n_qubits,
            n_layers=args.qcbm_layers,
            max_iter=args.qcbm_iter,
            seed=seed,
            spsa_a=args.spsa_a,
            spsa_c=args.spsa_c,
        )
        train_out = train_qcbm(bit_train_normal, config)
        thetas.append(train_out["theta"])
        model_dists.append(train_out["model_dist"])
        val_scores += score_samples(bit_val, train_out["model_dist"])
        test_scores += score_samples(bit_test, train_out["model_dist"])
        train_scores += score_samples(bit_train, train_out["model_dist"])

    val_scores /= ensemble
    test_scores /= ensemble
    train_scores /= ensemble

    normal_mask_train = (y_train.reset_index(drop=True).to_numpy() == 0)
    normal_scores_train = train_scores[normal_mask_train]
    mu = float(np.mean(normal_scores_train))
    sigma = float(np.std(normal_scores_train))
    val_scores_z = zscore(val_scores, mu, sigma)
    test_scores_z = zscore(test_scores, mu, sigma)
    train_scores_z = zscore(train_scores, mu, sigma)

    stage1_metrics = evaluate(y_test.to_numpy(), test_scores_z)
    if stage1_metrics["roc_auc"] < 0.5:
        test_scores_z = -test_scores_z
        val_scores_z = -val_scores_z
        train_scores_z = -train_scores_z
        stage1_metrics = evaluate(y_test.to_numpy(), test_scores_z)

    print("Stage 1 metrics:")
    print(f"ROC-AUC: {stage1_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {stage1_metrics['pr_auc']:.4f}")

    tail_t = float(np.quantile(train_scores_z[normal_mask_train], args.tail_percentile))
    best_t, best_f1 = find_best_threshold(y_val.to_numpy(), val_scores_z)

    print(f"Best val F1: {best_f1:.4f} at threshold {best_t:.6f}")
    print(f"Tail threshold (p={args.tail_percentile:.3f}): {tail_t:.6f}")

    pred_anom_train = train_scores_z >= best_t
    pred_anom_test = test_scores_z >= best_t

    return {
        "edges": edges,
        "qcbm_config": QCBMConfig(
            n_qubits=n_qubits,
            n_layers=args.qcbm_layers,
            max_iter=args.qcbm_iter,
            seed=args.seed,
            spsa_a=args.spsa_a,
            spsa_c=args.spsa_c,
        ),
        "qcbm_theta": np.asarray(thetas),
        "qcbm_model_dist": np.asarray(model_dists),
        "stage1_metrics": stage1_metrics,
        "pred_anom_train": pred_anom_train,
        "pred_anom_test": pred_anom_test,
        "best_threshold": best_t,
    }


def save_stage1_artifacts(out_dir, stage1_out):
    (out_dir / "hier_stage1_metrics.json").write_text(
        json.dumps(stage1_out["stage1_metrics"], indent=2)
    )
    (out_dir / "hier_qcbm_config.json").write_text(
        json.dumps(stage1_out["qcbm_config"].__dict__, indent=2)
    )
    np.save(out_dir / "hier_qcbm_theta.npy", stage1_out["qcbm_theta"])
    np.save(out_dir / "hier_qcbm_model_dist.npy", stage1_out["qcbm_model_dist"])
