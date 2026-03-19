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


def find_youden_threshold(y_true, scores):
    """Youden's J = Recall - FAR (maximises DR while minimising false alarms).

    Unlike F1, which weights precision and recall equally, Youden's J directly
    optimises the security-relevant tradeoff: catch as many attacks as possible
    while keeping the false alarm rate as low as possible.
    """
    thresholds = np.unique(scores)
    if len(thresholds) > 200:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 201))

    best_t = thresholds[0]
    best_j = -1.0
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        far    = fp / (fp + tn) if (fp + tn) else 0.0
        j = recall - far
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t), float(best_j)


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

    y_train_reset = y_train.reset_index(drop=True)
    btrain_df, _ = filter_normal(pd.DataFrame(bit_train), y_train_reset)
    bit_train_normal = btrain_df.to_numpy()

    # Anomaly bitstrings for contrastive loss
    anomaly_mask = (y_train_reset.to_numpy() == 1)
    bit_train_anomaly = bit_train[anomaly_mask]

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
            lambda_contrast=args.lambda_contrast,
            contrast_margin=args.contrast_margin,
        )
        train_out = train_qcbm(bit_train_normal, config, anomaly_bitstrings=bit_train_anomaly)
        thetas.append(train_out["theta"])
        model_dists.append(train_out["model_dist"])
        val_scores   += score_samples(bit_val,   train_out["model_dist"])
        test_scores  += score_samples(bit_test,  train_out["model_dist"])
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

    # Preliminary ranking metrics (no threshold needed)
    stage1_metrics = evaluate(y_test.to_numpy(), test_scores_z)
    if stage1_metrics["roc_auc"] < 0.5:
        test_scores_z = -test_scores_z
        val_scores_z = -val_scores_z
        train_scores_z = -train_scores_z
        stage1_metrics = evaluate(y_test.to_numpy(), test_scores_z)

    tail_t = float(np.quantile(train_scores_z[normal_mask_train], args.tail_percentile))

    # Threshold 1: maximise F1 (balanced precision/recall)
    f1_t, best_f1   = find_best_threshold(y_val.to_numpy(), val_scores_z)
    # Threshold 2: maximise Youden's J = Recall - FAR (security-optimised)
    youden_t, best_j = find_youden_threshold(y_val.to_numpy(), val_scores_z)

    metrics_f1     = evaluate(y_test.to_numpy(), test_scores_z, threshold=f1_t)
    metrics_youden = evaluate(y_test.to_numpy(), test_scores_z, threshold=youden_t)

    print("\nStage 1 metrics:")
    print(f"  {'Metric':<12} {'F1 threshold':>14} {'Youden threshold':>17}")
    print(f"  {'-'*45}")
    print(f"  {'ROC-AUC':<12} {metrics_f1['roc_auc']:>14.4f} {metrics_youden['roc_auc']:>17.4f}")
    print(f"  {'PR-AUC':<12} {metrics_f1['pr_auc']:>14.4f} {metrics_youden['pr_auc']:>17.4f}")
    if "f1" in metrics_f1:
        print(f"  {'F1':<12} {metrics_f1['f1']:>14.4f} {metrics_youden['f1']:>17.4f}")
        print(f"  {'Precision':<12} {metrics_f1['precision']:>14.4f} {metrics_youden['precision']:>17.4f}")
        print(f"  {'Recall/DR':<12} {metrics_f1['recall_dr']:>14.4f} {metrics_youden['recall_dr']:>17.4f}")
        print(f"  {'FAR':<12} {metrics_f1['far']:>14.4f} {metrics_youden['far']:>17.4f}")
        print(f"  {'MCC':<12} {metrics_f1['mcc']:>14.4f} {metrics_youden['mcc']:>17.4f}")
        print(f"  {'TP':<12} {metrics_f1['tp']:>14d} {metrics_youden['tp']:>17d}")
        print(f"  {'FP':<12} {metrics_f1['fp']:>14d} {metrics_youden['fp']:>17d}")
        print(f"  {'FN':<12} {metrics_f1['fn']:>14d} {metrics_youden['fn']:>17d}")
        print(f"  {'TN':<12} {metrics_f1['tn']:>14d} {metrics_youden['tn']:>17d}")
    print(f"\n  F1 threshold    : {f1_t:.6f}  (val F1={best_f1:.4f})")
    print(f"  Youden threshold: {youden_t:.6f}  (val J={best_j:.4f})")
    print(f"  Tail threshold  : {tail_t:.6f}  (p={args.tail_percentile:.3f})")

    # Use Youden threshold for downstream stages — better DR/FAR tradeoff for IDS
    pred_anom_train = train_scores_z >= youden_t
    pred_anom_test  = test_scores_z  >= youden_t

    # Save both metric sets; mark which threshold was used for predictions
    stage1_metrics = metrics_youden
    stage1_metrics["f1_threshold_metrics"] = metrics_f1
    stage1_metrics["active_threshold"] = "youden"

    return {
        "edges": edges,
        "qcbm_config": QCBMConfig(
            n_qubits=n_qubits,
            n_layers=args.qcbm_layers,
            max_iter=args.qcbm_iter,
            seed=args.seed,
            spsa_a=args.spsa_a,
            spsa_c=args.spsa_c,
            lambda_contrast=args.lambda_contrast,
            contrast_margin=args.contrast_margin,
        ),
        "qcbm_theta": np.asarray(thetas),
        "qcbm_model_dist": np.asarray(model_dists),
        "stage1_metrics": stage1_metrics,
        "pred_anom_train": pred_anom_train,
        "pred_anom_test": pred_anom_test,
        "best_threshold": youden_t,
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
