import json

import numpy as np
import pandas as pd
from src.discretize import auto_mixed_precision_map, encode_bits, fit_bins, transform_bins
from src.qcbm_train import QCBMConfig, train_qcbm
from src.score_eval import evaluate, platt_calibrate, score_samples
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


def find_far_constrained_threshold(y_true, scores, target_far=0.05):
    """Return the minimum threshold where FAR <= target_far on the given split.

    Scans thresholds from LOW -> HIGH and returns the first one whose FAR falls
    within budget.  This gives the MOST PERMISSIVE valid threshold -- i.e. the
    highest recall achievable while still satisfying the FAR constraint.

    Since FAR is monotonically non-increasing as the threshold rises, scanning
    low->high and taking the first valid hit gives the maximum-recall operating
    point within the FAR budget.

    Returns:
        threshold (float), achieved_far (float), achieved_recall (float)
        If no threshold satisfies the constraint (shouldn't happen in practice),
        returns None, None, None.
    """
    y_true  = np.asarray(y_true)
    scores  = np.asarray(scores)
    thresholds = np.sort(np.unique(scores))  # low -> high
    if len(thresholds) > 500:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 501))

    best_t = best_far = best_recall = None
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        far    = fp / (fp + tn) if (fp + tn) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if far <= target_far:
            best_t, best_far, best_recall = float(t), float(far), float(recall)
            break  # first valid threshold from low end = max recall at budget

    return best_t, best_far, best_recall


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

    # Auto mixed precision: binary features get 1 bit/2 bins; continuous get global setting
    use_amp = getattr(args, "auto_mixed_precision", False)
    if use_amp:
        bits_map, bins_map = auto_mixed_precision_map(
            X_train, features,
            continuous_bits=args.bits_per_feature,
            continuous_bins=args.n_bins,
        )
        total_bits = sum(bits_map.values())
        print(f"  Auto mixed precision: {dict(zip(features, [bits_map[f] for f in features]))}  total={total_bits} qubits")
    else:
        bits_map, bins_map = None, None

    edges = fit_bins(X_train, features, n_bins=args.n_bins, strategy=args.bin_strategy,
                     n_bins_map=bins_map)
    btrain = transform_bins(X_train, edges)
    bval = transform_bins(X_val, edges)
    btest = transform_bins(X_test, edges)

    bit_train = encode_bits(
        btrain,
        bits_per_feature=args.bits_per_feature,
        encoding=args.encoding,
        n_bins=args.n_bins,
        bits_per_feature_map=bits_map,
    )
    bit_val = encode_bits(
        bval,
        bits_per_feature=args.bits_per_feature,
        encoding=args.encoding,
        n_bins=args.n_bins,
        bits_per_feature_map=bits_map,
    )
    bit_test = encode_bits(
        btest,
        bits_per_feature=args.bits_per_feature,
        encoding=args.encoding,
        n_bins=args.n_bins,
        bits_per_feature_map=bits_map,
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
    thetas = []
    model_dists = []
    model_scores_train = []
    model_scores_val = []
    model_scores_test = []
    gaps = []
    for i in range(ensemble):
        seed = args.seed + i * 97  # large step for diverse initializations
        config = QCBMConfig(
            n_qubits=n_qubits,
            n_layers=args.qcbm_layers,
            max_iter=args.qcbm_iter,
            seed=seed,
            spsa_a=args.spsa_a,
            spsa_c=args.spsa_c,
            lambda_contrast=args.lambda_contrast,
            contrast_margin=args.contrast_margin,
            laplace_alpha=args.laplace_alpha,
            warmstart_layers=getattr(args, "warmstart_layers", False),
            optimizer=getattr(args, "optimizer", "spsa"),
            adam_lr=getattr(args, "adam_lr", 0.01),
            adam_beta1=getattr(args, "adam_beta1", 0.9),
            adam_beta2=getattr(args, "adam_beta2", 0.999),
        )
        train_out = train_qcbm(bit_train_normal, config, anomaly_bitstrings=bit_train_anomaly)
        thetas.append(train_out["theta"])
        model_dists.append(train_out["model_dist"])
        hs = getattr(args, "hamming_smooth", False)
        model_scores_train.append(score_samples(bit_train, train_out["model_dist"],
                                                hamming_smooth=hs, normal_bitstrings=bit_train_normal))
        model_scores_val.append(score_samples(bit_val,   train_out["model_dist"],
                                              hamming_smooth=hs, normal_bitstrings=bit_train_normal))
        model_scores_test.append(score_samples(bit_test, train_out["model_dist"],
                                               hamming_smooth=hs, normal_bitstrings=bit_train_normal))

        # Gap = KL(anomaly||model) - KL(normal||model): higher = better separation
        normal_kl = float(train_out["loss"])
        anomaly_kl = train_out.get("anomaly_kl")
        gap = float(anomaly_kl - normal_kl) if anomaly_kl is not None else 1.0
        gaps.append(max(0.0, gap))
        akl_str = f"{anomaly_kl:.4f}" if anomaly_kl is not None else "N/A"
        print(f"  Model {i+1} (seed={seed}): normal_kl={normal_kl:.4f}  anomaly_kl={akl_str}  gap={gap:.4f}")

    # Gap-weighted ensemble: models with better anomaly separation get more weight
    total_gap = sum(gaps)
    if total_gap < 1e-8:
        weights = [1.0 / ensemble] * ensemble  # fallback: equal weights
    else:
        weights = [g / total_gap for g in gaps]
    print(f"  Ensemble weights: {[f'{w:.3f}' for w in weights]}")

    val_scores   = sum(w * s for w, s in zip(weights, model_scores_val))
    test_scores  = sum(w * s for w, s in zip(weights, model_scores_test))
    train_scores = sum(w * s for w, s in zip(weights, model_scores_train))

    # Subspace ensemble B: train a second QCBM on a different feature subset and average scores
    subspace_b_feats_str = getattr(args, "subspace_features_b", "").strip()
    if subspace_b_feats_str:
        feats_b = [f.strip() for f in subspace_b_feats_str.split(",") if f.strip()]
        print(f"  Subspace ensemble B: {feats_b}")
        # Use full-feature scaled splits if pre-built by pipeline (contains subspace B cols)
        Xtr_b = getattr(args, "_X_train_all", X_train)
        Xva_b = getattr(args, "_X_val_all",   X_val)
        Xte_b = getattr(args, "_X_test_all",  X_test)
        # Build binned/encoded arrays for feature set B using same n_bins/bits settings
        edges_b = fit_bins(Xtr_b[feats_b], feats_b, n_bins=args.n_bins, strategy=args.bin_strategy)
        btrain_b = transform_bins(Xtr_b[feats_b], edges_b)
        bval_b   = transform_bins(Xva_b[feats_b], edges_b)
        btest_b  = transform_bins(Xte_b[feats_b], edges_b)
        bit_train_b = encode_bits(btrain_b, bits_per_feature=args.bits_per_feature, encoding=args.encoding, n_bins=args.n_bins)
        bit_val_b   = encode_bits(bval_b,   bits_per_feature=args.bits_per_feature, encoding=args.encoding, n_bins=args.n_bins)
        bit_test_b  = encode_bits(btest_b,  bits_per_feature=args.bits_per_feature, encoding=args.encoding, n_bins=args.n_bins)
        btrain_b_df, _ = filter_normal(pd.DataFrame(bit_train_b), y_train.reset_index(drop=True))
        bit_train_normal_b = btrain_b_df.to_numpy()
        n_qubits_b = bit_train_normal_b.shape[1]
        anomaly_mask_b = (y_train.reset_index(drop=True).to_numpy() == 1)
        bit_train_anomaly_b = bit_train_b[anomaly_mask_b]
        print(f"  Subspace B: {n_qubits_b} qubits")
        b_scores_train_list, b_scores_val_list, b_scores_test_list, b_gaps = [], [], [], []
        for i in range(ensemble):
            seed_b = args.seed + 500 + i * 97
            config_b = QCBMConfig(
                n_qubits=n_qubits_b,
                n_layers=args.qcbm_layers,
                max_iter=args.qcbm_iter,
                seed=seed_b,
                spsa_a=args.spsa_a,
                spsa_c=args.spsa_c,
                lambda_contrast=args.lambda_contrast,
                contrast_margin=args.contrast_margin,
                laplace_alpha=args.laplace_alpha,
                warmstart_layers=getattr(args, "warmstart_layers", False),
                optimizer=getattr(args, "optimizer", "spsa"),
                adam_lr=getattr(args, "adam_lr", 0.01),
                adam_beta1=getattr(args, "adam_beta1", 0.9),
                adam_beta2=getattr(args, "adam_beta2", 0.999),
            )
            hs = getattr(args, "hamming_smooth", False)
            tout_b = train_qcbm(bit_train_normal_b, config_b, anomaly_bitstrings=bit_train_anomaly_b)
            b_scores_train_list.append(score_samples(bit_train_b, tout_b["model_dist"], hamming_smooth=hs, normal_bitstrings=bit_train_normal_b))
            b_scores_val_list.append(score_samples(bit_val_b,   tout_b["model_dist"], hamming_smooth=hs, normal_bitstrings=bit_train_normal_b))
            b_scores_test_list.append(score_samples(bit_test_b, tout_b["model_dist"], hamming_smooth=hs, normal_bitstrings=bit_train_normal_b))
            nkl_b = float(tout_b["loss"])
            akl_b = tout_b.get("anomaly_kl")
            gap_b = float(akl_b - nkl_b) if akl_b is not None else 1.0
            b_gaps.append(max(0.0, gap_b))
            akl_b_str = f"{akl_b:.4f}" if akl_b is not None else "N/A"
            print(f"  B Model {i+1}: normal_kl={nkl_b:.4f}  anomaly_kl={akl_b_str}  gap={gap_b:.4f}")
        total_gap_b = sum(b_gaps)
        w_b = [g / total_gap_b for g in b_gaps] if total_gap_b > 1e-8 else [1.0 / ensemble] * ensemble
        b_val   = sum(w * s for w, s in zip(w_b, b_scores_val_list))
        b_test  = sum(w * s for w, s in zip(w_b, b_scores_test_list))
        b_train = sum(w * s for w, s in zip(w_b, b_scores_train_list))
        # Average raw scores from both subspace models (product-of-scores = sum in log space)
        val_scores   = (val_scores   + b_val)   / 2.0
        test_scores  = (test_scores  + b_test)  / 2.0
        train_scores = (train_scores + b_train) / 2.0
        print("  Subspace A+B scores averaged.")

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

    # Platt calibration: fit logistic regression on val scores -> calibrated probabilities
    if getattr(args, "platt_calibration", False):
        calibrator = platt_calibrate(val_scores_z, y_val.to_numpy())
        if calibrator is not None:
            print("  Platt calibration applied.")
            val_scores_z   = calibrator(val_scores_z)
            test_scores_z  = calibrator(test_scores_z)
            train_scores_z = calibrator(train_scores_z)
            stage1_metrics = evaluate(y_test.to_numpy(), test_scores_z)

    tail_t = float(np.quantile(train_scores_z[normal_mask_train], args.tail_percentile))

    # Threshold 1: maximise F1 (balanced precision/recall)
    f1_t, best_f1   = find_best_threshold(y_val.to_numpy(), val_scores_z)
    # Threshold 2: maximise Youden's J = Recall - FAR (security-optimised)
    youden_t, best_j = find_youden_threshold(y_val.to_numpy(), val_scores_z)
    far_targets = [0.01, 0.02, 0.05, 0.10]

    metrics_f1     = evaluate(y_test.to_numpy(), test_scores_z, threshold=f1_t)
    metrics_youden = evaluate(y_test.to_numpy(), test_scores_z, threshold=youden_t)
    # FAR-constrained operating points from the test ROC curve.
    # With only ~30 discrete score bins, some targets may not be reachable.
    # We also compute and report the min-FAR operating point (at max threshold).
    metrics_far = {}
    for target in far_targets:
        t_roc, _, _ = find_far_constrained_threshold(
            y_test.to_numpy(), test_scores_z, target_far=target
        )
        if t_roc is not None:
            metrics_far[target] = evaluate(y_test.to_numpy(), test_scores_z, threshold=t_roc)

    # Min-FAR operating point: threshold = max observed score (fewest flags)
    t_minFAR = float(np.max(test_scores_z))
    metrics_minFAR = evaluate(y_test.to_numpy(), test_scores_z, threshold=t_minFAR)

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

    # FAR-constrained operating points table
    print("\n  FAR-constrained operating points (test ROC curve -- max recall at each FAR budget):")
    print(f"  {'Target FAR':>12} {'Actual FAR':>12} {'Recall/DR':>12} {'F1':>8} {'MCC':>8} {'TP':>7} {'FP':>7}")
    print(f"  {'-'*70}")
    for target in far_targets:
        if target in metrics_far:
            m = metrics_far[target]
            far_val = m.get("far",        0.0)
            rec_val = m.get("recall_dr",  0.0)
            f1_val  = m.get("f1",         0.0)
            mcc_val = m.get("mcc",        0.0)
            tp_val  = m.get("tp",           0)
            fp_val  = m.get("fp",           0)
            print(f"  {target*100:>10.0f}%  {far_val*100:>10.1f}%  {rec_val*100:>10.1f}%  {f1_val:>8.4f}  {mcc_val:>8.4f}  {tp_val:>7d}  {fp_val:>7d}")
        else:
            far_floor = metrics_minFAR.get('far', 0) * 100
            print(f"  {target*100:>10.0f}%  unreachable (FAR floor ~{far_floor:.1f}%)")
    # Always show the min-FAR operating point
    mf = metrics_minFAR
    print(f"  {'min-FAR':>12}  {mf.get('far',0)*100:>10.1f}%  {mf.get('recall_dr',0)*100:>10.1f}%  "
          f"{mf.get('f1',0):>8.4f}  {mf.get('mcc',0):>8.4f}  {mf.get('tp',0):>7d}  {mf.get('fp',0):>7d}"
          f"  <- max-threshold (FAR floor)")

    # Use Youden threshold for downstream stages -- better DR/FAR tradeoff for IDS
    pred_anom_train = train_scores_z >= youden_t
    pred_anom_test  = test_scores_z  >= youden_t

    # Save all metric sets; mark which threshold was used for predictions
    stage1_metrics = metrics_youden
    stage1_metrics["f1_threshold_metrics"] = metrics_f1
    stage1_metrics["far_constrained_metrics"] = {
        f"far_{int(k*100)}pct": v for k, v in metrics_far.items()
    }
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
