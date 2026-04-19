import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Normalize duplicate label variants before any mapping
LABEL_NORMALIZE = {
    "Backdoors": "Backdoor",
}

# Maps specific UNSW-NB15 attack categories to broad families.
# Stage 2 classifies into these 4 families; Stage 3 resolves
# the specific attack type within each family.
ATTACK_FAMILY_MAP = {
    "Generic":        "Flood",
    "DoS":            "Flood",
    "Fuzzers":        "Probe",
    "Analysis":       "Probe",
    "Reconnaissance": "Probe",
    "Exploits":       "Exploit",
    "Shellcode":      "Exploit",
    "Backdoor":       "Persistence",
    "Worms":          "Persistence",
}


def normalize_labels(labels):
    return np.array([LABEL_NORMALIZE.get(l, l) for l in labels])


def map_to_family(labels):
    return np.array([ATTACK_FAMILY_MAP.get(l, "Unknown") for l in labels])


def train_xgboost_classifier(X_train, y_train):
    """XGBoost broad-family classifier with SMOTE + Bayesian prior correction.

    SMOTE oversamples minority classes to balance training. After training,
    we store the true class priors from the original data so predictions can
    be corrected at inference time (model learned uniform priors; test data
    has real priors ~68% Flood, 0.9% Persistence etc.).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    # True class priors from original (pre-SMOTE) distribution
    counts = np.bincount(y_enc, minlength=len(le.classes_))
    true_priors = counts / counts.sum()

    # SMOTE for minority classes
    min_samples = int(counts.min())
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
    if k_neighbors >= 1 and len(np.unique(y_enc)) > 1:
        try:
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_enc)
        except Exception:
            X_res, y_res = X_train, y_enc
    else:
        X_res, y_res = X_train, y_enc

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_res, y_res)

    return model, le, true_priors


def predict_with_priors(model, X: np.ndarray, true_priors: np.ndarray) -> np.ndarray:
    """Bayesian prior correction: scale model proba by true class priors.

    The model was trained on SMOTE-balanced data (uniform priors ≈ 1/n_classes).
    Multiplying by true_priors / (1/n_classes) ∝ true_priors restores the correct
    posterior, preventing rare classes like Persistence from being over-predicted.
    """
    proba = model.predict_proba(X)
    adjusted = proba * true_priors
    return adjusted.argmax(axis=1)


def run_stage2(X_train, X_test, y_train, y_test, attack_train, attack_test, pred_anom_test):
    print("Stage 2: Training XGBoost broad-family classifier (Flood / Probe / Exploit / Persistence)...")

    # Build training set: anomalies only, normalize then map to broad families
    train_anom_mask = (y_train.reset_index(drop=True).to_numpy() == 1)
    X_train_anom = X_train.reset_index(drop=True).iloc[train_anom_mask].reset_index(drop=True)
    y_train_cat = attack_train.reset_index(drop=True).iloc[train_anom_mask].astype(str).str.strip()
    keep = (y_train_cat.notna() & (y_train_cat != "") & (y_train_cat != "nan")).to_numpy()
    X_train_anom = X_train_anom.iloc[keep].reset_index(drop=True)
    y_train_cat = y_train_cat.iloc[keep].reset_index(drop=True)

    if len(y_train_cat.unique()) < 2:
        print("Not enough attack categories to train Stage 2.")
        return None, None, None

    y_train_norm = normalize_labels(y_train_cat.values)
    y_train_family = map_to_family(y_train_norm)
    model, le, true_priors = train_xgboost_classifier(X_train_anom.values, y_train_family)
    print(f"  True class priors: { {le.classes_[i]: round(true_priors[i], 4) for i in range(len(le.classes_))} }")

    # Evaluate on true test anomalies
    test_anom_mask = (y_test.reset_index(drop=True).to_numpy() == 1)
    X_test_anom = X_test.reset_index(drop=True).iloc[test_anom_mask].reset_index(drop=True)
    y_test_cat = attack_test.reset_index(drop=True).iloc[test_anom_mask].astype(str).str.strip()
    keep_test = (y_test_cat.notna() & (y_test_cat != "") & (y_test_cat != "nan")).to_numpy()
    X_test_anom = X_test_anom.iloc[keep_test].reset_index(drop=True)
    y_test_cat = y_test_cat.iloc[keep_test].reset_index(drop=True)

    if len(y_test_cat) > 0:
        y_test_norm = normalize_labels(y_test_cat.values)
        y_test_family = map_to_family(y_test_norm)
        known_mask = np.isin(y_test_family, le.classes_)
        y_test_family_safe = np.where(known_mask, y_test_family, le.classes_[0])
        y_pred_enc = predict_with_priors(model, X_test_anom.values, true_priors)
        y_pred_family = le.inverse_transform(y_pred_enc)
        y_true_family = le.inverse_transform(le.transform(y_test_family_safe))

        print("Stage 2 metrics (broad families, true anomalies only):")
        print(f"  Accuracy : {accuracy_score(y_true_family, y_pred_family):.4f}")
        print(f"  Macro-F1 : {f1_score(y_true_family, y_pred_family, average='macro', zero_division=0):.4f}")
        print("\nPer-family report:")
        print(classification_report(y_true_family, y_pred_family, zero_division=0))

    # Predict on all Stage 1 flagged anomalies
    y_test_r = y_test.reset_index(drop=True).to_numpy()
    attack_test_r = attack_test.reset_index(drop=True).astype(str).str.strip()
    X_test_r = X_test.reset_index(drop=True)

    X_test_pred_anom = X_test_r.loc[pred_anom_test].reset_index(drop=True)
    if len(X_test_pred_anom) > 0:
        y_pred_enc_all = predict_with_priors(model, X_test_pred_anom.values, true_priors)
        y_pred_family_all = le.inverse_transform(y_pred_enc_all)
        unique, counts = np.unique(y_pred_family_all, return_counts=True)
        print("Stage 2 predictions on Stage 1 flagged samples (broad families):")
        for k, v in zip(unique.tolist(), counts.tolist()):
            print(f"  {k}: {int(v)}")
    else:
        print("Stage 1 found no anomalies to pass to Stage 2.")

    # End-to-end combined metric
    # For each test sample: correctly detected = Stage1 flagged it AND Stage2 classified family correctly
    n_total_attacks = int(y_test_r.sum())
    flagged_idx = np.where(pred_anom_test)[0]
    flagged_true_labels = y_test_r[flagged_idx]          # 1=attack, 0=normal
    flagged_attack_cats = attack_test_r.iloc[flagged_idx].values

    # Among flagged attacks, classify with Stage 2
    flagged_attack_mask = (flagged_true_labels == 1)
    n_s1_tp = int(flagged_attack_mask.sum())              # Stage 1 true positives

    s2_correct = 0
    if n_s1_tp > 0:
        X_flagged_attacks = X_test_r.iloc[flagged_idx[flagged_attack_mask]].reset_index(drop=True)
        cats_flagged = flagged_attack_cats[flagged_attack_mask]
        cats_norm = normalize_labels(cats_flagged)
        families_true = map_to_family(cats_norm)
        known = np.isin(families_true, le.classes_)
        families_safe = np.where(known, families_true, "Unknown")
        y_pred_s2 = le.inverse_transform(predict_with_priors(model, X_flagged_attacks.values, true_priors))
        s2_correct = int(np.sum(y_pred_s2 == families_safe))

    e2e_recall    = s2_correct / n_total_attacks if n_total_attacks > 0 else 0.0
    s2_given_s1   = s2_correct / n_s1_tp if n_s1_tp > 0 else 0.0
    n_flagged     = int(pred_anom_test.sum())
    e2e_precision = s2_correct / n_flagged if n_flagged > 0 else 0.0

    print("\n" + "=" * 60)
    print("  END-TO-END PIPELINE (Stage 1 + Stage 2)")
    print("=" * 60)
    print(f"  Total test attacks          : {n_total_attacks:,}")
    print(f"  Stage 1 flagged (all)       : {n_flagged:,}  (incl. FP)")
    print(f"  Stage 1 true positives      : {n_s1_tp:,}  (attacks correctly flagged)")
    print(f"  Stage 1 recall              : {n_s1_tp/n_total_attacks:.4f}")
    print(f"  S2 correct given S1 flagged : {s2_correct:,}  (correct family classification)")
    print(f"  S2 accuracy | S1 TPs        : {s2_given_s1:.4f}")
    print(f"  End-to-end recall           : {e2e_recall:.4f}  (attacks correctly detected+classified)")
    print(f"  End-to-end precision        : {e2e_precision:.4f}  (of S1 flags, correctly classified attacks)")
    print("=" * 60)

    e2e_metrics = {
        "n_total_attacks":   n_total_attacks,
        "n_s1_flagged":      n_flagged,
        "n_s1_true_positive":n_s1_tp,
        "s1_recall":         round(n_s1_tp / n_total_attacks, 6) if n_total_attacks else 0,
        "s2_accuracy_given_s1_tp": round(s2_given_s1, 6),
        "e2e_recall":        round(e2e_recall, 6),
        "e2e_precision":     round(e2e_precision, 6),
    }

    # Pass normalized attack_cat labels to Stage 3
    y_train_cat_norm = y_train_cat.copy()
    y_train_cat_norm[:] = y_train_norm
    return model, le, y_train_cat_norm, e2e_metrics
