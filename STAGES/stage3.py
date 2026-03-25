import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

from STAGES.stage2 import ATTACK_FAMILY_MAP, map_to_family, normalize_labels


def train_xgboost_subtype(X_train, y_train):
    """XGBoost specific-category classifier within a broad family.

    Stage 3 resolves the exact UNSW-NB15 attack category given the
    broad family predicted by Stage 2. One model is trained per family.
    Uses balanced sample weights to handle within-family class imbalance.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    if len(np.unique(y_enc)) < 2:
        return None, le

    min_samples = int(np.bincount(y_enc).min())
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
    if k_neighbors >= 1:
        try:
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_enc)
        except Exception:
            X_res, y_res = X_train, y_enc
    else:
        X_res, y_res = X_train, y_enc

    sample_weights = compute_sample_weight("balanced", y_res)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_res, y_res, sample_weight=sample_weights)
    return model, le


def run_stage3(X_train_anom, y_train_cat, min_subtype_samples,
               X_test_anom=None, y_test_cat=None):
    """Train one XGBoost per broad family to classify specific attack types.
    Evaluates on test anomalies if provided.
    """
    print("Stage 3: Training XGBoost specific-category classifiers per attack family...")

    y_train_cat = y_train_cat.reset_index(drop=True)
    y_train_norm = normalize_labels(y_train_cat.values)
    y_train_family = map_to_family(y_train_norm)

    y_test_norm = normalize_labels(y_test_cat.values) if y_test_cat is not None else None
    y_test_family = map_to_family(y_test_norm) if y_test_norm is not None else None

    subtype_models = {}
    families = sorted(set(ATTACK_FAMILY_MAP.values()))

    for family in families:
        train_mask = (y_train_family == family)
        sub_x = X_train_anom.loc[train_mask].reset_index(drop=True)
        sub_y = np.array(y_train_norm)[train_mask]

        if len(sub_y) < min_subtype_samples:
            print(f"  [{family}] skipped — only {len(sub_y)} samples")
            continue
        if len(np.unique(sub_y)) < 2:
            print(f"  [{family}] skipped — only one category ({np.unique(sub_y)[0]})")
            continue

        model, le = train_xgboost_subtype(sub_x.values, sub_y)
        if model is None:
            continue

        subtype_models[family] = (model, le)
        cats = ", ".join(sorted(le.classes_.tolist()))
        print(f"  [{family}] trained on {len(sub_y)} samples | categories: {cats}")

        # Evaluate on test data for this family
        if X_test_anom is not None and y_test_family is not None:
            test_mask = (y_test_family == family)
            X_te = X_test_anom.loc[test_mask].reset_index(drop=True)
            y_te = np.array(y_test_norm)[test_mask]

            if len(X_te) == 0 or len(np.unique(y_te)) < 1:
                continue

            known = np.isin(y_te, le.classes_)
            y_te_safe = np.where(known, y_te, le.classes_[0])
            y_pred_enc = model.predict(X_te.values)
            y_pred = le.inverse_transform(y_pred_enc)
            y_true = y_te_safe

            acc = accuracy_score(y_true, y_pred)
            mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            print(f"    Test accuracy: {acc:.4f}  Macro-F1: {mf1:.4f}")
            report = classification_report(y_true, y_pred, zero_division=0)
            print("    " + report.replace("\n", "\n    "))

    if not subtype_models:
        print("No Stage 3 models trained (insufficient data per family).")
    else:
        print(f"Stage 3 complete: {len(subtype_models)} family models trained: {list(subtype_models.keys())}")

    return subtype_models
