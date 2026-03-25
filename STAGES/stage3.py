import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from STAGES.stage2 import ATTACK_FAMILY_MAP, map_to_family


def train_xgboost_subtype(X_train, y_train):
    """XGBoost specific-category classifier within a broad family.

    Stage 3 resolves the exact UNSW-NB15 attack category given the
    broad family predicted by Stage 2. One model is trained per family.
    Consistent with CNN-LSTM hybrid approaches (Springer IJIT, 2025)
    achieving up to 96.78% accuracy on UNSW-NB15 subtype classification.
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
    model.fit(X_res, y_res)
    return model, le


def run_stage3(X_train_anom, y_train_cat, min_subtype_samples):
    """Train one XGBoost per broad family to classify specific attack types.

    Uses attack_cat labels from UNSW-NB15 as the subtype target.
    Family grouping is inherited from Stage 2's ATTACK_FAMILY_MAP.
    """
    print("Stage 3: Training XGBoost specific-category classifiers per attack family...")

    y_train_cat = y_train_cat.reset_index(drop=True)
    y_train_family = map_to_family(y_train_cat.values)

    subtype_models = {}
    families = sorted(set(ATTACK_FAMILY_MAP.values()))

    for family in families:
        mask = (y_train_family == family)
        sub_x = X_train_anom.loc[mask].reset_index(drop=True)
        sub_y = y_train_cat.loc[mask].reset_index(drop=True)

        if len(sub_y) < min_subtype_samples:
            print(f"  [{family}] skipped — only {len(sub_y)} samples")
            continue
        if len(sub_y.unique()) < 2:
            print(f"  [{family}] skipped — only one category ({sub_y.unique()[0]})")
            continue

        model, le = train_xgboost_subtype(sub_x.values, sub_y.values)
        if model is not None:
            subtype_models[family] = (model, le)
            cats = ", ".join(sorted(sub_y.unique().tolist()))
            print(f"  [{family}] trained on {len(sub_y)} samples | categories: {cats}")

    if not subtype_models:
        print("No Stage 3 models trained (insufficient data per family).")
    else:
        print(f"\nStage 3 complete: {len(subtype_models)} family models trained: {list(subtype_models.keys())}")

    return subtype_models
