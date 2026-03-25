import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE


def train_xgboost_subtype(X_train, y_train):
    """XGBoost subtype classifier with SMOTE, per attack category.

    Consistent with CNN-LSTM hybrid approaches (Springer IJIT, 2025) that
    achieve up to 96.78% accuracy on UNSW-NB15 subtype classification using
    tabular feature representations.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    min_samples = int(np.bincount(y_enc).min())
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


def run_stage3(
    X_train_anom,
    y_train_cat,
    sub_train,
    min_subtype_samples,
):
    print("Stage 3: Training XGBoost subtype classifiers per attack category...")
    subtype_models = {}

    for attack_type in sorted(y_train_cat.unique().tolist()):
        mask = (y_train_cat == attack_type).values
        sub_y = sub_train.loc[mask].astype(str).str.strip()
        sub_x = X_train_anom.loc[mask].reset_index(drop=True)

        sub_keep = (sub_y.notna() & (sub_y != "") & (sub_y != "nan")).values
        sub_y = sub_y.loc[sub_keep].reset_index(drop=True)
        sub_x = sub_x.loc[sub_keep].reset_index(drop=True)

        if len(sub_y) < min_subtype_samples or len(sub_y.unique()) < 2:
            continue

        model, le = train_xgboost_subtype(sub_x.values, sub_y.values)
        subtype_models[attack_type] = (model, le)
        print(f"  [{attack_type}] trained on {len(sub_y)} samples, {len(sub_y.unique())} subtypes")

    if not subtype_models:
        print("No subtype models trained (insufficient data per category).")
    else:
        print(f"Trained {len(subtype_models)} subtype models: {list(subtype_models.keys())}")

    return subtype_models
