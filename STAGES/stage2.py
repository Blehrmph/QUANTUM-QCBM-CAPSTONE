import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE


def train_xgboost_classifier(X_train, y_train):
    """XGBoost multiclass classifier with SMOTE oversampling.

    Replicates the approach from Aldweesh et al. (2023) which achieves
    83.2% accuracy and macro F1 of 68.8% on UNSW-NB15 attack categories.
    SMOTE addresses the severe class imbalance across the 9 attack families.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    # Apply SMOTE only where minority classes have enough samples
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
    return model, le


def run_stage2(X_train, X_test, y_train, y_test, attack_train, attack_test, pred_anom_test):
    print("Stage 2: Training XGBoost attack category classifier (Aldweesh et al., 2023)...")

    train_anom_mask = (y_train.reset_index(drop=True).to_numpy() == 1)
    X_train_anom = X_train.reset_index(drop=True).iloc[train_anom_mask].reset_index(drop=True)
    y_train_cat = attack_train.reset_index(drop=True).iloc[train_anom_mask].astype(str).str.strip()
    keep = (y_train_cat.notna() & (y_train_cat != "") & (y_train_cat != "nan")).to_numpy()
    X_train_anom = X_train_anom.iloc[keep].reset_index(drop=True)
    y_train_cat = y_train_cat.iloc[keep].reset_index(drop=True)

    if len(y_train_cat.unique()) < 2:
        print("Not enough attack categories to train Stage 2.")
        return None, None

    model, le = train_xgboost_classifier(X_train_anom.values, y_train_cat.values)

    test_anom_mask = (y_test.reset_index(drop=True).to_numpy() == 1)
    X_test_anom = X_test.reset_index(drop=True).iloc[test_anom_mask].reset_index(drop=True)
    y_test_cat = attack_test.reset_index(drop=True).iloc[test_anom_mask].astype(str).str.strip()
    keep_test = (y_test_cat.notna() & (y_test_cat != "") & (y_test_cat != "nan")).to_numpy()
    X_test_anom = X_test_anom.iloc[keep_test].reset_index(drop=True)
    y_test_cat = y_test_cat.iloc[keep_test].reset_index(drop=True)

    if len(y_test_cat) > 0:
        y_enc_test = le.transform(
            np.where(np.isin(y_test_cat.values, le.classes_), y_test_cat.values, le.classes_[0])
        )
        y_pred_enc = model.predict(X_test_anom.values)
        y_pred_cat = le.inverse_transform(y_pred_enc)
        y_true_cat = le.inverse_transform(y_enc_test)

        print("Stage 2 metrics (true anomalies only):")
        print(f"  Accuracy : {accuracy_score(y_true_cat, y_pred_cat):.4f}")
        print(f"  Macro-F1 : {f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0):.4f}")
        print("\nPer-class report:")
        print(classification_report(y_true_cat, y_pred_cat, zero_division=0))
    else:
        print("No labeled anomalies available in test set for Stage 2 evaluation.")
        y_pred_cat = None

    X_test_pred_anom = X_test.loc[pred_anom_test].reset_index(drop=True)
    if len(X_test_pred_anom) > 0:
        y_pred_enc_all = model.predict(X_test_pred_anom.values)
        y_pred_cat_all = le.inverse_transform(y_pred_enc_all)
        unique, counts = np.unique(y_pred_cat_all, return_counts=True)
        print("Stage 2 predictions on Stage 1 anomalies:")
        for k, v in zip(unique.tolist(), counts.tolist()):
            print(f"  {k}: {int(v)}")
    else:
        print("Stage 1 found no anomalies to pass to Stage 2.")

    return model, le
