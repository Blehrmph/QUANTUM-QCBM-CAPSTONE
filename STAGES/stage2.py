import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Maps specific UNSW-NB15 attack categories to broad families.
# Stage 2 classifies into these 4 families; Stage 3 then resolves
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
    "Backdoors":      "Persistence",
    "Worms":          "Persistence",
}


def map_to_family(labels):
    return np.array([ATTACK_FAMILY_MAP.get(l, "Unknown") for l in labels])


def train_xgboost_classifier(X_train, y_train):
    """XGBoost broad-family classifier with SMOTE oversampling.

    Stage 2 classifies flagged anomalies into 4 broad attack families:
    Flood, Probe, Exploit, Persistence. Replicates the XGBoost approach
    from Aldweesh et al. (2023) which achieves 83.2% accuracy on UNSW-NB15.
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
    print("Stage 2: Training XGBoost broad-family classifier (Flood / Probe / Exploit / Persistence)...")

    # Build training set: anomalies only, mapped to broad families
    train_anom_mask = (y_train.reset_index(drop=True).to_numpy() == 1)
    X_train_anom = X_train.reset_index(drop=True).iloc[train_anom_mask].reset_index(drop=True)
    y_train_cat = attack_train.reset_index(drop=True).iloc[train_anom_mask].astype(str).str.strip()
    keep = (y_train_cat.notna() & (y_train_cat != "") & (y_train_cat != "nan")).to_numpy()
    X_train_anom = X_train_anom.iloc[keep].reset_index(drop=True)
    y_train_cat = y_train_cat.iloc[keep].reset_index(drop=True)

    if len(y_train_cat.unique()) < 2:
        print("Not enough attack categories to train Stage 2.")
        return None, None, None

    y_train_family = map_to_family(y_train_cat.values)
    model, le = train_xgboost_classifier(X_train_anom.values, y_train_family)

    # Evaluate on true test anomalies
    test_anom_mask = (y_test.reset_index(drop=True).to_numpy() == 1)
    X_test_anom = X_test.reset_index(drop=True).iloc[test_anom_mask].reset_index(drop=True)
    y_test_cat = attack_test.reset_index(drop=True).iloc[test_anom_mask].astype(str).str.strip()
    keep_test = (y_test_cat.notna() & (y_test_cat != "") & (y_test_cat != "nan")).to_numpy()
    X_test_anom = X_test_anom.iloc[keep_test].reset_index(drop=True)
    y_test_cat = y_test_cat.iloc[keep_test].reset_index(drop=True)

    if len(y_test_cat) > 0:
        y_test_family = map_to_family(y_test_cat.values)
        known_mask = np.isin(y_test_family, le.classes_)
        y_test_family_safe = np.where(known_mask, y_test_family, le.classes_[0])
        y_pred_enc = model.predict(X_test_anom.values)
        y_pred_family = le.inverse_transform(y_pred_enc)
        y_true_family = le.inverse_transform(le.transform(y_test_family_safe))

        print("Stage 2 metrics (broad families, true anomalies only):")
        print(f"  Accuracy : {accuracy_score(y_true_family, y_pred_family):.4f}")
        print(f"  Macro-F1 : {f1_score(y_true_family, y_pred_family, average='macro', zero_division=0):.4f}")
        print("\nPer-family report:")
        print(classification_report(y_true_family, y_pred_family, zero_division=0))

    # Predict on all Stage 1 flagged anomalies
    X_test_pred_anom = X_test.loc[pred_anom_test].reset_index(drop=True)
    if len(X_test_pred_anom) > 0:
        y_pred_enc_all = model.predict(X_test_pred_anom.values)
        y_pred_family_all = le.inverse_transform(y_pred_enc_all)
        unique, counts = np.unique(y_pred_family_all, return_counts=True)
        print("Stage 2 predictions on Stage 1 anomalies (broad families):")
        for k, v in zip(unique.tolist(), counts.tolist()):
            print(f"  {k}: {int(v)}")
    else:
        print("Stage 1 found no anomalies to pass to Stage 2.")

    # Return model, encoder, and the specific attack_cat labels for Stage 3
    return model, le, y_train_cat
