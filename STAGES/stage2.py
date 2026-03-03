import numpy as np

from src.classical_lr import _import_sklearn


def train_multiclass_logreg(X_train, y_train):
    LogisticRegression = _import_sklearn()
    model = LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=None)
    model.fit(X_train, y_train)
    return model


def run_stage2(X_train, X_test, y_train, y_test, attack_train, attack_test, pred_anom_test):
    print("Stage 2: Training attack category classifier on true anomalies...")
    train_anom_mask = (y_train.reset_index(drop=True).to_numpy() == 1)
    X_train_anom = X_train.reset_index(drop=True).iloc[train_anom_mask].reset_index(drop=True)
    y_train_cat = attack_train.reset_index(drop=True).iloc[train_anom_mask].astype(str).str.strip()
    keep = (y_train_cat.notna() & (y_train_cat != "")).to_numpy()
    X_train_anom = X_train_anom.reset_index(drop=True).iloc[keep].reset_index(drop=True)
    y_train_cat = y_train_cat.reset_index(drop=True).iloc[keep].reset_index(drop=True)

    if len(y_train_cat.unique()) < 2:
        print("Not enough attack categories to train Stage 2.")
        return None

    stage2_model = train_multiclass_logreg(X_train_anom, y_train_cat)

    test_anom_mask = (y_test.reset_index(drop=True).to_numpy() == 1)
    X_test_anom = X_test.reset_index(drop=True).iloc[test_anom_mask].reset_index(drop=True)
    y_test_cat = attack_test.reset_index(drop=True).iloc[test_anom_mask].astype(str).str.strip()
    keep_test = (y_test_cat.notna() & (y_test_cat != "")).to_numpy()
    X_test_anom = X_test_anom.reset_index(drop=True).iloc[keep_test].reset_index(drop=True)
    y_test_cat = y_test_cat.reset_index(drop=True).iloc[keep_test].reset_index(drop=True)

    if len(y_test_cat) > 0:
        y_pred_cat = stage2_model.predict(X_test_anom)
        from sklearn.metrics import accuracy_score, f1_score

        print("Stage 2 metrics (true anomalies only):")
        print(f"Accuracy: {accuracy_score(y_test_cat, y_pred_cat):.4f}")
        print(f"Macro-F1: {f1_score(y_test_cat, y_pred_cat, average='macro'):.4f}")
    else:
        print("No labeled anomalies available in test set for Stage 2 evaluation.")

    X_test_pred_anom = X_test.loc[pred_anom_test].reset_index(drop=True)
    if len(X_test_pred_anom) > 0:
        y_pred_cat_all = stage2_model.predict(X_test_pred_anom)
        unique, counts = np.unique(y_pred_cat_all, return_counts=True)
        print("Stage 2 predictions on Stage 1 anomalies:")
        for k, v in zip(unique.tolist(), counts.tolist()):
            print(f"{k}: {int(v)}")
    else:
        print("Stage 1 found no anomalies to pass to Stage 2.")

    return stage2_model
