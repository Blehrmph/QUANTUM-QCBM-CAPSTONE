from src.classical_lr import _import_sklearn


def train_multiclass_logreg(X_train, y_train):
    LogisticRegression = _import_sklearn()
    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train, y_train)
    return model


def run_stage3(
    X_train_anom,
    y_train_cat,
    sub_train,
    min_subtype_samples,
):
    print("Stage 3: Training subtype classifiers per attack category...")
    subtype_models = {}
    for attack_type in sorted(y_train_cat.unique().tolist()):
        mask = y_train_cat == attack_type
        sub_y = sub_train.loc[mask].astype(str).str.strip()
        sub_x = X_train_anom.loc[mask].reset_index(drop=True)
        sub_keep = sub_y.notna() & (sub_y != "")
        sub_y = sub_y.loc[sub_keep]
        sub_x = sub_x.loc[sub_keep]
        if len(sub_y) < min_subtype_samples or len(sub_y.unique()) < 2:
            continue
        subtype_models[attack_type] = train_multiclass_logreg(sub_x, sub_y)

    if not subtype_models:
        print("No subtype models trained (insufficient data).")
    else:
        print(f"Trained subtype models: {len(subtype_models)}")

    return subtype_models
