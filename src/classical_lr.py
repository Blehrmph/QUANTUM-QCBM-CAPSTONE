import numpy as np


def _import_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        raise ImportError(
            "scikit-learn is required for the classical logistic regression pipeline."
        ) from exc
    return LogisticRegression


def train_logistic_regression(X_train, y_train):
    LogisticRegression = _import_sklearn()
    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train, y_train)
    return model


def score_logistic_regression(model, X):
    probs = model.predict_proba(X)[:, 1]
    return probs


def model_to_dict(model):
    return {
        "coef_": np.asarray(model.coef_).tolist(),
        "intercept_": np.asarray(model.intercept_).tolist(),
        "classes_": np.asarray(model.classes_).tolist(),
    }
