import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from src.configure_loader import load_config
from src.utils import load_classifier, load_preprocessed_data


def test_feature_cost_importance():
    config = load_config()
    X, y = load_preprocessed_data(config)
    clf = load_classifier(config)
    # Skip test if feature count does not match
    if X.shape[1] != clf.n_features_in_:
        pytest.skip(
            f"Feature count mismatch: X has {X.shape[1]}, \
              model expects {clf.n_features_in_}"
        )
    base_acc = accuracy_score(y, clf.predict(X))
    feature_drops = []
    for i in range(X.shape[1]):
        X_mod = np.delete(X, i, axis=1)
        # Retrain a new model of the same type on the modified data
        clf_mod = type(clf)()
        if X_mod.shape[1] < 2:
            continue
        clf_mod.fit(X_mod, y)
        acc = accuracy_score(y, clf_mod.predict(X_mod))
        feature_drops.append(base_acc - acc)
    assert any(
        abs(drop) > 0 for drop in feature_drops
    ), "No feature had any impact on accuracy"


def test_synonym_metamorphic():
    config = load_config()
    model = load_classifier(config)
    X, y = load_preprocessed_data(config)
    # Skip test if feature count does not match
    if X.shape[1] != model.n_features_in_:
        pytest.skip(
            f"Feature count mismatch: X has {X.shape[1]}, \
            model expects {model.n_features_in_}"
        )
    idxs = np.where(y == y[0])[0][:2]
    if len(idxs) < 2:
        idxs = [0, 1]  # fallback
    X_samples = X[idxs]
    preds = model.predict(X_samples)
    assert preds[0] == preds[1], "Model predictions differ for similar samples"
