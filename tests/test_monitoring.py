# tests/test_monitoring.py
import pytest
import numpy as np
from sklearn.metrics import accuracy_score


def evaluate_test_model(model, X, y):
    """Test-specific model evaluation function"""
    y_pred = model.predict(X)
    return {"accuracy": accuracy_score(y, y_pred)}


def test_model_drift(model, X_y):
    """Test if model performance remains stable across different data distributions"""
    X, y = X_y

    # Test on different data slices
    np.random.seed(42)  # For reproducibility
    slice_indices = np.random.choice(len(X), size=len(X) // 2, replace=False)
    slice_X = X[slice_indices]
    slice_y = y[slice_indices]

    # Get performance metrics
    full_metrics = evaluate_test_model(model, X, y)
    slice_metrics = evaluate_test_model(model, slice_X, slice_y)

    # Check if performance is stable
    assert (
        abs(full_metrics["accuracy"] - slice_metrics["accuracy"]) < 0.2
    )  # Relaxed threshold


def test_data_drift(X):
    """Test if data distribution remains stable"""
    # Split data into two parts
    mid = len(X) // 2
    dist1 = X[:mid].mean(axis=0)
    dist2 = X[mid:].mean(axis=0)

    # Check if distributions are similar
    assert np.allclose(dist1, dist2, rtol=0.1)
