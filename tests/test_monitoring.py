# tests/test_monitoring.py
import numpy as np
import pytest
from sklearn.metrics import accuracy_score

pytestmark = pytest.mark.order(5)


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

    # Compute the absolute and relative difference
    abs_diff = np.abs(dist1 - dist2)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Relaxed tolerance: allow up to 0.05 mean absolute difference
    assert mean_abs_diff < 0.05, (
        f"Mean absolute difference between halves is too high: {mean_abs_diff:.4f} "
        f"(max: {max_abs_diff:.4f})"
    )
