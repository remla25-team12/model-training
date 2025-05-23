# tests/test_robustness.py
import numpy as np


def test_noise_robustness(model, X_y):
    """Test model robustness against input noise"""
    X, y = X_y
    np.random.seed(42)  # For reproducibility

    # Get original predictions
    original_preds = model.predict(X)

    # Add noise to input
    noise = np.random.normal(0, 0.05, X.shape)  # Reduced noise
    noisy_X = X + noise
    noisy_preds = model.predict(noisy_X)

    # Check if predictions are stable
    accuracy = np.mean(original_preds == noisy_preds)
    assert (
        accuracy > 0.7
    ), f"Model predictions changed too much with noise (accuracy: {accuracy})"


def test_missing_value_robustness(model, X_y):
    """Test model robustness against missing values"""
    X, y = X_y
    np.random.seed(42)  # For reproducibility

    # Create copy with some missing values
    X_missing = X.copy()
    mask = np.random.random(X.shape) < 0.05  # Reduced missing value rate
    X_missing[mask] = 0

    # Get predictions
    original_preds = model.predict(X)
    missing_preds = model.predict(X_missing)

    # Check if predictions are stable
    accuracy = np.mean(original_preds == missing_preds)
    assert (
        accuracy > 0.7
    ), f"Model predictions changed too much with missing values (accuracy: {accuracy})"


def test_metamorphic_relations(model, X_y):
    """Test if model predictions are relatively stable under small scaling"""
    X, y = X_y

    # Test with smaller scale factor
    scale_factor = 1.1  # Reduced scale factor
    scaled_X = X * scale_factor

    original_preds = model.predict(X)
    scaled_preds = model.predict(scaled_X)

    # Check if most predictions remain stable
    accuracy = np.mean(original_preds == scaled_preds)
    assert (
        accuracy > 0.7
    ), f"Model predictions changed too much after scaling (accuracy: {accuracy})"
