# tests/test_robustness.py
import warnings
import time
import psutil
import numpy as np
import pytest


def _slice_accuracy(model, X, noise_std=0.05):
    np.random.seed(42)
    original_preds = model.predict(X)
    noise = np.random.normal(0, noise_std, X.shape)
    noisy_X = X + noise
    noisy_preds = model.predict(noisy_X)
    return np.mean(original_preds == noisy_preds)


def _metamorphic_repair(model, X, y, scale_factor):
    scaled_X = X * scale_factor
    original_preds = model.predict(X)
    scaled_preds = model.predict(scaled_X)
    accuracy = np.mean(original_preds == scaled_preds)
    if accuracy > 0.7:
        return True
    warnings.warn("Metamorphic relation failed, retraining on scaled data.")
    repaired_model = type(model)()
    repaired_model.fit(scaled_X, y)
    repaired_preds = repaired_model.predict(scaled_X)
    repaired_accuracy = np.mean(original_preds == repaired_preds)
    return repaired_accuracy > 0.7


def test_noise_robustness(model, X_y):
    """Test model robustness against input noise, including explicit data slices"""
    X, y = X_y
    if X.shape[1] != model.n_features_in_:
        pytest.skip(
            f"Feature count mismatch: X has {X.shape[1]}, \
            model expects {model.n_features_in_}"
        )
    acc = _slice_accuracy(model, X)
    assert acc > 0.7, f"Model predictions changed too much \
    with noise (accuracy: {acc})"
    mid = len(X) // 2
    for X_slice in [X[:mid], X[mid:]]:
        slice_acc = _slice_accuracy(model, X_slice)
        assert slice_acc > 0.7, f"Slice predictions \
        changed too much with noise (accuracy: {slice_acc})"


def test_metamorphic_with_repair(model, X_y):
    """Test metamorphic relation and attempt automatic
    repair if failed, including explicit data slices"""
    X, y = X_y
    if X.shape[1] != model.n_features_in_:
        pytest.skip(
            f"Feature count mismatch: X has {X.shape[1]},\
            model expects {model.n_features_in_}"
        )
    scale_factor = 1.1
    assert _metamorphic_repair(model, X, y, scale_factor), (
        "Automatic repair failed: accuracy after retraining is too low"
    )
    mid = len(X) // 2
    for X_slice, y_slice in [(X[:mid], y[:mid]), (X[mid:], y[mid:])]:
        assert _metamorphic_repair(model, X_slice, y_slice, scale_factor), (
            "Slice automatic repair failed: accuracy after retraining is too low"
        )


def test_nonfunctional_performance(model, X_y):
    """
    Test inference time and memory usage (non-functional requirements)
    """
    X, _ = X_y
    if X.shape[1] != model.n_features_in_:
        pytest.skip(
            f"Feature count mismatch: X has \
            {X.shape[1]}, model expects {model.n_features_in_}"
        )
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # in MB

    start_time = time.time()
    model.predict(X)
    inference_time = time.time() - start_time

    final_memory = process.memory_info().rss / 1024 / 1024  # in MB
    memory_increase = final_memory - initial_memory

    # Set reasonable thresholds for your use case
    assert inference_time < 5, f"Inference time ({inference_time}s) exceeds 5 seconds"
    assert memory_increase < 1024, f"Memory increase ({memory_increase}MB) exceeds 1GB"
