import numpy as np
import pytest

pytestmark = pytest.mark.order(3)


def test_fairness_group_parity(model, X_y):
    """Test prediction parity across synthetic groups"""
    X, y = X_y
    # Create a synthetic group feature (e.g., even/odd index)
    groups = np.array([i % 2 for i in range(len(X))])
    preds = model.predict(X)
    pos_rate_0 = np.mean(preds[groups == 0])
    pos_rate_1 = np.mean(preds[groups == 1])
    diff = abs(pos_rate_0 - pos_rate_1)
    # Allow up to 0.2 difference in positive prediction rates
    assert diff < 0.2, f"Prediction rate difference between groups too high: {diff}"
