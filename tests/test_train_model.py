# model-training/tests/test_train_model.py
import os
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from src.configure_loader import load_config
from src.train_model import train_model

pytestmark = pytest.mark.order(10)


@pytest.fixture
def config():
    """Load test configuration"""
    return load_config()


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y


@patch("src.train_model.load_preprocessed_data")
def test_train_model_success(mock_load_data, sample_data, config):
    """Test successful model training based on is_alternative_model param"""
    # Setup mock
    mock_load_data.return_value = sample_data

    # Load parameter to determine expected classifier type
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    use_logistic = params["model"]["is_alternative_model"]
    expected_model = LogisticRegression if use_logistic else GaussianNB

    # Run training
    train_model()

    # Verify model file exists
    model_path = os.path.join(
        config["model"]["classifier"]["model_dir"],
        config["model"]["classifier"]["model_filename"],
    )
    assert os.path.exists(model_path), "Model file should be created"

    # Load and verify the saved model
    model = joblib.load(model_path)
    assert isinstance(
        model, expected_model
    ), f"Model should be {expected_model.__name__}"
    assert hasattr(model, "predict"), "Model should have a predict method"
    assert model.classes_.shape[0] == 2, "Model should be a binary classifier"
