import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

from src.configure_loader import load_config
from src.evaluate import evaluate_model

pytestmark = pytest.mark.order(2)


@pytest.fixture
def config():
    """Load test configuration"""
    return load_config()


@pytest.fixture
def sample_data():
    """Create sample preprocessed data"""
    # Create a simple dataset with clear patterns
    X = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    y = np.array([1, 0, 1, 0, 1, 0])  # Perfect separation
    return X, y


@pytest.fixture
def sample_classifier():
    """Create sample classifier"""
    clf = GaussianNB()
    X = np.array([[1, 0], [0, 1]])
    y = np.array([1, 0])
    clf.fit(X, y)
    return clf


@pytest.fixture
def load_vectorizer():
    """
    Fixture that loads the pre-trained CountVectorizer for testing.
    """
    vectorizer_path = (
        Path(__file__).parent.parent / "models" / "c1_BoW_Sentiment_Model.pkl"
    )
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


def test_vectorizer_dimension(load_vectorizer):
    vocab_size = len(load_vectorizer.get_feature_names_out())
    assert vocab_size == 1420, f"Expected 1420 features, got {vocab_size}"


@patch("src.evaluate.load_preprocessed_data")
@patch("src.evaluate.load_classifier")
def test_evaluate_model_success(
    mock_load_classifier, mock_load_data, config, sample_data, sample_classifier
):
    """Test successful model evaluation"""
    # Setup mocks
    mock_load_data.return_value = sample_data
    mock_load_classifier.return_value = sample_classifier

    # Run evaluation
    evaluate_model()

    # Load and verify metrics
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Since we used a perfect separation dataset, accuracy should be 1.0
    assert metrics["accuracy"] == 1.0  # Perfect accuracy
    assert "precision" in metrics["macro avg"]
    assert "recall" in metrics["macro avg"]
    assert "f1-score" in metrics["macro avg"]


@patch("src.evaluate.load_preprocessed_data")
@patch("src.evaluate.load_classifier")
@patch("src.evaluate.train_test_split")
def test_evaluate_model_with_imperfect_data(
    mock_split, mock_load_classifier, mock_load_data, config
):
    """Test model evaluation with imperfect predictions"""
    # Create a classifier that will make some mistakes
    clf = GaussianNB()
    X_train = np.array(
        [[1, 0], [0.9, 0.1], [0.1, 0.9], [0, 1]]  # Class 1 examples  # Class 0 examples
    )
    y_train = np.array([1, 1, 0, 0])
    clf.fit(X_train, y_train)

    # Create test data that will cause some mistakes
    X_test = np.array(
        [
            [0.6, 0.4],  # Ambiguous, model will predict class 1
            [0.4, 0.6],  # Ambiguous, model will predict class 0
            [0.55, 0.45],  # Ambiguous, model will predict class 1
            [0.45, 0.55],  # Ambiguous, model will predict class 0
        ]
    )
    y_test = np.array([0, 1, 0, 1])  # Opposite of what model will predict

    # Mock train_test_split to return our test data
    mock_split.return_value = (None, X_test, None, y_test)

    # Setup other mocks
    mock_load_data.return_value = (X_test, y_test)  # Doesn't matter what we return here
    mock_load_classifier.return_value = clf

    # Run evaluation
    evaluate_model()

    # Check metrics
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Should have exactly 0% accuracy since we designed the test data
    # to be classified opposite to the true labels
    assert metrics["accuracy"] == 0.0
