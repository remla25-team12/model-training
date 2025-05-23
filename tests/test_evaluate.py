import json
import os
import shutil
import time
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

from src.configure_loader import load_config
from src.evaluate import evaluate_model


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


def safe_cleanup(path):
    """Safely clean up a directory with retries for Windows"""
    max_retries = 3
    retry_delay = 1  # seconds

    for i in range(max_retries):
        try:
            if os.path.exists(path):
                # Reset permissions recursively
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        try:
                            os.chmod(os.path.join(root, name), 0o777)
                        except:
                            pass
                    for name in dirs:
                        try:
                            os.chmod(os.path.join(root, name), 0o777)
                        except:
                            pass
                    try:
                        os.chmod(root, 0o777)
                    except:
                        pass

                # Remove directory
                shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception as e:
            if i == max_retries - 1:  # Last attempt
                print(f"Warning: Could not clean up {path}: {str(e)}")
            else:
                time.sleep(retry_delay)
    return False


@pytest.fixture(autouse=True)
def cleanup_dirs():
    """Clean up test directories before and after tests"""
    dirs_to_clean = ["./data/model_eval"]

    # Clean before test
    for dir_path in dirs_to_clean:
        safe_cleanup(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    yield

    # Clean after test
    for dir_path in dirs_to_clean:
        safe_cleanup(dir_path)


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


@patch("src.evaluate.load_preprocessed_data")
def test_evaluate_model_missing_data(mock_load_data, config):
    """Test handling of missing data"""
    mock_load_data.side_effect = FileNotFoundError("Test error")

    with pytest.raises(FileNotFoundError):
        evaluate_model()


@patch("src.evaluate.load_preprocessed_data")
@patch("src.evaluate.load_classifier")
def test_evaluate_model_missing_classifier(
    mock_load_classifier, mock_load_data, config, sample_data
):
    """Test handling of missing classifier"""
    mock_load_data.return_value = sample_data
    mock_load_classifier.side_effect = FileNotFoundError("Test error")

    with pytest.raises(FileNotFoundError):
        evaluate_model()


@patch("src.evaluate.load_preprocessed_data")
@patch("src.evaluate.load_classifier")
def test_evaluate_model_invalid_data(
    mock_load_classifier, mock_load_data, config, sample_classifier
):
    """Test handling of invalid data format"""
    # Return invalid data format
    mock_load_data.return_value = (np.array([]), np.array([]))  # Empty arrays
    mock_load_classifier.return_value = sample_classifier

    with pytest.raises(ValueError):
        evaluate_model()


@patch("src.evaluate.load_preprocessed_data")
@patch("src.evaluate.load_classifier")
def test_evaluate_model_data_shape_mismatch(
    mock_load_classifier, mock_load_data, config, sample_classifier
):
    """Test handling of data shape mismatch"""
    # Return data with mismatched shapes
    mock_load_data.return_value = (
        np.array([[1, 2], [3, 4]]),
        np.array([1]),
    )  # Different lengths
    mock_load_classifier.return_value = sample_classifier

    with pytest.raises(ValueError):
        evaluate_model()


@patch("src.evaluate.load_preprocessed_data")
@patch("src.evaluate.load_classifier")
@patch("builtins.open", new_callable=mock_open)
def test_evaluate_model_metrics_file_error(
    mock_open,
    mock_load_classifier,
    mock_load_data,
    config,
    sample_data,
    sample_classifier,
):
    """Test handling of metrics file writing error"""
    mock_load_data.return_value = sample_data
    mock_load_classifier.return_value = sample_classifier

    # Make open raise PermissionError
    mock_open.side_effect = PermissionError("Test error")

    with pytest.raises(PermissionError):
        evaluate_model()
