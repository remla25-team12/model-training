# model-training/tests/test_train_model.py
import os
import shutil
import time
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

from src.configure_loader import load_config
from src.train_model import train_model


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
    dirs_to_clean = ["./models"]  # Match the config path

    # Clean before test
    for dir_path in dirs_to_clean:
        safe_cleanup(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    yield

    # Clean after test
    for dir_path in dirs_to_clean:
        safe_cleanup(dir_path)


@patch("src.train_model.load_preprocessed_data")
def test_train_model_success(mock_load_data, sample_data, config):
    """Test successful model training"""
    # Setup mock
    mock_load_data.return_value = sample_data

    # Run training
    train_model()

    # Verify model was saved
    model_path = os.path.join(
        config["model"]["classifier"]["model_dir"],
        config["model"]["classifier"]["model_filename"],
    )
    assert os.path.exists(model_path), "Model file should be created"

    # Load and verify the saved model
    model = joblib.load(model_path)
    assert isinstance(model, GaussianNB), "Saved model should be GaussianNB"
    assert hasattr(model, "predict"), "Model should have predict method"
    assert model.classes_.shape[0] == 2, "Model should be binary classifier"


@patch("src.train_model.load_preprocessed_data")
def test_train_model_with_empty_data(mock_load_data, config):
    """Test handling of empty dataset"""
    # Setup mock with empty data
    mock_load_data.return_value = (np.array([]), np.array([]))

    # Should raise ValueError for empty dataset
    with pytest.raises(ValueError):
        train_model()


@patch("src.train_model.load_preprocessed_data")
def test_train_model_with_mismatched_data(mock_load_data, config):
    """Test handling of mismatched X, y shapes"""
    # Setup mock with mismatched shapes
    mock_load_data.return_value = (np.random.rand(10, 5), np.array([0]))

    # Should raise ValueError for shape mismatch
    with pytest.raises(ValueError):
        train_model()


@patch("src.train_model.load_preprocessed_data")
def test_train_model_data_loading_error(mock_load_data, config):
    """Test handling of data loading error"""
    # Setup mock to raise error
    mock_load_data.side_effect = FileNotFoundError("Test error")

    # Should propagate FileNotFoundError
    with pytest.raises(FileNotFoundError):
        train_model()


@patch("src.train_model.load_preprocessed_data")
@patch("joblib.dump")
def test_train_model_saving_error(mock_dump, mock_load_data, sample_data, config):
    """Test handling of model saving error"""
    # Setup mocks
    mock_load_data.return_value = sample_data
    mock_dump.side_effect = PermissionError("Test error")

    # Should raise PermissionError when trying to save
    with pytest.raises(PermissionError):
        train_model()
