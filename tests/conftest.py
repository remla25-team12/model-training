# tests/conftest.py
import os

import joblib
import numpy as np
import pytest
from omegaconf import OmegaConf
from sklearn.naive_bayes import GaussianNB

from src.utils import load_classifier, load_preprocessed_data


@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    return OmegaConf.create(
        {
            "dataset": {
                "preprocessed": {
                    "output_dir": "./data/processed",
                    "X_filename": "X.joblib",
                    "y_filename": "y.joblib",
                }
            },
            "model": {
                "classifier": {
                    "model_dir": "./models",
                    "model_filename": "Classifier_Sentiment_Model.joblib",
                }
            },
        }
    )


@pytest.fixture
def mock_data():
    """Create mock data for testing with controlled distribution"""
    np.random.seed(42)  # Set seed for reproducibility
    n_samples = 100
    n_features = 10

    # Create data with similar distributions for both halves
    X1 = np.random.normal(0.5, 0.1, (n_samples // 2, n_features))
    X2 = np.random.normal(0.5, 0.1, (n_samples // 2, n_features))
    X = np.vstack([X1, X2])

    y = np.random.randint(0, 2, n_samples)
    return X, y


@pytest.fixture
def mock_model(mock_data):
    """Create and train a mock model"""
    X, y = mock_data
    model = GaussianNB()
    model.fit(X, y)
    return model


@pytest.fixture
def setup_test_files(test_config, mock_data, mock_model):
    """Setup test data and model files"""
    X, y = mock_data

    # Create directories
    os.makedirs(test_config.dataset.preprocessed.output_dir, exist_ok=True)
    os.makedirs(test_config.model.classifier.model_dir, exist_ok=True)

    # Save data files
    X_path = os.path.join(
        test_config.dataset.preprocessed.output_dir,
        test_config.dataset.preprocessed.X_filename,
    )
    y_path = os.path.join(
        test_config.dataset.preprocessed.output_dir,
        test_config.dataset.preprocessed.y_filename,
    )

    # Save model file
    model_path = os.path.join(
        test_config.model.classifier.model_dir,
        test_config.model.classifier.model_filename,
    )

    # Save all files
    joblib.dump(X, X_path)
    joblib.dump(y, y_path)
    joblib.dump(mock_model, model_path)

    yield  # Allow tests to run

    # Cleanup
    for path in [X_path, y_path, model_path]:
        try:
            os.remove(path)
        except:
            pass


@pytest.fixture
def data(test_config, setup_test_files):
    """Fixture to load preprocessed data"""
    return load_preprocessed_data(test_config)


@pytest.fixture
def model(test_config, setup_test_files):
    """Fixture to load classifier"""
    return load_classifier(test_config)


@pytest.fixture
def X_y(data):
    """Fixture to get X and y separately"""
    X, y = data
    return X, y


@pytest.fixture
def X(X_y):
    """Fixture to get just X"""
    X, _ = X_y
    return X
