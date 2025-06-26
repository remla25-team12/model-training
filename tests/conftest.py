# tests/conftest.py
import subprocess
import sys

import pytest

from src.configure_loader import load_config
from src.utils import load_classifier, load_preprocessed_data


def pytest_sessionstart(session):
    """
    Ensure data, preprocessing, and model files are present before any tests run.
    """
    subprocess.run([sys.executable, "-m", "src.get_data"], check=True)
    subprocess.run([sys.executable, "-m", "src.preprocess_data"], check=True)
    subprocess.run([sys.executable, "-m", "src.train_model"], check=True)


@pytest.fixture
def data():
    """Fixture to load preprocessed data"""
    return load_preprocessed_data(load_config())


@pytest.fixture
def model():
    """Fixture to load classifier"""
    return load_classifier(load_config())


@pytest.fixture
def X_y(data):
    X, y = data
    return X, y


@pytest.fixture
def X(X_y):
    X, _ = X_y
    return X
