import os
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from src.configure_loader import load_config
from src.preprocess_data import (
    _load_raw_dataset,
    _save_preprocessed_data,
    _save_vectorizer,
    preprocess,
)


@pytest.fixture
def config():
    """Load test configuration"""
    return load_config()


@pytest.fixture
def sample_raw_data():
    """Create sample raw dataset"""
    return pd.DataFrame(
        {
            "Review": ["Great food!", "Terrible service", "Amazing experience"],
            "Sentiment": [1, 0, 1],
        }
    )


@pytest.fixture
def sample_processed_data():
    """Create sample processed data"""
    X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])  # Example BOW vectors
    y = np.array([1, 0, 1])
    return X, y


@pytest.fixture
def real_vectorizer():
    """Create a real CountVectorizer instance"""
    vectorizer = CountVectorizer()
    # Fit it with some sample data
    vectorizer.fit(["Great food!", "Terrible service", "Amazing experience"])
    return vectorizer


@pytest.fixture
def cleanup_dirs():
    """Clean up test directories before and after tests"""
    dirs_to_clean = ["./data/processed", "./data/initial_raw", "./models"]

    # Clean before test
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    yield

    # Clean after test
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


def test_load_raw_dataset(config, sample_raw_data, cleanup_dirs):
    """Test loading raw dataset"""
    # Create test data file
    os.makedirs(config["dataset"]["raw"]["output_dir"], exist_ok=True)
    file_path = os.path.join(
        config["dataset"]["raw"]["output_dir"],
        config["dataset"]["raw"]["output_filename"],
    )
    sample_raw_data.to_csv(file_path, sep="\t", index=False, quoting=3)

    # Test loading
    loaded_data = _load_raw_dataset(config)
    assert isinstance(loaded_data, pd.DataFrame)
    assert loaded_data.shape == sample_raw_data.shape
    assert all(loaded_data.columns == sample_raw_data.columns)


def test_save_vectorizer(config, real_vectorizer, cleanup_dirs):
    """Test saving CountVectorizer"""
    _save_vectorizer(real_vectorizer, config)

    # Check if file exists
    bow_path = os.path.join(
        config["model"]["count_vectorizer"]["bow_dir"],
        config["model"]["count_vectorizer"]["bow_filename"],
    )
    assert os.path.exists(bow_path)
    assert os.path.getsize(bow_path) > 0


def test_save_preprocessed_data(config, sample_processed_data, cleanup_dirs):
    """Test saving preprocessed data"""
    X, y = sample_processed_data
    _save_preprocessed_data(X, y, config)

    # Check if files exist
    pre_dir = config["dataset"]["preprocessed"]["output_dir"]
    X_path = os.path.join(pre_dir, config["dataset"]["preprocessed"]["X_filename"])
    y_path = os.path.join(pre_dir, config["dataset"]["preprocessed"]["y_filename"])

    assert os.path.exists(X_path)
    assert os.path.exists(y_path)
    assert os.path.getsize(X_path) > 0
    assert os.path.getsize(y_path) > 0


@patch("src.preprocess_data.libml")
def test_preprocess_pipeline(
    mock_libml,
    config,
    sample_raw_data,
    sample_processed_data,
    real_vectorizer,
    cleanup_dirs,
):
    """Test the complete preprocessing pipeline"""
    # Setup mock libml
    X, y = sample_processed_data
    mock_libml.preprocess_dataset.return_value = (X, y, real_vectorizer)

    # Create raw data file
    os.makedirs(config["dataset"]["raw"]["output_dir"], exist_ok=True)
    file_path = os.path.join(
        config["dataset"]["raw"]["output_dir"],
        config["dataset"]["raw"]["output_filename"],
    )
    sample_raw_data.to_csv(file_path, sep="\t", index=False, quoting=3)

    # Run preprocessing
    preprocess()

    # Check if all files were created
    bow_path = os.path.join(
        config["model"]["count_vectorizer"]["bow_dir"],
        config["model"]["count_vectorizer"]["bow_filename"],
    )
    pre_dir = config["dataset"]["preprocessed"]["output_dir"]
    X_path = os.path.join(pre_dir, config["dataset"]["preprocessed"]["X_filename"])
    y_path = os.path.join(pre_dir, config["dataset"]["preprocessed"]["y_filename"])

    assert os.path.exists(bow_path)
    assert os.path.exists(X_path)
    assert os.path.exists(y_path)

    # Verify libml was called correctly
    mock_libml.preprocess_dataset.assert_called_once()
    df_arg = mock_libml.preprocess_dataset.call_args[0][0]
    assert isinstance(df_arg, pd.DataFrame)
    assert df_arg.shape == sample_raw_data.shape


def test_load_raw_dataset_missing_file(config, cleanup_dirs):
    """Test handling of missing raw dataset"""
    with pytest.raises(FileNotFoundError):
        _load_raw_dataset(config)


def test_save_preprocessed_data_invalid_shapes(config, cleanup_dirs):
    """Test handling of invalid data shapes"""
    X = np.array([[1, 2], [3, 4]])  # 2x2
    y = np.array([1])  # 1D with different length

    with pytest.raises(ValueError):
        _save_preprocessed_data(X, y, config)


def test_save_vectorizer_invalid_type(config, cleanup_dirs):
    """Test handling of invalid vectorizer type"""
    invalid_vectorizer = {"vocabulary": {}}  # Not a proper CountVectorizer

    with pytest.raises(AttributeError):
        _save_vectorizer(invalid_vectorizer, config)
