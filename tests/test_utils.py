import os
import pickle
import shutil

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

from src.configure_loader import load_config
from src.utils import load_classifier, load_preprocessed_data, load_vectorizer


@pytest.fixture
def config():
    """Load test configuration"""
    return load_config()


@pytest.fixture
def sample_data():
    """Create sample preprocessed data"""
    X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
    y = np.array([1, 0, 1])
    return X, y


@pytest.fixture
def sample_classifier():
    """Create sample classifier"""
    clf = GaussianNB()
    X = np.array([[1, 1], [0, 0], [1, 0]])
    y = np.array([1, 0, 1])
    clf.fit(X, y)
    return clf


@pytest.fixture
def sample_vectorizer():
    """Create sample vectorizer"""
    vectorizer = CountVectorizer()
    vectorizer.fit(["Great food!", "Terrible service", "Amazing experience"])
    return vectorizer


@pytest.fixture(autouse=True)
def cleanup_dirs():
    """Clean up test directories before and after tests"""
    dirs_to_clean = ["./data/processed", "./models"]

    # Clean before test
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    yield

    # Clean after test
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


def setup_test_files(
    config, sample_data=None, sample_classifier=None, sample_vectorizer=None
):
    """Setup test files"""
    # Setup preprocessed data if provided
    if sample_data is not None:
        X, y = sample_data
        pre_dir = config["dataset"]["preprocessed"]["output_dir"]
        os.makedirs(pre_dir, exist_ok=True)
        X_path = os.path.join(pre_dir, config["dataset"]["preprocessed"]["X_filename"])
        y_path = os.path.join(pre_dir, config["dataset"]["preprocessed"]["y_filename"])
        joblib.dump(X, X_path)
        joblib.dump(y, y_path)

    # Setup classifier if provided
    if sample_classifier is not None:
        model_dir = config["model"]["classifier"]["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(
            model_dir, config["model"]["classifier"]["model_filename"]
        )
        joblib.dump(sample_classifier, model_path)

    # Setup vectorizer if provided
    if sample_vectorizer is not None:
        bow_dir = config["model"]["count_vectorizer"]["bow_dir"]
        os.makedirs(bow_dir, exist_ok=True)
        bow_path = os.path.join(
            bow_dir, config["model"]["count_vectorizer"]["bow_filename"]
        )
        with open(bow_path, "wb") as f:
            pickle.dump(sample_vectorizer, f)


def test_load_preprocessed_data(config, sample_data):
    """Test loading preprocessed data"""
    X_expected, y_expected = sample_data
    setup_test_files(config, sample_data, None, None)

    X, y = load_preprocessed_data(config)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == X_expected.shape
    assert y.shape == y_expected.shape
    assert np.array_equal(X, X_expected)
    assert np.array_equal(y, y_expected)


def test_load_preprocessed_data_missing_files(config):
    """Test handling of missing preprocessed data files"""
    with pytest.raises(FileNotFoundError):
        load_preprocessed_data(config)


def test_load_preprocessed_data_corrupted(config, sample_data):
    """Test handling of corrupted preprocessed data files"""
    # First create valid files
    setup_test_files(config, sample_data, None, None)

    # Then corrupt the X file
    X_path = os.path.join(
        config["dataset"]["preprocessed"]["output_dir"],
        config["dataset"]["preprocessed"]["X_filename"],
    )
    with open(X_path, "wb") as f:
        f.write(b"corrupted data")

    with pytest.raises(Exception):
        load_preprocessed_data(config)


def test_load_classifier(config, sample_classifier):
    """Test loading classifier"""
    setup_test_files(config, None, sample_classifier, None)
    clf = load_classifier(config)

    assert isinstance(clf, GaussianNB)
    # Compare predictions to ensure it's the same model
    X_test = np.array([[1, 1], [0, 0]])
    assert np.array_equal(clf.predict(X_test), sample_classifier.predict(X_test))


def test_load_classifier_missing_file(config):
    """Test handling of missing classifier file"""
    with pytest.raises(FileNotFoundError):
        load_classifier(config)


def test_load_classifier_corrupted(config, sample_classifier):
    """Test handling of corrupted classifier file"""
    # First create valid file
    setup_test_files(config, None, sample_classifier, None)

    # Then corrupt it
    model_path = os.path.join(
        config["model"]["classifier"]["model_dir"],
        config["model"]["classifier"]["model_filename"],
    )
    with open(model_path, "wb") as f:
        f.write(b"corrupted data")

    with pytest.raises(Exception):
        load_classifier(config)


def test_load_vectorizer(config, sample_vectorizer):
    """Test loading vectorizer"""
    setup_test_files(config, None, None, sample_vectorizer)
    cv = load_vectorizer(config)

    assert isinstance(cv, CountVectorizer)
    assert cv.vocabulary_ == sample_vectorizer.vocabulary_
    # Compare transformations to ensure it's the same vectorizer
    test_text = ["Test food service"]
    assert np.array_equal(
        cv.transform(test_text).toarray(),
        sample_vectorizer.transform(test_text).toarray(),
    )


def test_load_vectorizer_missing_file(config):
    """Test handling of missing vectorizer file"""
    with pytest.raises(FileNotFoundError):
        load_vectorizer(config)


def test_load_vectorizer_corrupted(config, sample_vectorizer):
    """Test handling of corrupted vectorizer file"""
    # First create valid file
    setup_test_files(config, None, None, sample_vectorizer)

    # Then corrupt it
    bow_path = os.path.join(
        config["model"]["count_vectorizer"]["bow_dir"],
        config["model"]["count_vectorizer"]["bow_filename"],
    )
    with open(bow_path, "wb") as f:
        f.write(b"corrupted data")

    with pytest.raises(Exception):
        load_vectorizer(config)
