"""
Helper functions for loading processed data and models
"""

import os
import pickle

import joblib


def load_preprocessed_data(config):
    """
    Loads the preprocessed dataset

    Args:
        config (dict): Configuration dictionary
    Returns:
        tuple: (X, y) preprocessed features and labels
    Raises:
        FileNotFoundError: If preprocessed data files don't exist
        Exception: If data files are corrupted
    """
    preprocessed_dir = config["dataset"]["preprocessed"]["output_dir"]
    X_filename = config["dataset"]["preprocessed"]["X_filename"]
    y_filename = config["dataset"]["preprocessed"]["y_filename"]

    X_path = os.path.join(preprocessed_dir, X_filename)
    y_path = os.path.join(preprocessed_dir, y_filename)

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Preprocessed data files not found in {preprocessed_dir}"
        )

    try:
        X = joblib.load(X_path)
        y = joblib.load(y_path)
        return X, y
    except (OSError, ValueError) as e:
        raise OSError(f"Error loading preprocessed data: {str(e)}") from e


def load_classifier(config):
    """
    Loads the Naive Bayes Classifier model

    Args:
        config (dict): Configuration dictionary
    Returns:
        GaussianNB: Trained classifier model
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model file is corrupted
    """
    model_dir = config["model"]["classifier"]["model_dir"]
    model_filename = config["model"]["classifier"]["model_filename"]
    model_path = os.path.join(model_dir, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model not found at {model_path}")

    try:
        return joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Error loading classifier model: {str(e)}")


def load_vectorizer(config):
    """
    Loads the CountVectorizer model

    Args:
        config (dict): Configuration dictionary
    Returns:
        CountVectorizer: Fitted vectorizer model
    Raises:
        FileNotFoundError: If vectorizer file doesn't exist
        Exception: If vectorizer file is corrupted
    """
    bow_dir = config["model"]["count_vectorizer"]["bow_dir"]
    bow_filename = config["model"]["count_vectorizer"]["bow_filename"]
    bow_path = os.path.join(bow_dir, bow_filename)

    if not os.path.exists(bow_path):
        raise FileNotFoundError(f"Vectorizer model not found at {bow_path}")

    try:
        with open(bow_path, "rb") as f:
            cv = pickle.load(f)
        return cv
    except Exception as e:
        raise Exception(f"Error loading vectorizer model: {str(e)}")
