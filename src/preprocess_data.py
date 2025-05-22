"""
Preprocesses the training data using the libml package
"""

import os
import pickle
import pandas as pd
import joblib
import libml
import configure_loader

def _load_raw_dataset(config):
    """
    Load the raw, unprocessed dataset according to the settings specified in the config
    
    Args: 
        config (dict): See config.yaml for contents)
    Returns: 
        raw_dataset (pd.DataFrame): The raw, unprocessed dataset
    """
    output_dir = config["dataset"]["raw"]["output_dir"]
    output_filename = config["dataset"]["raw"]["output_filename"]
    raw_dataset = pd.read_csv(
        os.path.join(output_dir, output_filename),
        delimiter = '\t', quoting = 3)
    
    print(f"Loaded raw dataset from {output_dir}")
    return raw_dataset

def _save_vectorizer(cv, config):
    """
    Save the vectorizer according to the settings specified in the config

    Args: 
        - cv (CountVectorizer)
        - config (dict): See config.yaml for contents
    """
    bow_dir = config["model"]["count_vectorizer"]["bow_dir"]
    bow_filename = config["model"]["count_vectorizer"]["bow_filename"]
    bow_path = os.path.join(bow_dir, bow_filename)

    os.makedirs(bow_dir, exist_ok=True)
    with open(bow_path, "wb") as f:
        pickle.dump(cv, f)

    print(f"Saved CountVectorizer model in {bow_dir}")


def _save_preprocessed_data(X, y, config):
    """
    Save the preprocessed data to the file directories specified in the config

    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Labels corresponding to the input features
        config (dict): See config.yaml for contents
    """
    pre_dir = config["dataset"]["preprocessed"]["output_dir"]
    X_filename = config["dataset"]["preprocessed"]["X_filename"]
    y_filename = config["dataset"]["preprocessed"]["y_filename"]

    os.makedirs(pre_dir, exist_ok=True)

    X_path = os.path.join(pre_dir, X_filename)
    y_path = os.path.join(pre_dir, y_filename)

    joblib.dump(X, X_path)
    joblib.dump(y, y_path)

    print(f"Saved processed dataset (with {X.shape[0]} samples) in {pre_dir}")


def preprocess():
    """
    Main function for preprocessesing the training data
    """
    # Intiial step: load configs
    config = configure_loader.load_config()

    # Step 1: Load the raw dataset
    raw_dataset = _load_raw_dataset(config)

    # Step 2: Preprocess the dataset using libml to obtain
    # samples X, labels y, and CountVectorizer cv
    X, y, cv = libml.preprocess_dataset(raw_dataset)

    # Step 3: Save the CountVectorizer model for later use during inference
    _save_vectorizer(cv, config)

    # Step 4: Save the preprocessed data
    _save_preprocessed_data(X, y, config)


if __name__ == "__main__":
    preprocess()
