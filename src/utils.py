from configure_loader import load_config
import os
import joblib
import pickle

def load_preprocessed_data():
    config = load_config()

    preprocessed_dir = config["dataset"]["preprocessed"]["output_dir"]
    X_filename = config["dataset"]["preprocessed"]["X_filename"]
    y_filename = config["dataset"]["preprocessed"]["y_filename"]

    X_path = os.path.join(preprocessed_dir, X_filename)
    y_path = os.path.join(preprocessed_dir, y_filename)
    
    X = joblib.load(X_path)
    y = joblib.load(y_path)

    return X, y


def load_classifier():
    config = load_config()
    model_dir = config["model"]["classifier"]["model_dir"]
    model_filename = config["model"]["classifier"]["model_filename"]
    model_path = os.path.join(model_dir, model_filename)
    return joblib.load(model_path)


def load_vectorizer():
    config = load_config()
    bow_dir = config["model"]["count_vectorizer"]["bow_dir"]
    bow_filename = config["model"]["count_vectorizer"]["bow_filename"]
    bow_path = os.path.join(bow_dir, bow_filename)

    with open(bow_path, "rb") as f:
        cv = pickle.load(f)

    return cv