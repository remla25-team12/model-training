import os
import pandas as pd
import pickle
import joblib
from libml import preprocess_dataset
from get_data import download_data
from configure_loader import load_config

def preprocess():
    # Load the raw dataset
    config = load_config()
    output_dir = config["dataset"]["raw"]["output_dir"]
    output_filename = config["dataset"]["raw"]["output_filename"]
    raw_dataset = pd.read_csv(os.path.join(output_dir, output_filename), delimiter = '\t', quoting = 3)

    # Preprocess the dataset
    X, y, cv = preprocess_dataset(raw_dataset)

    # Save the CountVectorizer model for later use during inference
    bow_dir = config["model"]["count_vectorizer"]["bow_dir"]
    bow_filename = config["model"]["count_vectorizer"]["bow_filename"]
    bow_path = os.path.join(bow_dir, bow_filename)
    
    os.makedirs(bow_dir, exist_ok=True)
    with open(bow_path, "wb") as f:
        pickle.dump(cv, f)

    # Save the preprocessed data
    preprocessed_dir = config["dataset"]["preprocessed"]["output_dir"]
    os.makedirs(preprocessed_dir, exist_ok=True)

    X_filename = config["dataset"]["preprocessed"]["X_filename"]
    y_filename = config["dataset"]["preprocessed"]["y_filename"]

    X_path = os.path.join(preprocessed_dir, X_filename)
    y_path = os.path.join(preprocessed_dir, y_filename)
    
    joblib.dump(X, X_path)
    joblib.dump(y, y_path)


if __name__ == "__main__":
    preprocess()