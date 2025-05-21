import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
import pickle
from configure_loader import load_config

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


def load_vectorizer():
    config = load_config()
    bow_dir = config["model"]["count_vectorizer"]["bow_dir"]
    bow_filename = config["model"]["count_vectorizer"]["bow_filename"]
    bow_path = os.path.join(bow_dir, bow_filename)

    with open(bow_path, "rb") as f:
        cv = pickle.load(f)

    return cv


def train_model():
    config = load_config()

    # Load the preprocessed dataset and vectorizer
    X, y = load_preprocessed_data()
    cv = load_vectorizer()

    # Divide dataset into training and test set
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size = test_size, random_state = random_state)

    # Train and fit a Naive Bayes classifier 
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Exporting NB Classifier to later use in prediction
    classifier_dir = config["model"]["classifier"]["model_dir"]
    os.makedirs(classifier_dir, exist_ok=True)

    classifier_filename = config["model"]["classifier"]["model_filename"]
    classifier_path = os.path.join(classifier_dir, classifier_filename)

    joblib.dump(classifier, classifier_path)


if __name__ == "__main__":
    train_model()