import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
import pickle
from configure_loader import load_config
from utils import load_preprocessed_data

def train_model():
    config = load_config()

    # Load the preprocessed dataset and vectorizer
    X, y = load_preprocessed_data()

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