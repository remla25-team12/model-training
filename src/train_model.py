"""
Trains a Naive Bayes or Logistic Regression Classifier model
"""

import os

import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.configure_loader import load_config
from src.utils import load_preprocessed_data


def train_model():
    """
    Main function for training the Naive Bayes Classifier
    """
    # Load YAML parameter for classifier selection
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    use_logistic = params["model"]["is_alternative_model"]

    # Load configuration
    config = load_config()

    # Load the preprocessed dataset
    print("Loading dataset...")
    X, y = load_preprocessed_data(config)

    # Divide dataset into training and test set
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train and fit a classifier
    if use_logistic:
        print("Training Logistic Regression classifier...")
        classifier = LogisticRegression(max_iter=1000)
    else:
        print("Training Naive Bayes classifier...")
        classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Exporting NB Classifier to later use in prediction
    classifier_dir = config["model"]["classifier"]["model_dir"]
    os.makedirs(classifier_dir, exist_ok=True)

    classifier_filename = config["model"]["classifier"]["model_filename"]
    classifier_path = os.path.join(classifier_dir, classifier_filename)

    joblib.dump(classifier, classifier_path)
    print(f"Saved trained classifier model in {classifier_dir}")


if __name__ == "__main__":
    train_model()
