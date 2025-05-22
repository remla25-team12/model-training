"""
Evaluates the model's performance
"""
import json
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import load_classifier, load_preprocessed_data
from configure_loader import load_config

def evaluate_model():
    """
    Main function for evaluating the model using the test set
    """
    # Load configuration
    config = load_config()

    # Load the preprocessed data and trained classifier
    X, y = load_preprocessed_data(config)
    classifier = load_classifier(config)

    # Divide dataset into training and test set
    # using the same test size and random state as used during training
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    _, X_test, _, y_test = train_test_split(X, y,
                                            test_size = test_size,
                                            random_state = random_state)

    # Evaluate model performance on the test set
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred)


    print("--- Evaluation results: ---")
    print(report)

    with open("data/model_eval/metrics.json", "w",encoding="utf-8") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    evaluate_model()
