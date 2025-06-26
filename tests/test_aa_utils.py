import pickle
from pathlib import Path

import joblib
import numpy as np
import pytest
from libml import preprocess_input
from sklearn.inspection import permutation_importance

pytestmark = pytest.mark.order(9)


@pytest.fixture
def load_model():
    """Load the pre-trained model."""
    model_path = (
        Path(__file__).parent.parent / "models" / "Classifier_Sentiment_Model.joblib"
    )
    with open(model_path, "rb") as f:
        return joblib.load(f)


@pytest.fixture
def load_vectorizer():
    """Load the pre-trained CountVectorizer."""
    vectorizer_path = (
        Path(__file__).parent.parent / "models" / "c1_BoW_Sentiment_Model.pkl"
    )
    with open(vectorizer_path, "rb") as f:
        return pickle.load(f)


def test_feature_cost_importance(load_model, load_vectorizer):
    """
    Use permutation importance to evaluate feature cost.
    """
    texts = [
        "Wow... Loved this place.",
        "Crust is not good.",
        "I really enjoyed the food.",
        "Service was very bad.",
        "Delicious pizza and friendly staff.",
        "Terrible and slow experience.",
        "Great ambiance and tasty dishes.",
        "Awful customer service ruined it.",
        "Yummy pasta and quick service!",
        "Would not recommend.",
    ]
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    X = np.vstack([preprocess_input(text, load_vectorizer) for text in texts])
    y = labels

    result = permutation_importance(load_model, X, y, n_repeats=5, random_state=42)
    importances = result.importances_mean

    # Check that at least one feature has non-zero importance
    assert np.any(importances != 0), "No feature had any measurable importance"


def test_synonym_metamorphic(load_model, load_vectorizer):
    """
    Ensure the model gives consistent predictions for synonymous sentences.
    """
    pairs = [
        ("Wow... Loved this place.", "Wow... Adored this place."),
        ("Crust is not good.", "Crust is terrible."),
        ("I enjoyed the food.", "I liked the food."),
        ("Service was very bad.", "Service was awful."),
    ]

    for sent1, sent2 in pairs:
        vec1 = preprocess_input(sent1, load_vectorizer)
        vec2 = preprocess_input(sent2, load_vectorizer)
        pred1 = load_model.predict(vec1)[0]
        pred2 = load_model.predict(vec2)[0]
        assert pred1 == pred2, f"Predictions differ: '{sent1}' → '{sent2}'"
