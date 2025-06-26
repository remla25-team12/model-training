import pytest
import pickle
import joblib
import numpy as np
from pathlib import Path
from libml import preprocess_input

pytestmark = pytest.mark.order(7)
@pytest.fixture
def load_model():
    """
    Fixture that loads the pre-trained model for testing.
    """
    model_path = Path(__file__).parent.parent / "models" / "Classifier_Sentiment_Model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model


@pytest.fixture
def load_vectorizer():
    """
    Fixture that loads the pre-trained CountVectorizer for testing.
    """
    vectorizer_path = Path(__file__).parent.parent / "models" / "c1_BoW_Sentiment_Model.pkl"
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


def test_vectorizer_feature_count(load_vectorizer):
    """
    Ensure the loaded vectorizer has the expected number of features (vocabulary size).
    """
    vocab_size = len(load_vectorizer.get_feature_names_out())
    assert vocab_size == 1420, f"Expected 1420 features, got {vocab_size}"


@pytest.mark.parametrize("pair", [
    ("The food was great", "The food was excellent"),
    ("Terrible service", "Awful service"),
    ("I enjoyed the meal", "I liked the meal"),
    ("Service was slow", "Service was sluggish"),
])
def test_semantic_equivalence_predictions(pair, load_vectorizer, load_model):
    """
    The model should give consistent predictions for semantically similar sentences.
    """
    sent1, sent2 = pair
    vec1 = preprocess_input(sent1, load_vectorizer)
    vec2 = preprocess_input(sent2, load_vectorizer)
    pred1 = load_model.predict(vec1)[0]
    pred2 = load_model.predict(vec2)[0]
    assert pred1 == pred2, f"Inconsistent prediction for: '{sent1}' vs '{sent2}'"


@pytest.mark.parametrize("original,variant", [
    ("I really liked the food", "really liked food"),
    ("The service was not good", "service not good"),
    ("Staff were very helpful", "staff helpful"),
])
def test_stopword_robustness(original, variant, load_vectorizer, load_model):
    """
    Model should not change prediction when only stopwords are removed.
    """
    vec1 = preprocess_input(original, load_vectorizer)
    vec2 = preprocess_input(variant, load_vectorizer)
    pred1 = load_model.predict(vec1)[0]
    pred2 = load_model.predict(vec2)[0]
    assert pred1 == pred2, f"Prediction changed with stopword removal: '{original}' vs '{variant}'"
