# tests/test_robustness.py
import pickle
import random
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
import pytest
from libml import preprocess_input

pytestmark = pytest.mark.order(8)


@pytest.fixture
def load_model():
    """
    Fixture that loads the pre-trained model for testing.
    """
    model_path = (
        Path(__file__).parent.parent / "models" / "Classifier_Sentiment_Model.joblib"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


@pytest.fixture
def load_vectorizer():
    """
    Fixture that loads the pre-trained CountVectorizer for testing.
    """
    vectorizer_path = (
        Path(__file__).parent.parent / "models" / "c1_BoW_Sentiment_Model.pkl"
    )
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


def test_vectorizer_dimension(load_vectorizer):
    vocab_size = len(load_vectorizer.get_feature_names_out())
    assert vocab_size == 1420, f"Expected 1420 features, got {vocab_size}"


def replace_with_synonyms(sentence, synonym_map):
    return " ".join([synonym_map.get(word, word) for word in sentence.split()])


def try_repair(mutated_sentence, original_pred, mutant_map, model, vectorizer):
    words = mutated_sentence.split()
    for j, word in enumerate(words):
        if word in mutant_map:
            for mutant in mutant_map[word]:
                words[j] = mutant
                repaired_sentence = " ".join(words)
                repaired_feature = preprocess_input(repaired_sentence, vectorizer)
                repaired_pred = model.predict(repaired_feature)[0]
                if repaired_pred == original_pred:
                    return repaired_sentence
    return None


def test_mutamorphic_with_synonym_replacement(load_vectorizer, load_model):
    """
    Metamorphic test with synonym replacement to check model consistency.
    """
    test_sentences = [
        "This restaurant is amazing",
        "The food was terrible",
        "I love the ambiance",
        "Service was slow",
        "The staff was friendly",
    ]
    synonym_map = {
        "amazing": "fantastic",
        "terrible": "awful",
        "love": "adore",
        "slow": "sluggish",
        "friendly": "welcoming",
    }
    mutant_map = {
        "fantastic": ["incredible", "wonderful", "marvelous"],
        "awful": ["dreadful", "horrible", "appalling"],
        "adore": ["cherish", "appreciate", "treasure"],
        "sluggish": ["lazy", "unhurried", "bad"],
        "welcoming": ["hospitable", "warm", "accommodating"],
    }

    for sentence in test_sentences:
        mutated_sentence = replace_with_synonyms(sentence, synonym_map)
        original_vector = preprocess_input(sentence, load_vectorizer)
        mutated_vector = preprocess_input(mutated_sentence, load_vectorizer)

        original_pred = load_model.predict(original_vector)[0]
        mutated_pred = load_model.predict(mutated_vector)[0]

        if original_pred != mutated_pred:
            repaired_sentence = try_repair(
                mutated_sentence, original_pred, mutant_map, load_model, load_vectorizer
            )
            assert repaired_sentence is not None, (
                f"Inconsistency found between original and mutated "
                f"predictions for: {mutated_sentence}"
            )


def test_noise_robustness_text(load_vectorizer, load_model):
    """
    Test model robustness against character-level noise: allow a small number of mismatches.
    """
    test_sentences = [
        "This restaurant is amazing",
        "The food was terrible",
        "I love the ambiance",
        "Service was slow",
        "The staff was friendly",
    ]

    def add_typo_noise(sentence, noise_level=0.15):

        chars = list(sentence)
        n_noisy = int(len(chars) * noise_level)
        for _ in range(n_noisy):
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
        return "".join(chars)

    mismatches = 0

    for sentence in test_sentences:
        noisy_sentence = add_typo_noise(sentence)
        orig_vec = preprocess_input(sentence, load_vectorizer)
        noisy_vec = preprocess_input(noisy_sentence, load_vectorizer)
        orig_pred = load_model.predict(orig_vec)[0]
        noisy_pred = load_model.predict(noisy_vec)[0]
        if orig_pred != noisy_pred:
            print(f"Mismatch: '{sentence}' -> '{noisy_sentence}'")
            mismatches += 1

    # Allow up to 40% mismatch (2 out of 5)
    assert (
        mismatches <= 2
    ), f"Too many prediction changes under noise: {mismatches} mismatches"


def test_nonfunctional_performance_text(load_vectorizer, load_model):
    """
    Test inference time and memory usage (non-functional requirements) for text input.
    """
    test_sentences = [
        "This restaurant is amazing",
        "The food was terrible",
        "I love the ambiance",
        "Service was slow",
        "The staff was friendly",
    ] * 100  # Increase for a more substantial test
    X = [preprocess_input(sentence, load_vectorizer)[0] for sentence in test_sentences]
    X = np.array(X)
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # in MB
    start_time = time.time()
    load_model.predict(X)
    inference_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 / 1024  # in MB
    memory_increase = final_memory - initial_memory
    assert inference_time < 5, f"Inference time ({inference_time}s) exceeds 5 seconds"
    assert memory_increase < 1024, f"Memory increase ({memory_increase}MB) exceeds 1GB"
