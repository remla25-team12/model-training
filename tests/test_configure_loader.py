import pytest

from src.configure_loader import load_config


def test_load_config():
    """Test if configuration can be loaded successfully"""
    config = load_config()
    assert config is not None
    assert isinstance(config, dict)

    # Test required configuration sections
    assert "dataset" in config
    assert "model" in config
    assert "training" in config


def test_config_structure():
    """Test if configuration has the required structure"""
    config = load_config()

    # Test dataset section
    assert "raw" in config["dataset"]
    assert "preprocessed" in config["dataset"]
    assert "url" in config["dataset"]["raw"]
    assert "output_dir" in config["dataset"]["raw"]

    # Test model section
    assert "classifier" in config["model"]
    assert "model_dir" in config["model"]["classifier"]

    # Test training section
    assert "test_size" in config["training"]
    assert "random_state" in config["training"]
