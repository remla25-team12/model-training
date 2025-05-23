"""Module for loading YAML configuration."""

import yaml


def load_config(path="./config.yaml"):
    """
    Function to load a YAML config file from the given path.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        dict: Parsed YAML contents
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
