"""
Downloads the training dataset
"""

import os

import requests

from src.configure_loader import load_config


def download_data(config=None):
    """
    Main function that downloads the training dataset

    Args:
        config (dict, optional): Configuration dictionary. If None, loads from file.
    """
    if config is None:
        config = load_config()

    url = config["dataset"]["raw"]["url"]
    output_dir = config["dataset"]["raw"]["output_dir"]
    output_filename = config["dataset"]["raw"]["output_filename"]

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_filename)

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
                print("Dataset downloaded successfully.")
        else:
            print(
                f"Failed to download the dataset with status code: {response.status_code}"
            )
    except requests.exceptions.Timeout:
        print("Request timed out while downloading the dataset")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")


if __name__ == "__main__":
    download_data()
