"""
Downloads the training dataset 
"""

import os
import requests
from configure_loader import load_config

def download_data():
    """
    Main function that downloads the training dataset 
    """
    config = load_config()
    url = config["dataset"]["raw"]["url"]
    output_dir = config["dataset"]["raw"]["output_dir"]
    output_filename = config["dataset"]["raw"]["output_filename"]

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_filename)

    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
            print("Dataset downloaded successfully.")
    else:
        print(f"Failed to download the dataset with status code: {response.status_code}")


if __name__ == "__main__":
    download_data()
