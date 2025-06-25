import os
import shutil
from unittest.mock import Mock, patch

import pytest

from src.configure_loader import load_config
from src.get_data import download_data


@pytest.fixture
def config():
    """Load test configuration"""
    return load_config()


def test_download_data_creates_directory(config):
    """Test if download_data creates the output directory"""
    output_dir = config["dataset"]["raw"]["output_dir"]
    # Remove the directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    assert not os.path.exists(output_dir), "Directory should not exist before test"

    with patch("requests.get") as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test data"
        mock_get.return_value = mock_response

        download_data(config)

        assert os.path.exists(output_dir), "Directory should be created"
        assert os.path.isdir(output_dir), "Should be a directory"


def test_download_data_saves_file(config):
    """Test if download_data saves the file correctly"""
    output_dir = config["dataset"]["raw"]["output_dir"]
    output_file = config["dataset"]["raw"]["output_filename"]
    file_path = os.path.join(output_dir, output_file)

    with patch("requests.get") as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test,data\n1,2\n3,4"
        mock_get.return_value = mock_response

        download_data(config)

        assert os.path.exists(file_path), "File should exist"
        assert os.path.getsize(file_path) > 0, "File should not be empty"

        # Verify content
        with open(file_path, "rb") as f:
            content = f.read()
            assert content == b"test,data\n1,2\n3,4", "File content should match"
