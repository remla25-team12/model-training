import os
import shutil
from unittest.mock import Mock, patch

import pytest
import requests

from src.configure_loader import load_config
from src.get_data import download_data


@pytest.fixture
def config():
    """Load test configuration"""
    return load_config()


@pytest.fixture
def cleanup_data_dir():
    """Fixture to clean up test data directory before and after tests"""
    # Setup: Remove the directory if it exists
    test_dir = "./data/initial_raw"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    yield  # Run the test

    # Teardown: Clean up after test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def test_download_data_creates_directory(config, cleanup_data_dir):
    """Test if download_data creates the output directory"""
    output_dir = config["dataset"]["raw"]["output_dir"]
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


def test_download_data_saves_file(config, cleanup_data_dir):
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


def test_download_data_handles_error(config, cleanup_data_dir):
    """Test if download_data handles HTTP errors correctly"""
    with patch("requests.get") as mock_get:
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        download_data(config)  # Should not raise exception

        # Check that the function handled the error (didn't create file)
        output_dir = config["dataset"]["raw"]["output_dir"]
        output_file = config["dataset"]["raw"]["output_filename"]
        file_path = os.path.join(output_dir, output_file)
        assert not os.path.exists(file_path), "File should not exist on error"


def test_download_data_uses_config_params(config, cleanup_data_dir):
    """Test if download_data uses the correct config parameters"""
    with patch("requests.get") as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test data"
        mock_get.return_value = mock_response

        download_data(config)

        # Verify the URL from config was used
        mock_get.assert_called_once_with(config["dataset"]["raw"]["url"], timeout=30)


def test_download_data_timeout(config, cleanup_data_dir):
    """Test if download_data handles timeout correctly"""
    with patch("requests.get") as mock_get:
        # Mock timeout
        mock_get.side_effect = requests.exceptions.Timeout

        download_data(config)  # Should not raise exception

        # Check that the function handled the timeout (didn't create file)
        output_dir = config["dataset"]["raw"]["output_dir"]
        output_file = config["dataset"]["raw"]["output_filename"]
        file_path = os.path.join(output_dir, output_file)
        assert not os.path.exists(file_path), "File should not exist on timeout"


def test_download_data_request_exception(config, cleanup_data_dir):
    """Test if download_data handles general request exceptions"""
    with patch("requests.get") as mock_get:
        # Mock general request exception
        mock_get.side_effect = requests.exceptions.RequestException("Test error")

        download_data(config)  # Should not raise exception

        # Check that the function handled the error (didn't create file)
        output_dir = config["dataset"]["raw"]["output_dir"]
        output_file = config["dataset"]["raw"]["output_filename"]
        file_path = os.path.join(output_dir, output_file)
        assert not os.path.exists(
            file_path
        ), "File should not exist on request exception"


def test_download_data_no_config():
    """Test if download_data works without config parameter"""
    with patch("requests.get") as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test data"
        mock_get.return_value = mock_response

        download_data()  # Should use default config loading


def test_data_format(config, cleanup_data_dir):
    """Test if downloaded data has correct format"""
    output_dir = config["dataset"]["raw"]["output_dir"]
    output_file = config["dataset"]["raw"]["output_filename"]
    file_path = os.path.join(output_dir, output_file)

    # Create mock TSV data
    mock_tsv_data = "Review\tSentiment\nGreat food!\t1\nTerrible service\t0"

    with patch("requests.get") as mock_get:
        # Mock successful response with TSV data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_tsv_data.encode("utf-8")
        mock_get.return_value = mock_response

        download_data(config)

        # Read and verify format
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            assert "\t" in first_line, "File is not in TSV format"
            assert first_line.count("\t") == 1, "Should have exactly one tab separator"
            assert first_line == "Review\tSentiment", "Header format incorrect"
