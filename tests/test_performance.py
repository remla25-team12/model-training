# tests/test_performance.py
import time
import pytest
import psutil
import numpy as np


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_memory_usage(model, X):
    """Test if model memory usage stays within acceptable bounds"""
    initial_memory = get_memory_usage()

    # Make predictions
    model.predict(X)

    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory

    # Memory increase should be less than 1GB
    assert memory_increase < 1024, f"Memory increase ({memory_increase}MB) exceeds 1GB"


def test_inference_time(model, X):
    """Test if model inference time stays within acceptable bounds"""
    # Measure inference time
    start_time = time.time()
    model.predict(X)
    inference_time = time.time() - start_time

    # Inference should take less than 5 seconds
    assert inference_time < 5, f"Inference time ({inference_time}s) exceeds 5 seconds"


def test_batch_processing(model, X):
    """Test if model can handle different batch sizes efficiently"""
    batch_sizes = [1, 10, 100]
    times = []

    for batch_size in batch_sizes:
        start_time = time.time()
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            model.predict(batch)
        times.append(time.time() - start_time)

    # Check if larger batches are more efficient
    assert times[0] > times[1] > times[2], "Larger batches should be more efficient"
