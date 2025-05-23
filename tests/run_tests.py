import sys
import time
import json
from collections import defaultdict
from pathlib import Path
import pytest
import psutil
import numpy as np
from coverage import Coverage


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class MLTestScore:
    """ML Test Score calculator and reporter"""

    CATEGORIES = {
        # Feature and Data Tests
        "data": {
            "name": "Data Quality & Features",
            "files": [
                "test_data_quality.py",
                "test_feature_engineering.py",
                "test_get_data.py",
                "test_preprocess_data.py",
            ],
            "weight": 0.25,
        },
        # Model Development Tests
        "model": {
            "name": "Model Development",
            "files": ["test_train_model.py", "test_evaluate.py"],
            "weight": 0.25,
        },
        # ML Infrastructure Tests
        "infrastructure": {
            "name": "ML Infrastructure",
            "files": ["test_utils.py", "test_configure_loader.py"],
            "weight": 0.2,
        },
        # Monitoring Tests
        "monitoring": {
            "name": "Monitoring & Performance",
            "files": [
                "test_monitoring.py",
                "test_performance.py",
                "test_robustness.py",
            ],
            "weight": 0.3,
        },
    }

    def __init__(self):
        self.results = defaultdict(
            lambda: {
                "total_tests": 0,
                "passed_tests": 0,
                "coverage": 0.0,
                "score": 0.0,
                "performance": {},
            }
        )
        self.cov = Coverage()

    def run_tests(self):
        """Run all tests and collect results"""
        # Start coverage measurement
        self.cov.start()

        start_time = time.time()
        initial_memory = get_memory_usage()

        # Run tests with pytest
        pytest_args = ["--verbose", "--cov=src", "--cov-report=term-missing", "tests/"]
        exit_code = pytest.main(pytest_args)

        # Collect performance metrics
        end_time = time.time()
        final_memory = get_memory_usage()

        # Stop coverage measurement
        self.cov.stop()
        self.cov.save()

        # Calculate metrics for each category
        for category, info in self.CATEGORIES.items():
            category_results = self.results[category]

            # Calculate test results
            test_files = [f for f in info["files"] if Path("tests/" + f).exists()]
            category_results["total_tests"] = len(test_files)
            category_results["passed_tests"] = len(test_files) if exit_code == 0 else 0

            # Add performance metrics
            category_results["performance"] = {
                "execution_time": end_time - start_time,
                "memory_usage": final_memory - initial_memory,
            }

            # Calculate coverage
            category_results["coverage"] = self.cov.report(include="src/*")

            # Calculate final score
            if category_results["total_tests"] > 0:
                test_score = (
                    category_results["passed_tests"] / category_results["total_tests"]
                )
                coverage_score = category_results["coverage"] / 100.0
                category_results["score"] = (
                    (test_score * 0.6) + (coverage_score * 0.4)
                ) * info["weight"]

        return exit_code

    def print_report(self):
        """Print detailed test adequacy report"""
        print("\nML Test Score Summary:")
        print("=" * 50)

        total_score = 0
        for category, info in self.CATEGORIES.items():
            results = self.results[category]
            score = results["score"] * 100
            total_score += score

            print(f"\n{info['name']}:")
            print(f"  Score: {score:.1f}%")
            print(f"  Tests: {results['passed_tests']}/{results['total_tests']}")
            print(f"  Coverage: {results['coverage']:.1f}%")
            print("  Performance:")
            print(f"    Time: {results['performance'].get('execution_time', 0):.2f}s")
            print(f"    Memory: {results['performance'].get('memory_usage', 0):.1f}MB")

        print("\n" + "=" * 50)
        print(f"Overall ML Test Score: {total_score:.1f}%")

        # Provide recommendations
        if total_score < 60:
            print("\nRecommendations:")
            for category, info in self.CATEGORIES.items():
                results = self.results[category]
                if results["score"] * 100 < 15:
                    print(f"- Add more tests for {info['name']}")
                if results["coverage"] < 50:
                    print(f"- Improve coverage for {info['name']}")

    def save_report(self, filename="test_adequacy_scores.json"):
        """Save detailed report to JSON file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)


def main():
    """Main function to run tests and generate reports"""
    test_score = MLTestScore()
    exit_code = test_score.run_tests()

    if exit_code == 0:
        test_score.print_report()
        test_score.save_report()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
