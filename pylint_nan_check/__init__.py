from .nan_comparison_checker import NaNComparisonChecker

def register(linter):
    linter.register_checker(NaNComparisonChecker(linter))