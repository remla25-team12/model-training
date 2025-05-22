'''
Register the NaNComparisonChecker with pylint
'''
from .nan_comparison_checker import NaNComparisonChecker

def register(linter):
    '''
    Register the NaNComparisonChecker with pylint
    '''
    linter.register_checker(NaNComparisonChecker(linter))
