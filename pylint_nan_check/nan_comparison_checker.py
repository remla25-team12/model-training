'''
Custom Pylint checker to check for NaN Equivalence Comparison Misused.
'''
from pylint.checkers import BaseChecker
import astroid

class NaNComparisonChecker(BaseChecker):
    '''
    Custom NaNComparisonChecker to check for NaN equivalence comparison misuse.
    '''
    name = "nan-comparison-checker"
    msgs = {
        "W5501": (
            "NaN compared using '=='. Use numpy.isnan() instead.",
            "nan-comparison-pd-np",
            "Used '== np.nan', which always returns False. Use np.isnan(x) instead.",
        )
    }

    def visit_compare(self, node):
        '''
        Checks for comparisons using '== np.nan' or '!= np.nan' and reports a warning,
        as these comparisons always return False. Suggests using np.isnan() instead.
        '''
        for side in (node.left, node.ops[0][1]):
            if isinstance(side, astroid.Attribute) \
                    and side.attrname == "nan" \
                    and side.expr.name == "np":
                self.add_message("nan-comparison-pd-np", node=node)
                return
