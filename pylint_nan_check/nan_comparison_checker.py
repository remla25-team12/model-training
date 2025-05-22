from pylint.checkers import BaseChecker
import astroid

class NaNComparisonChecker(BaseChecker):

    name = "nan-comparison-checker"
    msgs = {
        "W5501": (
            "NaN compared using '=='. Use numpy.isnan() instead.",
            "nan-comparison-pd-np",
            "Used '== np.nan', which always returns False. Use np.isnan(x) instead.",
        )
    }

    def visit_compare(self, node):
        for side in (node.left, node.ops[0][1]):
            if isinstance(side, astroid.Attribute) \
                    and side.attrname == "nan" \
                    and side.expr.name == "np":
                self.add_message("nan-comparison-pd-np", node=node)
                return
