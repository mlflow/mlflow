import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker

from .errors import USE_F_STRING, to_msgs


class StringChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "string-checker"
    msgs = to_msgs(USE_F_STRING)
    priority = -1

    def visit_call(self, node: astroid.Call):
        if (
            isinstance(node.func, astroid.Attribute)
            and node.func.attrname == "format"
            and isinstance(node.func.expr, astroid.Const)
            and isinstance(node.func.expr.value, str)
        ):
            if node.kwargs or node.starargs:
                return

            if all(isinstance(a, astroid.Name) for a in node.args) and all(
                isinstance(kw.value, astroid.Name) for kw in node.keywords
            ):
                self.add_message(USE_F_STRING.name, node=node)
