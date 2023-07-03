import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker

from .errors import USE_F_STRING, to_msgs


class StringChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "string-checker"
    msgs = to_msgs(USE_F_STRING)
    priority = -1

    def _is_simple_expression(node: astroid.NodeNG):
        """
        Returns True if the node is a Name or an Attribute containing only names.

        Examples
        - `a`
        - `a.b`
        - `a.b.c`
        """
        return isinstance(node, astroid.Name) or (
            isinstance(node, astroid.Attribute) and StringChecker._is_simple_expression(node.expr)
        )

    def visit_call(self, node: astroid.Call):
        if (
            isinstance(node.func, astroid.Attribute)
            and node.func.attrname == "format"
            and isinstance(node.func.expr, astroid.Const)
            and isinstance(node.func.expr.value, str)
        ):
            if node.kwargs or node.starargs:
                return

            if all(StringChecker._is_simple_expression(arg) for arg in node.args) and (
                all(StringChecker._is_simple_expression(kwarg.value) for kwarg in node.keywords)
            ):
                self.add_message(USE_F_STRING.name, node=node)
