import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker

from .errors import USE_SET_LITERAL, to_msgs


class SetChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "set-checker"
    msgs = to_msgs(USE_SET_LITERAL)
    priority = -1

    def visit_call(self, node: astroid.Call):
        if (
            node.func.as_string() == "set"
            and node.args
            and isinstance(node.args[0], (astroid.List, astroid.Tuple))
        ):
            self.add_message(USE_SET_LITERAL.name, node=node)
