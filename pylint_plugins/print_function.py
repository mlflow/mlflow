import os

import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker


IGNORE_DIRS = list(map(os.path.abspath, ["dev"]))


def _is_print_function(node: astroid.Call):
    return isinstance(node.func, astroid.Name) and node.func.name == "print"


def _should_ignore(node: astroid.Call):
    path = node.root().file
    return any(path.startswith(d) for d in IGNORE_DIRS)


class PrintFunction(BaseChecker):
    __implements__ = IAstroidChecker

    name = "print-function"
    msgs = {
        "W0002": (
            "print function should not be used",
            name,
            "Use logging methods instead",
        ),
    }
    priority = -1

    def visit_call(self, node: astroid.Call):
        if not _should_ignore(node) and _is_print_function(node):
            self.add_message(self.name, node=node)
