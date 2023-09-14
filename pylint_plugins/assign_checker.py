from typing import List

import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

from pylint_plugins.errors import USELESS_ASSIGNMENT, to_msgs


class AssignChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "assign-checker"
    msgs = to_msgs(USELESS_ASSIGNMENT)
    priority = -1

    def __init__(self, linter):
        super().__init__(linter)
        self.non_local = set()
        self.funcs = []

    def visit_global(self, node: astroid.Global):
        # Assignment to a global variable is not useless
        self.non_local.update(node.names)

    def visit_nonlocal(self, node: astroid.Nonlocal):
        # Assignment to a nonlocal variable is not useless
        self.non_local.update(node.names)

    def visit_functiondef(self, node: astroid.FunctionDef):
        self.funcs.append(node)
        self.useless_assign(node.body)

    def leave_functiondef(self, node: astroid.Module):
        self.funcs.pop()
        if not self.funcs:
            self.non_local.clear()

    def visit_with(self, node: astroid.With):
        self.useless_assign(node.body)

    def visit_if(self, node: astroid.If):
        self.useless_assign(node.body)
        self.useless_assign(node.orelse)

    def visit_while(self, node: astroid.While):
        self.useless_assign(node.body)
        self.useless_assign(node.orelse)

    def visit_for(self, node: astroid.For):
        self.useless_assign(node.body)
        self.useless_assign(node.orelse)

    def visit_tryexcept(self, node: astroid.TryExcept):
        self.useless_assign(node.body)
        for handler in node.handlers:
            self.useless_assign(handler.body)
        self.useless_assign(node.orelse)

    def useless_assign(self, body: List[astroid.NodeNG]):
        """
        ```
        def f():
            ...
            a = 1
            ^^^^^ Find useless assignment like this
            return a
        ```
        """
        if len(body) < 2:
            return

        second_last, last = body[-2:]
        # Is the last statement a return statement?
        if not isinstance(last, astroid.Return):
            return

        # Does the return statement have a value (e.g. `return a`)?
        if not last.value:
            return

        # Is the value a name (e.g. `return a`)?
        if not isinstance(last.value, astroid.Name):
            return

        # Is the second last statement an assignment?
        second_last = body[-2]
        if not isinstance(second_last, astroid.Assign):
            return

        # Does the assignment have a single target (e.g. `a = 1`)?
        if len(second_last.targets) != 1:
            return

        # Is the target a name (e.g. `a = 1`)?
        target = second_last.targets[0]
        if not isinstance(target, astroid.AssignName):
            return

        if target.name in self.non_local:
            return

        # Is the name in the assignment the same as the name in the return statement?
        if target.name != last.value.name:
            return

        self.add_message(USELESS_ASSIGNMENT.name, node=second_last)
