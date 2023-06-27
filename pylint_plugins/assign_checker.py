import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker

from .errors import USELESS_ASSIGNMENT, to_msgs


class AssignChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "assign-checker"
    msgs = to_msgs(USELESS_ASSIGNMENT)
    priority = -1

    def visit_functiondef(self, node: astroid.FunctionDef):
        """
        ```
        def f():
            a = 1
            ^^^^^ Find useless assignment like this
            return a
        ```
        """
        body = [s for s in node.body if not isinstance(s, (astroid.Import, astroid.ImportFrom))]
        if len(body) != 2:
            return

        second_last, last = body
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
        second_last = node.body[-2]
        if not isinstance(second_last, astroid.Assign):
            return

        # Does the assignment have a single target (e.g. `a = 1`)?
        if len(second_last.targets) != 1:
            return

        # Is the target a name (e.g. `a = 1`)?
        target = second_last.targets[0]
        if not isinstance(target, astroid.AssignName):
            return

        # Is the name in the assignment the same as the name in the return statement?
        if target.name != last.value.name:
            return

        self.add_message(USELESS_ASSIGNMENT.name, node=second_last)
