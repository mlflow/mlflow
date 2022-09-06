import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker

from ..errors import PYTEST_RAISES_WITHOUT_MATCH, PYTEST_RAISES_MULTIPLE_STATEMENTS, to_msgs


def _is_pytest_raises_call(node: astroid.NodeNG):
    if not isinstance(node, astroid.Call):
        return False
    if not isinstance(node.func, astroid.Attribute) or not isinstance(node.func.expr, astroid.Name):
        return False
    return node.func.expr.name == "pytest" and node.func.attrname == "raises"


def _called_with_match(node: astroid.Call):
    # Note `match` is a keyword-only argument:
    # https://docs.pytest.org/en/latest/reference/reference.html#pytest.raises
    return any(k.arg == "match" for k in node.keywords)


def _contains_multiple_statements(raises_with: astroid.With):
    return len(raises_with.body) > 1


class PytestRaisesChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "pytest-raises-checker"
    msgs = to_msgs(PYTEST_RAISES_WITHOUT_MATCH, PYTEST_RAISES_MULTIPLE_STATEMENTS)
    priority = -1

    def visit_call(self, node: astroid.Call):
        if not _is_pytest_raises_call(node):
            return

        if not _called_with_match(node):
            self.add_message(PYTEST_RAISES_WITHOUT_MATCH.name, node=node)

    def visit_with(self, node: astroid.With):
        if any(_is_pytest_raises_call(item[0]) for item in node.items) and (
            _contains_multiple_statements(node)
        ):
            self.add_message(PYTEST_RAISES_MULTIPLE_STATEMENTS.name, node=node)
