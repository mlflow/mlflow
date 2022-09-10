import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker


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
    WITHOUT_MATCH = "pytest-raises-without-match"
    MULTIPLE_STATEMENTS = "pytest-raises-multiple-statements"
    msgs = {
        "W0001": (
            "`pytest.raises` must be called with `match` argument`",
            WITHOUT_MATCH,
            "Use `pytest.raises(<exception>, match=...)`",
        ),
        "W0004": (
            "`pytest.raises` block should not contain multiple statements."
            " It should only contain a single statement that throws an exception.",
            MULTIPLE_STATEMENTS,
            "Any initialization/finalization code should be moved outside of `pytest.raises` block",
        ),
    }
    priority = -1

    def visit_call(self, node: astroid.Call):
        if not _is_pytest_raises_call(node):
            return

        if not _called_with_match(node):
            self.add_message(PytestRaisesChecker.WITHOUT_MATCH, node=node)

    def visit_with(self, node: astroid.With):
        if any(_is_pytest_raises_call(item[0]) for item in node.items) and (
            _contains_multiple_statements(node)
        ):
            self.add_message(PytestRaisesChecker.MULTIPLE_STATEMENTS, node=node)
