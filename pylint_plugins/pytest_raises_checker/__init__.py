import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter


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


class PytestRaisesChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "pytest-raises-checker"
    WITHOUT_MATCH = "pytest-raises-without-match"
    CONTAINS_ASSERTIONS = "pytest-raises-contains-assertions"
    msgs = {
        "W0001": (
            "`pytest.raises` must be called with `match` argument`",
            WITHOUT_MATCH,
            "Use `pytest.raises(<exception>, match=...)`",
        ),
        "W0004": (
            (
                "`pytest.raises` should not contain assertion."
                " See https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-raises"
                " for more information."
            ),
            CONTAINS_ASSERTIONS,
            "Remove assertions from `pytest.raises`",
        ),
    }
    priority = -1

    def __init__(self, linter: PyLinter) -> None:
        super().__init__(linter)
        self._is_in_pytest_raises = False

    def visit_call(self, node: astroid.Call):
        if not _is_pytest_raises_call(node):
            return

        if not _called_with_match(node):
            self.add_message(PytestRaisesChecker.WITHOUT_MATCH, node=node)

    def visit_assert(self, node: astroid.Assert):
        if self._is_in_pytest_raises:
            self.add_message(PytestRaisesChecker.CONTAINS_ASSERTIONS, node=node)

    def visit_with(self, node: astroid.With):
        if any(_is_pytest_raises_call(item[0]) for item in node.items):
            self._is_in_pytest_raises = True

    def leave_with(self, node: astroid.With):
        if any(_is_pytest_raises_call(item[0]) for item in node.items):
            self._is_in_pytest_raises = False
