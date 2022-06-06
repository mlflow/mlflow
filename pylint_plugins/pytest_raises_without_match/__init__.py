import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker


class PytestRaisesWithoutMatch(BaseChecker):
    __implements__ = IAstroidChecker

    name = "pytest-raises-without-match"
    msgs = {
        "W0001": (
            "`pytest.raises` must be called with `match` argument` ",
            name,
            "Use `pytest.raises(<exception>, match=...)`",
        ),
    }
    priority = -1

    @staticmethod
    def _is_pytest_raises_call(node: astroid.Call):
        if not isinstance(node.func, astroid.Attribute) or not isinstance(
            node.func.expr, astroid.Name
        ):
            return False
        return node.func.expr.name == "pytest" and node.func.attrname == "raises"

    @staticmethod
    def _called_with_match(node: astroid.Call):
        # Note `match` is a keyword-only argument:
        # https://docs.pytest.org/en/latest/reference/reference.html#pytest.raises
        return any(k.arg == "match" for k in node.keywords)

    def visit_call(self, node: astroid.Call):
        if not PytestRaisesWithoutMatch._is_pytest_raises_call(node):
            return

        if not PytestRaisesWithoutMatch._called_with_match(node):
            self.add_message(self.name, node=node)
