import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker


def _is_unittest_assert_raises(node: astroid.Call):
    return isinstance(node.func, astroid.Attribute) and node.func.as_string() == "self.assertRaises"


class UnittestAssertRaises(BaseChecker):
    __implements__ = IAstroidChecker

    name = "unittest-assert-raises"
    msgs = {
        "W0003": (
            "`assertRaises` must be replaced with `assertRaisesRegex`",
            name,
            "Use `assertRaisesRegex` instead",
        ),
    }
    priority = -1

    def visit_call(self, node: astroid.Call):
        if _is_unittest_assert_raises(node):
            self.add_message(self.name, node=node)
