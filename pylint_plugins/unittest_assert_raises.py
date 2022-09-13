import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker

from .errors import UNITTEST_PYTEST_RAISES, to_msgs


def _is_unittest_assert_raises(node: astroid.Call):
    return isinstance(node.func, astroid.Attribute) and (
        node.func.as_string() in ("self.assertRaises", "self.assertRaisesRegex")
    )


class UnittestAssertRaises(BaseChecker):
    __implements__ = IAstroidChecker

    name = "unittest-assert-raises"
    msgs = to_msgs(UNITTEST_PYTEST_RAISES)
    priority = -1

    def visit_call(self, node: astroid.Call):
        if _is_unittest_assert_raises(node):
            self.add_message(UNITTEST_PYTEST_RAISES.name, node=node)
