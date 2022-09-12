import pytest
import astroid
from pylint.testutils import MessageTest
from pylint.testutils import CheckerTestCase

from pylint_plugins import SetChecker


@pytest.fixture(scope="module")
def test_case():
    class TestSetChecker(CheckerTestCase):
        CHECKER_CLASS = SetChecker

    test_case = TestSetChecker()
    test_case.setup_method()
    return test_case


def test_use_set_literal(test_case):
    node = astroid.extract_node("set(['a', 'b'])")
    with test_case.assertAddsMessages(
        MessageTest("use-set-literal", node=node, line=1, col_offset=0)
    ):
        test_case.walk(node)

    node = astroid.extract_node("set(('a', 'b'))")
    with test_case.assertAddsMessages(
        MessageTest("use-set-literal", node=node, line=1, col_offset=0)
    ):
        test_case.walk(node)

    node = astroid.extract_node("set([x for x in range(3)])")
    with test_case.assertNoMessages():
        test_case.walk(node)

    node = astroid.extract_node("set(numbers)")
    with test_case.assertNoMessages():
        test_case.walk(node)
