import pytest
import astroid
from pylint.testutils import MessageTest
from pylint.testutils import CheckerTestCase

from pylint_plugins import PrintFunction


@pytest.fixture(scope="module")
def test_case():
    class TestPrintFunction(CheckerTestCase):
        CHECKER_CLASS = PrintFunction

    test_case = TestPrintFunction()
    test_case.setup_method()
    return test_case


def test_print_function(test_case):
    node = astroid.extract_node("print('hello')")
    with test_case.assertAddsMessages(
        MessageTest(test_case.CHECKER_CLASS.name, node=node, line=1, col_offset=0)
    ):
        test_case.walk(node)

    node = astroid.extract_node("module.print('hello')")
    with test_case.assertNoMessages():
        test_case.walk(node)
