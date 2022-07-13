import pytest

from tests.pylint_plugins.utils import create_message, extract_node, skip_if_pylint_unavailable

pytestmark = skip_if_pylint_unavailable()


@pytest.fixture(scope="module")
def test_case():
    import pylint.testutils
    from pylint_plugins import PrintFunction

    class TestPrintFunction(pylint.testutils.CheckerTestCase):
        CHECKER_CLASS = PrintFunction

    test_case = TestPrintFunction()
    test_case.setup_method()
    return test_case


def test_print_function(test_case):
    node = extract_node("print('hello')")
    with test_case.assertAddsMessages(create_message(test_case.CHECKER_CLASS.name, node)):
        test_case.walk(node)

    node = extract_node("module.print('hello')")
    with test_case.assertNoMessages():
        test_case.walk(node)
