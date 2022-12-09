import pytest
import astroid
from pylint.testutils import MessageTest, CheckerTestCase
from pylint_plugins import StringChecker, errors


@pytest.fixture(scope="module")
def test_case():
    class TestUnittestAssertRaises(CheckerTestCase):
        CHECKER_CLASS = StringChecker

    test_case = TestUnittestAssertRaises()
    test_case.setup_method()
    return test_case


def test_use_f_string(test_case):
    node = astroid.extract_node('"{} {}".format(a, b)')
    with test_case.assertAddsMessages(
        MessageTest(errors.USE_F_STRING.name, node=node, line=1, col_offset=0)
    ):
        test_case.walk(node)

    node = astroid.extract_node('"{a} {b}".format(a=a, b=b)')
    with test_case.assertAddsMessages(
        MessageTest(errors.USE_F_STRING.name, node=node, line=1, col_offset=0)
    ):
        test_case.walk(node)

    node = astroid.extract_node('"{} {} {}".format(*a)')
    with test_case.assertNoMessages():
        test_case.walk(node)

    node = astroid.extract_node('"{a} {b}".format(**a)')
    with test_case.assertNoMessages():
        test_case.walk(node)

    node = astroid.extract_node('"{}".format(func())')
    with test_case.assertNoMessages():
        test_case.walk(node)

    node = astroid.extract_node('"{}".format(a.b)')
    with test_case.assertNoMessages():
        test_case.walk(node)
