import pytest
import astroid
from pylint.testutils import MessageTest
from pylint.testutils import CheckerTestCase
from pylint_plugins import UnittestAssertRaises


@pytest.fixture(scope="module")
def test_case():
    class TestUnittestAssertRaises(CheckerTestCase):
        CHECKER_CLASS = UnittestAssertRaises

    test_case = TestUnittestAssertRaises()
    test_case.setup_method()
    return test_case


def test_unittest_assert_raises(test_case):
    node = astroid.extract_node("self.assertRaises(Exception)")
    with test_case.assertAddsMessages(
        MessageTest(test_case.CHECKER_CLASS.name, node=node, line=1, col_offset=0)
    ):
        test_case.walk(node)

    node = astroid.extract_node("self.assertRaisesRegex(Exception, 'error message')")
    with test_case.assertAddsMessages(
        MessageTest(test_case.CHECKER_CLASS.name, node=node, line=1, col_offset=0)
    ):
        test_case.walk(node)

    node = astroid.extract_node("pytest.raises(Exception, 'error message')")
    with test_case.assertNoMessages():
        test_case.walk(node)
