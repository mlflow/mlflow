import pytest

from tests.pylint_plugins.utils import create_message, extract_node, skip_if_pylint_unavailable

pytestmark = skip_if_pylint_unavailable()


@pytest.fixture(scope="module")
def test_case():
    import pylint.testutils
    from pylint_plugins import UnittestAssertRaises

    class TestUnittestAssertRaises(pylint.testutils.CheckerTestCase):
        CHECKER_CLASS = UnittestAssertRaises

    test_case = TestUnittestAssertRaises()
    test_case.setup_method()
    return test_case


def test_unittest_assert_raises(test_case):
    node = extract_node("self.assertRaises(Exception)")
    with test_case.assertAddsMessages(create_message(test_case.CHECKER_CLASS.name, node)):
        test_case.walk(node)

    node = extract_node("self.assertRaisesRegex(Exception, 'error message')")
    with test_case.assertNoMessages():
        test_case.walk(node)

    node = extract_node("pandas.assertRaises(Exception)")
    with test_case.assertNoMessages():
        test_case.walk(node)
