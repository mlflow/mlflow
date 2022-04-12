import pytest

from tests.pylint_plugins.utils import create_message, extract_node, skip_if_pylint_unavailable

pytestmark = skip_if_pylint_unavailable()


@pytest.fixture(scope="module")
def test_case():
    import pylint.testutils
    from pylint_plugins import AssertRaisesWithoutMsg

    class TestAssertRaisesWithoutMsg(pylint.testutils.CheckerTestCase):
        CHECKER_CLASS = AssertRaisesWithoutMsg

    test_case = TestAssertRaisesWithoutMsg()
    test_case.setup_method()
    return test_case


def test_assert_raises_without_msg(test_case):
    node = extract_node("self.assertRaises(Exception)")
    with test_case.assertAddsMessages(create_message(test_case.CHECKER_CLASS.name, node)):
        test_case.walk(node)

    node = extract_node("self.assertRaises(Exception, msg='test')")
    print(node)
    with test_case.assertNoMessages():
        test_case.walk(node)

    node = extract_node("pandas.assertRaises(Exception)")
    with test_case.assertNoMessages():
        test_case.walk(node)
