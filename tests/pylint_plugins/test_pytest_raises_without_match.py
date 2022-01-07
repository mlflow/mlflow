import pytest

from tests.helper_functions import _is_importable

pytestmark = pytest.mark.skipif(
    not _is_importable("pylint"), reason="pylint is required to run tests in this module"
)


@pytest.fixture(scope="module")
def test_case():
    # Ref: https://pylint.pycqa.org/en/latest/how_tos/custom_checkers.html#testing-a-checker
    import pylint.testutils
    from pylint_plugins import PytestRaisesWithoutMatch

    class TestPytestRaisesWithoutMatch(pylint.testutils.CheckerTestCase):
        CHECKER_CLASS = PytestRaisesWithoutMatch

    test_case = TestPytestRaisesWithoutMatch()
    test_case.setup_method()
    return test_case


def create_message(msg_id, node):
    import pylint.testutils

    return pylint.testutils.Message(msg_id=msg_id, node=node)


def extract_node(code):
    import astroid

    return astroid.extract_node(code)


def iter_bad_cases():
    # Single context manager
    root_node = extract_node(
        """
with pytest.raises(Exception):
    raise Exception("failed")
"""
    )
    yield root_node, root_node.items[0][0]

    # Multiple context managers
    root_node = extract_node(
        """
with context_manager, pytest.raises(Exception):
    raise Exception("failed")
"""
    )
    yield root_node, root_node.items[1][0]

    # Without `with`
    root_node = extract_node(
        """
pytest.raises(Exception)
"""
    )
    yield root_node, root_node


def test_bad_cases(test_case):
    for root_node, error_node in iter_bad_cases():
        with test_case.assertAddsMessages(create_message(test_case.CHECKER_CLASS.name, error_node)):
            test_case.walk(root_node)


def iter_good_cases():
    # Single context manager
    yield extract_node(
        """
with pytest.raises(Exception, match="failed"):
    raise Exception("failed")
"""
    )

    # Multiple context managers
    yield extract_node(
        """
with context_manager, pytest.raises(Exception, match="failed"):
    raise Exception("failed")
"""
    )

    # Without `with`
    yield extract_node(
        """
pytest.raises(Exception, match="failed")
"""
    )


def test_good_cases(test_case):
    for root_node in iter_good_cases():
        with test_case.assertNoMessages():
            test_case.walk(root_node)
