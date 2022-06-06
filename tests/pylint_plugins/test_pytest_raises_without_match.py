import pytest

from tests.pylint_plugins.utils import create_message, extract_node, skip_if_pylint_unavailable

pytestmark = skip_if_pylint_unavailable()


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
